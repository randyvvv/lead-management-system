"""LLM provider chain: fallback, caching, and the offline guarantee.

No test here touches the network. The point of the chain is that a missing or
broken provider degrades instead of failing, and these tests pin that.
"""

import tempfile
import unittest
from pathlib import Path

from leadms import llm as llm_module
from leadms.llm import (
    CachedClient,
    ChainClient,
    HeuristicClient,
    LLMUnavailable,
)


class Boom:
    """A provider that always fails, standing in for an expired key."""

    name = "boom"
    is_model_backed = True
    model = "boom-1"

    def __init__(self):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.attempts = 0

    def adjudicate_duplicate(self, a, b, evidence):
        self.attempts += 1
        raise LLMUnavailable("simulated outage")

    def extract_source(self, text, hints=None):
        self.attempts += 1
        raise LLMUnavailable("simulated outage")


class Working:
    name = "working"
    is_model_backed = True
    model = "working-1"

    def __init__(self):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.calls = 0

    def adjudicate_duplicate(self, a, b, evidence):
        self.calls += 1
        return {"verdict": "same", "confidence": 0.9, "reason": "ok"}

    def extract_source(self, text, hints=None):
        self.calls += 1
        return {"channel": "Event", "detail": "x", "confidence": 0.9, "reason": "ok"}


class TestHeuristicClient(unittest.TestCase):
    """The offline stub must be honest about what it is."""

    def setUp(self):
        self.client = HeuristicClient()

    def test_it_does_not_claim_to_be_a_model(self):
        self.assertFalse(self.client.is_model_backed)

    def test_decides_only_on_hard_identifiers(self):
        verdict = self.client.adjudicate_duplicate({}, {}, {"phone_exact": 1.0})
        self.assertEqual(verdict["verdict"], "same")

        verdict = self.client.adjudicate_duplicate({}, {}, {"name_sim": 0.4})
        self.assertEqual(verdict["verdict"], "different")

    def test_abstains_on_the_ambiguous_band(self):
        """Re-deriving an answer from the same features the scorer already
        used would add no information, so it declines."""
        verdict = self.client.adjudicate_duplicate(
            {}, {}, {"phone_exact": 0.0, "email_exact": 0.0, "name_sim": 1.0}
        )
        self.assertEqual(verdict["verdict"], "unsure")
        self.assertEqual(verdict["confidence"], 0.0)

    def test_extraction_falls_back_to_keywords(self):
        self.assertEqual(
            self.client.extract_source("saw them at the expo booth")["channel"], "Event"
        )
        self.assertEqual(
            self.client.extract_source("no signal at all here")["channel"], "Other"
        )


class TestChain(unittest.TestCase):
    def test_falls_through_to_the_next_provider(self):
        broken, working = Boom(), Working()
        chain = ChainClient([broken, working])
        result = chain.adjudicate_duplicate({}, {}, {})
        self.assertEqual(result["verdict"], "same")
        self.assertEqual(result["provider"], "working")
        self.assertEqual(broken.attempts, 1)
        self.assertEqual(working.calls, 1)

    def test_heuristic_is_the_final_safety_net(self):
        chain = ChainClient([Boom(), HeuristicClient()])
        result = chain.adjudicate_duplicate({}, {}, {"phone_exact": 1.0})
        self.assertEqual(result["provider"], "heuristic")

    def test_raises_only_when_every_provider_fails(self):
        chain = ChainClient([Boom(), Boom()])
        with self.assertRaises(LLMUnavailable):
            chain.extract_source("anything")

    def test_a_malformed_response_is_treated_as_a_failure(self):
        class Garbage:
            name = "garbage"
            is_model_backed = True
            usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

            def extract_source(self, text, hints=None):
                return "not a dict"

        chain = ChainClient([Garbage(), Working()])
        self.assertEqual(chain.extract_source("x")["provider"], "working")

    def test_provider_name_is_attached_to_every_result(self):
        chain = ChainClient([Working()])
        self.assertEqual(chain.extract_source("x")["provider"], "working")
        self.assertEqual(chain.extract_source("x")["model"], "working-1")


class TestCache(unittest.TestCase):
    def test_repeated_requests_hit_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Working()
            cached = CachedClient(ChainClient([working]), cache_dir=Path(tmp))
            cached.extract_source("same text")
            cached.extract_source("same text")
            self.assertEqual(working.calls, 1)
            self.assertEqual(cached.hits, 1)
            self.assertEqual(cached.misses, 1)

    def test_different_inputs_are_cached_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Working()
            cached = CachedClient(ChainClient([working]), cache_dir=Path(tmp))
            cached.extract_source("one")
            cached.extract_source("two")
            self.assertEqual(working.calls, 2)

    def test_a_corrupt_cache_entry_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Working()
            cached = CachedClient(ChainClient([working]), cache_dir=Path(tmp))
            cached.extract_source("text")
            for path in Path(tmp).glob("*.json"):
                path.write_text("{ not json", encoding="utf-8")
            cached.extract_source("text")
            self.assertEqual(working.calls, 2)


class TestBuildClient(unittest.TestCase):
    def test_always_returns_something_usable(self):
        """With no API key at all, the service must still answer."""
        client = llm_module.build_client(mode="off", use_cache=False)
        self.assertFalse(client.is_model_backed)
        result = client.extract_source("saw them at the conference booth")
        self.assertEqual(result["provider"], "heuristic")

    def test_describe_reports_the_chain(self):
        described = llm_module.describe_client(
            llm_module.build_client(mode="off", use_cache=False)
        )
        self.assertEqual(described["providers"], ["heuristic"])
        self.assertFalse(described["model_backed"])


class TestSchemaTranslation(unittest.TestCase):
    def test_json_schema_maps_onto_the_gemini_dialect(self):
        converted = llm_module._to_gemini_schema(llm_module.DEDUPE_SCHEMA)
        self.assertEqual(converted["type"], "OBJECT")
        self.assertEqual(converted["properties"]["verdict"]["type"], "STRING")
        self.assertEqual(
            converted["properties"]["verdict"]["enum"], ["same", "different", "unsure"]
        )
        self.assertIn("confidence", converted["required"])

    def test_fenced_json_is_unwrapped(self):
        self.assertEqual(
            llm_module._strip_json_fence('```json\n{"a": 1}\n```'), '{"a": 1}'
        )
        self.assertEqual(llm_module._strip_json_fence('{"a": 1}'), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()

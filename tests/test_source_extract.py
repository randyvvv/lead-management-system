"""Source extraction, including the cases the rules are meant to get wrong
gracefully rather than confidently."""

import csv
import unittest

from leadms import config
from leadms.source_extract import (
    CHANNELS,
    ORIGINAL_SOURCE_EXPECTATION,
    extract_source,
)


class FakeLLM:
    """Records what it was asked and returns a canned structured answer."""

    name = "fake"
    is_model_backed = True

    def __init__(self, response=None):
        self.response = response or {
            "channel": "Referral",
            "detail": "Partner introduction",
            "confidence": 0.8,
            "reason": "test",
            "provider": "fake",
        }
        self.calls = []

    def extract_source(self, text, hints=None):
        self.calls.append((text, hints))
        return self.response


class TestChannelMapping(unittest.TestCase):
    CASES = [
        ("He scanned our QR code at the SaaStr Annual booth.", "Event"),
        ("Met her at the Mobile World Congress booth, scanned our QR code.", "Event"),
        ("Met at the booth during Dubai FinTech Week, said they'd follow up over email.", "Event"),
        ("Spoke with them at our TechCrunch Disrupt booth, no QR scan logged.", "Event"),
        ("Referred by Michael Zhang, warm intro.", "Referral"),
        ("Linkedin dm inbound asking about pricing.", "LinkedIn"),
        ("Connected on LinkedIn after commenting on our post.", "LinkedIn"),
        ("Found us through organic google search then landed on the contact page.", "Organic Search"),
        ("Googled us and ended up on the pricing page before booking a demo.", "Organic Search"),
        ("Filled out the form on the comparison-vs-hubspot page.", "Website"),
        ("Manual - added after inbound phone call.", "Manual/Sales"),
        ("Manually added by sales after a phone call from a cold outreach list.", "Manual/Sales"),
        ("Other - reached out via our general info@ inbox.", "Other"),
        ("Other - walked into our office without an appointment.", "Other"),
    ]

    def test_each_template_maps_to_the_right_channel(self):
        for text, expected in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(extract_source(text)["channel"], expected)

    def test_channel_is_always_from_the_allowed_set(self):
        for text, _ in self.CASES:
            self.assertIn(extract_source(text)["channel"], CHANNELS)


class TestDetailExtraction(unittest.TestCase):
    def test_matches_the_shape_in_the_brief(self):
        result = extract_source(
            "He scanned our QR code at the Singapore FinTech Festival 2026 booth."
        )
        self.assertEqual(result["channel"], "Event")
        self.assertEqual(
            result["detail"], "Singapore FinTech Festival 2026 — Booth QR Code"
        )

    def test_booth_conversation_without_a_scan_is_distinguished(self):
        result = extract_source(
            "Spoke with them at our Web Summit 2026 booth, no QR scan logged."
        )
        self.assertIn("no QR scan logged", result["detail"])

    def test_referrer_name_is_captured(self):
        self.assertEqual(
            extract_source("Referred by Michael Zhang, warm intro.")["detail"],
            "Warm intro from Michael Zhang",
        )

    def test_sales_status_tail_is_not_treated_as_source_text(self):
        """Notes append progress sentences that say nothing about channel."""
        plain = extract_source("Filled out the form on the pricing page.")
        tailed = extract_source(
            "Filled out the form on the pricing page. Great fit, prioritizing."
        )
        self.assertEqual(plain["detail"], tailed["detail"])

    def test_duplicate_marker_does_not_reach_the_detail(self):
        result = extract_source(
            "Filled out the form on the contact page. "
            "possible duplicate — verify before contacting."
        )
        self.assertNotIn("duplicate", result["detail"].lower())


class TestAmbiguityHandling(unittest.TestCase):
    def test_paid_ads_are_flagged_as_a_taxonomy_gap(self):
        """The required enum has Organic Search but no Paid Search bucket.

        Folding paid traffic into Website or Organic would silently corrupt
        attribution, so we answer Other and say why.
        """
        result = extract_source(
            "Booked a demo via the book-a-demo page after clicking a google ad."
        )
        self.assertEqual(result["channel"], "Other")
        self.assertEqual(result["taxonomy_gap"], "Paid Search")
        self.assertIn("Google Ads", result["detail"])

    def test_paid_rule_beats_the_landing_page_rule(self):
        """The note mentions a page too; acquisition channel must win."""
        result = extract_source(
            "Booked a demo via the book-a-demo page after clicking a google ad."
        )
        self.assertNotEqual(result["channel"], "Website")

    def test_platform_inference_lowers_confidence(self):
        """'Saw our post and commented' never names LinkedIn."""
        named = extract_source("Connected on LinkedIn after commenting on our post.")
        inferred = extract_source("Saw our post about replacing hubspot and commented.")
        self.assertEqual(inferred["channel"], "LinkedIn")
        self.assertLess(inferred["confidence"], named["confidence"])

    def test_uninformative_text_is_not_forced_into_a_channel(self):
        result = extract_source(
            "Following up after our earlier conversation, please send more info."
        )
        self.assertEqual(result["channel"], "Other")
        self.assertLessEqual(result["confidence"], 0.3)
        self.assertEqual(result["method"], "fallback:unclassified")

    def test_empty_input(self):
        self.assertEqual(extract_source("")["channel"], "Other")
        self.assertEqual(extract_source(None)["channel"], "Other")


class TestStructuredFallbacks(unittest.TestCase):
    def test_page_url_beats_an_uninformative_message(self):
        result = extract_source(
            "Following up after our earlier conversation, please send more info.",
            hints={"page_url": "/book-a-demo"},
        )
        self.assertEqual(result["channel"], "Website")
        self.assertIn("book a demo", result["detail"])

    def test_llm_is_only_called_when_rules_fail(self):
        fake = FakeLLM()
        extract_source("Referred by Michael Zhang, warm intro.", llm=fake)
        self.assertEqual(fake.calls, [], "rules should have handled this")

        extract_source("A totally novel phrasing with no known markers", llm=fake)
        self.assertEqual(len(fake.calls), 1)

    def test_llm_result_is_adopted_when_valid(self):
        fake = FakeLLM()
        result = extract_source("Novel phrasing here", llm=fake)
        self.assertEqual(result["channel"], "Referral")
        self.assertTrue(result["method"].startswith("llm:"))

    def test_invalid_llm_channel_is_rejected(self):
        fake = FakeLLM({"channel": "Carrier Pigeon", "detail": "", "confidence": 1.0})
        result = extract_source("Novel phrasing here", llm=fake)
        self.assertEqual(result["channel"], "Other")
        self.assertEqual(result["method"], "fallback:unclassified")

    def test_llm_failure_does_not_propagate(self):
        class Boom:
            def extract_source(self, text, hints=None):
                raise RuntimeError("provider down")

        result = extract_source("Novel phrasing here", llm=Boom())
        self.assertEqual(result["channel"], "Other")


class TestAgainstTheWholeFixture(unittest.TestCase):
    """Corpus-level checks, which is where a rules pass usually breaks."""

    @classmethod
    def setUpClass(cls):
        with open(config.SEED_CSV, newline="", encoding="utf-8-sig") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.results = [extract_source(row["Notes"]) for row in cls.rows]

    def test_rules_cover_every_seed_note(self):
        unclassified = [
            r for r in self.results if r["method"] == "fallback:unclassified"
        ]
        self.assertEqual(
            len(unclassified), 0,
            "rules should classify all 2,049 notes; " + str(len(unclassified)) + " fell through",
        )

    def test_agrees_with_the_populated_original_source_column(self):
        """An independent cross-check: Original Source is never an input.

        It is blank on ~50% of rows and too coarse to use directly, but where
        it is populated it should never contradict the extracted channel.
        """
        checked = mismatches = 0
        for row, result in zip(self.rows, self.results):
            original = (row["Original Source"] or "").strip()
            if not original:
                continue
            checked += 1
            if original not in ORIGINAL_SOURCE_EXPECTATION[result["channel"]]:
                mismatches += 1
        self.assertGreater(checked, 1000)
        self.assertEqual(mismatches, 0, str(mismatches) + " of " + str(checked) + " disagree")


if __name__ == "__main__":
    unittest.main()

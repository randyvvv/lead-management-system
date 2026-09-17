"""LLM access with a provider chain and a deterministic offline fallback.

Provider order (``LEADMS_LLM=auto``):

1. **Anthropic** when ``ANTHROPIC_API_KEY`` is set.
2. **Gemini** when ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) is set - the
   free-tier fallback, so the project is runnable without paid credits.
3. **Heuristic** - a documented, deterministic rule set that runs offline.

Every provider implements the same two task methods, so the dedupe and
extraction code never branches on which one is active. Responses are cached
on disk keyed by the request payload, which makes repeated runs free and
keeps a re-run of the evaluation reproducible.

Raw HTTP via ``urllib`` is a deliberate choice: this project ships with zero
third-party dependencies so a reviewer can clone and run it with nothing but
a Python interpreter. In a codebase that already had dependencies, the
official ``anthropic`` SDK would be the better call.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request

from . import config

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

CHANNELS = [
    "Website",
    "Event",
    "LinkedIn",
    "Organic Search",
    "Referral",
    "Manual/Sales",
    "Other",
]

# A forced boolean on a genuinely 50/50 pair produces noise, so the
# adjudicator is allowed to abstain. "unsure" leaves the statistical score
# untouched instead of pushing it in an arbitrary direction.
DEDUPE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["same", "different", "unsure"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "channel": {"type": "string", "enum": CHANNELS},
        "detail": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["channel", "detail", "confidence", "reason"],
    "additionalProperties": False,
}

DEDUPE_SYSTEM = (
    "You are a CRM data steward deciding whether two lead records describe "
    "the same real person. Two records at the same company with similar but "
    "distinct names are usually DIFFERENT colleagues, not duplicates. "
    "Treat a shared phone number or a shared exact email as strong evidence "
    "of the same person; treat a shared company domain alone as weak. "
    "Answer 'unsure' when the evidence genuinely does not settle it. "
    "Reply with JSON only."
)

SOURCE_SYSTEM = (
    "You classify where a sales lead came from, using a free-text note from a "
    "CRM. Pick exactly one channel from: " + ", ".join(CHANNELS) + ". "
    "'detail' should name the specific event, page, referrer or campaign in a "
    "short human-readable phrase, or be an empty string if the note names "
    "nothing specific. Do not invent specifics the note does not contain. "
    "Reply with JSON only."
)


class LLMUnavailable(RuntimeError):
    """Raised when a provider cannot serve a request; the chain moves on."""


def _strip_json_fence(text):
    """Remove a ```json ... ``` wrapper if the model added one."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _post_json(url, payload, headers, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LLMUnavailable("HTTP " + str(exc.code) + ": " + detail) from exc
    except Exception as exc:  # network down, DNS failure, timeout
        raise LLMUnavailable(str(exc)) from exc


class BaseLLMClient:
    """Interface shared by every provider."""

    name = "base"

    def __init__(self):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def adjudicate_duplicate(self, record_a, record_b, evidence):
        raise NotImplementedError

    def extract_source(self, text, hints=None):
        raise NotImplementedError

    @property
    def is_model_backed(self):
        """False for the offline stub - the README and API responses say so."""
        return True


class HeuristicClient(BaseLLMClient):
    """Offline, deterministic stand-in. Not a model - and it says so.

    This exists so the whole service runs, and the test suite passes, with no
    API key and no network. It deliberately *abstains* on ambiguous duplicate
    pairs rather than inventing a verdict: re-deriving an answer from the same
    features the statistical scorer already used would add no information and
    would make the offline numbers look better than they are.
    """

    name = "heuristic"

    _SOURCE_KEYWORDS = [
        ("Event", ("booth", "conference", "expo", "summit", "trade show", "festival")),
        ("LinkedIn", ("linkedin", "dm", "our post", "commented")),
        ("Referral", ("referred", "referral", "warm intro", "introduced by")),
        ("Organic Search", ("google search", "googled", "organic", "searched for")),
        ("Website", ("form", "webinar", "download", "signed up", "book-a-demo")),
        ("Manual/Sales", ("cold call", "outreach", "manually", "phone call", "sdr")),
    ]

    @property
    def is_model_backed(self):
        return False

    def adjudicate_duplicate(self, record_a, record_b, evidence):
        self.usage["calls"] += 1
        if evidence.get("phone_exact") or evidence.get("email_exact"):
            return {
                "verdict": "same",
                "confidence": 0.95,
                "reason": "Shares an exact phone number or email address.",
            }
        if evidence.get("name_sim", 0.0) < 0.80:
            return {
                "verdict": "different",
                "confidence": 0.8,
                "reason": "Names are too dissimilar to be the same person.",
            }
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "reason": (
                "Offline heuristic adjudicator abstains; no model was available "
                "to judge this pair."
            ),
        }

    def extract_source(self, text, hints=None):
        self.usage["calls"] += 1
        lowered = (text or "").lower()
        for channel, keywords in self._SOURCE_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                return {
                    "channel": channel,
                    "detail": "",
                    "confidence": 0.4,
                    "reason": "Offline keyword match (no model available).",
                }
        return {
            "channel": "Other",
            "detail": "",
            "confidence": 0.2,
            "reason": "Offline heuristic found no channel signal in the text.",
        }


class AnthropicClient(BaseLLMClient):
    """Anthropic Messages API over raw HTTPS."""

    name = "anthropic"

    def __init__(self, api_key, model=None, timeout=None):
        super().__init__()
        self.api_key = api_key
        self.model = model or config.ANTHROPIC_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    def _complete(self, system, user, schema):
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # effort 'low' keeps a short classification cheap; thinking stays
            # on (the default) because disabling it has known failure modes.
            "output_config": {
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
            # Server-side refusal fallback: if a safety classifier declines the
            # request, Anthropic re-runs it on a recommended fallback model
            # rather than handing back an unusable refusal.
            "fallbacks": "default",
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": "server-side-fallback-2026-07-01",
        }
        response = _post_json(ANTHROPIC_URL, payload, headers, self.timeout)

        # Check stop_reason before touching content: on a refusal the content
        # array is empty or partial.
        if response.get("stop_reason") == "refusal":
            raise LLMUnavailable("request refused by safety classifiers")

        usage = response.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["output_tokens"] += usage.get("output_tokens", 0)

        for block in response.get("content", []):
            if block.get("type") == "text":
                return json.loads(_strip_json_fence(block.get("text", "")))
        raise LLMUnavailable("no text block in response")

    def adjudicate_duplicate(self, record_a, record_b, evidence):
        return self._complete(
            DEDUPE_SYSTEM, _dedupe_prompt(record_a, record_b, evidence), DEDUPE_SCHEMA
        )

    def extract_source(self, text, hints=None):
        return self._complete(
            SOURCE_SYSTEM, _source_prompt(text, hints), SOURCE_SCHEMA
        )


class GeminiClient(BaseLLMClient):
    """Google Gemini generateContent over raw HTTPS (free-tier fallback)."""

    name = "gemini"

    def __init__(self, api_key, model=None, timeout=None):
        super().__init__()
        self.api_key = api_key
        self.model = model or config.GEMINI_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    def _complete(self, system, user, schema):
        url = GEMINI_BASE + "/models/" + self.model + ":generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _to_gemini_schema(schema),
            },
        }
        headers = {
            "content-type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        response = _post_json(url, payload, headers, self.timeout)

        usage = response.get("usageMetadata") or {}
        self.usage["calls"] += 1
        self.usage["input_tokens"] += usage.get("promptTokenCount", 0)
        self.usage["output_tokens"] += usage.get("candidatesTokenCount", 0)

        candidates = response.get("candidates") or []
        if not candidates:
            raise LLMUnavailable("no candidates returned")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        for part in parts:
            if "text" in part:
                return json.loads(_strip_json_fence(part["text"]))
        raise LLMUnavailable("no text part in response")

    def adjudicate_duplicate(self, record_a, record_b, evidence):
        return self._complete(
            DEDUPE_SYSTEM, _dedupe_prompt(record_a, record_b, evidence), DEDUPE_SCHEMA
        )

    def extract_source(self, text, hints=None):
        return self._complete(
            SOURCE_SYSTEM, _source_prompt(text, hints), SOURCE_SCHEMA
        )

    def list_models(self):
        """Names the key can actually use - model IDs move between releases."""
        request = urllib.request.Request(
            GEMINI_BASE + "/models", headers={"x-goog-api-key": self.api_key}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
        return [
            m.get("name", "").replace("models/", "")
            for m in data.get("models", [])
            if "generateContent" in (m.get("supportedGenerationMethods") or [])
        ]


def _to_gemini_schema(schema):
    """Translate our JSON Schema subset to Gemini's OpenAPI-ish dialect."""
    type_map = {
        "object": "OBJECT",
        "string": "STRING",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }
    converted = {"type": type_map.get(schema.get("type"), "STRING")}
    if "enum" in schema:
        converted["enum"] = schema["enum"]
    if "properties" in schema:
        converted["properties"] = {
            key: _to_gemini_schema(value)
            for key, value in schema["properties"].items()
        }
    if "required" in schema:
        converted["required"] = schema["required"]
    return converted


class ChainClient(BaseLLMClient):
    """Try each provider in order; fall through on failure.

    A failing paid provider degrades to the free one, and a failing free one
    degrades to the offline rules, so the endpoint never returns a 500 just
    because a key expired.
    """

    name = "chain"

    def __init__(self, providers):
        super().__init__()
        self.providers = list(providers)
        self.errors = []

    @property
    def active(self):
        return self.providers[0].name if self.providers else "none"

    @property
    def is_model_backed(self):
        return any(p.is_model_backed for p in self.providers[:-1]) or (
            bool(self.providers) and self.providers[0].is_model_backed
        )

    def _dispatch(self, method, *args, **kwargs):
        for provider in self.providers:
            try:
                result = getattr(provider, method)(*args, **kwargs)
                result["provider"] = provider.name
                if provider.is_model_backed:
                    result["model"] = getattr(provider, "model", None)
                return result
            except LLMUnavailable as exc:
                self.errors.append(provider.name + ": " + str(exc))
            except (ValueError, KeyError, TypeError) as exc:
                # Malformed JSON or an unexpected shape - treat like a failure.
                self.errors.append(provider.name + ": bad response: " + str(exc))
        raise LLMUnavailable("; ".join(self.errors) or "no providers configured")

    def adjudicate_duplicate(self, record_a, record_b, evidence):
        return self._dispatch("adjudicate_duplicate", record_a, record_b, evidence)

    def extract_source(self, text, hints=None):
        return self._dispatch("extract_source", text, hints)

    @property
    def aggregate_usage(self):
        return {
            provider.name: dict(provider.usage)
            for provider in self.providers
            if provider.usage["calls"]
        }


class CachedClient(BaseLLMClient):
    """Disk cache in front of any provider, keyed by the request payload."""

    def __init__(self, inner, cache_dir=None):
        super().__init__()
        self.inner = inner
        self.cache_dir = cache_dir or config.LLM_CACHE_DIR
        self.hits = 0
        self.misses = 0

    name = "cached"

    @property
    def is_model_backed(self):
        return self.inner.is_model_backed

    def _path(self, key_material):
        digest = hashlib.sha256(
            json.dumps(key_material, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / (digest + ".json")

    def _cached(self, key_material, produce):
        path = self._path(key_material)
        if path.exists():
            try:
                self.hits += 1
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass  # corrupt cache entry - fall through and recompute
        self.misses += 1
        result = produce()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result), encoding="utf-8")
        except OSError:
            pass  # a read-only checkout should not break the request
        return result

    def adjudicate_duplicate(self, record_a, record_b, evidence):
        key = ["dedupe", self.inner.name, record_a, record_b]
        return self._cached(
            key, lambda: self.inner.adjudicate_duplicate(record_a, record_b, evidence)
        )

    def extract_source(self, text, hints=None):
        key = ["source", self.inner.name, text, hints]
        return self._cached(key, lambda: self.inner.extract_source(text, hints))

    @property
    def aggregate_usage(self):
        usage = getattr(self.inner, "aggregate_usage", {})
        return {"cache": {"hits": self.hits, "misses": self.misses}, **usage}


def _dedupe_prompt(record_a, record_b, evidence):
    return (
        "Record A:\n"
        + json.dumps(record_a, indent=2, ensure_ascii=False)
        + "\n\nRecord B:\n"
        + json.dumps(record_b, indent=2, ensure_ascii=False)
        + "\n\nPre-computed field comparisons:\n"
        + json.dumps(evidence, indent=2, sort_keys=True)
        + "\n\nAre these the same person?"
    )


def _source_prompt(text, hints=None):
    prompt = "Note text:\n" + json.dumps(text or "", ensure_ascii=False)
    if hints:
        prompt += "\n\nAdditional context:\n" + json.dumps(
            hints, indent=2, ensure_ascii=False
        )
    return prompt + "\n\nWhich channel did this lead come from?"


def build_client(mode=None, use_cache=True):
    """Assemble the provider chain from environment configuration."""
    mode = (mode or config.LLM_MODE).lower()
    providers = []

    if mode in ("auto", "anthropic") and config.ANTHROPIC_API_KEY:
        providers.append(AnthropicClient(config.ANTHROPIC_API_KEY))
    if mode in ("auto", "gemini") and config.GEMINI_API_KEY:
        providers.append(GeminiClient(config.GEMINI_API_KEY))

    # Always last: guarantees the service answers even with no keys at all.
    providers.append(HeuristicClient())

    chain = ChainClient(providers)
    return CachedClient(chain) if use_cache else chain


def describe_client(client):
    """Short, honest summary for /health and the CLI."""
    inner = getattr(client, "inner", client)
    providers = getattr(inner, "providers", [inner])
    return {
        "providers": [p.name for p in providers],
        "model_backed": client.is_model_backed,
        "primary": providers[0].name if providers else "none",
        "models": {
            p.name: getattr(p, "model", None)
            for p in providers
            if getattr(p, "model", None)
        },
    }

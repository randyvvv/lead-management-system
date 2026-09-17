"""AI-assisted lead source extraction (assignment part 3).

Takes messy free text - the CRM ``Notes`` column, or a website form message -
and returns a structured channel plus a human-readable detail string:

    {"channel": "Event", "detail": "Singapore FinTech Festival 2026 - Booth QR Code"}

Design: **rules first, model second.** A deterministic pass handles the
phrasings we can recognise, and anything it cannot classify is handed to the
LLM. That ordering is driven by measurement, not taste - see
``docs`` in the README: the rule pass already classifies 100% of the 2,049
seed notes, so paying for a model call per row would buy nothing. The model
earns its place on genuinely unseen phrasing, which is exactly what the
website form messages contain (49 of 90 form messages are a generic
"following up..." line that no rule should pretend to classify).

Note that ``Original Source`` is deliberately **not** an input here. The
assignment flags it as untrustworthy, and we keep it clean as an independent
cross-check for evaluation (see ``eval/run_eval.py``).
"""

from __future__ import annotations

import re

from .normalize import EM_DASH, strip_dup_marker

CHANNELS = [
    "Website",
    "Event",
    "LinkedIn",
    "Organic Search",
    "Referral",
    "Manual/Sales",
    "Other",
]

# Sales-progress sentences that the fixture appends to the source sentence.
# They say nothing about acquisition channel, so we trim them before parsing
# to keep the extracted `detail` clean.
_STATUS_TAILS = (
    "Great fit, prioritizing.",
    "Qualifying now.",
    "Very interested, wants pricing call.",
    "Connected, sending proposal.",
    "Left voicemail, will retry.",
    "No response yet.",
    "Not interested for now.",
)


def _clean(text):
    cleaned = strip_dup_marker(text or "").strip()
    changed = True
    while changed:
        changed = False
        for tail in _STATUS_TAILS:
            if cleaned.lower().endswith(tail.lower()):
                cleaned = cleaned[: -len(tail)].strip()
                changed = True
    return cleaned


def _titlecase_event(name):
    """Tidy a captured event name without destroying existing casing."""
    return re.sub(r"\s+", " ", (name or "").strip(" .,")).strip()


def _result(channel, detail, confidence, method, evidence, taxonomy_gap=None):
    return {
        "channel": channel,
        "detail": detail,
        "confidence": round(float(confidence), 3),
        "method": method,
        "evidence": evidence,
        "taxonomy_gap": taxonomy_gap,
    }


# --- individual rules --------------------------------------------------------
# Order matters: the first match wins. Paid advertising has to be tested
# before the website-page and demo-booking rules, because the paid notes
# mention a landing page too ("Booked a demo via the book-a-demo page after
# clicking a google ad").

_PAID_RE = re.compile(
    r"\b(google ad|google ads|adwords|paid ad|ppc|retarget\w*|sponsored)\b", re.I
)
_PAID_PAGE_RE = re.compile(r"\bvia the ([\w\-/]+) page\b", re.I)

_EVENT_BOOTH_RE = re.compile(r"\bat (?:the|our) (.+?) booth\b", re.I)
_EVENT_DURING_RE = re.compile(r"\bbooth during ([^,.]+)", re.I)
_EVENT_GENERIC_RE = re.compile(
    r"\b(booth|trade ?show|expo|summit|conference|festival|meetup)\b", re.I
)
_NO_QR_RE = re.compile(r"\bno qr\b", re.I)
_QR_RE = re.compile(r"\bqr\b", re.I)

_REFERRAL_RE = re.compile(
    r"\b(?:referred by|introduced by|intro from|referral from)\s+([^,.;]+)", re.I
)
_REFERRAL_GENERIC_RE = re.compile(r"\b(referral|warm intro|word of mouth)\b", re.I)

_LINKEDIN_DM_RE = re.compile(r"\blinked ?in\b.*\b(dm|message|inmail)\b", re.I)
_LINKEDIN_RE = re.compile(r"\blinked ?in\b", re.I)
_OUR_POST_RE = re.compile(r"\bour post\b", re.I)

_ORGANIC_RE = re.compile(
    r"\b(organic (?:google )?search|googled us|found us (?:through|via) "
    r"(?:organic )?(?:google )?search|search engine)\b",
    re.I,
)
_LANDED_RE = re.compile(r"\b(?:landed on|ended up on) the ([^.,]+?)(?:\s+before|\s*[.,]|$)", re.I)

_FORM_RE = re.compile(r"\bfilled out the form on (?:the )?([^.]+)", re.I)
_FORM_GENERIC_RE = re.compile(r"\b(form submission|submitted the form|web form)\b", re.I)

_MANUAL_RE = re.compile(
    r"\b(manual\b|manually added|added by sales|cold (?:outreach|call|list)|"
    r"inbound phone call|sdr)\b",
    re.I,
)
_COLD_RE = re.compile(r"\bcold\b", re.I)

_INBOX_RE = re.compile(r"\b(info@|general inbox|hello@|contact@)", re.I)
_WALKIN_RE = re.compile(r"\bwalked into (?:our )?office\b|\bwalk-?in\b", re.I)


def _apply_rules(text):
    """Return a result dict, or None when no rule matches."""
    if not text:
        return None

    # 1. Paid advertising. The required channel enum has no Paid Search bucket
    #    (it has Organic Search only), so we return Other and flag the gap
    #    rather than silently folding paid traffic into Website or Organic.
    if _PAID_RE.search(text):
        page = _PAID_PAGE_RE.search(text)
        detail = "Paid Search " + EM_DASH + " Google Ads"
        if page:
            detail += " (" + page.group(1) + " page)"
        return _result(
            "Other", detail, 0.9, "rules:paid_search", _PAID_RE.search(text).group(0),
            taxonomy_gap="Paid Search",
        )

    # 2. Events.
    event_match = _EVENT_BOOTH_RE.search(text) or _EVENT_DURING_RE.search(text)
    if event_match:
        event = _titlecase_event(event_match.group(1))
        if _QR_RE.search(text) and not _NO_QR_RE.search(text):
            detail = event + " " + EM_DASH + " Booth QR Code"
        elif _NO_QR_RE.search(text):
            detail = event + " " + EM_DASH + " Booth conversation (no QR scan logged)"
        else:
            detail = event + " " + EM_DASH + " Booth conversation"
        return _result("Event", detail, 0.95, "rules:event_booth", event_match.group(0))

    # 3. Referrals - capture the referrer's name, which is the useful part.
    referral_match = _REFERRAL_RE.search(text)
    if referral_match:
        referrer = _titlecase_event(referral_match.group(1))
        return _result(
            "Referral",
            "Warm intro from " + referrer,
            0.95,
            "rules:referral_named",
            referral_match.group(0),
        )

    # 4. LinkedIn. "Saw our post ... and commented" never names the platform;
    #    we infer LinkedIn from context and lower the confidence to say so.
    if _LINKEDIN_DM_RE.search(text):
        return _result(
            "LinkedIn",
            "Inbound LinkedIn DM",
            0.95,
            "rules:linkedin_dm",
            _LINKEDIN_RE.search(text).group(0),
        )
    if _LINKEDIN_RE.search(text):
        detail = "Comment on our post" if _OUR_POST_RE.search(text) else "LinkedIn touchpoint"
        return _result(
            "LinkedIn", detail, 0.9, "rules:linkedin", _LINKEDIN_RE.search(text).group(0)
        )
    if _OUR_POST_RE.search(text) and re.search(r"\bcomment", text, re.I):
        return _result(
            "LinkedIn",
            "Comment on our social post (platform inferred)",
            0.6,
            "rules:social_post_inferred",
            _OUR_POST_RE.search(text).group(0),
        )

    # 5. Organic search.
    if _ORGANIC_RE.search(text):
        landing = _LANDED_RE.search(text)
        detail = "Google organic search"
        if landing:
            detail += " " + EM_DASH + " " + _titlecase_event(landing.group(1))
        return _result(
            "Organic Search", detail, 0.95, "rules:organic", _ORGANIC_RE.search(text).group(0)
        )

    # 6. Website forms.
    form_match = _FORM_RE.search(text)
    if form_match:
        page = _titlecase_event(form_match.group(1))
        return _result(
            "Website",
            "Website form " + EM_DASH + " " + page,
            0.95,
            "rules:website_form",
            form_match.group(0),
        )
    if _FORM_GENERIC_RE.search(text):
        return _result(
            "Website",
            "Website form",
            0.8,
            "rules:website_form_generic",
            _FORM_GENERIC_RE.search(text).group(0),
        )

    # 7. Manually created by sales.
    if _MANUAL_RE.search(text):
        detail = (
            "Cold outreach call"
            if _COLD_RE.search(text)
            else "Inbound phone call, logged manually"
        )
        return _result(
            "Manual/Sales", detail, 0.9, "rules:manual_sales", _MANUAL_RE.search(text).group(0)
        )

    # 8. Named "Other" flavours worth keeping distinct in the detail string.
    if _INBOX_RE.search(text):
        return _result(
            "Other",
            "Inbound email to general inbox",
            0.85,
            "rules:general_inbox",
            _INBOX_RE.search(text).group(0),
        )
    if _WALKIN_RE.search(text):
        return _result(
            "Other", "Office walk-in", 0.85, "rules:walk_in", _WALKIN_RE.search(text).group(0)
        )
    if _EVENT_GENERIC_RE.search(text):
        return _result(
            "Event",
            "Event touchpoint",
            0.6,
            "rules:event_generic",
            _EVENT_GENERIC_RE.search(text).group(0),
        )
    if _REFERRAL_GENERIC_RE.search(text):
        return _result(
            "Referral",
            "Referral",
            0.7,
            "rules:referral_generic",
            _REFERRAL_GENERIC_RE.search(text).group(0),
        )
    return None


def _from_page_url(page_url):
    """A form submission's landing page is better evidence than a vague message."""
    page = (page_url or "").strip().strip("/")
    if not page:
        return None
    label = page.replace("-", " ").replace("/", " / ") or "homepage"
    return _result(
        "Website",
        "Website form " + EM_DASH + " " + label + " page",
        0.8,
        "rules:page_url",
        page_url,
    )


def extract_source(text, hints=None, llm=None):
    """Extract ``{channel, detail, ...}`` from free text.

    ``hints`` may carry structured context from a form submission
    (``page_url``, ``form_name``). ``llm`` is any client from ``leadms.llm``;
    when omitted, the function is pure and offline.
    """
    hints = hints or {}
    cleaned = _clean(text)

    matched = _apply_rules(cleaned)
    if matched:
        return matched

    # No rule fired. Prefer hard structured context over a model guess.
    page_result = _from_page_url(hints.get("page_url"))
    if page_result:
        page_result["method"] = "rules:page_url_fallback"
        return page_result

    if llm is not None and cleaned:
        try:
            response = llm.extract_source(cleaned, hints or None)
            channel = response.get("channel")
            if channel in CHANNELS:
                return _result(
                    channel,
                    (response.get("detail") or "").strip(),
                    response.get("confidence", 0.5),
                    "llm:" + str(response.get("provider", "unknown")),
                    response.get("reason", ""),
                )
        except Exception:
            # A model failure must not take the endpoint down; fall through
            # to the honest "Unknown" answer below.
            pass

    return _result(
        "Other",
        "",
        0.2,
        "fallback:unclassified",
        cleaned[:160],
    )


# --- evaluation support ------------------------------------------------------
# Which raw `Original Source` values are consistent with each extracted
# channel. Used only by eval/run_eval.py to score the extractor against the
# 1,035 rows where that column is populated - never as an input feature.
ORIGINAL_SOURCE_EXPECTATION = {
    "Website": {"Direct Traffic"},
    "Organic Search": {"Organic Search"},
    "LinkedIn": {"Social Media"},
    "Referral": {"Referrals"},
    "Event": {"Offline Sources", "Other Campaigns"},
    "Manual/Sales": {"Other Campaigns"},
    "Other": {"Other Campaigns", "Paid Search", "Offline Sources"},
}

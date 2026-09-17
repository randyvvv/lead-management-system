"""Field-level cleaning for the raw HubSpot-shaped export.

Every function here is pure and individually testable. The messiness this
module is written against was measured on the supplied ``leads_seed.csv``
(2,049 rows); the numbers quoted in the docstrings come from that profiling
pass and are reproducible with ``python -m leadms.cli profile``.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

EM_DASH = "—"

# --- lead status -------------------------------------------------------------
# The export carries 35 distinct spellings of 7 real statuses: casing varies
# ("New"/"new"/"NEW") and leading/trailing whitespace is common (" New", "New ").
CANONICAL_STATUSES = [
    "New",
    "Contacted",
    "Connected",
    "Qualified",
    "Opportunity",
    "Closed Won",
    "Closed Lost",
]


def _squash(value):
    """Lowercase and drop everything that is not a letter.

    "` Closed Lost `" and "CLOSED LOST" both collapse to "closedlost".
    """
    return re.sub(r"[^a-z]", "", (value or "").lower())


_STATUS_LOOKUP = {_squash(s): s for s in CANONICAL_STATUSES}


def normalize_status(raw):
    """Map any observed spelling onto one of CANONICAL_STATUSES.

    Returns None for blank or unrecognised input rather than guessing, so an
    unexpected status surfaces as a null in the store instead of being
    silently bucketed into "New".
    """
    return _STATUS_LOOKUP.get(_squash(raw or ""))


def normalize_country(raw):
    """Title-case country names while preserving acronyms.

    62 distinct values appear and 112 rows are lowercased ("china", "united
    kingdom"). A naive ``.title()`` would also turn "UAE" into "Uae", so
    short all-caps tokens are left alone.
    """
    value = re.sub(r"\s+", " ", (raw or "").strip())
    if not value:
        return None
    words = []
    for word in value.split(" "):
        if word.isupper() and len(word) <= 3:
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)


def normalize_owner(raw):
    """Collapse internal and trailing whitespace in Contact Owner.

    10 real owners appear as 20 distinct strings because some rows carry a
    trailing space ("Marcus Wong " vs "Marcus Wong").
    """
    cleaned = re.sub(r"\s+", " ", (raw or "").strip())
    return cleaned or None


# --- dates -------------------------------------------------------------------
# Three formats are mixed inside the same column:
#   2026-06-02            (ISO date)
#   2026-05-20T00:00:00Z  (ISO datetime)
#   6/4/2026              (US slash)
#
# The slash format is month-first. Evidence from the seed file: across 1,078
# slash-formatted dates the first component never exceeds 12 while the second
# reaches 31, which is only consistent with M/D/YYYY.
_DATE_PATTERNS = ("%Y-%m-%d", "%m/%d/%Y")


def parse_date(raw):
    """Return an ISO ``YYYY-MM-DD`` string, or None if blank/unparseable."""
    value = (raw or "").strip()
    if not value:
        return None
    # ISO datetime: keep the date part, tolerating 'Z' and explicit offsets.
    if "T" in value:
        value = value.split("T", 1)[0]
    for pattern in _DATE_PATTERNS:
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def parse_timestamp(raw):
    """Return a full ISO-8601 timestamp when the input carries a time part."""
    value = (raw or "").strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).isoformat()
    except ValueError:
        date_only = parse_date(value)
        return date_only + "T00:00:00" if date_only else None


# --- phone -------------------------------------------------------------------
def phone_digits(raw):
    """Strip every non-digit character.

    "+1 693 555 0198", "+46 70 460 23 83" and "16935550198" all appear in the
    export; reducing to digits makes the reformatted duplicates comparable.
    Every number in the seed data carries a country code (lengths 10-13, none
    start with a trunk '0'), so no country-code inference is needed.
    """
    return re.sub(r"\D", "", raw or "")


def phone_key(raw):
    """Last 9 digits: a looser blocking key that survives country-code drift."""
    digits = phone_digits(raw)
    return digits[-9:] if len(digits) >= 9 else digits


# --- email -------------------------------------------------------------------
def split_email(raw):
    """Return ``(localpart, domain)``, both lowercased. ``("", "")`` if absent."""
    value = (raw or "").strip().lower()
    if "@" not in value:
        return (value, "")
    local, _, domain = value.partition("@")
    return (local, domain)


def domain_root(domain):
    """First label of the domain: ``bluepeak.com.au`` -> ``bluepeak``.

    Duplicates keep the company's domain root even when the localpart is
    rewritten, which makes this the cheapest reliable company blocking key.
    """
    return (domain or "").strip().lower().split(".")[0]


def email_local_variants(local):
    """Normalised forms of a localpart: ``erik.a`` -> ``{"erik.a", "erika"}``."""
    base = (local or "").strip().lower()
    if not base:
        return set()
    return {base, re.sub(r"[^a-z0-9]", "", base)}


# --- names -------------------------------------------------------------------
def _ascii_fold(value):
    """Drop accents so "Ines" and "Ines" with a grave accent compare equal."""
    decomposed = unicodedata.normalize("NFKD", value or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def name_tokens(value):
    """Lowercased tokens; hyphens are kept so "ji-woo" stays a single token."""
    folded = _ascii_fold(value or "").lower()
    return [t for t in re.split(r"[^a-z0-9-]+", folded) if t]


def resolve_name(first, last, full):
    """Reconcile the First/Last vs Full Name split.

    ~95% of rows use First Name + Last Name and ~5% use Full Name only; the
    seed data never populates both. Duplicates exploit this: the same person
    appears once as ("Erik", "Almeida") and once as Full Name "Erik Almeida".
    We always derive a single ``display`` string and a whitespace-free ``key``.
    """
    first = (first or "").strip()
    last = (last or "").strip()
    full = (full or "").strip()

    if not first and not last and full:
        tokens = full.split()
        if len(tokens) >= 2:
            first, last = tokens[0], tokens[-1]
        elif tokens:
            last = tokens[0]

    display = re.sub(r"\s+", " ", (full or (first + " " + last)).strip())
    tokens = name_tokens(display)
    return {
        "first": first or None,
        "last": last or None,
        "display": display or None,
        "tokens": tokens,
        # "J. Yoon" and "Ji-woo Yoon" differ here, which is why the dedupe
        # scorer also runs an initial-compatibility check rather than relying
        # on this key alone.
        "key": "".join(tokens),
    }


def is_initial(token):
    """True for a single-letter token, i.e. the "J." in "J. Yoon"."""
    return len(token.rstrip(".")) == 1


# --- company -----------------------------------------------------------------
# Every company name in the seed data ends in one of a small pool of legal or
# descriptor suffixes, and the duplicate generator swaps them freely
# ("Lotus Finance Pte. Ltd." / "Lotus Finance Analytics" / "Lotus Finance & Co").
# We therefore never treat company equality as hard evidence: these tokens are
# down-weighted during scoring.
COMPANY_SUFFIX_TOKENS = {
    "and", "co", "company", "corp", "corporation", "inc", "incorporated",
    "llc", "ltd", "limited", "plc", "ag", "gmbh", "srl", "bv", "nv", "sa",
    "pte", "pty", "sdn", "bhd", "kk", "oy", "ab", "as",
    "group", "holdings", "holding", "solutions", "studio", "labs", "lab",
    "robotics", "analytics", "consulting", "trading", "freight", "ventures",
    "partners", "digital", "retail", "textiles", "imports", "logistics",
    "legal", "finance", "bros", "technologies", "tech", "international",
    "global", "enterprises", "services", "systems",
}


def company_tokens(raw):
    folded = _ascii_fold(raw or "").lower()
    return [t for t in re.split(r"[^a-z0-9]+", folded) if t]


def company_core(raw):
    """Strip trailing suffix tokens, keeping at least one token.

    "Huang Analytics Consulting" and "Huang Analytics Robotics" both reduce to
    "huang". This is a display/grouping label only: similarity scoring uses the
    weighted token comparison in ``similarity.company_similarity``, because the
    same token can be core in one name and a suffix in another.
    """
    tokens = company_tokens(raw)
    while len(tokens) > 1 and tokens[-1] in COMPANY_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


# --- notes -------------------------------------------------------------------
# 136 rows in the seed file carry an annotation reading
# "possible duplicate <em dash> verify before contacting." in the Notes column.
# That is a hint left in the fixture, not a signal a production system would
# ever see, and it covers only part of the real duplicate population. The
# dedupe pipeline strips it from every text feature so it cannot leak into a
# score or into an LLM prompt; tests/test_dedupe.py asserts that.
_DUP_MARKER_RE = re.compile(
    r"\s*possible duplicate\s*[" + EM_DASH + r"\-]?\s*verify before contacting\.?",
    re.IGNORECASE,
)


def strip_dup_marker(notes):
    return _DUP_MARKER_RE.sub("", notes or "").strip()


def has_dup_marker(notes):
    return bool(_DUP_MARKER_RE.search(notes or ""))

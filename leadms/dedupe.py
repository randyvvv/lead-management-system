"""AI-assisted lead deduplication (assignment part 2).

Three stages, each one cheaper than the one after it:

1. **Candidate generation** - deterministic blocking keys (phone digits,
   exact email, company email-domain root, name key) unioned with top-K
   neighbours from a TF-IDF character n-gram index. This is what keeps the
   problem tractable: at 2,049 records a full pairwise sweep is 2,098,176
   comparisons; blocking brings it under ~30k, a >98% reduction, and the
   n-gram index recovers the duplicates that share no exact key.

2. **Vectorised scoring** - a transparent log-odds model over ~12 field
   comparison features. Every surviving candidate gets a score and a
   human-readable list of reasons. No network calls, milliseconds for the
   whole corpus.

3. **LLM adjudication of the ambiguous band only** - pairs scoring between
   ``DEDUPE_REVIEW_LOW`` and ``DEDUPE_AUTO_MERGE`` are the ones where field
   comparisons genuinely do not settle it (same name, same company, different
   phone and email: colleague or re-entry?). Only those go to a model, which
   is a few dozen calls rather than two million.

Finally, pairs above the reporting threshold are merged into clusters with a
union-find pass, so "A duplicates B" and "B duplicates C" surface as one
three-record group rather than two disjoint pairs.

**Leakage note.** 136 seed rows carry a "possible duplicate - verify before
contacting." annotation in ``Notes``. It is a fixture artefact, it covers only
part of the duplicate population, and nothing in this module reads it: notes
are passed through ``strip_dup_marker`` before they reach a feature or a
prompt. ``tests/test_dedupe.py`` asserts that.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from . import config
from .normalize import is_initial, strip_dup_marker
from .similarity import (
    TfidfNgramIndex,
    build_company_token_weights,
    company_similarity,
    jaro_winkler,
)

# --- scoring model -----------------------------------------------------------
# Log-odds weights. These are hand-set from the field semantics rather than
# fitted (2k rows with no ground-truth labels does not support fitting), then
# checked against the hand-labelled pairs in eval/dedupe_labels.csv. Keeping
# them in one dict means the README can show the model and a reviewer can
# argue with any single number.
WEIGHTS = {
    "bias": -6.0,
    "phone_exact": 5.0,       # strongest single identifier in this dataset
    "phone_suffix": 3.5,      # same national number, different country prefix
    # Two different reachable numbers is evidence against, but not decisive:
    # people change phones, and a CRM re-entry years later legitimately
    # carries a new number. Setting this too punitive (-2.5 in an earlier
    # pass) collapsed every different-phone pair to ~0.0 and left the LLM
    # adjudication band empty, which defeats the point of the hybrid.
    "phone_conflict": -1.8,
    "email_exact": 5.0,
    "domain_exact": 1.2,
    "domain_root": 1.2,       # same employer - weak on its own, colleagues share it
    "name_sim": 7.0,          # applied to (jaro_winkler - 0.72)
    "name_initial_compat": 1.5,   # "J. Yoon" vs "Ji-woo Yoon"
    "local_consistent": 2.2,  # both localparts derive from the same person name
    "company_sim": 1.5,       # applied to (similarity - 0.35)
    "country_match": 0.8,
    # Deliberately softer than phone_conflict. The two fire together on almost
    # every ambiguous pair, and they are correlated rather than independent
    # evidence, so stacking two hard penalties double-counts one observation.
    # A country field is also genuinely mutable and often just wrong.
    "country_conflict": -0.8,
    "title_match": 0.4,
    # Interaction term. An identical name at the same employer is the exact
    # configuration where field comparison runs out of signal: it is equally
    # consistent with "one person entered twice" and with "two colleagues who
    # share a name". Without this, such pairs score ~0.13 and get silently
    # discarded as non-duplicates - an overconfident answer. With it they land
    # in the review band and get a model's judgement, which is the whole point
    # of the hybrid.
    "same_name_same_company": 1.2,
}

_NAME_SIM_CENTER = 0.72
_COMPANY_SIM_CENTER = 0.35
_ALNUM_RE = re.compile(r"[^a-z0-9]")


def _squash(value):
    return _ALNUM_RE.sub("", (value or "").lower())


def name_local_patterns(first, last):
    """Localparts a human would plausibly build from this name.

    ``("Erik", "Almeida")`` yields {erikalmeida, almeidaerik, ealmeida, erika,
    almeidae, ea, erik, almeida}, which covers ``erik.a@``, ``erika@`` and
    ``erikalmeida@`` - the three spellings the fixture actually uses for one
    person.
    """
    f, l = _squash(first), _squash(last)
    patterns = set()
    if f and l:
        patterns.update({f + l, l + f, f[0] + l, f + l[0], l + f[0], f[0] + l[0]})
    if f:
        patterns.add(f)
    if l:
        patterns.add(l)
    return patterns


def initial_compatible(tokens_a, tokens_b):
    """1.0 when the two token lists differ only by an abbreviated given name."""
    if not tokens_a or not tokens_b or len(tokens_a) != len(tokens_b):
        return 0.0
    for ta, tb in zip(tokens_a, tokens_b):
        if ta == tb:
            continue
        if is_initial(ta) and tb.startswith(ta.rstrip(".")):
            continue
        if is_initial(tb) and ta.startswith(tb.rstrip(".")):
            continue
        return 0.0
    return 1.0


def local_consistency(a, b):
    """Does each record's email localpart spell out the *other* record's name?

    Cross-checking catches the common duplicate pattern where the same person
    is entered as ``erik.a@lotusfinance.biz`` and ``erikalmeida@lotusfinance.biz``.
    """
    score = 0.0
    if _squash(a.get("email_local")) in name_local_patterns(
        b.get("first_name"), b.get("last_name")
    ):
        score += 0.5
    if _squash(b.get("email_local")) in name_local_patterns(
        a.get("first_name"), a.get("last_name")
    ):
        score += 0.5
    return score


def pair_features(a, b, company_weights=None):
    """Field-by-field comparison of two lead records.

    Reads only identity fields. ``Notes`` is never consulted - see the module
    docstring on marker leakage.
    """
    features = {}

    phone_a, phone_b = a.get("phone_digits") or "", b.get("phone_digits") or ""
    key_a, key_b = a.get("phone_key") or "", b.get("phone_key") or ""
    features["phone_exact"] = 1.0 if phone_a and phone_a == phone_b else 0.0
    features["phone_suffix"] = (
        1.0 if not features["phone_exact"] and key_a and key_a == key_b else 0.0
    )
    features["phone_conflict"] = (
        1.0 if key_a and key_b and key_a != key_b else 0.0
    )

    email_a, email_b = a.get("email") or "", b.get("email") or ""
    features["email_exact"] = 1.0 if email_a and email_a == email_b else 0.0
    features["domain_exact"] = (
        1.0
        if a.get("email_domain") and a.get("email_domain") == b.get("email_domain")
        else 0.0
    )
    features["domain_root"] = (
        1.0
        if a.get("domain_root") and a.get("domain_root") == b.get("domain_root")
        else 0.0
    )

    features["name_sim"] = jaro_winkler(a.get("name_key") or "", b.get("name_key") or "")
    features["name_initial_compat"] = initial_compatible(
        (a.get("name_tokens") or []), (b.get("name_tokens") or [])
    )
    features["local_consistent"] = local_consistency(a, b)
    features["company_sim"] = company_similarity(
        a.get("company_name"), b.get("company_name"), company_weights
    )

    country_a = (a.get("country") or "").strip().lower()
    country_b = (b.get("country") or "").strip().lower()
    features["country_match"] = 1.0 if country_a and country_a == country_b else 0.0
    features["country_conflict"] = (
        1.0 if country_a and country_b and country_a != country_b else 0.0
    )

    title_a = (a.get("job_title") or "").strip().lower()
    title_b = (b.get("job_title") or "").strip().lower()
    features["title_match"] = 1.0 if title_a and title_a == title_b else 0.0

    features["same_name_same_company"] = (
        1.0
        if features["name_sim"] >= 0.97
        and (features["domain_root"] or features["company_sim"] >= 0.9)
        else 0.0
    )

    return features


def score_features(features):
    """Combine features into a confidence in [0, 1] via a logistic link."""
    z = WEIGHTS["bias"]
    z += WEIGHTS["phone_exact"] * features["phone_exact"]
    z += WEIGHTS["phone_suffix"] * features["phone_suffix"]
    z += WEIGHTS["phone_conflict"] * features["phone_conflict"]
    z += WEIGHTS["email_exact"] * features["email_exact"]
    z += WEIGHTS["domain_exact"] * features["domain_exact"]
    z += WEIGHTS["domain_root"] * features["domain_root"]
    z += WEIGHTS["name_sim"] * (features["name_sim"] - _NAME_SIM_CENTER)
    z += WEIGHTS["name_initial_compat"] * features["name_initial_compat"]
    z += WEIGHTS["local_consistent"] * features["local_consistent"]
    z += WEIGHTS["company_sim"] * (features["company_sim"] - _COMPANY_SIM_CENTER)
    z += WEIGHTS["country_match"] * features["country_match"]
    z += WEIGHTS["country_conflict"] * features["country_conflict"]
    z += WEIGHTS["title_match"] * features["title_match"]
    z += WEIGHTS["same_name_same_company"] * features["same_name_same_company"]
    # Guard against overflow on extreme inputs.
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def explain(features):
    """Human-readable reasons, strongest evidence first."""
    reasons = []
    if features["email_exact"]:
        reasons.append("identical email address")
    if features["phone_exact"]:
        reasons.append("identical phone number")
    elif features["phone_suffix"]:
        reasons.append("same phone number in a different format")
    if features["name_sim"] >= 0.97:
        reasons.append("identical name")
    elif features["name_initial_compat"]:
        reasons.append("name differs only by an abbreviated first name")
    elif features["name_sim"] >= 0.85:
        reasons.append("very similar name")
    if features["local_consistent"] >= 1.0:
        reasons.append("both email localparts spell out the same person")
    elif features["local_consistent"] > 0:
        reasons.append("one email localpart matches the other record's name")
    if features["domain_exact"]:
        reasons.append("same company email domain")
    elif features["domain_root"]:
        reasons.append("same company domain root, different TLD")
    if features["company_sim"] >= 0.6:
        reasons.append("company names share their distinctive tokens")
    if features.get("same_name_same_company") and not (
        features["email_exact"] or features["phone_exact"]
    ):
        reasons.append(
            "AMBIGUOUS: identical name at the same employer, but no shared "
            "phone or email - could equally be two colleagues"
        )
    if features["phone_conflict"]:
        reasons.append("CONFLICT: different phone numbers")
    if features["country_conflict"]:
        reasons.append("CONFLICT: different countries")
    return reasons


# --- candidate generation ----------------------------------------------------
def _blocking_keys(lead):
    """Cheap exact keys under which two records are worth comparing."""
    keys = []
    phone = lead.get("phone_digits") or ""
    if len(phone) >= 7:
        keys.append(("phone", phone))
    phone_key = lead.get("phone_key") or ""
    if len(phone_key) >= 9:
        keys.append(("phone9", phone_key))
    email = lead.get("email") or ""
    if email:
        keys.append(("email", email))
    root = lead.get("domain_root") or ""
    if root:
        keys.append(("domain", root))
    tokens = lead.get("name_tokens") or []
    if tokens:
        # first initial + surname: survives "Ji-woo Yoon" -> "J. Yoon"
        keys.append(("name", tokens[0][:1] + tokens[-1]))
    return keys


def _index_text(lead):
    """The blob the TF-IDF index sees for a record."""
    return " ".join(
        str(part)
        for part in (
            lead.get("display_name"),
            lead.get("company_core") or lead.get("company_name"),
            lead.get("email_local"),
            lead.get("domain_root"),
        )
        if part
    )


def generate_candidates(leads, top_k=None, max_block_size=None):
    """Return ``(pairs, stats)`` where pairs is a set of ordered id tuples."""
    top_k = top_k or config.ANN_TOP_K
    max_block_size = max_block_size or config.MAX_BLOCK_SIZE

    by_id = {lead["id"]: lead for lead in leads}
    pairs = set()
    provenance = defaultdict(set)

    # -- exact blocks
    blocks = defaultdict(list)
    for lead in leads:
        for key in _blocking_keys(lead):
            blocks[key].append(lead["id"])

    oversized = 0
    for (kind, _value), members in blocks.items():
        if len(members) < 2:
            continue
        if len(members) > max_block_size:
            # A single huge block would reintroduce quadratic cost. The n-gram
            # neighbours below still cover these records.
            oversized += 1
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = tuple(sorted((members[i], members[j])))
                pairs.add(pair)
                provenance[pair].add(kind)

    # -- approximate neighbours
    index = TfidfNgramIndex().build((lead["id"], _index_text(lead)) for lead in leads)
    for id_a, id_b, _score in index.all_neighbour_pairs(top_k=top_k, min_score=0.35):
        pair = tuple(sorted((id_a, id_b)))
        if pair[0] != pair[1]:
            pairs.add(pair)
            provenance[pair].add("ngram")

    total = len(leads)
    stats = {
        "records": total,
        "full_pairwise_comparisons": total * (total - 1) // 2,
        "candidate_pairs": len(pairs),
        "reduction_ratio": (
            1.0 - len(pairs) / (total * (total - 1) / 2) if total > 1 else 0.0
        ),
        "oversized_blocks_skipped": oversized,
    }
    return pairs, provenance, stats, by_id


# --- union-find --------------------------------------------------------------
class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


# --- LLM adjudication --------------------------------------------------------
def _llm_summary(lead):
    """The view of a record we hand to the adjudicator.

    Notes go through ``strip_dup_marker`` so the fixture's "possible duplicate"
    annotation can never reach the prompt.
    """
    return {
        "name": lead.get("display_name"),
        "job_title": lead.get("job_title"),
        "company": lead.get("company_name"),
        "email": lead.get("email"),
        "phone": lead.get("phone_raw"),
        "country": lead.get("country"),
        "created": lead.get("created_at"),
        "note": strip_dup_marker(lead.get("notes"))[:200],
    }


def adjudicate(pair_results, by_id, llm, low=None, high=None):
    """Send only the ambiguous band to the model and fold verdicts back in."""
    low = config.DEDUPE_REVIEW_LOW if low is None else low
    high = config.DEDUPE_AUTO_MERGE if high is None else high

    adjudicated = 0
    for result in pair_results:
        if not (low <= result["score"] < high):
            continue
        record_a = _llm_summary(by_id[result["a"]])
        record_b = _llm_summary(by_id[result["b"]])
        try:
            verdict = llm.adjudicate_duplicate(record_a, record_b, result["features"])
        except Exception as exc:
            result["llm"] = {"error": str(exc)}
            continue

        adjudicated += 1
        confidence = float(verdict.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        decision = (verdict.get("verdict") or "unsure").lower()
        result["llm"] = verdict
        result["score_before_llm"] = result["score"]

        # An abstention deliberately leaves the statistical score untouched.
        if decision == "same":
            result["score"] = max(result["score"], high + (1.0 - high) * confidence)
        elif decision == "different":
            result["score"] = min(result["score"], low * (1.0 - confidence))
        result["adjudicated_by"] = verdict.get("provider", getattr(llm, "name", "llm"))

    return adjudicated


# --- top-level ---------------------------------------------------------------
def find_duplicate_candidates(
    leads,
    llm=None,
    min_confidence=None,
    use_llm=True,
    top_k=None,
    pair_floor=None,
):
    """Full pipeline. Returns ``{"groups": [...], "stats": {...}}``.

    Groups are ranked by confidence, highest first. Nothing is merged: the
    caller gets candidates, scores and explanations, and a human decides.
    """
    min_confidence = (
        config.DEDUPE_REPORT_MIN if min_confidence is None else min_confidence
    )
    # Pairs below this are discarded before scoring is recorded. Defaults to
    # the review floor; the evaluation harness passes a lower value so that
    # near-misses can be inspected when measuring recall.
    pair_floor = config.DEDUPE_REVIEW_LOW if pair_floor is None else pair_floor
    leads = list(leads)
    if len(leads) < 2:
        return {"groups": [], "stats": {"records": len(leads), "candidate_pairs": 0}}

    pairs, provenance, stats, by_id = generate_candidates(leads, top_k=top_k)
    company_weights = build_company_token_weights(
        lead.get("company_name") for lead in leads
    )

    scored = []
    for id_a, id_b in pairs:
        features = pair_features(by_id[id_a], by_id[id_b], company_weights)
        score = score_features(features)
        if score < pair_floor:
            continue  # not worth reporting or paying a model for
        scored.append(
            {
                "a": id_a,
                "b": id_b,
                "score": score,
                "features": features,
                "reasons": explain(features),
                "blocked_by": sorted(provenance[(id_a, id_b)]),
            }
        )

    stats["pairs_above_review_floor"] = len(scored)
    stats["llm_adjudicated_pairs"] = 0
    if use_llm and llm is not None:
        stats["llm_adjudicated_pairs"] = adjudicate(scored, by_id, llm)

    # Cluster whatever survives the reporting threshold.
    keep = [p for p in scored if p["score"] >= min_confidence]
    union_find = _UnionFind()
    for pair in keep:
        union_find.union(pair["a"], pair["b"])

    clusters = defaultdict(list)
    for pair in keep:
        clusters[union_find.find(pair["a"])].append(pair)

    groups = []
    for root, members in clusters.items():
        ids = sorted({pair["a"] for pair in members} | {pair["b"] for pair in members})
        scores = [pair["score"] for pair in members]
        groups.append(
            {
                "group_id": "dup-" + str(root),
                "lead_ids": ids,
                "size": len(ids),
                # min is the conservative number: it is the weakest link
                # holding the cluster together.
                "confidence": round(min(scores), 4),
                "max_pair_confidence": round(max(scores), 4),
                "leads": [
                    {
                        "id": lead_id,
                        "name": by_id[lead_id].get("display_name"),
                        "company": by_id[lead_id].get("company_name"),
                        "email": by_id[lead_id].get("email"),
                        "phone": by_id[lead_id].get("phone_raw"),
                        "status": by_id[lead_id].get("status"),
                        "created_at": by_id[lead_id].get("created_at"),
                    }
                    for lead_id in ids
                ],
                "pairs": [
                    {
                        "a": pair["a"],
                        "b": pair["b"],
                        "confidence": round(pair["score"], 4),
                        "reasons": pair["reasons"],
                        "blocked_by": pair["blocked_by"],
                        "llm": pair.get("llm"),
                    }
                    for pair in sorted(members, key=lambda p: -p["score"])
                ],
            }
        )

    groups.sort(key=lambda g: (-g["confidence"], g["lead_ids"][0]))
    stats["groups"] = len(groups)
    stats["records_in_groups"] = sum(g["size"] for g in groups)
    stats["min_confidence"] = min_confidence
    return {"groups": groups, "stats": stats}


def rank_matches(candidate, candidate_leads, company_weights=None, limit=5):
    """Score an incoming record against an already-narrowed candidate list.

    Used by ``POST /leads/ingest``. The narrowing happens in SQL
    (``store.find_blocking_candidates``) using the same blocking keys as the
    batch path, so a single ingest touches a handful of rows instead of
    rebuilding the corpus index. Scoring is the identical function the batch
    report uses, so the two cannot drift apart.
    """
    scored = []
    for other in candidate_leads:
        if other.get("id") is not None and other["id"] == candidate.get("id"):
            continue
        features = pair_features(candidate, other, company_weights)
        scored.append(
            {
                "lead_id": other["id"],
                "score": score_features(features),
                "features": features,
                "reasons": explain(features),
                "lead": {
                    "id": other["id"],
                    "name": other.get("display_name"),
                    "company": other.get("company_name"),
                    "email": other.get("email"),
                    "phone": other.get("phone_raw"),
                },
            }
        )
    scored.sort(key=lambda m: -m["score"])
    return scored[:limit]


def best_match_for(candidate, candidate_leads, company_weights=None):
    """The single most likely existing lead, or None."""
    matches = rank_matches(candidate, candidate_leads, company_weights, limit=1)
    return matches[0] if matches else None

"""Measure the two AI-assisted features.

    python eval/run_eval.py              # everything, offline
    python eval/run_eval.py --dedupe     # duplicates only
    python eval/run_eval.py --source     # source extraction only

Reports, in order:

1. **Candidate generation** - how much of the pairwise space was avoided, and
   whether the surviving set still contains every true duplicate (blocking
   recall; a miss here can never be recovered downstream).
2. **Pair scoring** - precision/recall/F1 across thresholds against the
   independent ground truth in ``ground_truth.py``.
3. **Clustering** - exact-match rate of predicted groups against true groups.
4. **Source extraction** - rule coverage, and agreement with the ``Original
   Source`` column on the rows where it is populated. That column is never an
   input to the extractor, so it is a genuine held-out check.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.ground_truth import true_clusters, true_pairs  # noqa: E402
from leadms import config, dedupe, store  # noqa: E402
from leadms.normalize import has_dup_marker  # noqa: E402
from leadms.similarity import build_company_token_weights  # noqa: E402
from leadms.source_extract import (  # noqa: E402
    ORIGINAL_SOURCE_EXPECTATION,
    extract_source,
)

THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def _rule(title):
    print("\n" + title)
    print("-" * len(title))


def _prf(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def load_leads():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    store.load_seed(conn)
    return store.leads_for_dedupe(conn)


def evaluate_dedupe(leads):
    by_id = {int(lead["id"]): lead for lead in leads}
    gold_pairs = true_pairs(leads)
    gold_clusters = true_clusters(leads)

    _rule("1. Ground truth")
    marked = sum(1 for lead in leads if has_dup_marker(lead.get("notes")))
    print("records                      : " + str(len(leads)))
    print("true duplicate pairs         : " + str(len(gold_pairs)))
    print("true duplicate clusters      : " + str(len(gold_clusters)))
    print("records inside a cluster     : " + str(sum(len(c) for c in gold_clusters)))
    print("rows carrying the fixture's 'possible duplicate' note: " + str(marked))
    covered = sum(
        1 for lead in leads
        if has_dup_marker(lead.get("notes"))
        and any(int(lead["id"]) in c for c in gold_clusters)
    )
    print("  ...of which the ground truth also finds: " + str(covered))
    print(
        "  => that annotation labels only "
        + str(round(100 * marked / max(1, sum(len(c) for c in gold_clusters)), 1))
        + "% of the records that are actually duplicated, which is why it is"
    )
    print("     used as a cross-check and never as a label or a feature.")

    _rule("2. Candidate generation (scale)")
    start = time.time()
    pairs, _provenance, stats, _index = dedupe.generate_candidates(leads)
    elapsed = time.time() - start
    print("full pairwise comparisons    : " + format(stats["full_pairwise_comparisons"], ","))
    print("candidate pairs generated    : " + format(stats["candidate_pairs"], ","))
    print("comparisons avoided          : " + format(stats["reduction_ratio"] * 100, ".2f") + "%")
    print("wall clock                   : " + format(elapsed, ".2f") + "s")
    recovered = len(gold_pairs & pairs)
    print(
        "blocking recall              : "
        + str(recovered) + "/" + str(len(gold_pairs))
        + " (" + format(100 * recovered / max(1, len(gold_pairs)), ".1f") + "%)"
    )
    missed = gold_pairs - pairs
    if missed:
        print("  missed pairs (blocking can never recover these):")
        for a, b in sorted(missed)[:10]:
            print("    " + str(a) + " / " + str(b) + "  "
                  + str(by_id[a].get("display_name")) + " | " + str(by_id[b].get("display_name")))

    _rule("3. Pair scoring")
    weights = build_company_token_weights(lead.get("company_name") for lead in leads)
    scores = {}
    for id_a, id_b in pairs:
        features = dedupe.pair_features(by_id[id_a], by_id[id_b], weights)
        scores[(id_a, id_b)] = dedupe.score_features(features)

    print("threshold   TP     FP     FN   precision   recall      F1")
    for threshold in THRESHOLDS:
        predicted = {pair for pair, score in scores.items() if score >= threshold}
        true_positive = len(predicted & gold_pairs)
        false_positive = len(predicted - gold_pairs)
        false_negative = len(gold_pairs - predicted)
        precision, recall, f1 = _prf(true_positive, false_positive, false_negative)
        marker = "  <- default" if abs(threshold - config.DEDUPE_REPORT_MIN) < 1e-9 else ""
        print(
            format(threshold, "<11.2f")
            + format(true_positive, "<7") + format(false_positive, "<7") + format(false_negative, "<6")
            + format(precision, "<12.4f") + format(recall, "<11.4f") + format(f1, ".4f")
            + marker
        )

    default = {p for p, s in scores.items() if s >= config.DEDUPE_REPORT_MIN}
    for label, pairs_to_show in (
        ("false positives at the default threshold", sorted(default - gold_pairs)),
        ("false negatives at the default threshold", sorted(gold_pairs - default)),
    ):
        print("\n" + label + ": " + str(len(pairs_to_show)))
        for a, b in pairs_to_show[:8]:
            lead_a, lead_b = by_id[a], by_id[b]
            print(
                "  " + format(scores.get((a, b), 0.0), ".3f") + "  "
                + str(lead_a.get("display_name")) + " / " + str(lead_b.get("display_name"))
                + "  |  " + str(lead_a.get("email")) + " / " + str(lead_b.get("email"))
                + "  |  " + str(lead_a.get("phone_raw")) + " / " + str(lead_b.get("phone_raw"))
            )

    _rule("4. Ambiguous band (what an LLM is asked to adjudicate)")
    band = sorted(
        (
            (score, pair) for pair, score in scores.items()
            if config.DEDUPE_REVIEW_LOW <= score < config.DEDUPE_AUTO_MERGE
        ),
        reverse=True,
    )
    print(
        "pairs in [" + str(config.DEDUPE_REVIEW_LOW) + ", "
        + str(config.DEDUPE_AUTO_MERGE) + "): " + str(len(band))
        + "  (vs " + format(stats["full_pairwise_comparisons"], ",") + " if every pair were sent to a model)"
    )
    for score, (a, b) in band:
        verdict = "DUPLICATE" if (a, b) in gold_pairs else "different people"
        print(
            "  " + format(score, ".3f") + "  " + str(by_id[a].get("display_name"))
            + " @ " + str(by_id[a].get("company_name"))
            + "  vs  " + str(by_id[b].get("company_name"))
            + "   [ground truth: " + verdict + "]"
        )

    _rule("5. Clustering")
    result = dedupe.find_duplicate_candidates(leads, use_llm=False)
    predicted_clusters = {tuple(g["lead_ids"]) for g in result["groups"]}
    gold_set = {tuple(c) for c in gold_clusters}
    exact = len(predicted_clusters & gold_set)
    print("predicted clusters           : " + str(len(predicted_clusters)))
    print("true clusters                : " + str(len(gold_set)))
    print(
        "exactly matching             : " + str(exact)
        + " (" + format(100 * exact / max(1, len(gold_set)), ".1f") + "%)"
    )


def evaluate_source():
    _rule("6. Source extraction")
    with open(config.SEED_CSV, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    results = [extract_source(row["Notes"]) for row in rows]
    unclassified = sum(1 for r in results if r["method"] == "fallback:unclassified")
    print("notes processed              : " + str(len(rows)))
    print(
        "classified by rules          : " + str(len(rows) - unclassified)
        + " (" + format(100 * (len(rows) - unclassified) / len(rows), ".1f") + "%)"
    )
    print("needing an LLM fallback      : " + str(unclassified))

    checked = agree = 0
    disagreements = []
    for row, result in zip(rows, results):
        original = (row["Original Source"] or "").strip()
        if not original:
            continue
        checked += 1
        if original in ORIGINAL_SOURCE_EXPECTATION[result["channel"]]:
            agree += 1
        else:
            disagreements.append((original, result["channel"], row["Notes"][:70]))

    print(
        "\nheld-out check against the populated `Original Source` column"
        "\n(never an input to the extractor; blank on ~50% of rows)"
    )
    print("rows with a populated value  : " + str(checked))
    print(
        "consistent with extraction   : " + str(agree)
        + " (" + format(100 * agree / max(1, checked), ".1f") + "%)"
    )
    for original, channel, note in disagreements[:10]:
        print("  original=" + original + " extracted=" + channel + "  note=" + note)

    channels = {}
    gaps = 0
    for result in results:
        channels[result["channel"]] = channels.get(result["channel"], 0) + 1
        if result["taxonomy_gap"]:
            gaps += 1
    print("\nchannel distribution:")
    for channel, count in sorted(channels.items(), key=lambda kv: -kv[1]):
        print("  " + format(channel, "<16") + format(count, ">5"))
    print(
        "\n" + str(gaps) + " leads carry a taxonomy gap (real channel is Paid Search,"
        " which the required enum has no bucket for)."
    )

    _rule("7. Form submission messages (unseen phrasing)")
    import json

    with open(config.FORM_SUBMISSIONS_JSON, encoding="utf-8") as handle:
        submissions = json.load(handle)
    by_method = {}
    for submission in submissions:
        result = extract_source(
            submission["message"], hints={"page_url": submission.get("page_url")}
        )
        by_method[result["method"]] = by_method.get(result["method"], 0) + 1
    print("submissions                  : " + str(len(submissions)))
    for method, count in sorted(by_method.items(), key=lambda kv: -kv[1]):
        print("  " + format(method, "<32") + format(count, ">4"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--source", action="store_true")
    args = parser.parse_args()
    run_all = not (args.dedupe or args.source)

    if run_all or args.dedupe:
        evaluate_dedupe(load_leads())
    if run_all or args.source:
        evaluate_source()
    print()


if __name__ == "__main__":
    main()

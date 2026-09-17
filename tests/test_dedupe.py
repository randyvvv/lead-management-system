"""Deduplication: scoring behaviour, scale, clustering, and leakage."""

import json
import unittest

from leadms import config, dedupe, store
from leadms.similarity import build_company_token_weights


def lead(lead_id, name, company, email, phone, country="Spain", title=None, notes=""):
    """Build a record in the shape ``leadms.dedupe`` consumes."""
    from leadms import normalize

    resolved = normalize.resolve_name(None, None, name)
    local, domain = normalize.split_email(email)
    return {
        "id": lead_id,
        "display_name": resolved["display"],
        "first_name": resolved["first"],
        "last_name": resolved["last"],
        "name_key": resolved["key"],
        "name_tokens": resolved["tokens"],
        "company_name": company,
        "company_core": normalize.company_core(company),
        "email": email.lower(),
        "email_local": local,
        "email_domain": domain,
        "domain_root": normalize.domain_root(domain),
        "phone_raw": phone,
        "phone_digits": normalize.phone_digits(phone),
        "phone_key": normalize.phone_key(phone),
        "country": country,
        "job_title": title,
        "status": "New",
        "created_at": "2026-01-01",
        "notes": notes,
    }


def score(a, b, weights=None):
    return dedupe.score_features(dedupe.pair_features(a, b, weights))


class TestScoring(unittest.TestCase):
    def test_reformatted_duplicate_scores_high(self):
        """Same person: email localpart rewritten, phone reformatted,
        company legal suffix swapped, name moved to Full Name."""
        a = lead(1, "Erik Almeida", "Lotus Finance Pte. Ltd.",
                 "erik.a@lotusfinance.biz", "+1 693 555 0198", "United States")
        b = lead(2, "Erik Almeida", "Lotus Finance Analytics",
                 "erika@lotusfinance.biz", "16935550198", "United States")
        self.assertGreaterEqual(score(a, b), config.DEDUPE_AUTO_MERGE)

    def test_abbreviated_first_name_still_matches(self):
        a = lead(1, "Ji-woo Yoon", "Foster Partners", "ji-woo.yoon@foster.biz", "34696235827")
        b = lead(2, "J. Yoon", "Foster Trading", "j.yoon@foster.biz", "34696235827")
        self.assertGreaterEqual(score(a, b), config.DEDUPE_AUTO_MERGE)

    def test_colleagues_sharing_a_surname_score_low(self):
        """Different given names at one company are not duplicates."""
        a = lead(1, "Ama Bianchi", "Kilat Retail Solutions",
                 "ama.bianchi@kilatretail.co", "+63 957 700 4578", "Philippines")
        b = lead(2, "Farah Bianchi", "Kilat Retail Solutions",
                 "farahb@kilatretail.co", "+234 806 965 2341", "Nigeria")
        self.assertLess(score(a, b), 0.1)

    def test_similar_but_different_surnames_score_low(self):
        a = lead(1, "Sven Han", "Tanaka Partners", "sven.han@tanaka.com.au",
                 "+234 807 897 6106", "Nigeria")
        b = lead(2, "Sven Tan", "Tanaka Inc", "svent@tanaka.biz",
                 "+351 91 436 7318", "Portugal")
        self.assertLess(score(a, b), 0.1)

    def test_same_name_same_company_no_shared_contact_is_ambiguous(self):
        """The case field comparison genuinely cannot settle.

        Equally consistent with one person re-entered and with two colleagues
        who share a name, so it must land in the band that gets a model's
        opinion - not be silently resolved either way.
        """
        a = lead(1, "Lucas Schmidt", "Chua Textiles Retail Group",
                 "lucas.schmidt@chuatextiles.biz", "+234 803 135 4251", "Nigeria")
        b = lead(2, "Lucas Schmidt", "Chua Textiles Retail Group",
                 "lucas.s@chuatextiles.com", "+84 91 985 5253", "Vietnam")
        value = score(a, b)
        self.assertGreaterEqual(value, config.DEDUPE_REVIEW_LOW)
        self.assertLess(value, config.DEDUPE_AUTO_MERGE)

    def test_same_name_different_company_is_not_a_duplicate(self):
        a = lead(1, "Sanjay Martins", "Chen Digital Inc.",
                 "sanjay.martins@chendigital.net", "+52 55 9007 3072", "Mexico")
        b = lead(2, "Sanjay Martins", "Huang Analytics Group",
                 "s.martins@huanganalytics.co", "+55 11 97670-3363", "Brazil")
        self.assertLess(score(a, b), 0.1)

    def test_score_is_a_probability(self):
        a = lead(1, "A B", "C Ltd", "a.b@c.com", "+1 555 000 0001")
        b = lead(2, "X Y", "Z Ltd", "x.y@z.com", "+44 555 000 0002")
        for value in (score(a, a), score(a, b)):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestExplanations(unittest.TestCase):
    def test_reasons_name_the_evidence(self):
        a = lead(1, "Erik Almeida", "Lotus Ltd", "erik.a@lotus.biz", "+1 693 555 0198")
        b = lead(2, "Erik Almeida", "Lotus Analytics", "erika@lotus.biz", "16935550198")
        reasons = dedupe.explain(dedupe.pair_features(a, b))
        self.assertIn("identical phone number", reasons)
        self.assertTrue(any("localpart" in r for r in reasons))

    def test_conflicts_are_surfaced(self):
        a = lead(1, "Lucas Schmidt", "Chua Ltd", "l.schmidt@chua.biz",
                 "+234 803 135 4251", "Nigeria")
        b = lead(2, "Lucas Schmidt", "Chua Ltd", "lucas.s@chua.com",
                 "+84 91 985 5253", "Vietnam")
        reasons = dedupe.explain(dedupe.pair_features(a, b))
        self.assertTrue(any(r.startswith("CONFLICT") for r in reasons))
        self.assertTrue(any(r.startswith("AMBIGUOUS") for r in reasons))


class TestLocalpartLogic(unittest.TestCase):
    def test_patterns_cover_the_observed_spellings(self):
        patterns = dedupe.name_local_patterns("Erik", "Almeida")
        for spelling in ("erika", "erikalmeida", "ealmeida", "erik", "almeida"):
            self.assertIn(spelling, patterns)

    def test_cross_consistency_requires_both_directions(self):
        a = lead(1, "Erik Almeida", "X", "erika@x.com", "+1 555 000 0001")
        b = lead(2, "Erik Almeida", "X", "erikalmeida@x.com", "+1 555 000 0001")
        self.assertEqual(dedupe.local_consistency(a, b), 1.0)

        c = lead(3, "Erik Almeida", "X", "webmaster@x.com", "+1 555 000 0001")
        self.assertLess(dedupe.local_consistency(a, c), 1.0)

    def test_initial_compatibility(self):
        self.assertEqual(dedupe.initial_compatible(["j", "yoon"], ["ji-woo", "yoon"]), 1.0)
        self.assertEqual(dedupe.initial_compatible(["ama", "bianchi"], ["farah", "bianchi"]), 0.0)
        self.assertEqual(dedupe.initial_compatible(["a"], ["a", "b"]), 0.0)


class TestNoMarkerLeakage(unittest.TestCase):
    """The fixture's 'possible duplicate' annotation must not influence
    anything. It is a hint that would not exist in production, and it labels
    only part of the duplicate population."""

    MARKER = " possible duplicate — verify before contacting."

    def test_notes_do_not_change_the_score(self):
        a = lead(1, "Erik Almeida", "Lotus Ltd", "erik.a@lotus.biz",
                 "+1 693 555 0198", notes="Filled out the form.")
        b_clean = lead(2, "Erik Almeida", "Lotus Analytics", "erika@lotus.biz",
                       "16935550198", notes="Filled out the form.")
        b_marked = lead(2, "Erik Almeida", "Lotus Analytics", "erika@lotus.biz",
                        "16935550198", notes="Filled out the form." + self.MARKER)
        self.assertEqual(
            dedupe.pair_features(a, b_clean), dedupe.pair_features(a, b_marked)
        )

    def test_features_never_read_notes(self):
        a = lead(1, "Erik Almeida", "Lotus Ltd", "erik.a@lotus.biz", "+1 693 555 0198")
        b = lead(2, "Erik Almeida", "Lotus Analytics", "erika@lotus.biz", "16935550198")
        del a["notes"], b["notes"]
        dedupe.pair_features(a, b)  # must not raise

    def test_marker_is_stripped_before_reaching_a_prompt(self):
        record = lead(1, "Erik Almeida", "Lotus Ltd", "erik.a@lotus.biz",
                      "+1 693 555 0198", notes="Filled out the form." + self.MARKER)
        summary = dedupe._llm_summary(record)
        self.assertNotIn("possible duplicate", json.dumps(summary).lower())


class TestCandidateGeneration(unittest.TestCase):
    """Scale is an explicit requirement: no full pairwise sweep."""

    @classmethod
    def setUpClass(cls):
        conn = store.connect(":memory:")
        store.init_schema(conn)
        store.load_seed(conn)
        cls.leads = store.leads_for_dedupe(conn)

    def test_candidates_are_a_small_fraction_of_all_pairs(self):
        _pairs, _prov, stats, _by_id = dedupe.generate_candidates(self.leads)
        self.assertEqual(stats["records"], 2049)
        self.assertEqual(stats["full_pairwise_comparisons"], 2049 * 2048 // 2)
        self.assertGreater(stats["reduction_ratio"], 0.95)
        self.assertLess(
            stats["candidate_pairs"], stats["full_pairwise_comparisons"] * 0.05
        )

    def test_blocking_finds_duplicates_that_share_no_exact_field(self):
        """A pair whose email and phone both differ must still be generated."""
        subset = self.leads[:200] + [
            lead(999001, "Lucas Schmidt", "Chua Textiles Retail Group",
                 "lucas.schmidt@chuatextiles.biz", "+234 803 135 4251"),
            lead(999002, "Lucas Schmidt", "Chua Textiles Retail Group",
                 "lucas.s@chuatextiles.com", "+84 91 985 5253"),
        ]
        pairs, _prov, _stats, _by_id = dedupe.generate_candidates(subset)
        self.assertIn((999001, 999002), pairs)

    def test_oversized_blocks_do_not_explode(self):
        """A shared domain across many records must not go quadratic."""
        crowd = [
            lead(900000 + i, "Person " + str(i), "Big Corp",
                 "user" + str(i) + "@bigcorp.com", "+1 555 100 " + str(1000 + i))
            for i in range(120)
        ]
        _pairs, _prov, stats, _by_id = dedupe.generate_candidates(
            crowd, max_block_size=80
        )
        self.assertGreater(stats["oversized_blocks_skipped"], 0)
        self.assertLess(stats["candidate_pairs"], 120 * 119 / 2)


class TestClustering(unittest.TestCase):
    def test_transitive_pairs_become_one_group(self):
        """A-B and B-C must surface as a single three-record group."""
        leads = [
            lead(1, "Erik Almeida", "Lotus Pte. Ltd.", "erik.a@lotus.biz", "+1 693 555 0198"),
            lead(2, "Erik Almeida", "Lotus Analytics", "erika@lotus.biz", "16935550198"),
            lead(3, "Erik Almeida", "Lotus & Co", "erikalmeida@lotus.biz", "+1 693 555 0198"),
            lead(4, "Farah Bianchi", "Kilat Ltd", "farahb@kilat.co", "+63 957 700 4578"),
        ]
        result = dedupe.find_duplicate_candidates(leads, use_llm=False)
        self.assertEqual(len(result["groups"]), 1)
        group = result["groups"][0]
        self.assertEqual(group["lead_ids"], [1, 2, 3])
        self.assertEqual(group["size"], 3)
        self.assertNotIn(4, group["lead_ids"])

    def test_group_confidence_is_the_weakest_link(self):
        leads = [
            lead(1, "Erik Almeida", "Lotus Ltd", "erik.a@lotus.biz", "+1 693 555 0198"),
            lead(2, "Erik Almeida", "Lotus Analytics", "erika@lotus.biz", "16935550198"),
        ]
        group = dedupe.find_duplicate_candidates(leads, use_llm=False)["groups"][0]
        self.assertLessEqual(group["confidence"], group["max_pair_confidence"])

    def test_empty_and_single_record_inputs(self):
        self.assertEqual(dedupe.find_duplicate_candidates([])["groups"], [])
        one = [lead(1, "A B", "C", "a@c.com", "+1 555 000 0001")]
        self.assertEqual(dedupe.find_duplicate_candidates(one)["groups"], [])


class RecordingLLM:
    """Adjudicator stub that records which pairs it was asked about."""

    name = "recording"
    is_model_backed = True

    def __init__(self, verdict="different", confidence=0.9):
        self.verdict, self.confidence = verdict, confidence
        self.seen = []

    def adjudicate_duplicate(self, a, b, evidence):
        self.seen.append((a, b))
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reason": "stub",
            "provider": self.name,
        }


class TestAdjudication(unittest.TestCase):
    AMBIGUOUS = [
        lead(1, "Lucas Schmidt", "Chua Textiles Retail Group",
             "lucas.schmidt@chuatextiles.biz", "+234 803 135 4251", "Nigeria"),
        lead(2, "Lucas Schmidt", "Chua Textiles Retail Group",
             "lucas.s@chuatextiles.com", "+84 91 985 5253", "Vietnam"),
    ]
    CERTAIN = [
        lead(3, "Erik Almeida", "Lotus Ltd", "erik.a@lotus.biz", "+1 693 555 0198"),
        lead(4, "Erik Almeida", "Lotus Analytics", "erik.a@lotus.biz", "16935550198"),
    ]

    def test_only_the_ambiguous_band_is_sent_to_the_model(self):
        client = RecordingLLM()
        dedupe.find_duplicate_candidates(
            self.AMBIGUOUS + self.CERTAIN, llm=client, use_llm=True
        )
        self.assertEqual(len(client.seen), 1, "only the ambiguous pair should be sent")
        names = {client.seen[0][0]["name"], client.seen[0][1]["name"]}
        self.assertEqual(names, {"Lucas Schmidt"})

    def test_a_same_verdict_promotes_the_pair(self):
        client = RecordingLLM(verdict="same", confidence=0.9)
        result = dedupe.find_duplicate_candidates(
            list(self.AMBIGUOUS), llm=client, use_llm=True
        )
        self.assertEqual(len(result["groups"]), 1)
        self.assertGreaterEqual(
            result["groups"][0]["confidence"], config.DEDUPE_AUTO_MERGE
        )

    def test_a_different_verdict_demotes_the_pair(self):
        client = RecordingLLM(verdict="different", confidence=0.9)
        result = dedupe.find_duplicate_candidates(
            list(self.AMBIGUOUS), llm=client, use_llm=True
        )
        self.assertEqual(result["groups"], [])

    def test_an_abstention_leaves_the_score_untouched(self):
        client = RecordingLLM(verdict="unsure", confidence=0.0)
        with_llm = dedupe.find_duplicate_candidates(
            list(self.AMBIGUOUS), llm=client, use_llm=True,
            min_confidence=config.DEDUPE_REVIEW_LOW,
        )
        without = dedupe.find_duplicate_candidates(
            list(self.AMBIGUOUS), use_llm=False,
            min_confidence=config.DEDUPE_REVIEW_LOW,
        )
        self.assertEqual(
            with_llm["groups"][0]["confidence"], without["groups"][0]["confidence"]
        )

    def test_a_failing_model_does_not_break_the_report(self):
        class Boom:
            name = "boom"
            is_model_backed = True

            def adjudicate_duplicate(self, a, b, evidence):
                raise RuntimeError("provider down")

        result = dedupe.find_duplicate_candidates(
            list(self.AMBIGUOUS), llm=Boom(), use_llm=True,
            min_confidence=config.DEDUPE_REVIEW_LOW,
        )
        self.assertEqual(len(result["groups"]), 1)


class TestAgainstGroundTruth(unittest.TestCase):
    """Regression guard on the headline numbers.

    Ground truth comes from ``eval/ground_truth.py``, which labels pairs using
    the fixture's Record ID insertion order - information the scorer never
    reads. See that module for why this is not circular.
    """

    @classmethod
    def setUpClass(cls):
        from eval.ground_truth import true_clusters, true_pairs

        conn = store.connect(":memory:")
        store.init_schema(conn)
        store.load_seed(conn)
        cls.leads = store.leads_for_dedupe(conn)
        cls.gold_pairs = true_pairs(cls.leads)
        cls.gold_clusters = {tuple(c) for c in true_clusters(cls.leads)}

    def test_blocking_loses_no_true_duplicate(self):
        """A pair dropped here can never be recovered by any later stage."""
        pairs, _prov, _stats, _by_id = dedupe.generate_candidates(self.leads)
        missed = self.gold_pairs - pairs
        self.assertEqual(missed, set(), "blocking dropped " + str(len(missed)) + " true pairs")

    def test_precision_and_recall_at_the_default_threshold(self):
        result = dedupe.find_duplicate_candidates(self.leads, use_llm=False)
        predicted = {
            tuple(sorted((p["a"], p["b"])))
            for group in result["groups"]
            for p in group["pairs"]
        }
        false_positives = predicted - self.gold_pairs
        false_negatives = self.gold_pairs - predicted
        self.assertEqual(false_positives, set(), "unexpected false positives")
        self.assertEqual(false_negatives, set(), "missed true duplicates")

    def test_clusters_match_exactly(self):
        result = dedupe.find_duplicate_candidates(self.leads, use_llm=False)
        predicted = {tuple(g["lead_ids"]) for g in result["groups"]}
        self.assertEqual(predicted, self.gold_clusters)

    def test_the_planted_lookalikes_are_never_grouped_together(self):
        """The brief plants distinct people sharing a company and a name.

        Each of these records may legitimately belong to its own duplicate
        group (100236210 really is a duplicate of its ID-neighbour); what must
        not happen is the two lookalikes landing in the *same* group. These
        are also the pairs that fall into the LLM adjudication band.
        """
        result = dedupe.find_duplicate_candidates(self.leads, use_llm=False)
        lookalike_pairs = [
            (100235532, 100235902),  # Lucas Schmidt @ Chua Textiles
            (100235561, 100236210),  # Marcus Ho @ Asante Retail
            (100236269, 100236741),  # Arjun Carvalho @ Lotus Finance
        ]
        for group in result["groups"]:
            members = set(group["lead_ids"])
            for a, b in lookalike_pairs:
                self.assertFalse(
                    {a, b} <= members,
                    str(a) + " and " + str(b) + " are different people but were grouped",
                )


class TestIngestMatching(unittest.TestCase):
    def test_rank_matches_puts_the_real_person_first(self):
        existing = [
            lead(1, "Karim Toure", "Liu Trading Studio", "k.toure@liutrading.biz",
                 "+61 462 210 338", "Australia"),
            lead(2, "Other Person", "Liu Trading Ltd", "other@liutrading.biz",
                 "+61 400 000 000", "Australia"),
        ]
        incoming = lead(None, "Karim Toure", "Liu Trading Studio",
                        "k.toure@liutrading.biz", "+61 462 210 338", "Australia")
        weights = build_company_token_weights(l["company_name"] for l in existing)
        matches = dedupe.rank_matches(incoming, existing, weights)
        self.assertEqual(matches[0]["lead_id"], 1)
        self.assertGreaterEqual(matches[0]["score"], config.DEDUPE_AUTO_MERGE)

    def test_no_candidates_returns_none(self):
        incoming = lead(None, "Nobody Here", "Nowhere", "n@nowhere.com", "+1 555 000 0000")
        self.assertIsNone(dedupe.best_match_for(incoming, []))


if __name__ == "__main__":
    unittest.main()

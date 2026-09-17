"""Store and HTTP API behaviour, exercised through the router."""

import json
import unittest

from leadms import api, config, store


class APITestCase(unittest.TestCase):
    """Fresh in-memory database per test class."""

    @classmethod
    def setUpClass(cls):
        cls.conn = store.connect(":memory:")
        store.init_schema(cls.conn)
        cls.load_result = store.load_seed(cls.conn)
        # llm_client=None keeps every test hermetic and offline.
        cls.app = api.LeadAPI(cls.conn, llm_client=None)

    def call(self, method, path, query=None, body=None):
        return self.app.handle(method, path, query or {}, body)


class TestLoading(APITestCase):
    def test_all_rows_loaded(self):
        self.assertEqual(self.load_result["inserted"], 2049)
        self.assertEqual(self.load_result["skipped_without_id"], 0)

    def test_normalised_and_raw_are_both_kept(self):
        row = self.conn.execute(
            "SELECT status, status_raw, owner, owner_raw, raw_json FROM leads"
            " WHERE status_raw != status LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row, "fixture should contain messy status spellings")
        self.assertIn(row["status"], ["New", "Contacted", "Connected", "Qualified",
                                      "Opportunity", "Closed Won", "Closed Lost"])
        self.assertTrue(json.loads(row["raw_json"]))

    def test_always_empty_columns_are_not_in_the_schema(self):
        columns = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(leads)").fetchall()
        }
        for dropped in ("city", "annual_revenue", "gdpr_consent"):
            self.assertNotIn(dropped, columns)


class TestListAndFilter(APITestCase):
    def test_unfiltered_total(self):
        status, payload, _ = self.call("GET", "/leads")
        self.assertEqual(status, 200)
        self.assertEqual(payload["total"], 2049)

    def test_status_filter_is_case_and_whitespace_insensitive(self):
        totals = set()
        for spelling in ("Qualified", "qualified", " QUALIFIED "):
            _, payload, _ = self.call("GET", "/leads", {"status": [spelling]})
            totals.add(payload["total"])
        self.assertEqual(len(totals), 1, "spellings should collapse to one result set")
        self.assertGreater(totals.pop(), 0)

    def test_owner_filter_ignores_trailing_space_variants(self):
        _, payload, _ = self.call("GET", "/leads", {"owner": ["marcus wong"]})
        self.assertGreater(payload["total"], 190)

    def test_country_filter_matches_normalised_casing(self):
        _, payload, _ = self.call("GET", "/leads", {"country": ["uae"]})
        self.assertGreater(payload["total"], 0)

    def test_free_text_search_spans_name_company_and_email(self):
        for needle in ("lotus", "almeida", "@bluepeak"):
            _, payload, _ = self.call("GET", "/leads", {"q": [needle]})
            self.assertGreater(payload["total"], 0, needle)

    def test_filters_combine(self):
        _, both, _ = self.call(
            "GET", "/leads", {"status": ["New"], "channel": ["Referral"]}
        )
        _, one, _ = self.call("GET", "/leads", {"status": ["New"]})
        self.assertLessEqual(both["total"], one["total"])

    def test_pagination(self):
        _, first, _ = self.call("GET", "/leads", {"limit": ["5"]})
        _, second, _ = self.call("GET", "/leads", {"limit": ["5"], "offset": ["5"]})
        self.assertEqual(len(first["leads"]), 5)
        self.assertNotEqual(
            [l["id"] for l in first["leads"]], [l["id"] for l in second["leads"]]
        )

    def test_invalid_parameters_are_rejected(self):
        for query in ({"limit": ["abc"]}, {"limit": ["9999"]}, {"offset": ["-1"]}):
            status, payload, _ = self.call("GET", "/leads", query)
            self.assertEqual(status, 400, query)
            self.assertIn("error", payload)


class TestSingleLead(APITestCase):
    def test_get_existing(self):
        status, payload, _ = self.call("GET", "/leads/100234811")
        self.assertEqual(status, 200)
        self.assertEqual(payload["id"], 100234811)

    def test_get_missing(self):
        status, _, _ = self.call("GET", "/leads/999999999")
        self.assertEqual(status, 404)


class TestPatch(APITestCase):
    def test_status_is_normalised_and_raw_preserved(self):
        status, payload, _ = self.call(
            "PATCH", "/leads/100234812", body={"status": " closed won "}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "Closed Won")
        self.assertEqual(payload["status_raw"], " closed won ")

    def test_owner_and_notes(self):
        _, payload, _ = self.call(
            "PATCH", "/leads/100234813",
            body={"owner": "Wei Chen ", "notes": "Call booked"},
        )
        self.assertEqual(payload["owner"], "Wei Chen")
        self.assertEqual(payload["notes"], "Call booked")

    def test_updated_at_is_touched(self):
        _, payload, _ = self.call("PATCH", "/leads/100234814", body={"status": "New"})
        self.assertIsNotNone(payload["updated_at"])

    def test_unknown_status_is_rejected(self):
        status, payload, _ = self.call(
            "PATCH", "/leads/100234815", body={"status": "Frozen"}
        )
        self.assertEqual(status, 400)
        self.assertIn("unrecognised status", payload["error"])

    def test_unsupported_field_is_rejected_not_ignored(self):
        status, payload, _ = self.call(
            "PATCH", "/leads/100234815", body={"email": "new@example.com"}
        )
        self.assertEqual(status, 400)
        self.assertIn("email", payload["error"])

    def test_patch_missing_lead(self):
        status, _, _ = self.call("PATCH", "/leads/999999999", body={"status": "New"})
        self.assertEqual(status, 404)

    def test_empty_body(self):
        status, _, _ = self.call("PATCH", "/leads/100234815", body={})
        self.assertEqual(status, 400)


class TestExport(APITestCase):
    def test_export_is_not_parsed_as_a_lead_id(self):
        """/leads/export must not collide with /leads/{id}."""
        status, payload, kind = self.call("GET", "/leads/export")
        self.assertEqual(status, 200)
        self.assertEqual(kind, "csv")

    def test_export_respects_the_current_filter(self):
        _, listing, _ = self.call("GET", "/leads", {"status": ["New"]})
        _, csv_text, _ = self.call("GET", "/leads/export", {"status": ["New"]})
        # header + one line per matching row
        self.assertEqual(len(csv_text.strip().splitlines()), listing["total"] + 1)

    def test_export_header_matches_the_list_columns(self):
        _, csv_text, _ = self.call("GET", "/leads/export", {"limit": ["1"]})
        self.assertEqual(
            csv_text.splitlines()[0].split(","), store.LIST_COLUMNS
        )


class TestRouting(APITestCase):
    def test_unknown_path(self):
        status, _, _ = self.call("GET", "/nope")
        self.assertEqual(status, 404)

    def test_method_not_allowed_lists_allowed_methods(self):
        status, payload, _ = self.call("POST", "/leads/100234811")
        self.assertEqual(status, 405)
        self.assertIn("GET", payload["allowed"])
        self.assertIn("PATCH", payload["allowed"])

    def test_health_reports_llm_state(self):
        status, payload, _ = self.call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertIn("llm", payload)
        self.assertIn("providers", payload["llm"])


class TestDashboard(APITestCase):
    def test_counts_by_status_and_channel(self):
        status, payload, _ = self.call("GET", "/dashboard")
        self.assertEqual(status, 200)
        self.assertEqual(sum(payload["by_status"].values()), payload["total_leads"])
        self.assertEqual(
            sum(payload["by_source_channel"].values()), payload["total_leads"]
        )

    def test_statuses_are_canonical(self):
        _, payload, _ = self.call("GET", "/dashboard")
        self.assertEqual(len(payload["by_status"]), 7)

    def test_paid_search_gap_is_surfaced(self):
        _, payload, _ = self.call("GET", "/dashboard")
        self.assertIn("Paid Search", payload["taxonomy_gaps"])


class TestExtractEndpoint(APITestCase):
    def test_extracts_channel_and_detail(self):
        status, payload, _ = self.call(
            "POST", "/extract-source",
            body={"text": "Met him at the SFF booth, scanned our QR code"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["channel"], "Event")
        self.assertIn("Booth QR Code", payload["detail"])

    def test_blank_text_is_rejected(self):
        status, _, _ = self.call("POST", "/extract-source", body={"text": "   "})
        self.assertEqual(status, 400)


class TestIngest(unittest.TestCase):
    """Ingest mutates data, so this class gets its own database."""

    SUBMISSION = {
        "form_id": "form_pricing",
        "form_name": "Contact Us",
        "page_url": "/pricing",
        "submitted_at": "2026-06-12T18:17:00Z",
        "name": "Karim Toure",
        "email": "k.toure@liutrading.biz",
        "phone": "+61 462 210 338",
        "company": "Liu Trading Studio",
        "country": "Australia",
        "message": "Following up after our earlier conversation, please send more info.",
    }

    def setUp(self):
        self.conn = store.connect(":memory:")
        store.init_schema(self.conn)
        store.load_seed(self.conn)
        self.app = api.LeadAPI(self.conn, llm_client=None)

    def call(self, body):
        return self.app.handle("POST", "/leads/ingest", {}, body)

    def test_known_person_updates_instead_of_creating(self):
        before = self.conn.execute("SELECT COUNT(*) c FROM leads").fetchone()["c"]
        status, payload, _ = self.call(self.SUBMISSION)
        self.assertEqual(status, 200)
        self.assertEqual(payload["action"], "updated")
        self.assertGreaterEqual(payload["confidence"], config.DEDUPE_AUTO_MERGE)
        after = self.conn.execute("SELECT COUNT(*) c FROM leads").fetchone()["c"]
        self.assertEqual(after, before)

    def test_update_appends_the_message_without_clobbering_sales_fields(self):
        _, before, _ = self.app.handle("GET", "/leads/100235478", {}, None)
        self.call(self.SUBMISSION)
        _, after, _ = self.app.handle("GET", "/leads/100235478", {}, None)
        self.assertEqual(after["status"], before["status"])
        self.assertEqual(after["owner"], before["owner"])
        self.assertIn(self.SUBMISSION["message"], after["notes"])

    def test_new_person_creates_a_lead(self):
        submission = dict(
            self.SUBMISSION,
            name="Zelda Nakamura",
            email="zelda.nakamura@brandnewco.example",
            phone="+81 90 1111 2222",
            company="Brand New Co",
            message="Filled out the form on the pricing page.",
        )
        status, payload, _ = self.call(submission)
        self.assertEqual(status, 200)
        self.assertEqual(payload["action"], "created")
        self.assertEqual(payload["source"]["channel"], "Website")
        self.assertEqual(payload["lead"]["origin"], "form")
        self.assertEqual(payload["lead"]["status"], "New")

    def test_submission_without_contact_details_is_rejected(self):
        status, payload, _ = self.call({"name": "No Contact Details"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["action"], "rejected")

    def test_uninformative_message_falls_back_to_the_page_url(self):
        submission = dict(
            self.SUBMISSION,
            name="Brand New Person",
            email="bnp@somewhereelse.example",
            phone="+1 202 555 0143",
            company="Somewhere Else",
            page_url="/book-a-demo",
        )
        _, payload, _ = self.call(submission)
        self.assertEqual(payload["source"]["channel"], "Website")
        self.assertIn("book a demo", payload["source"]["detail"])

    def test_batch_payload_returns_a_summary(self):
        with open(config.FORM_SUBMISSIONS_JSON, encoding="utf-8") as handle:
            submissions = json.load(handle)
        status, payload, _ = self.call(submissions)
        self.assertEqual(status, 200)
        self.assertEqual(payload["processed"], len(submissions))
        self.assertEqual(
            payload["created"] + payload["updated"] + payload["rejected"],
            len(submissions),
        )
        # About half of the fixture is people already in the CSV.
        self.assertGreater(payload["updated"], 30)
        self.assertGreater(payload["created"], 10)

    def test_ingest_is_idempotent_for_the_same_submission(self):
        self.call(self.SUBMISSION)
        after_first = self.conn.execute("SELECT COUNT(*) c FROM leads").fetchone()["c"]
        self.call(self.SUBMISSION)
        after_second = self.conn.execute("SELECT COUNT(*) c FROM leads").fetchone()["c"]
        self.assertEqual(after_first, after_second)

    def test_bad_body_shape(self):
        status, _, _ = self.call("not an object")
        self.assertEqual(status, 400)


class TestDedupeEndpoint(APITestCase):
    def test_returns_ranked_groups_and_scale_stats(self):
        status, payload, _ = self.call(
            "POST", "/leads/dedupe-candidates",
            body={"use_llm": False, "limit": 5},
        )
        self.assertEqual(status, 200)
        self.assertLessEqual(len(payload["groups"]), 5)
        confidences = [g["confidence"] for g in payload["groups"]]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

        stats = payload["stats"]
        self.assertLess(stats["candidate_pairs"], stats["full_pairwise_comparisons"])
        self.assertGreater(stats["reduction_ratio"], 0.95)

    def test_groups_carry_explanations(self):
        _, payload, _ = self.call(
            "POST", "/leads/dedupe-candidates", body={"use_llm": False, "limit": 1}
        )
        group = payload["groups"][0]
        self.assertGreaterEqual(len(group["lead_ids"]), 2)
        self.assertTrue(group["pairs"][0]["reasons"])

    def test_threshold_is_respected(self):
        _, high, _ = self.call(
            "POST", "/leads/dedupe-candidates",
            body={"use_llm": False, "min_confidence": 0.99},
        )
        _, low, _ = self.call(
            "POST", "/leads/dedupe-candidates",
            body={"use_llm": False, "min_confidence": 0.5},
        )
        self.assertLessEqual(len(high["groups"]), len(low["groups"]))

    def test_invalid_threshold(self):
        status, _, _ = self.call(
            "POST", "/leads/dedupe-candidates", body={"min_confidence": "high"}
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()

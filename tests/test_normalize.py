"""Field normalisation, aimed at the messy cases rather than the happy path."""

import csv
import unittest

from leadms import config, normalize


class TestStatus(unittest.TestCase):
    def test_all_observed_spellings_collapse_to_seven(self):
        """The export uses 35 spellings for 7 statuses."""
        spellings = [
            "New", "new", "NEW", " New", "New ",
            "Contacted", "contacted", "CONTACTED", " Contacted", "Contacted ",
            "Closed Won", "closed won", "CLOSED WON", " Closed Won", "Closed Won ",
            "Closed Lost", "closed lost", "CLOSED LOST", " Closed Lost", "Closed Lost ",
        ]
        results = {normalize.normalize_status(s) for s in spellings}
        self.assertEqual(
            results, {"New", "Contacted", "Closed Won", "Closed Lost"}
        )

    def test_whole_seed_file_normalises(self):
        """No row in the real file falls through to None."""
        with open(config.SEED_CSV, newline="", encoding="utf-8-sig") as handle:
            raw = {row["Lead Status"] for row in csv.DictReader(handle)}
        self.assertGreater(len(raw), 20, "fixture should contain many spellings")
        for value in raw:
            self.assertIn(
                normalize.normalize_status(value),
                normalize.CANONICAL_STATUSES,
                "failed to normalise " + repr(value),
            )

    def test_unknown_status_returns_none_rather_than_guessing(self):
        self.assertIsNone(normalize.normalize_status("Frozen"))
        self.assertIsNone(normalize.normalize_status(""))
        self.assertIsNone(normalize.normalize_status(None))


class TestDates(unittest.TestCase):
    def test_three_formats_in_one_column(self):
        self.assertEqual(normalize.parse_date("2026-06-02"), "2026-06-02")
        self.assertEqual(normalize.parse_date("2026-05-20T00:00:00Z"), "2026-05-20")
        self.assertEqual(normalize.parse_date("6/4/2026"), "2026-06-04")

    def test_slash_dates_are_month_first(self):
        """6/4/2026 is 4 June, not 6 April - justified in normalize.py."""
        self.assertEqual(normalize.parse_date("12/31/2025"), "2025-12-31")
        self.assertEqual(normalize.parse_date("1/29/2026"), "2026-01-29")

    def test_unparseable_returns_none(self):
        for value in ("", None, "not a date", "31/12/2025"):
            self.assertIsNone(normalize.parse_date(value), value)

    def test_timestamp_keeps_time_part(self):
        self.assertTrue(
            normalize.parse_timestamp("2026-06-12T18:17:00Z").startswith(
                "2026-06-12T18:17"
            )
        )


class TestPhone(unittest.TestCase):
    def test_reformatted_numbers_compare_equal(self):
        pairs = [
            ("+1 693 555 0198", "16935550198"),
            ("+46 70 460 23 83", "46704602383"),
            ("+62 897-3519-1389", "628973519 1389"),
        ]
        for a, b in pairs:
            self.assertEqual(
                normalize.phone_digits(a), normalize.phone_digits(b), (a, b)
            )

    def test_phone_key_is_last_nine_digits(self):
        self.assertEqual(normalize.phone_key("+1 693 555 0198"), "935550198")
        self.assertEqual(normalize.phone_key("short"), "")


class TestNames(unittest.TestCase):
    def test_first_last_form(self):
        resolved = normalize.resolve_name("Erik", "Almeida", "")
        self.assertEqual(resolved["display"], "Erik Almeida")
        self.assertEqual(resolved["key"], "erikalmeida")

    def test_full_name_only_form_is_split(self):
        """~5% of rows use Full Name instead of First/Last."""
        resolved = normalize.resolve_name("", "", "Erik Almeida")
        self.assertEqual(resolved["first"], "Erik")
        self.assertEqual(resolved["last"], "Almeida")
        self.assertEqual(resolved["key"], "erikalmeida")

    def test_initial_form_keeps_the_initial(self):
        resolved = normalize.resolve_name("", "", "J. Yoon")
        self.assertEqual(resolved["tokens"], ["j", "yoon"])
        self.assertTrue(normalize.is_initial("J."))
        self.assertFalse(normalize.is_initial("Ji-woo"))

    def test_single_token_name(self):
        resolved = normalize.resolve_name("", "", "Prince")
        self.assertEqual(resolved["display"], "Prince")

    def test_accents_fold(self):
        self.assertEqual(
            normalize.resolve_name("", "", "Inès Danso")["key"], "inesdanso"
        )


class TestCompany(unittest.TestCase):
    def test_suffix_swaps_reduce_to_the_same_core(self):
        variants = [
            "Lotus Finance Pte. Ltd.",
            "Lotus Finance Analytics",
            "Lotus Finance & Co",
        ]
        cores = {normalize.company_core(v) for v in variants}
        self.assertEqual(len(cores), 1, cores)

    def test_core_never_empties(self):
        self.assertTrue(normalize.company_core("Solutions Ltd"))
        self.assertEqual(normalize.company_core(""), "")


class TestCountry(unittest.TestCase):
    def test_casing_is_repaired(self):
        self.assertEqual(normalize.normalize_country("china"), "China")
        self.assertEqual(normalize.normalize_country("united kingdom"), "United Kingdom")

    def test_short_acronyms_are_preserved(self):
        """A naive .title() would produce 'Uae'."""
        self.assertEqual(normalize.normalize_country("UAE"), "UAE")


class TestEmail(unittest.TestCase):
    def test_split_and_domain_root(self):
        local, domain = normalize.split_email("B.Ba@BluePeak.com.au")
        self.assertEqual((local, domain), ("b.ba", "bluepeak.com.au"))
        self.assertEqual(normalize.domain_root(domain), "bluepeak")

    def test_missing_at_sign(self):
        self.assertEqual(normalize.split_email("garbage"), ("garbage", ""))

    def test_local_variants_include_punctuation_stripped_form(self):
        self.assertEqual(
            normalize.email_local_variants("erik.a"), {"erik.a", "erika"}
        )


class TestDuplicateMarker(unittest.TestCase):
    """The fixture's 'possible duplicate' hint must be strippable."""

    MARKED = (
        "Filled out the form on the contact page. Qualifying now. "
        "possible duplicate — verify before contacting."
    )

    def test_detected_and_removed(self):
        self.assertTrue(normalize.has_dup_marker(self.MARKED))
        cleaned = normalize.strip_dup_marker(self.MARKED)
        self.assertNotIn("possible duplicate", cleaned.lower())
        self.assertIn("Filled out the form", cleaned)

    def test_clean_note_is_untouched(self):
        note = "Filled out the form on the contact page."
        self.assertFalse(normalize.has_dup_marker(note))
        self.assertEqual(normalize.strip_dup_marker(note), note)

    def test_marker_count_in_fixture(self):
        with open(config.SEED_CSV, newline="", encoding="utf-8-sig") as handle:
            marked = sum(
                1 for row in csv.DictReader(handle)
                if normalize.has_dup_marker(row["Notes"])
            )
        self.assertEqual(marked, 136)


if __name__ == "__main__":
    unittest.main()

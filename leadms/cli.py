"""Command line entry points.

    python -m leadms.cli init          # create the DB and load the seed CSV
    python -m leadms.cli serve         # start the REST API
    python -m leadms.cli dedupe        # duplicate report
    python -m leadms.cli extract TEXT  # channel + detail for one note
    python -m leadms.cli ingest        # replay the form-submission fixture
    python -m leadms.cli dashboard     # counts by status / channel
    python -m leadms.cli profile       # data-quality profile of the raw CSV
    python -m leadms.cli llm-check     # which LLM provider is configured
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys

from . import api, config, dedupe, llm as llm_module, store
from .source_extract import extract_source


def _print(payload):
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _open_db(args, require_data=True):
    conn = store.connect(getattr(args, "db", None))
    store.init_schema(conn)
    if require_data:
        total = conn.execute("SELECT COUNT(*) AS c FROM leads").fetchone()["c"]
        if total == 0:
            print("Database is empty; loading seed CSV first...", file=sys.stderr)
            store.load_seed(conn)
    return conn


def cmd_init(args):
    conn = store.connect(args.db)
    store.init_schema(conn)
    result = store.load_seed(conn, args.csv, replace=not args.append)
    result["db"] = str(args.db or config.DB_PATH)
    _print(result)


def cmd_serve(args):
    api.serve(args.host, args.port, args.db)


def cmd_dedupe(args):
    conn = _open_db(args)
    client = None if args.no_llm else llm_module.build_client()
    leads = store.leads_for_dedupe(conn)
    result = dedupe.find_duplicate_candidates(
        leads,
        llm=client,
        min_confidence=args.min_confidence,
        use_llm=not args.no_llm,
        pair_floor=args.pair_floor,
    )
    if args.limit:
        result["groups"] = result["groups"][: args.limit]
    if not args.no_llm:
        result["llm"] = llm_module.describe_client(client)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, default=str)
        print("wrote " + args.out)
        _print(result["stats"])
    else:
        _print(result)


def cmd_extract(args):
    client = None if args.no_llm else llm_module.build_client()
    _print(extract_source(args.text, llm=client))


def cmd_ingest(args):
    conn = _open_db(args)
    with open(args.file, encoding="utf-8") as handle:
        submissions = json.load(handle)
    if args.limit:
        submissions = submissions[: args.limit]

    handler = api.LeadAPI(conn, llm_client=None if args.no_llm else llm_module.build_client())
    status, payload, _kind = handler.ingest(body=submissions)
    if args.verbose:
        _print(payload)
    else:
        _print({k: v for k, v in payload.items() if k != "results"})
    return 0 if status == 200 else 1


def cmd_dashboard(args):
    _print(store.dashboard(_open_db(args)))


def cmd_llm_check(args):
    client = llm_module.build_client(use_cache=False)
    described = llm_module.describe_client(client)
    print(json.dumps(described, indent=2))
    if not client.is_model_backed:
        print(
            "\nNo API key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY to enable\n"
            "model-backed adjudication and extraction. The service runs without\n"
            "one, using the documented offline heuristics."
        )
        return 0

    inner = getattr(client, "inner", client)
    for provider in getattr(inner, "providers", []):
        if provider.name == "gemini":
            try:
                models = provider.list_models()
            except llm_module.LLMUnavailable as exc:
                print("\nGemini: could not list models: " + str(exc))
                continue
            print("\nGemini models available to this key (" + str(len(models)) + "):")
            for name in models[:25]:
                marker = "  <- configured" if name == provider.model else ""
                print("  " + name + marker)
            if provider.model not in models:
                print(
                    "\nWARNING: LEADMS_GEMINI_MODEL=" + provider.model
                    + " is not in the list above. Set it to one of these."
                )

    print("\nSending one live test request...")
    try:
        result = client.extract_source("Met them at the Web Summit booth, scanned our QR code.")
        _print(result)
    except Exception as exc:
        print("Live call failed: " + type(exc).__name__ + ": " + str(exc))
        return 1
    return 0


def cmd_profile(args):
    """Data-quality profile of the raw CSV.

    Every "the data looks like this" claim in the README is produced here, so
    a reviewer can re-derive the numbers rather than take them on trust.
    """
    path = args.csv or config.SEED_CSV
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    total = len(rows)
    columns = list(rows[0].keys()) if rows else []
    fill = {
        column: sum(1 for r in rows if (r.get(column) or "").strip())
        for column in columns
    }
    always_empty = [c for c in columns if fill[c] == 0]

    def slash_components():
        first, second = [], []
        for row in rows:
            for column in ("Create Date", "Last Modified Date"):
                match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", (row.get(column) or "").strip())
                if match:
                    first.append(int(match.group(1)))
                    second.append(int(match.group(2)))
        return first, second

    first, second = slash_components()
    statuses = collections.Counter(r.get("Lead Status") for r in rows)
    owners = collections.Counter(r.get("Contact Owner") for r in rows)

    _print(
        {
            "rows": total,
            "columns": len(columns),
            "always_empty_columns": always_empty,
            "fill_rate_pct": {
                c: round(100 * fill[c] / total, 1) for c in columns if fill[c]
            },
            "lead_status": {
                "distinct_raw_spellings": len(statuses),
                "canonical_values": 7,
            },
            "contact_owner": {
                "distinct_raw_spellings": len(owners),
                "after_trimming": len({(o or "").strip() for o in owners}),
            },
            "slash_dates": {
                "count": len(first),
                "max_first_component": max(first) if first else None,
                "max_second_component": max(second) if second else None,
                "conclusion": "month-first (M/D/YYYY): first component never exceeds 12",
            },
            "duplicate_marker_rows": sum(
                1 for r in rows if "possible duplicate" in (r.get("Notes") or "").lower()
            ),
        }
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="leadms", description=__doc__)
    parser.add_argument("--db", help="SQLite path (default: ./leads.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("init", help="create the schema and load the seed CSV")
    p.add_argument("--csv", help="path to leads_seed.csv")
    p.add_argument("--append", action="store_true", help="keep existing rows")
    p.set_defaults(func=cmd_init)

    p = subparsers.add_parser("serve", help="start the REST API")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = subparsers.add_parser("dedupe", help="duplicate candidate report")
    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--pair-floor", type=float, default=None,
                   help="keep pairs scoring above this (default: the review floor)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help="skip model adjudication")
    p.add_argument("--out", help="write the full report to a JSON file")
    p.set_defaults(func=cmd_dedupe)

    p = subparsers.add_parser("extract", help="extract a source channel from text")
    p.add_argument("text")
    p.add_argument("--no-llm", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = subparsers.add_parser("ingest", help="replay the form submissions fixture")
    p.add_argument("--file", default=str(config.FORM_SUBMISSIONS_JSON))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = subparsers.add_parser("dashboard", help="counts by status and channel")
    p.set_defaults(func=cmd_dashboard)

    p = subparsers.add_parser("profile", help="data-quality profile of the raw CSV")
    p.add_argument("--csv", help="path to leads_seed.csv")
    p.set_defaults(func=cmd_profile)

    p = subparsers.add_parser("llm-check", help="show which LLM provider is active")
    p.set_defaults(func=cmd_llm_check)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

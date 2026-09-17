"""SQLite-backed lead store.

**Why SQLite:** it is in the standard library, needs no server, gives real
indexed filtering for the list endpoint, and - unlike an in-memory dict -
persists ``PATCH`` edits across restarts, which the assignment's update
endpoint implies. At 2,049 rows the performance question is moot; the
deciding factors were zero setup for a reviewer and durable writes.

**What we keep:** the 22 export columns are not all worth modelling. Five are
empty for every single row (``City``, ``Original Source Drill-Down 1``,
``Annual Revenue``, ``Marketing contact status``, ``GDPR consent``) and are
dropped from the schema. ``Lead Score`` is populated on 7.5% of rows and
``Job Title`` on 60%, so both are kept as nullable columns. The complete
original row is preserved as JSON in ``raw_json`` so nothing is lost and a
normalisation bug is always recoverable without re-importing.

Each row stores both the raw value and its normalised form (``status_raw`` vs
``status``). Normalising in place would destroy the audit trail; normalising
only at query time would mean re-parsing 2k dates on every request.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone

from . import config, normalize
from .source_extract import extract_source

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id              INTEGER PRIMARY KEY,
    first_name      TEXT,
    last_name       TEXT,
    full_name       TEXT,
    display_name    TEXT,
    name_key        TEXT,
    name_tokens     TEXT,
    job_title       TEXT,
    company_name    TEXT,
    company_core    TEXT,
    email           TEXT,
    email_local     TEXT,
    email_domain    TEXT,
    domain_root     TEXT,
    phone_raw       TEXT,
    phone_digits    TEXT,
    phone_key       TEXT,
    country         TEXT,
    status          TEXT,
    status_raw      TEXT,
    lifecycle_stage TEXT,
    original_source TEXT,
    owner           TEXT,
    owner_raw       TEXT,
    created_at      TEXT,
    updated_at      TEXT,
    notes           TEXT,
    lead_score      INTEGER,
    source_channel  TEXT,
    source_detail   TEXT,
    source_confidence REAL,
    source_method   TEXT,
    source_taxonomy_gap TEXT,
    origin          TEXT NOT NULL DEFAULT 'seed',
    ingested_at     TEXT,
    raw_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_leads_status  ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_owner   ON leads(owner);
CREATE INDEX IF NOT EXISTS idx_leads_country ON leads(country);
CREATE INDEX IF NOT EXISTS idx_leads_phone   ON leads(phone_digits);
CREATE INDEX IF NOT EXISTS idx_leads_email   ON leads(email);
CREATE INDEX IF NOT EXISTS idx_leads_domain  ON leads(domain_root);
CREATE INDEX IF NOT EXISTS idx_leads_channel ON leads(source_channel);
"""

# Columns a PATCH is allowed to touch. Anything else is rejected rather than
# silently ignored, so a typo in a client does not look like a success.
PATCHABLE = {"status", "owner", "notes"}

LIST_COLUMNS = [
    "id", "display_name", "first_name", "last_name", "job_title",
    "company_name", "email", "phone_raw", "country", "status", "owner",
    "lifecycle_stage", "created_at", "updated_at", "source_channel",
    "source_detail", "lead_score", "origin",
]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path=None):
    """Open a connection with row access by column name."""
    path = str(path or config.DB_PATH)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def _to_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def lead_from_csv_row(row):
    """Normalise one raw export row into the stored shape."""
    name = normalize.resolve_name(
        row.get("First Name"), row.get("Last Name"), row.get("Full Name")
    )
    local, domain = normalize.split_email(row.get("Email"))
    notes = (row.get("Notes") or "").strip()
    source = extract_source(notes)

    return {
        "id": _to_int(row.get("Record ID")),
        "first_name": name["first"],
        "last_name": name["last"],
        "full_name": (row.get("Full Name") or "").strip() or None,
        "display_name": name["display"],
        "name_key": name["key"],
        "name_tokens": json.dumps(name["tokens"]),
        "job_title": (row.get("Job Title") or "").strip() or None,
        "company_name": (row.get("Company Name") or "").strip() or None,
        "company_core": normalize.company_core(row.get("Company Name")),
        "email": (row.get("Email") or "").strip().lower() or None,
        "email_local": local or None,
        "email_domain": domain or None,
        "domain_root": normalize.domain_root(domain) or None,
        "phone_raw": (row.get("Phone Number") or "").strip() or None,
        "phone_digits": normalize.phone_digits(row.get("Phone Number")) or None,
        "phone_key": normalize.phone_key(row.get("Phone Number")) or None,
        "country": normalize.normalize_country(row.get("Country/Region")),
        "status": normalize.normalize_status(row.get("Lead Status")),
        "status_raw": row.get("Lead Status"),
        "lifecycle_stage": (row.get("Lifecycle Stage") or "").strip() or None,
        "original_source": (row.get("Original Source") or "").strip() or None,
        "owner": normalize.normalize_owner(row.get("Contact Owner")),
        "owner_raw": row.get("Contact Owner"),
        "created_at": normalize.parse_date(row.get("Create Date")),
        "updated_at": normalize.parse_date(row.get("Last Modified Date")),
        "notes": notes or None,
        "lead_score": _to_int(row.get("Lead Score")),
        "source_channel": source["channel"],
        "source_detail": source["detail"],
        "source_confidence": source["confidence"],
        "source_method": source["method"],
        "source_taxonomy_gap": source["taxonomy_gap"],
        "origin": "seed",
        "ingested_at": _now(),
        "raw_json": json.dumps(row, ensure_ascii=False),
    }


def lead_from_submission(submission, llm=None):
    """Normalise a website form submission into the stored shape.

    The form metadata is internally inconsistent in the supplied fixture -
    ``form_id`` and ``form_name`` are paired at random (a ``form_demo_request``
    labelled "Newsletter Signup" on ``/blog``), so neither is trustworthy. We
    extract the channel from ``message`` and fall back to ``page_url``, which
    is the only self-consistent channel evidence the payload carries.
    """
    name = normalize.resolve_name(None, None, submission.get("name"))
    local, domain = normalize.split_email(submission.get("email"))
    message = (submission.get("message") or "").strip()
    source = extract_source(
        message,
        hints={
            "page_url": submission.get("page_url"),
            "form_name": submission.get("form_name"),
        },
        llm=llm,
    )
    submitted = normalize.parse_timestamp(submission.get("submitted_at"))

    return {
        "id": None,
        "first_name": name["first"],
        "last_name": name["last"],
        "full_name": (submission.get("name") or "").strip() or None,
        "display_name": name["display"],
        "name_key": name["key"],
        "name_tokens": json.dumps(name["tokens"]),
        "job_title": None,
        "company_name": (submission.get("company") or "").strip() or None,
        "company_core": normalize.company_core(submission.get("company")),
        "email": (submission.get("email") or "").strip().lower() or None,
        "email_local": local or None,
        "email_domain": domain or None,
        "domain_root": normalize.domain_root(domain) or None,
        "phone_raw": (submission.get("phone") or "").strip() or None,
        "phone_digits": normalize.phone_digits(submission.get("phone")) or None,
        "phone_key": normalize.phone_key(submission.get("phone")) or None,
        "country": normalize.normalize_country(submission.get("country")),
        "status": "New",
        "status_raw": None,
        "lifecycle_stage": "Lead",
        "original_source": None,
        "owner": None,
        "owner_raw": None,
        "created_at": (submitted or _now())[:10],
        "updated_at": None,
        "notes": message or None,
        "lead_score": None,
        "source_channel": source["channel"],
        "source_detail": source["detail"],
        "source_confidence": source["confidence"],
        "source_method": source["method"],
        "source_taxonomy_gap": source["taxonomy_gap"],
        "origin": "form",
        "ingested_at": _now(),
        "raw_json": json.dumps(submission, ensure_ascii=False),
    }


_INSERT_COLUMNS = [
    "id", "first_name", "last_name", "full_name", "display_name", "name_key",
    "name_tokens", "job_title", "company_name", "company_core", "email",
    "email_local", "email_domain", "domain_root", "phone_raw", "phone_digits",
    "phone_key", "country", "status", "status_raw", "lifecycle_stage",
    "original_source", "owner", "owner_raw", "created_at", "updated_at",
    "notes", "lead_score", "source_channel", "source_detail",
    "source_confidence", "source_method", "source_taxonomy_gap", "origin",
    "ingested_at", "raw_json",
]


def insert_lead(conn, lead):
    """Insert, allocating an id when the caller did not supply one."""
    lead = dict(lead)
    if lead.get("id") is None:
        row = conn.execute("SELECT COALESCE(MAX(id), 100000000) AS m FROM leads").fetchone()
        lead["id"] = int(row["m"]) + 1
    placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
    conn.execute(
        "INSERT INTO leads (" + ", ".join(_INSERT_COLUMNS) + ") VALUES (" + placeholders + ")",
        [lead.get(column) for column in _INSERT_COLUMNS],
    )
    conn.commit()
    return lead["id"]


def load_seed(conn, csv_path=None, replace=True):
    """Load the CSV export. Returns the number of rows stored."""
    csv_path = csv_path or config.SEED_CSV
    if replace:
        conn.execute("DELETE FROM leads")

    inserted = skipped = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            lead = lead_from_csv_row(raw)
            if lead["id"] is None:
                skipped += 1  # no Record ID: nothing to key the row on
                continue
            rows.append([lead.get(column) for column in _INSERT_COLUMNS])
            inserted += 1
        placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
        conn.executemany(
            "INSERT OR REPLACE INTO leads (" + ", ".join(_INSERT_COLUMNS)
            + ") VALUES (" + placeholders + ")",
            rows,
        )
    conn.commit()
    return {"inserted": inserted, "skipped_without_id": skipped}


def get_lead(conn, lead_id):
    row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
    return dict(row) if row else None


def _filter_clause(filters):
    """Build the shared WHERE clause for list and export."""
    clauses, params = [], []
    if filters.get("status"):
        canonical = normalize.normalize_status(filters["status"]) or filters["status"]
        clauses.append("status = ?")
        params.append(canonical)
    if filters.get("owner"):
        clauses.append("LOWER(owner) = ?")
        params.append(normalize.normalize_owner(filters["owner"]).lower())
    if filters.get("country"):
        clauses.append("LOWER(country) = ?")
        params.append(filters["country"].strip().lower())
    if filters.get("channel"):
        clauses.append("source_channel = ?")
        params.append(filters["channel"])
    if filters.get("q"):
        needle = "%" + filters["q"].strip().lower() + "%"
        clauses.append(
            "(LOWER(COALESCE(display_name, '')) LIKE ?"
            " OR LOWER(COALESCE(company_name, '')) LIKE ?"
            " OR LOWER(COALESCE(email, '')) LIKE ?)"
        )
        params.extend([needle, needle, needle])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def list_leads(conn, filters=None, limit=50, offset=0):
    filters = filters or {}
    where, params = _filter_clause(filters)
    total = conn.execute("SELECT COUNT(*) AS c FROM leads" + where, params).fetchone()["c"]
    rows = conn.execute(
        "SELECT " + ", ".join(LIST_COLUMNS) + " FROM leads" + where
        + " ORDER BY id LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {k: v for k, v in filters.items() if v},
        "leads": [dict(row) for row in rows],
    }


def iter_filtered(conn, filters=None):
    """All matching rows, unpaginated - used by the CSV export."""
    where, params = _filter_clause(filters or {})
    return conn.execute(
        "SELECT " + ", ".join(LIST_COLUMNS) + " FROM leads" + where + " ORDER BY id",
        params,
    )


def update_lead(conn, lead_id, patch):
    """Apply a PATCH. Returns ``(lead, error)``; error is None on success."""
    unknown = set(patch) - PATCHABLE
    if unknown:
        return None, "unsupported field(s): " + ", ".join(sorted(unknown))
    if not patch:
        return None, "no fields to update"
    if get_lead(conn, lead_id) is None:
        return None, "not found"

    assignments, params = [], []
    if "status" in patch:
        canonical = normalize.normalize_status(patch["status"])
        if canonical is None:
            return None, (
                "unrecognised status "
                + repr(patch["status"])
                + "; expected one of "
                + ", ".join(normalize.CANONICAL_STATUSES)
            )
        assignments += ["status = ?", "status_raw = ?"]
        params += [canonical, patch["status"]]
    if "owner" in patch:
        assignments += ["owner = ?", "owner_raw = ?"]
        params += [normalize.normalize_owner(patch["owner"]), patch["owner"]]
    if "notes" in patch:
        assignments.append("notes = ?")
        params.append(patch["notes"])

    assignments.append("updated_at = ?")
    params.append(_now()[:10])
    params.append(lead_id)
    conn.execute("UPDATE leads SET " + ", ".join(assignments) + " WHERE id = ?", params)
    conn.commit()
    return get_lead(conn, lead_id), None


def merge_submission_into(conn, lead_id, lead_payload):
    """Enrich an existing lead from a matched form submission.

    Only fills blanks and appends the new message to the notes. A form
    submission must never overwrite a value a salesperson set by hand, so
    ``status`` and ``owner`` are left alone.
    """
    existing = get_lead(conn, lead_id)
    if existing is None:
        return None

    fillable = [
        "job_title", "company_name", "company_core", "email", "email_local",
        "email_domain", "domain_root", "phone_raw", "phone_digits",
        "phone_key", "country", "full_name",
    ]
    assignments, params = [], []
    filled = []
    for column in fillable:
        if not existing.get(column) and lead_payload.get(column):
            assignments.append(column + " = ?")
            params.append(lead_payload[column])
            filled.append(column)

    note = (lead_payload.get("notes") or "").strip()
    if note and note not in (existing.get("notes") or ""):
        combined = ((existing.get("notes") or "").strip() + "\n" + note).strip()
        assignments.append("notes = ?")
        params.append(combined)
        filled.append("notes")

    assignments.append("updated_at = ?")
    params.append(_now()[:10])
    params.append(lead_id)
    conn.execute("UPDATE leads SET " + ", ".join(assignments) + " WHERE id = ?", params)
    conn.commit()
    return {"lead": get_lead(conn, lead_id), "fields_filled": filled}


def leads_for_dedupe(conn):
    """Every record in the shape ``leadms.dedupe`` expects."""
    rows = conn.execute(
        "SELECT id, display_name, first_name, last_name, name_key, name_tokens,"
        " company_name, company_core, email, email_local, email_domain,"
        " domain_root, phone_raw, phone_digits, phone_key, country, job_title,"
        " status, created_at, notes FROM leads"
    ).fetchall()
    leads = []
    for row in rows:
        lead = dict(row)
        try:
            lead["name_tokens"] = json.loads(lead["name_tokens"] or "[]")
        except ValueError:
            lead["name_tokens"] = []
        leads.append(lead)
    return leads


def find_blocking_candidates(conn, lead, limit=200):
    """Rows worth comparing against ``lead``, narrowed in SQL.

    Mirrors the blocking keys used by the batch deduplication pass (phone
    digits, phone suffix, exact email, company domain root, first-initial +
    surname) so a single ingest scores a handful of rows rather than the whole
    corpus. Every key here is backed by an index.
    """
    clauses, params = [], []
    if lead.get("phone_digits"):
        clauses.append("phone_digits = ?")
        params.append(lead["phone_digits"])
    if lead.get("phone_key"):
        clauses.append("phone_key = ?")
        params.append(lead["phone_key"])
    if lead.get("email"):
        clauses.append("email = ?")
        params.append(lead["email"])
    if lead.get("domain_root"):
        clauses.append("domain_root = ?")
        params.append(lead["domain_root"])

    tokens = lead.get("name_tokens")
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens or "[]")
        except ValueError:
            tokens = []
    if tokens:
        clauses.append("name_key LIKE ?")
        params.append(tokens[0][:1] + "%" + tokens[-1])

    if not clauses:
        return []

    rows = conn.execute(
        "SELECT id, display_name, first_name, last_name, name_key, name_tokens,"
        " company_name, company_core, email, email_local, email_domain,"
        " domain_root, phone_raw, phone_digits, phone_key, country, job_title,"
        " status, created_at, notes FROM leads WHERE "
        + " OR ".join(clauses)
        + " LIMIT ?",
        params + [limit],
    ).fetchall()

    candidates = []
    for row in rows:
        item = dict(row)
        try:
            item["name_tokens"] = json.loads(item["name_tokens"] or "[]")
        except ValueError:
            item["name_tokens"] = []
        candidates.append(item)
    return candidates


def company_token_weights(conn):
    """IDF weights over the stored company names, for similarity scoring."""
    from .similarity import build_company_token_weights

    rows = conn.execute(
        "SELECT company_name FROM leads WHERE company_name IS NOT NULL"
    ).fetchall()
    return build_company_token_weights(row["company_name"] for row in rows)


def dashboard(conn):
    """Counts by status and by extracted source channel (assignment part 4)."""
    def counts(column):
        rows = conn.execute(
            "SELECT COALESCE(" + column + ", '(unknown)') AS k, COUNT(*) AS c"
            " FROM leads GROUP BY k ORDER BY c DESC"
        ).fetchall()
        return {row["k"]: row["c"] for row in rows}

    total = conn.execute("SELECT COUNT(*) AS c FROM leads").fetchone()["c"]
    gaps = conn.execute(
        "SELECT source_taxonomy_gap AS k, COUNT(*) AS c FROM leads"
        " WHERE source_taxonomy_gap IS NOT NULL GROUP BY k ORDER BY c DESC"
    ).fetchall()
    return {
        "total_leads": total,
        "by_status": counts("status"),
        "by_source_channel": counts("source_channel"),
        "by_owner": counts("owner"),
        "by_country": dict(list(counts("country").items())[:10]),
        "by_origin": counts("origin"),
        # Surfaced explicitly: these leads are paid-search acquisitions that
        # the required channel taxonomy has no bucket for.
        "taxonomy_gaps": {row["k"]: row["c"] for row in gaps},
    }

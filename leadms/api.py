"""REST API over the lead store, on ``http.server``.

A hand-rolled router on the stdlib HTTP server keeps the project at zero
dependencies. It is not what you would deploy - there is no auth, no rate
limiting, no graceful shutdown - but the assignment explicitly puts
production concerns out of scope, and a reviewer can start it with one
command and no virtualenv.

Endpoints
---------
``GET    /health``                  service + LLM provider status
``GET    /leads``                   filter by status, owner, country, channel, q
``GET    /leads/export``            CSV of the current filtered view
``GET    /leads/{id}``              single lead
``PATCH  /leads/{id}``              update status, owner or notes
``POST   /leads/ingest``            create or update from a form submission
``POST   /leads/dedupe-candidates`` ranked duplicate groups
``POST   /extract-source``          channel + detail from free text
``GET    /dashboard``               counts by status and source channel
``GET    /``                        a minimal single-page UI
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, dedupe, llm as llm_module, store
from .source_extract import extract_source

MAX_BODY_BYTES = 4 * 1024 * 1024
UI_PATH = Path(__file__).resolve().parent / "ui.html"


class LeadAPI:
    """Request handling, kept separate from the HTTP plumbing so tests can
    exercise routes without binding a socket."""

    def __init__(self, conn, llm_client=None):
        self.conn = conn
        self.llm = llm_client if llm_client is not None else llm_module.build_client()
        self.routes = [
            ("GET", re.compile(r"^/health$"), self.health),
            ("GET", re.compile(r"^/$"), self.ui),
            ("GET", re.compile(r"^/dashboard$"), self.dashboard),
            ("GET", re.compile(r"^/leads/export$"), self.export),
            ("GET", re.compile(r"^/leads$"), self.list_leads),
            # The int-only pattern is what keeps /leads/export from being
            # parsed as a lead id.
            ("GET", re.compile(r"^/leads/(?P<lead_id>\d+)$"), self.get_lead),
            ("PATCH", re.compile(r"^/leads/(?P<lead_id>\d+)$"), self.patch_lead),
            ("POST", re.compile(r"^/leads/ingest$"), self.ingest),
            ("POST", re.compile(r"^/leads/dedupe-candidates$"), self.dedupe_candidates),
            ("POST", re.compile(r"^/extract-source$"), self.extract_source_endpoint),
        ]

    # --- dispatch ------------------------------------------------------------
    def handle(self, method, path, query, body):
        """Returns ``(status, payload, content_type)``.

        A path can be registered under several methods (``/leads/{id}`` takes
        both GET and PATCH), so we collect every pattern that matches the path
        before deciding: 405 is only correct once we know no registered method
        for this path fits.
        """
        allowed = []
        for route_method, pattern, handler in self.routes:
            match = pattern.match(path)
            if not match:
                continue
            if route_method == method:
                return handler(query=query, body=body, **match.groupdict())
            allowed.append(route_method)
        if allowed:
            return (
                405,
                {
                    "error": "method not allowed on " + path,
                    "allowed": sorted(set(allowed)),
                },
                "json",
            )
        return 404, {"error": "no route for " + method + " " + path}, "json"

    # --- endpoints -----------------------------------------------------------
    def health(self, **_):
        total = self.conn.execute("SELECT COUNT(*) AS c FROM leads").fetchone()["c"]
        return (
            200,
            {
                "status": "ok",
                "leads_loaded": total,
                "llm": llm_module.describe_client(self.llm),
                "dedupe_thresholds": {
                    "auto_merge": config.DEDUPE_AUTO_MERGE,
                    "review_low": config.DEDUPE_REVIEW_LOW,
                    "report_min": config.DEDUPE_REPORT_MIN,
                },
            },
            "json",
        )

    def ui(self, **_):
        try:
            return 200, UI_PATH.read_text(encoding="utf-8"), "html"
        except OSError:
            return 404, {"error": "UI not available"}, "json"

    def list_leads(self, query=None, **_):
        query = query or {}
        limit, error = _int_param(query, "limit", 50, 1, 500)
        if error:
            return 400, {"error": error}, "json"
        offset, error = _int_param(query, "offset", 0, 0, 10_000_000)
        if error:
            return 400, {"error": error}, "json"
        return 200, store.list_leads(self.conn, _filters(query), limit, offset), "json"

    def get_lead(self, lead_id=None, **_):
        lead = store.get_lead(self.conn, int(lead_id))
        if lead is None:
            return 404, {"error": "lead " + str(lead_id) + " not found"}, "json"
        return 200, lead, "json"

    def patch_lead(self, lead_id=None, body=None, **_):
        if not isinstance(body, dict):
            return 400, {"error": "body must be a JSON object"}, "json"
        lead, error = store.update_lead(self.conn, int(lead_id), body)
        if error == "not found":
            return 404, {"error": "lead " + str(lead_id) + " not found"}, "json"
        if error:
            return 400, {"error": error}, "json"
        return 200, lead, "json"

    def export(self, query=None, **_):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=store.LIST_COLUMNS)
        writer.writeheader()
        for row in store.iter_filtered(self.conn, _filters(query or {})):
            writer.writerow(dict(row))
        return 200, buffer.getvalue(), "csv"

    def dashboard(self, **_):
        return 200, store.dashboard(self.conn), "json"

    def extract_source_endpoint(self, body=None, **_):
        if not isinstance(body, dict):
            return 400, {"error": "body must be a JSON object"}, "json"
        text = body.get("text") or body.get("notes") or ""
        if not str(text).strip():
            return 400, {"error": "field 'text' is required"}, "json"
        use_llm = bool(body.get("use_llm", True))
        result = extract_source(
            str(text),
            hints=body.get("hints"),
            llm=self.llm if use_llm else None,
        )
        return 200, result, "json"

    def dedupe_candidates(self, body=None, **_):
        body = body if isinstance(body, dict) else {}
        min_confidence = body.get("min_confidence", config.DEDUPE_REPORT_MIN)
        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            return 400, {"error": "min_confidence must be a number"}, "json"
        use_llm = bool(body.get("use_llm", True))
        limit = body.get("limit")

        leads = store.leads_for_dedupe(self.conn)
        result = dedupe.find_duplicate_candidates(
            leads,
            llm=self.llm if use_llm else None,
            min_confidence=min_confidence,
            use_llm=use_llm,
        )
        if limit:
            try:
                result["groups"] = result["groups"][: int(limit)]
            except (TypeError, ValueError):
                return 400, {"error": "limit must be an integer"}, "json"
        result["llm"] = llm_module.describe_client(self.llm) if use_llm else None
        return 200, result, "json"

    def ingest(self, body=None, **_):
        """Create a lead, or enrich an existing one when it is the same person.

        Three outcomes, driven by the same scorer the batch report uses:

        * ``score >= auto_merge`` - update the existing lead in place.
        * ``review_low <= score < auto_merge`` - create the lead but flag it
          for review with the candidate attached. Silently merging an
          uncertain match is the more expensive mistake: a wrong merge
          destroys two records' history, a wrong split is fixed later by the
          dedupe report.
        * otherwise - create a new lead.
        """
        submissions = body if isinstance(body, list) else [body]
        if not submissions or not all(isinstance(s, dict) for s in submissions):
            return 400, {"error": "body must be a form submission object or a list"}, "json"

        weights = store.company_token_weights(self.conn)
        results = []
        for submission in submissions:
            if not (submission.get("email") or submission.get("phone")):
                results.append(
                    {"action": "rejected", "error": "email or phone is required"}
                )
                continue

            payload = store.lead_from_submission(submission, llm=self.llm)
            candidates = store.find_blocking_candidates(self.conn, payload)
            matches = dedupe.rank_matches(payload, candidates, weights, limit=3)
            best = matches[0] if matches else None

            if best and best["score"] >= config.DEDUPE_AUTO_MERGE:
                merged = store.merge_submission_into(self.conn, best["lead_id"], payload)
                results.append(
                    {
                        "action": "updated",
                        "lead_id": best["lead_id"],
                        "confidence": round(best["score"], 4),
                        "reasons": best["reasons"],
                        "fields_filled": merged["fields_filled"],
                        "lead": merged["lead"],
                    }
                )
                continue

            new_id = store.insert_lead(self.conn, payload)
            outcome = {
                "action": "created",
                "lead_id": new_id,
                "lead": store.get_lead(self.conn, new_id),
                "source": {
                    "channel": payload["source_channel"],
                    "detail": payload["source_detail"],
                    "method": payload["source_method"],
                },
            }
            if best and best["score"] >= config.DEDUPE_REVIEW_LOW:
                outcome["needs_review"] = True
                outcome["possible_duplicate_of"] = [
                    {
                        "lead_id": m["lead_id"],
                        "confidence": round(m["score"], 4),
                        "reasons": m["reasons"],
                        "lead": m["lead"],
                    }
                    for m in matches
                    if m["score"] >= config.DEDUPE_REVIEW_LOW
                ]
            results.append(outcome)

        if isinstance(body, list):
            summary = {"processed": len(results), "results": results}
            for action in ("created", "updated", "rejected"):
                summary[action] = sum(1 for r in results if r.get("action") == action)
            summary["needs_review"] = sum(
                1 for r in results if r.get("needs_review")
            )
            return 200, summary, "json"
        status = 400 if results[0].get("action") == "rejected" else 200
        return status, results[0], "json"


# --- helpers -----------------------------------------------------------------
def _filters(query):
    return {
        "status": _first(query, "status"),
        "owner": _first(query, "owner"),
        "country": _first(query, "country"),
        "channel": _first(query, "channel"),
        "q": _first(query, "q"),
    }


def _first(query, key):
    values = query.get(key)
    if not values:
        return None
    value = values[0] if isinstance(values, list) else values
    return value.strip() or None


def _int_param(query, key, default, low, high):
    raw = _first(query, key)
    if raw is None:
        return default, None
    try:
        value = int(raw)
    except ValueError:
        return None, key + " must be an integer"
    if not low <= value <= high:
        return None, key + " must be between " + str(low) + " and " + str(high)
    return value, None


class _Handler(BaseHTTPRequestHandler):
    server_version = "leadms/1.0"
    api = None  # injected by serve()

    def _respond(self, status, payload, kind):
        if kind == "json":
            data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif kind == "csv":
            data = payload.encode("utf-8")
            content_type = "text/csv; charset=utf-8"
        else:
            data = payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if kind == "csv":
            self.send_header("Content-Disposition", 'attachment; filename="leads.csv"')
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, method):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        body = None
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            if length > MAX_BODY_BYTES:
                self._respond(413, {"error": "request body too large"}, "json")
                return
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                self._respond(400, {"error": "invalid JSON body: " + str(exc)}, "json")
                return

        try:
            status, payload, kind = self.api.handle(
                method, parsed.path.rstrip("/") or "/", query, body
            )
        except Exception as exc:  # never leak a traceback to the client
            self._respond(500, {"error": type(exc).__name__ + ": " + str(exc)}, "json")
            return
        self._respond(status, payload, kind)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def log_message(self, fmt, *args):
        print("[api] " + (fmt % args))


def serve(host=None, port=None, db_path=None):
    conn = store.connect(db_path)
    store.init_schema(conn)
    total = conn.execute("SELECT COUNT(*) AS c FROM leads").fetchone()["c"]
    if total == 0:
        print("[api] empty database; loading seed CSV...")
        print("[api] " + json.dumps(store.load_seed(conn)))

    handler = type("_BoundHandler", (_Handler,), {"api": LeadAPI(conn)})
    address = (host or config.HOST, port or config.PORT)
    httpd = ThreadingHTTPServer(address, handler)
    print("[api] listening on http://" + address[0] + ":" + str(address[1]))
    print("[api] llm: " + json.dumps(llm_module.describe_client(handler.api.llm)))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] shutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()

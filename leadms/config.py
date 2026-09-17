"""Runtime configuration, all overridable by environment variable."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("LEADMS_DATA_DIR", PROJECT_DIR / "data"))
SEED_CSV = DATA_DIR / "leads_seed.csv"
FORM_SUBMISSIONS_JSON = DATA_DIR / "website_form_submissions.json"

DB_PATH = Path(os.environ.get("LEADMS_DB", PROJECT_DIR / "leads.db"))
LLM_CACHE_DIR = Path(os.environ.get("LEADMS_LLM_CACHE", PROJECT_DIR / ".llm_cache"))

# --- dedupe thresholds -------------------------------------------------------
# Pairs scoring at or above AUTO_MERGE are treated as the same person without
# asking a model. Pairs in [REVIEW_LOW, AUTO_MERGE) are the ambiguous band that
# gets sent to the LLM adjudicator. Below REVIEW_LOW we drop the pair entirely.
DEDUPE_AUTO_MERGE = float(os.environ.get("LEADMS_AUTO_MERGE", "0.90"))
DEDUPE_REVIEW_LOW = float(os.environ.get("LEADMS_REVIEW_LOW", "0.40"))
# Default cut-off for what /leads/dedupe-candidates reports as a group.
DEDUPE_REPORT_MIN = float(os.environ.get("LEADMS_REPORT_MIN", "0.70"))

# Top-K neighbours pulled out of the TF-IDF index per lead during candidate
# generation. 8 is enough to cover every duplicate cluster we observed
# (largest cluster in the seed data is 3 records).
ANN_TOP_K = int(os.environ.get("LEADMS_ANN_TOP_K", "8"))
# Blocks larger than this are skipped for exhaustive in-block pairing; the
# TF-IDF neighbours still cover them. Guards against a single huge block
# (e.g. a shared gmail.com domain in real data) turning into O(n^2).
MAX_BLOCK_SIZE = int(os.environ.get("LEADMS_MAX_BLOCK", "80"))

# --- LLM ---------------------------------------------------------------------
# auto  -> Anthropic if ANTHROPIC_API_KEY, else Gemini if GEMINI_API_KEY,
#          else the offline heuristic stub
# off   -> never call a network model
LLM_MODE = os.environ.get("LEADMS_LLM", "auto").strip().lower()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("LEADMS_ANTHROPIC_MODEL", "claude-opus-5").strip()
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
).strip()
GEMINI_MODEL = os.environ.get("LEADMS_GEMINI_MODEL", "gemini-2.5-flash").strip()
LLM_TIMEOUT = float(os.environ.get("LEADMS_LLM_TIMEOUT", "30"))

HOST = os.environ.get("LEADMS_HOST", "127.0.0.1")
PORT = int(os.environ.get("LEADMS_PORT", "8000"))

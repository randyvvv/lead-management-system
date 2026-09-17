# Mini Lead Management System

A small backend for a messy CRM export, with two AI-assisted features:
deduplication and lead-source extraction.

**Zero third-party dependencies.** Everything runs on the Python 3.9+ standard
library — `sqlite3`, `http.server`, `urllib`, `unittest`. No `pip install`, no
virtualenv, no API key required.

```bash
cd lead-management
python -m leadms.cli init      # build leads.db from data/leads_seed.csv  (~1s)
python -m leadms.cli serve     # http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000> for a small built-in UI (lead list with
filters, dashboard, duplicate review, source extraction), or use the API
directly.

```bash
python -m unittest discover -s tests -t .   # 134 tests
python eval/run_eval.py                     # measured results, reproduced
```

---

## Contents

- [Running it](#running-it)
- [API](#api)
- [How the data was read](#how-the-data-was-read)
- [Design: storage](#design-storage)
- [Design: deduplication](#design-deduplication)
- [Design: source extraction](#design-source-extraction)
- [LLM usage and cost](#llm-usage-and-cost)
- [Results](#results)
- [Tests](#tests)
- [What I cut, and what I'd do next](#what-i-cut-and-what-id-do-next)

---

## Running it

| Command | What it does |
|---|---|
| `python -m leadms.cli init` | Create the schema and load the seed CSV |
| `python -m leadms.cli serve` | Start the REST API + UI on `:8000` |
| `python -m leadms.cli dedupe --limit 20` | Duplicate report to stdout |
| `python -m leadms.cli dedupe --out report.json` | Full report to a file |
| `python -m leadms.cli extract "Met her at the Web Summit booth"` | One-off extraction |
| `python -m leadms.cli ingest` | Replay all 90 form submissions |
| `python -m leadms.cli dashboard` | Counts by status and channel |
| `python -m leadms.cli profile` | Re-derive every data-quality number quoted below |
| `python -m leadms.cli llm-check` | Show which LLM provider is active; list available models |

Configuration is by environment variable — `LEADMS_DB`, `LEADMS_PORT`,
`LEADMS_LLM`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` — see `leadms/config.py`.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/leads` | `status`, `owner`, `country`, `channel`, `q`, `limit`, `offset` |
| `GET` | `/leads/{id}` | Full record including the preserved raw row |
| `PATCH` | `/leads/{id}` | `status`, `owner`, `notes` only |
| `GET` | `/leads/export` | CSV of the current filtered view |
| `POST` | `/leads/ingest` | One submission or a list; creates **or** updates |
| `POST` | `/leads/dedupe-candidates` | Ranked duplicate groups with explanations |
| `POST` | `/extract-source` | `{channel, detail, confidence, method, ...}` |
| `GET` | `/dashboard` | Counts by status and source channel |
| `GET` | `/health` | Row count, active LLM provider, thresholds |

```bash
curl 'localhost:8000/leads?status=qualified&country=uae&limit=3'
curl -X PATCH localhost:8000/leads/100234811 -d '{"status":" closed won "}'
curl -X POST localhost:8000/leads/dedupe-candidates -d '{"min_confidence":0.9,"limit":5}'
curl -X POST localhost:8000/extract-source \
     -d '{"text":"He scanned our QR code at the Singapore FinTech Festival 2026 booth."}'
# => {"channel":"Event","detail":"Singapore FinTech Festival 2026 — Booth QR Code", ...}
```

Filters accept the data's own messiness: `status=qualified`, `status=QUALIFIED`
and `status=%20Qualified%20` all return the same 321 rows.

---

## How the data was read

I profiled the CSV before modelling it. Everything below is reproducible with
`python -m leadms.cli profile`.

| Observation | Consequence |
|---|---|
| 2,049 rows, 22 columns | — |
| **5 columns are empty in every row** (`City`, `Original Source Drill-Down 1`, `Annual Revenue`, `Marketing contact status`, `GDPR consent`) | Dropped from the schema |
| `Job Title` 59.6%, `Lead Score` 7.5%, `Full Name` 5.2% filled | Kept, nullable |
| **`Lead Status` has 35 spellings for 7 values** (`New`/`new`/`NEW`/`" New"`) | Normalised on write, raw kept alongside |
| `Contact Owner` has 20 spellings for 10 people (trailing spaces) | Whitespace collapsed |
| 3 date formats in one column; 1,078 slash dates | Parsed to ISO. Slash dates are **month-first**: across 1,078 of them the first component never exceeds 12 while the second reaches 31 |
| `Country/Region` has 62 values, 112 rows mis-cased (`china`) | Title-cased, but short acronyms preserved so `UAE` doesn't become `Uae` |
| `Original Source` blank on 49.5% of rows, and coarse when present | **Never used as an input.** Held back as an independent check on the extractor |
| Names split across `First`/`Last` **or** `Full Name`, never both | Reconciled into one `display_name` + token list |
| Phone numbers in mixed formats, all with country codes | Compared on digits only |

Two things I found that aren't in the brief:

**1. The `Notes` column contains a planted hint.** 136 rows end with
`"possible duplicate — verify before contacting."`. It is tempting to use, and
wrong to: it is a fixture artefact that no production system would ever see,
and it labels only **27.5%** of the records that are actually duplicated. The
dedupe pipeline strips it from every text feature and every LLM prompt, and
`tests/test_dedupe.py::TestNoMarkerLeakage` asserts that the score is
identical with and without it.

**2. The form fixture's metadata is internally inconsistent.** `form_id` and
`form_name` are paired at random — a `form_demo_request` labelled
`"Newsletter Signup"` on `/blog`. Neither is trustworthy, so ingest derives
the channel from `message`, falling back to `page_url`, which is the only
self-consistent channel evidence in the payload.

---

## Design: storage

**SQLite**, because it is in the standard library, needs no server, gives real
indexed filtering for `GET /leads`, and persists `PATCH` edits across restarts
(an in-memory dict would not). At 2,049 rows performance is irrelevant; the
deciding factors were zero setup for a reviewer and durable writes.

Each row keeps **both** the raw value and its normalised form — `status_raw`
next to `status` — plus the complete original row as JSON in `raw_json`.
Normalising in place would destroy the audit trail; normalising only at query
time would mean re-parsing 2,000 dates on every request. Derived comparison
keys (`phone_digits`, `domain_root`, `name_key`, `company_core`) are computed
once on write and indexed.

---

## Design: deduplication

Three stages, each more expensive than the last, so the expensive one runs on
almost nothing.

### 1. Candidate generation — 2,098,176 → 29,652 pairs (98.59% avoided)

A full pairwise sweep is 2.1M comparisons. Two mechanisms narrow it:

- **Exact blocking keys**: phone digits, last-9 phone digits, exact email,
  company email-domain root, first-initial + surname. Blocks larger than 80
  records are skipped to stop one huge block (a shared `gmail.com` in real
  data) from going quadratic — the n-gram pass still covers them.
- **A TF-IDF character 3-gram index** over `name + company + email`, taking
  each record's top-8 cosine neighbours from an inverted index with
  high-document-frequency grams pruned.

Why character n-grams and not embeddings: the signal separating duplicates
here is surface-level — transposed characters, dropped initials, reformatted
punctuation — which n-grams capture directly. It needs no API call, no model
download and no vector store, and it stays exact and reproducible in tests.
`TfidfNgramIndex.neighbours()` is a narrow interface; swapping in embeddings +
ANN behind it is a contained change if the corpus outgrows this.

**Measured blocking recall: 294/294 (100%).** This is the number that matters
most — a true pair dropped here can never be recovered downstream.

### 2. Vectorised scoring

Every surviving pair gets a log-odds score over ~13 field comparisons, in
`dedupe.WEIGHTS` — one dict, so any single number is arguable:

```
phone_exact +5.0   email_exact +5.0   name_sim +7.0×(jw − 0.72)
local_consistent +2.2   domain_root +1.2   company_sim +1.5×(sim − 0.35)
phone_conflict −1.8   country_conflict −0.8   same_name_same_company +1.2
```

Weights are hand-set from field semantics rather than fitted — 2k rows with no
labels does not support fitting — then checked against ground truth.

Three features are worth calling out:

- **`local_consistent`** cross-checks whether each record's email localpart
  spells out the *other* record's name. `erik.a@`, `erika@` and
  `erikalmeida@` are all generated by `("Erik", "Almeida")`, so this fires
  strongly on the exact perturbation the fixture uses.
- **`company_sim`** uses IDF weights derived from the corpus, not a
  hand-written suffix list. A curated list cannot work, because the same token
  is core in one name and decoration in another: *Analytics* is the identity
  of "Huang Analytics Consulting" but throwaway in "Lotus Finance Analytics".
  IDF settles it from the data.
- **`same_name_same_company`** is an interaction term, added after measurement.
  Without it, an identical name at the same employer with no shared phone or
  email scored ~0.13 and was silently discarded as a non-duplicate — a
  confidently wrong answer on the one configuration where field comparison
  genuinely runs out of signal. With it, those pairs land in the review band
  instead. (An earlier weighting also had `phone_conflict` at −1.8 stacking
  with `country_conflict` at −1.5; since those two fire together on nearly
  every ambiguous pair and are correlated rather than independent evidence,
  the country penalty was softened to avoid double-counting one observation.)

### 3. LLM adjudication — of 3 pairs

Pairs scoring in `[0.40, 0.90)` go to a model. On this dataset that is
**3 pairs**, not 2.1 million:

| Score | Pair | Ground truth |
|---|---|---|
| 0.607 | Arjun Carvalho @ Lotus Finance Partners (×2) | different people |
| 0.607 | Lucas Schmidt @ Chua Textiles Retail Group (×2) | different people |
| 0.417 | Marcus Ho @ Asante Retail Ltd / and Co | different people |

These are precisely the brief's *"genuinely different people who happen to
share a company and a similar-sounding name"*. Identical name, identical
employer, email localparts that both legitimately spell that name, different
phone and country. No field comparison settles it, which is exactly when a
model's judgement is worth paying for.

The adjudicator returns `same` / `different` / **`unsure`**. Allowing
abstention is deliberate: forcing a binary on a genuinely 50/50 pair produces
noise, and an `unsure` verdict leaves the statistical score untouched rather
than nudging it in an arbitrary direction.

Finally, pairs above the reporting threshold are merged with union-find, so
"A duplicates B" and "B duplicates C" surface as one three-record group.

**Nothing is auto-merged.** The report returns candidates, confidences and
plain-English reasons; a human decides. On ingest, a match at ≥0.90 updates in
place, `[0.40, 0.90)` creates the lead but flags it with the candidate
attached, and below that creates normally. Silently merging an uncertain match
is the more expensive mistake — a wrong merge destroys two records' history,
a wrong split is caught later by the dedupe report.

---

## Design: source extraction

**Rules first, model second**, driven by measurement rather than taste: the
rule pass classifies **100% of the 2,049 seed notes**, so a model call per row
would buy nothing and cost real money. The model earns its place on unseen
phrasing, which is what the form submissions contain.

Ordered regex rules, first match wins. Order matters: paid advertising is
tested before the landing-page rules, because those notes mention a page too
("Booked a demo via the book-a-demo page after clicking a google ad") and the
acquisition channel must win over the conversion page.

The extractor also:

- strips the appended sales-progress sentences ("Great fit, prioritizing.")
  before parsing, so they never leak into `detail`;
- captures the specific thing — event name, referrer name, landing page —
  rather than just the channel;
- distinguishes a scanned booth QR from a booth conversation with no scan;
- **lowers its confidence when it infers**. "Saw our post and commented" never
  names a platform; it returns `LinkedIn` at 0.6 rather than 0.95.

### The taxonomy has a real gap

119 leads came from **Google Ads**. The required channel enum has
`Organic Search` but no paid-search bucket. Folding them into `Website` or
`Organic Search` would silently corrupt exactly the distinction marketing
cares about, so they return:

```json
{"channel": "Other",
 "detail": "Paid Search — Google Ads (book-a-demo page)",
 "taxonomy_gap": "Paid Search"}
```

`taxonomy_gap` is machine-readable and surfaced on `GET /dashboard`, so the
count is visible rather than buried in `Other`. **Recommendation: add
`Paid Search` to the taxonomy** — it is 5.8% of the dataset, and the raw
`Original Source` column independently confirms the label on all 49 rows where
it is populated.

---

## LLM usage and cost

Three providers, tried in order, configured by environment variable:

| Order | Provider | When |
|---|---|---|
| 1 | **Anthropic** (`claude-opus-5`) | `ANTHROPIC_API_KEY` set |
| 2 | **Gemini** (`gemini-2.5-flash`) | `GEMINI_API_KEY` set — the free-tier fallback |
| 3 | **Heuristic** | always available, offline, deterministic |

```bash
export GEMINI_API_KEY=...            # free tier is plenty for this workload
python -m leadms.cli llm-check       # verifies the key and lists usable models
```

**Cost: $0.00.** Everything reported here was produced offline. The workload
is 3 adjudication calls plus a handful of extraction calls — roughly 4k input
and 1k output tokens for a full run. Even on the most expensive option
(Opus 5, $5/$25 per MTok) a complete run is well under one US cent; on the
Gemini free tier it is free. Responses are cached on disk by request hash, so
re-runs cost nothing.

The chain degrades rather than fails: a broken or expired key falls through to
the next provider, and finally to offline rules, so an endpoint never returns
a 500 because a provider is down (`tests/test_llm.py`). The offline stub is
labelled `model_backed: false` in `/health` and **abstains** on ambiguous
pairs rather than re-deriving an answer from the same features the scorer
already used — which would make the offline numbers look better than they are.

Raw `urllib` instead of the official SDK is a consequence of the
zero-dependency choice. In a codebase that already had dependencies I would
use the `anthropic` SDK.

---

## Results

`python eval/run_eval.py`

### Where the labels come from

The dataset ships no ground truth, and grading a model against its own output
is worthless. The fixture generator inserted each duplicate **immediately
after** the record it copies, so true duplicate pairs have adjacent
`Record ID`s. That is verified, not assumed: of the 294 pairs the scorer ranks
≥0.90, *every one* has an ID gap of 1 or 2, and no distant-ID pair reaches
that score.

This is not circular, because **`Record ID` is never read by the scorer** — not
as a feature, not as a blocking key, not in a prompt. Adjacency alone is also
not sufficient (two unrelated people can hold consecutive IDs), so a pair is
labelled a duplicate only if it is adjacent *and* the identity fields agree.
I reviewed every disagreement between the labels and the scorer by hand; the
first pass had two labelling errors, both fixed in `eval/ground_truth.py` with
the reasoning recorded there:

- *Jia Hui Ferrari* vs *J. Ferrari* — one person (they share a phone number),
  wrongly rejected by a whole-string similarity test.
- *Ahmed Fischer* vs *Meera Fischer* — two people at consecutive IDs, wrongly
  accepted by the same test.

Comparing given names and surnames separately fixes both.

### Deduplication

| Metric | Value |
|---|---|
| True duplicate pairs / clusters | 294 / 232 (495 of 2,049 records) |
| Full pairwise comparisons | 2,098,176 |
| Candidate pairs generated | 29,652 (**98.59% avoided**) |
| **Blocking recall** | **294 / 294 (100%)** |
| Precision / Recall / F1 @ 0.70 | **1.000 / 1.000 / 1.000** |
| Clusters matched exactly | **232 / 232 (100%)** |
| Pairs sent to an LLM | 3 |
| Wall clock, full corpus | ~1.5 s |

**A perfect F1 deserves scepticism, so: this is a property of the fixture, not
evidence the approach would hit 1.0 on a real CRM.** Every planted duplicate
preserves its phone digits exactly, so once phone formatting is normalised the
task is close to separable. Real data has people who change numbers, shared
switchboards, and personal email addresses at a corporate domain. What I would
claim to generalise is the *structure*: blocking recall measured separately
from precision, an explicit uncertain band, and abstention instead of a forced
binary. The 3-pair ambiguous band is the honest part of this result — the
three hardest pairs in the dataset are correctly held out of the report rather
than confidently mis-resolved.

### Source extraction

| Metric | Value |
|---|---|
| Notes classified by rules | 2,049 / 2,049 (100%) |
| Agreement with held-out `Original Source` | **1,035 / 1,035 (100%)** |
| Leads flagged with a taxonomy gap | 119 (Paid Search) |

`Original Source` is blank on half the rows and too coarse to use directly,
but it is never an input to the extractor, which makes it a genuine held-out
check on the 1,035 rows where it is populated.

### Ingest

Replaying all 90 form submissions against the loaded corpus: **49 updated,
41 created, 0 rejected**. I inspected every sub-threshold near-match by hand;
all are genuinely different people (same company domain, different person —
e.g. *Nur Bennett* vs *Vikram Agyemang* at `lau.com`). Updating fills blanks
and appends the message, and never overwrites a `status` or `owner` a
salesperson set by hand.

---

## Tests

134 tests, `python -m unittest discover -s tests -t .` (~35s, fully offline).

They target the ambiguous cases rather than the happy path:

- `test_normalize.py` — all 35 status spellings; month-first date
  disambiguation; `UAE` not becoming `Uae`; unknown status returning `None`
  instead of guessing.
- `test_source_extract.py` — each channel; paid-ads beating the landing-page
  rule; inference lowering confidence; uninformative text **not** being forced
  into a channel; an invalid LLM channel being rejected; a model failure not
  propagating; and corpus-level coverage and agreement.
- `test_dedupe.py` — reformatted duplicates scoring high; colleagues and
  similar-surname pairs scoring low; the ambiguous band landing *between* the
  thresholds; **no-marker-leakage**; blocking not going quadratic on a large
  block; transitive clustering; that only the band reaches the model; and that
  abstention leaves the score unchanged.
- `test_api.py` — filters against the data's real messiness; `/leads/export`
  not being parsed as a lead id; `PATCH` rejecting unknown fields rather than
  ignoring them; ingest create/update/reject and idempotency.
- `test_llm.py` — provider fallthrough, malformed responses, cache hits,
  corrupt cache entries, and the guarantee that no key still yields an answer.

Two bugs the tests caught while building, both fixed:

1. **Router method dispatch.** `PATCH /leads/{id}` returned 405 because the
   path matched the `GET` route first and returned early. The router now
   collects every path match before deciding.
2. **An empty LLM band.** The first weighting was so punitive on conflicting
   fields that the score distribution was bimodal — 0.0 or 1.0 — and *no pair
   ever reached the adjudication band*. The hybrid design was decorative until
   that was found and fixed.

---

## What I cut, and what I'd do next

**Out of scope by instruction, and not built:** auth/roles, analytics
integration, webhook infrastructure beyond the one ingest endpoint, audit-log
UI, HubSpot migration tooling, deployment/scaling/monitoring.

**Cut by my own judgment:**

- *Auto-merge.* The brief said surfacing candidates is enough, and a wrong
  merge is expensive to undo. Ingest merges only above 0.90.
- *Fitted scoring weights.* No labels and 2k rows; hand-set weights with
  documented reasoning are more honest and much easier to argue with.
- *Embeddings.* They would not have helped here (the signal is
  character-level) and would have cost a dependency, an API key, and
  reproducibility. The interface is ready for them.
- *A polished UI.* The single page covers the workflows; it isn't a product.

**Next, roughly in order of value:**

1. **A real labelled eval set.** The ID-adjacency trick is a property of this
   fixture. Production needs a few hundred human-labelled pairs, ideally
   sampled from the uncertain band where the labels are worth the most.
2. **Calibrate the score.** It is a log-odds value passed through a sigmoid,
   not a calibrated probability — 0.7 does not mean "70% likely". With labels,
   isotonic regression would make the threshold mean something, and would let
   the weights be fitted rather than chosen.
3. **A merge workflow with survivorship rules and an undo.** Which field wins
   when two records disagree, who approved it, and how to reverse it.
4. **Incremental dedupe.** Today the batch report rescores the whole corpus.
   Ingest already uses the cheap SQL-blocked path; the batch job should
   maintain the index and only rescore touched blocks.
5. **Widen the ambiguous band deliberately and measure it.** Three pairs is
   suspiciously few. On real data I would tune the band by cost: what does a
   missed duplicate cost versus a model call?
6. **Add `Paid Search` to the channel taxonomy**, and re-check the mapping
   against the raw `Original Source` values as a regression test.
7. **Person-level identity separate from lead-level records**, so the same
   human at a new employer links rather than duplicates — the *Sanjay Martins*
   case in the fixture, where one name appears at two unrelated companies.

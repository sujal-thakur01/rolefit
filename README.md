# RoleFit

Semantic job search and resume matching over Hacker News "Who is hiring" threads.

Hundreds of people write those postings in whatever format they feel like. RoleFit
turns that into a queryable index: an LLM pulls each posting into a strict schema,
the result is embedded and stored in Postgres with pgvector, and search runs as
hybrid retrieval with an LLM reranker on top. A small evaluation harness measures
whether each stage actually helps.

Live: `TODO add your URL`

---

## Why the architecture looks like this

**Why an LLM extractor and not a parser.** Roughly 10% of postings ignore the
`Company | Role | Location | MODE` convention entirely, and some are job seekers
posting in the wrong thread. A regex handles the tidy 90% and silently mangles the
rest. The extractor returns a strict Pydantic schema where every field is required
and unknowns are explicit, so a bad extraction is countable instead of invisible.
`is_job_posting` is a field rather than a filter applied afterwards because the
model is better at that judgement than any heuristic.

**Why the embedding is not the raw posting.** Raw postings are mostly boilerplate
about funding rounds, perks and how to apply. Embedding that drags unrelated
companies together in vector space. Instead a composed document is embedded:
title, company, seniority, work mode, location, stack and a two sentence summary.
Tighter, cheaper, and it retrieves noticeably better.

**Why hybrid and not just vectors.** Dense retrieval understands meaning and is
weak on exact tokens, so a query for "rust" pulls in "systems programming" and can
rank an actual Rust job below it. Lexical search is the mirror image. Fusing the
two fixes both failure modes. They are combined with Reciprocal Rank Fusion:

```
score(doc) = Σ  1 / (60 + rank_in_list)
```

Rank based, not score based, on purpose. Cosine similarity and `ts_rank_cd` sit on
different scales and neither is calibrated, so adding the raw numbers lets whichever
has the wider range dominate. Ranks are comparable by construction.

**Why reranking is a separate stage.** Retrieval optimises recall cheaply: get the
right job somewhere in the top 50. Reranking optimises precision: spend one model
call to order the top candidates properly and explain each one. Running the LLM over
the whole index would cost ~900 calls per query. This costs one.

**Why the resume is never stored.** The match endpoint holds resume text in memory
for the length of the request and writes it nowhere, not to the database and not to
logs. The response does not echo it back either. There is no resume data at rest,
so there is nothing to leak.

---

## Results

Pooled LLM judging over 15 queries. For each query the top 10 from every arm are
unioned, every pair in that pool is graded 0 to 3 once at temperature 0, and all
arms are then scored against identical labels. This is the standard IR pooling
method with an LLM in place of human assessors.

| arm | recall@10 | nDCG@10 |
|---|---|---|
| lexical only | `TODO` | `TODO` |
| dense only | `TODO` | `TODO` |
| hybrid (RRF) | `TODO` | `TODO` |
| hybrid + rerank | `TODO` | `TODO` |

Run `python -m evals.run_eval` to regenerate. Numbers land in `evals/results/`.

**The honest caveat.** The judge and the reranker come from the same model family,
which can flatter the rerank arm. The harness writes `evals/results/sample_to_check.csv`
with 30 random judgements for hand grading so the agreement rate can be reported
alongside the table. Quote the numbers with that caveat attached.

---

## Stack

Python 3.12, FastAPI, SQLAlchemy 2.0, PostgreSQL 16 with pgvector (HNSW index),
Postgres full text search, Docker, deployed on Railway.

---

## Running it

```bash
cp .env.example .env          # add your API key
docker compose up -d db       # Postgres with pgvector
pip install -r requirements.txt

python -m app.ingest --threads 4      # fetch, extract, embed, store
uvicorn app.main:app --reload         # http://localhost:8000

python scripts/selftest.py            # logic tests, no DB or key needed
python -m evals.run_eval              # retrieval evaluation
```

Ingest cost for ~900 postings is a few cents on a small model. Start with
`--cap 100` to sanity check before the full run.

---

## Layout

```
app/
  config.py     settings
  db.py         engine, schema DDL, generated tsvector column, indexes
  sources.py    HN fetch and HTML cleanup
  schemas.py    Pydantic contracts for extraction, reranking and the API
  llm.py        provider wrapper: retries, batching, bounded concurrency
  ingest.py     fetch -> extract -> embed -> upsert
  search.py     dense, lexical, RRF fusion, filters
  rerank.py     LLM reranking with why-it-fits and what-is-missing
  main.py       FastAPI app, rate limits, CORS, input caps
evals/
  metrics.py    nDCG@k and recall@k, pure functions
  run_eval.py   pooled judging harness
  queries.json  the golden query set
scripts/
  selftest.py   24 assertions over fusion, filters, metrics and cleaning
```

## Security notes

- API keys server side only, read from env, never reach the browser
- CORS locked to an explicit origin list, no wildcard, because `/api/match` takes resume text
- Per IP rate limits on both model backed endpoints
- Length caps on query and resume before any model call, so a large body cannot run up cost
- All SQL goes through bound parameters, no string interpolation
- Container runs as a non-root user that owns nothing writable
- `/health` returns liveness only and leaks no config or version detail
- Reranker output is validated against the candidate id set, so a hallucinated id is dropped rather than trusted

## Limits

Single ingest process, no incremental scheduling yet. Index is a snapshot, not live.
Pooled judging measures recall against what the arms surfaced, not against the whole
index. The HNSW index is built with default parameters and has not been tuned.

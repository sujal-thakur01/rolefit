"""Retrieval evaluation with pooled LLM judging.

Method, and why it is this one:

1. Pool.   For each query, take the top 10 from every arm and union them.
           Arms are therefore judged on the same candidate set and no arm is
           penalised for surfacing a document the others never showed a judge.
2. Judge.  Grade every (query, document) pair in the pool once, 0 to 3, with a
           fixed prompt at temperature 0. Cache to disk so reruns are free and
           so every arm is scored against identical labels.
3. Score.  recall@10 and nDCG@10, with relevant defined as grade >= 2.

This is the standard pooling approach used in IR benchmarks, with an LLM in
place of human assessors. The honest caveat is that the judge and the reranker
come from the same model family, which can flatter the rerank arm. Hand check a
sample before quoting the numbers anywhere: see --sample.

Run:  python -m evals.run_eval --limit 15
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import SessionLocal
from app.llm import embed_one, gather_limited, structured
from app.rerank import rerank
from app.search import apply_filters, dense, lexical, rrf
from evals.metrics import K, ndcg_at_k, recall_at_k

settings = get_settings()
HERE = Path(__file__).parent
RESULTS = HERE / "results"
CACHE = RESULTS / "judgements.json"
ARMS = ["lexical", "dense", "hybrid", "rerank"]


class Judgement(BaseModel):
    relevance: int = Field(ge=0, le=3)
    reason: str


JUDGE_SYSTEM = """You are grading whether a job posting answers a search query.

3 = directly what the query asked for, on role and on technology.
2 = clearly relevant with one real mismatch, for example right role wrong seniority.
1 = same broad field but not what was asked for.
0 = different role, different field, or the posting does not support the query at all.

Grade only on the posting content given. Ignore how well written it is. Most
postings in a pool are not relevant, so 0 and 1 should be common grades."""


async def judge(query: str, row: dict) -> int:
    doc = json.dumps(
        {
            "title": row.get("title"),
            "company": row.get("company"),
            "location": row.get("location"),
            "work_mode": row.get("work_mode"),
            "seniority": row.get("seniority"),
            "stack": row.get("stack") or [],
            "summary": row.get("summary"),
        },
        ensure_ascii=False,
    )
    j = await structured(
        model=settings.rerank_model,
        system=JUDGE_SYSTEM,
        user=f"Query: {query}\n\nPosting:\n{doc}",
        schema=Judgement,
    )
    return j.relevance


async def run_arm(session, query: str, arm: str) -> list[int]:
    pool = 50
    if arm == "lexical":
        rows = lexical(session, query, k=pool)
    elif arm == "dense":
        rows = dense(session, await embed_one(query), k=pool)
    else:
        rows = rrf(
            dense(session, await embed_one(query), k=pool),
            lexical(session, query, k=pool),
            k=pool,
        )
    rows = apply_filters(rows)
    if arm == "rerank":
        rows = await rerank(query, rows, top_n=K)
    return [r["id"] for r in rows[:K]]




async def main(limit: int, sample: int) -> None:
    RESULTS.mkdir(exist_ok=True)
    queries = json.loads((HERE / "queries.json").read_text())[:limit]
    cache: dict[str, int] = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    with SessionLocal() as session:
        # 1. run every arm, collect the pool
        runs: dict[str, dict[str, list[int]]] = {}
        rowmap: dict[int, dict] = {}
        for q in queries:
            runs[q["id"]] = {}
            for arm in ARMS:
                ids = await run_arm(session, q["text"], arm)
                runs[q["id"]][arm] = ids
            print(f"ran arms for {q['id']}")

        pool_pairs = []
        for q in queries:
            pooled = {jid for arm in ARMS for jid in runs[q["id"]][arm]}
            for jid in pooled:
                if f"{q['id']}:{jid}" not in cache:
                    pool_pairs.append((q, jid))

        if pool_pairs:
            ids = sorted({jid for _, jid in pool_pairs})
            from sqlalchemy import text as sql

            rows = session.execute(
                sql(
                    "SELECT id,title,company,location,work_mode,seniority,stack,summary "
                    "FROM jobs WHERE id = ANY(:ids)"
                ),
                {"ids": ids},
            )
            rowmap = {r._mapping["id"]: dict(r._mapping) for r in rows}

            print(f"judging {len(pool_pairs)} uncached pairs")
            results = await gather_limited(
                [judge(q["text"], rowmap[jid]) for q, jid in pool_pairs if jid in rowmap],
                12,
            )
            for (q, jid), g in zip(
                [p for p in pool_pairs if p[1] in rowmap], results, strict=True
            ):
                if isinstance(g, int):
                    cache[f"{q['id']}:{jid}"] = g
            CACHE.write_text(json.dumps(cache, indent=1))

    # 2. score
    table: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rec, nd = [], []
        for q in queries:
            pooled = {jid for a in ARMS for jid in runs[q["id"]][a]}
            grades = {jid: cache.get(f"{q['id']}:{jid}", 0) for jid in pooled}
            rec.append(recall_at_k(runs[q["id"]][arm], grades))
            nd.append(ndcg_at_k(runs[q["id"]][arm], grades))
        table[arm] = {
            "recall@10": round(sum(rec) / len(rec), 3),
            "ndcg@10": round(sum(nd) / len(nd), 3),
        }

    print(f"\n{'arm':<10}{'recall@10':>12}{'nDCG@10':>12}")
    for arm in ARMS:
        print(f"{arm:<10}{table[arm]['recall@10']:>12}{table[arm]['ndcg@10']:>12}")

    (RESULTS / "metrics.json").write_text(
        json.dumps({"queries": len(queries), "k": K, "arms": table}, indent=1)
    )

    # 3. hand check sample so the numbers are defensible
    if sample:
        keys = random.Random(0).sample(sorted(cache), min(sample, len(cache)))
        lines = ["query_id,job_id,llm_grade,your_grade"]
        lines += [f"{k.split(':')[0]},{k.split(':')[1]},{cache[k]}," for k in keys]
        (RESULTS / "sample_to_check.csv").write_text("\n".join(lines))
        print(f"\nwrote {len(keys)} rows to evals/results/sample_to_check.csv")
        print("fill in your_grade by hand, then report the agreement rate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--sample", type=int, default=30)
    a = ap.parse_args()
    asyncio.run(main(a.limit, a.sample))

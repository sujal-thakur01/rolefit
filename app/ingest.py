"""Ingest pipeline: fetch -> LLM structured extraction -> embed -> upsert.

Run as a module:  python -m app.ingest --threads 4 --cap 800
"""

import argparse
import asyncio
import time

from sqlalchemy import text

from app import sources
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.llm import embed_batch, gather_limited, structured
from app.schemas import JobExtraction

settings = get_settings()

EXTRACT_SYSTEM = """You read one comment from a Hacker News "Who is hiring" thread and turn it into structured data.

Rules:
- Set is_job_posting false for meta commentary, replies, job-seeker posts, or anything without an actual open role.
- Never invent a value. If the posting does not say it, use an empty string or "unknown".
- stack is only technologies the posting actually names. Lowercase. At most 12.
- intern_friendly is true only if interns, new grads or students are explicitly mentioned as welcome.
- summary is two plain sentences on what the person would actually do. No marketing language.
- If a posting lists several roles, describe the most senior engineering role."""


def embed_doc(e: JobExtraction) -> str:
    """What actually gets embedded.

    Deliberately not the raw posting. Raw text is full of boilerplate about
    funding, perks and application instructions, which drags unrelated postings
    together in vector space. A tight composed document retrieves noticeably
    better and is cheaper to embed.
    """
    parts = [
        e.title,
        e.company,
        f"{e.seniority} level",
        e.work_mode,
        e.location,
        " ".join(e.stack),
        e.summary,
    ]
    return " | ".join(p for p in parts if p and p != "unknown")


async def extract_all(postings: list[sources.RawPosting]) -> list[tuple[sources.RawPosting, JobExtraction | None]]:
    coros = [
        structured(
            model=settings.extract_model,
            system=EXTRACT_SYSTEM,
            user=p.text[:6000],
            schema=JobExtraction,
        )
        for p in postings
    ]
    results = await gather_limited(coros, settings.ingest_concurrency)
    out = []
    for p, r in zip(postings, results, strict=True):
        out.append((p, r if isinstance(r, JobExtraction) else None))
    return out


async def embed_all(docs: list[str]) -> list[list[float] | None]:
    """Embed in batches of 96. One call per batch, not one per document."""
    batches = [docs[i : i + 96] for i in range(0, len(docs), 96)]
    results = await gather_limited([embed_batch(b) for b in batches], 4)
    out: list[list[float] | None] = []
    for b, r in zip(batches, results, strict=True):
        if isinstance(r, list):
            out.extend(r)
        else:
            out.extend([None] * len(b))
    return out


UPSERT = text(
    """
    INSERT INTO jobs (source, source_id, posted_at, raw_text, title, company,
                      location, work_mode, seniority, stack, comp, visa, summary,
                      intern_friendly, extraction_ok, embed_text, embedding)
    VALUES (:source, :source_id, :posted_at, :raw_text, :title, :company,
            :location, :work_mode, :seniority, :stack, :comp, :visa, :summary,
            :intern_friendly, :extraction_ok, :embed_text, :embedding)
    ON CONFLICT (source, source_id) DO UPDATE SET
        title = EXCLUDED.title,
        company = EXCLUDED.company,
        location = EXCLUDED.location,
        work_mode = EXCLUDED.work_mode,
        seniority = EXCLUDED.seniority,
        stack = EXCLUDED.stack,
        comp = EXCLUDED.comp,
        visa = EXCLUDED.visa,
        summary = EXCLUDED.summary,
        intern_friendly = EXCLUDED.intern_friendly,
        extraction_ok = EXCLUDED.extraction_ok,
        embed_text = EXCLUDED.embed_text,
        embedding = EXCLUDED.embedding
    """
)


async def run(threads: int, cap: int | None) -> dict:
    t0 = time.time()
    init_db()

    postings = await sources.collect(threads=threads, cap=cap)
    print(f"fetched {len(postings)} raw postings")

    pairs = await extract_all(postings)
    kept = [(p, e) for p, e in pairs if e and e.is_job_posting]
    failed = sum(1 for _, e in pairs if e is None)
    rejected = sum(1 for _, e in pairs if e and not e.is_job_posting)
    print(f"extracted ok={len(kept)} rejected_not_a_job={rejected} failed={failed}")

    docs = [embed_doc(e) for _, e in kept]
    vectors = await embed_all(docs)
    embedded = sum(1 for v in vectors if v)
    print(f"embedded {embedded}/{len(docs)}")

    written = 0
    with SessionLocal() as s:
        for (p, e), doc, vec in zip(kept, docs, vectors, strict=True):
            if vec is None:
                continue
            s.execute(
                UPSERT,
                {
                    "source": "hn",
                    "source_id": p.source_id,
                    "posted_at": p.posted_at,
                    "raw_text": p.text,
                    "title": e.title or None,
                    "company": e.company or None,
                    "location": e.location or None,
                    "work_mode": e.work_mode,
                    "seniority": e.seniority,
                    "stack": e.stack[:12],
                    "comp": e.comp or None,
                    "visa": e.visa or None,
                    "summary": e.summary or None,
                    "intern_friendly": e.intern_friendly,
                    "extraction_ok": True,
                    "embed_text": doc,
                    "embedding": str(vec),
                },
            )
            written += 1
        s.commit()

    stats = {
        "fetched": len(postings),
        "kept": len(kept),
        "rejected_not_a_job": rejected,
        "extraction_failed": failed,
        "written": written,
        "seconds": round(time.time() - t0, 1),
    }
    print(stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4, help="how many monthly threads")
    ap.add_argument("--cap", type=int, default=None, help="stop after N raw postings")
    a = ap.parse_args()
    asyncio.run(run(a.threads, a.cap))


if __name__ == "__main__":
    main()

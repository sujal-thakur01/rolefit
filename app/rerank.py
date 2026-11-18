"""LLM reranking over the fused candidate list.

Retrieval is recall oriented: get the right job somewhere in the top 50 cheaply.
Reranking is precision oriented: spend one model call to order the top handful
properly and say why each one fits. Doing this for all 900 jobs would be absurd,
which is exactly why the architecture is two stage.
"""

import json

from app.config import get_settings
from app.llm import structured
from app.schemas import RerankResponse

settings = get_settings()

RERANK_SYSTEM = """You rank job postings against what a candidate is looking for.

For each posting give:
- relevance 0 to 3. 3 means a strong match on role, stack and seniority. 2 means a good match with one real gap. 1 means loosely related. 0 means wrong role or wrong field.
- why_it_fits: one concrete sentence citing something actually in the posting. Empty string when relevance is 0.
- whats_missing: one honest sentence on the gap between the candidate and the posting. Empty string when there is no real gap.

Judge only on what the posting says. Do not reward marketing language. Be willing to give 0 and 1 scores; a list where everything is a 3 is useless."""


def _candidate_block(rows: list[dict]) -> str:
    items = []
    for r in rows:
        items.append(
            {
                "job_id": r["id"],
                "title": r.get("title"),
                "company": r.get("company"),
                "location": r.get("location"),
                "work_mode": r.get("work_mode"),
                "seniority": r.get("seniority"),
                "stack": r.get("stack") or [],
                "summary": r.get("summary"),
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=1)


async def rerank(profile: str, rows: list[dict], top_n: int = 10) -> list[dict]:
    """Score candidates and return them reordered, highest relevance first.

    One call for the whole batch rather than one per candidate. The model sees
    the candidates side by side, which produces more consistent scoring than
    judging each in isolation, and costs 1 request instead of N.
    """
    if not rows:
        return []

    pool = rows[: max(top_n * 2, 20)]
    user = (
        f"Candidate is looking for:\n{profile.strip()[:6000]}\n\n"
        f"Postings to rank:\n{_candidate_block(pool)}"
    )

    judged = await structured(
        model=settings.rerank_model,
        system=RERANK_SYSTEM,
        user=user,
        schema=RerankResponse,
    )

    by_id = {r["id"]: r for r in pool}
    out = []
    for j in judged.results:
        row = by_id.get(j.job_id)
        if row is None:  # model hallucinated an id, drop it rather than trust it
            continue
        row = dict(row)
        row["score"] = float(j.relevance)
        row["why_it_fits"] = j.why_it_fits or None
        row["whats_missing"] = j.whats_missing or None
        out.append(row)

    # Keep retrieval order as the tiebreak inside an equal relevance band.
    order = {r["id"]: i for i, r in enumerate(pool)}
    out.sort(key=lambda r: (-r["score"], order.get(r["id"], 999)))
    return [r for r in out if r["score"] > 0][:top_n]

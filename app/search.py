"""Retrieval. Four arms so the eval can compare them honestly.

  dense   - pgvector cosine over the composed embedding document
  lexical - Postgres full text, ts_rank_cd over a weighted tsvector
  hybrid  - the two fused with Reciprocal Rank Fusion
  rerank  - hybrid, then an LLM scores the top candidates (see rerank.py)

Why hybrid at all: dense retrieval is good at meaning and bad at exact tokens.
A query for "rust" gets semantically near matches like "systems programming"
but can rank an actual Rust job below them. Lexical search has the opposite
problem: it nails the token and misses paraphrase. Fusing them fixes both,
which the eval numbers in the README show.
"""

from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

RRF_K = 60  # standard constant, damps the influence of top-1 in either list


@dataclass
class Candidate:
    id: int
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    row: dict = field(default_factory=dict)


SELECT_COLS = """
    id, title, company, location, work_mode, seniority, stack, comp,
    summary, intern_friendly, posted_at, source_id, embed_text
"""

DENSE_SQL = text(
    f"""
    SELECT {SELECT_COLS}, 1 - (embedding <=> CAST(:qvec AS vector)) AS score
    FROM jobs
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:qvec AS vector)
    LIMIT :k
    """
)

LEXICAL_SQL = text(
    f"""
    SELECT {SELECT_COLS},
           ts_rank_cd(search_doc, websearch_to_tsquery('english', :q)) AS score
    FROM jobs
    WHERE search_doc @@ websearch_to_tsquery('english', :q)
    ORDER BY score DESC
    LIMIT :k
    """
)


def _rows(result) -> list[dict]:
    return [dict(r._mapping) for r in result]


def dense(session: Session, qvec: list[float], k: int = 50) -> list[dict]:
    return _rows(session.execute(DENSE_SQL, {"qvec": str(qvec), "k": k}))


def lexical(session: Session, q: str, k: int = 50) -> list[dict]:
    if not q.strip():
        return []
    return _rows(session.execute(LEXICAL_SQL, {"q": q, "k": k}))


def rrf(dense_rows: list[dict], lexical_rows: list[dict], k: int = 50) -> list[dict]:
    """Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (RRF_K + rank_in_that_list)

    Rank based rather than score based on purpose. Cosine similarity and
    ts_rank_cd are on completely different scales and neither is calibrated,
    so adding or averaging the raw numbers lets whichever has the wider range
    dominate. Ranks are comparable by construction.
    """
    acc: dict[int, Candidate] = {}

    for rank, row in enumerate(dense_rows, start=1):
        c = acc.setdefault(row["id"], Candidate(id=row["id"], score=0.0, row=row))
        c.dense_rank = rank
        c.score += 1.0 / (RRF_K + rank)

    for rank, row in enumerate(lexical_rows, start=1):
        c = acc.setdefault(row["id"], Candidate(id=row["id"], score=0.0, row=row))
        c.lexical_rank = rank
        c.score += 1.0 / (RRF_K + rank)
        if not c.row:
            c.row = row

    ordered = sorted(acc.values(), key=lambda c: c.score, reverse=True)[:k]
    out = []
    for c in ordered:
        row = dict(c.row)
        row["score"] = round(c.score, 6)
        row["dense_rank"] = c.dense_rank
        row["lexical_rank"] = c.lexical_rank
        out.append(row)
    return out


def apply_filters(
    rows: list[dict],
    work_mode: str | None = None,
    intern_only: bool = False,
) -> list[dict]:
    """Post filter rather than pre filter.

    Filtering inside the SQL would shrink the candidate pool before fusion and
    make the two lists disagree on what they were ranking. Candidate pools here
    are small enough (<= 50 each) that post filtering costs nothing.
    """
    out = rows
    if work_mode:
        out = [r for r in out if (r.get("work_mode") or "") == work_mode]
    if intern_only:
        out = [r for r in out if r.get("intern_friendly")]
    return out

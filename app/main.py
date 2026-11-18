import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import search as S
from app.config import get_settings
from app.db import engine, get_session, init_db
from app.llm import embed_one
from app.rerank import rerank
from app.schemas import JobOut, SearchResponse

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield
    engine.dispose()


app = FastAPI(
    title="RoleFit",
    description="Semantic job search and resume matching over Hacker News hiring threads.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter

# Locked to an explicit list. No wildcard, because the match endpoint takes
# resume text and a wildcard origin would let any page on the internet post to it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    raise HTTPException(status_code=429, detail="Too many requests. Slow down.")


class SearchRequest(BaseModel):
    q: str = Field(min_length=2)
    mode: str = Field(default="hybrid", pattern="^(dense|lexical|hybrid|rerank)$")
    top_k: int = Field(default=10, ge=1, le=25)
    work_mode: str | None = Field(default=None, pattern="^(remote|hybrid|onsite)$")
    intern_only: bool = False


class MatchRequest(BaseModel):
    resume: str = Field(min_length=50)
    top_k: int = Field(default=8, ge=1, le=15)
    work_mode: str | None = Field(default=None, pattern="^(remote|hybrid|onsite)$")
    intern_only: bool = False


def _to_out(rows: list[dict]) -> list[JobOut]:
    out = []
    for r in rows:
        out.append(
            JobOut(
                id=r["id"],
                title=r.get("title"),
                company=r.get("company"),
                location=r.get("location"),
                work_mode=r.get("work_mode"),
                seniority=r.get("seniority"),
                stack=list(r.get("stack") or []),
                comp=r.get("comp"),
                summary=r.get("summary"),
                intern_friendly=r.get("intern_friendly"),
                posted_at=r["posted_at"].isoformat() if r.get("posted_at") else None,
                source_url=(
                    f"https://news.ycombinator.com/item?id={r['source_id']}"
                    if r.get("source_id")
                    else None
                ),
                score=float(r.get("score") or 0.0),
                why_it_fits=r.get("why_it_fits"),
                whats_missing=r.get("whats_missing"),
            )
        )
    return out


async def _retrieve(
    session: Session, query: str, mode: str, top_k: int,
    work_mode: str | None, intern_only: bool,
) -> list[dict]:
    pool = 50
    if mode == "lexical":
        rows = S.lexical(session, query, k=pool)
    elif mode == "dense":
        rows = S.dense(session, await embed_one(query), k=pool)
    else:
        rows = S.rrf(
            S.dense(session, await embed_one(query), k=pool),
            S.lexical(session, query, k=pool),
            k=pool,
        )
    rows = S.apply_filters(rows, work_mode=work_mode, intern_only=intern_only)
    if mode == "rerank":
        return await rerank(query, rows, top_n=top_k)
    return rows[:top_k]


@app.post("/api/search", response_model=SearchResponse)
@limiter.limit(settings.rate_limit_search)
async def api_search(
    request: Request, body: SearchRequest, session: Session = Depends(get_session)
):
    if len(body.q) > settings.max_query_chars:
        raise HTTPException(400, f"Query over {settings.max_query_chars} characters.")
    t0 = time.perf_counter()
    rows = await _retrieve(
        session, body.q, body.mode, body.top_k, body.work_mode, body.intern_only
    )
    return SearchResponse(
        query=body.q,
        mode=body.mode,
        took_ms=int((time.perf_counter() - t0) * 1000),
        results=_to_out(rows),
    )


@app.post("/api/match", response_model=SearchResponse)
@limiter.limit(settings.rate_limit_match)
async def api_match(
    request: Request, body: MatchRequest, session: Session = Depends(get_session)
):
    """Match a resume against the index.

    The resume is held in memory for the length of this request and never
    written to the database or to logs. Nothing about the person who pasted it
    is stored, so there is no resume data at rest to leak.
    """
    if len(body.resume) > settings.max_resume_chars:
        raise HTTPException(
            400, f"Resume over {settings.max_resume_chars} characters."
        )
    t0 = time.perf_counter()
    rows = await _retrieve(
        session, body.resume, "rerank", body.top_k, body.work_mode, body.intern_only
    )
    return SearchResponse(
        query="[resume]",  # never echo the resume back in the response
        mode="rerank",
        took_ms=int((time.perf_counter() - t0) * 1000),
        results=_to_out(rows),
    )


@app.get("/api/stats")
def api_stats(session: Session = Depends(get_session)):
    row = session.execute(
        text(
            """
            SELECT count(*) AS jobs,
                   count(*) FILTER (WHERE intern_friendly) AS intern_friendly,
                   count(*) FILTER (WHERE work_mode = 'remote') AS remote,
                   count(DISTINCT company) AS companies,
                   max(posted_at) AS newest
            FROM jobs
            """
        )
    ).one()._mapping
    return {
        "jobs": row["jobs"],
        "companies": row["companies"],
        "intern_friendly": row["intern_friendly"],
        "remote": row["remote"],
        "newest": row["newest"].isoformat() if row["newest"] else None,
    }


@app.get("/health")
def health():
    """Liveness only. Deliberately returns no config or version detail."""
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        raise HTTPException(503, "database unavailable")


@app.get("/")
def index():
    return FileResponse("web/index.html")

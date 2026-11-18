from typing import Literal

from pydantic import BaseModel, Field

WorkMode = Literal["remote", "hybrid", "onsite", "unknown"]
Seniority = Literal["intern", "junior", "mid", "senior", "staff+", "unknown"]


class JobExtraction(BaseModel):
    """The structured shape we force the LLM to return for a raw posting.

    Every field is required so the model cannot quietly skip one. Unknowns are
    explicit rather than missing, which makes bad extractions easy to count.
    """

    is_job_posting: bool = Field(
        description="False for meta comments, replies and anything that is not an actual job ad."
    )
    title: str = Field(description="Role title, normalised. Empty string if unknown.")
    company: str = Field(description="Hiring company. Empty string if unknown.")
    location: str = Field(description="City and country if given, else empty string.")
    work_mode: WorkMode
    seniority: Seniority
    stack: list[str] = Field(
        default_factory=list,
        description="Technologies named in the posting, lowercase, max 12.",
    )
    comp: str = Field(description="Compensation as written. Empty string if absent.")
    visa: str = Field(description="Visa or sponsorship note. Empty string if absent.")
    summary: str = Field(description="Two sentences on what the role actually involves.")
    intern_friendly: bool = Field(
        description="True only if interns or new grads are explicitly welcomed."
    )


class RankedJudgement(BaseModel):
    """What the reranker returns per candidate."""

    job_id: int
    relevance: int = Field(ge=0, le=3, description="0 irrelevant, 3 strong match.")
    why_it_fits: str = Field(description="One sentence. Empty if relevance is 0.")
    whats_missing: str = Field(description="One sentence on the gap. Empty if none.")


class RerankResponse(BaseModel):
    results: list[RankedJudgement]


class JobOut(BaseModel):
    id: int
    title: str | None
    company: str | None
    location: str | None
    work_mode: str | None
    seniority: str | None
    stack: list[str]
    comp: str | None
    summary: str | None
    intern_friendly: bool | None
    posted_at: str | None
    source_url: str | None
    score: float
    why_it_fits: str | None = None
    whats_missing: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    took_ms: int
    results: list[JobOut]

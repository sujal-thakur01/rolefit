"""Fetch raw postings from Hacker News 'Who is hiring' threads.

Chosen on purpose: the comments are free text written by hundreds of different
people with no schema at all. Half of extraction is deciding what is even a job
posting. That is the same problem a real jobs index has.
"""

import html
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

ALGOLIA = "https://hn.algolia.com/api/v1"
MIN_CHARS = 180  # below this it is almost always a reply, not a posting

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


@dataclass
class RawPosting:
    source_id: str
    text: str
    posted_at: datetime | None

    @property
    def url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.source_id}"


def clean(raw: str) -> str:
    """HN serves comment HTML. Turn it back into something a model can read."""
    s = raw.replace("<p>", "\n\n").replace("</p>", "")
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    s = _TAG.sub(" ", s)
    s = html.unescape(s)
    s = _WS.sub(" ", s)
    s = _NL.sub("\n\n", s)
    return s.strip()


async def find_threads(client: httpx.AsyncClient, limit: int = 6) -> list[str]:
    """Most recent 'Ask HN: Who is hiring?' story ids, newest first."""
    r = await client.get(
        f"{ALGOLIA}/search_by_date",
        params={
            "query": "Ask HN: Who is hiring?",
            "tags": "story,author_whoishiring",
            "hitsPerPage": limit,
        },
    )
    r.raise_for_status()
    hits = r.json().get("hits", [])
    out = []
    for h in hits:
        title = (h.get("title") or "").lower()
        if "who is hiring" in title and "wants to be hired" not in title:
            out.append(str(h["objectID"]))
    return out


async def fetch_thread(client: httpx.AsyncClient, story_id: str) -> list[RawPosting]:
    """Top level comments of one thread. Each is one company's posting."""
    out: list[RawPosting] = []
    page = 0
    while True:
        r = await client.get(
            f"{ALGOLIA}/search_by_date",
            params={
                "tags": f"comment,story_{story_id}",
                "hitsPerPage": 1000,
                "page": page,
            },
        )
        r.raise_for_status()
        data = r.json()
        hits = data.get("hits", [])
        for h in hits:
            # parent_id == story_id keeps top level only, so replies and
            # follow up questions never enter the index.
            if str(h.get("parent_id")) != str(story_id):
                continue
            body = clean(h.get("comment_text") or "")
            if len(body) < MIN_CHARS:
                continue
            ts = h.get("created_at")
            out.append(
                RawPosting(
                    source_id=str(h["objectID"]),
                    text=body,
                    posted_at=datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None,
                )
            )
        page += 1
        if page >= data.get("nbPages", 1) or page > 10:
            break
    return out


async def collect(threads: int = 4, cap: int | None = None) -> list[RawPosting]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        story_ids = await find_threads(client, limit=threads * 2)
        postings: list[RawPosting] = []
        seen: set[str] = set()
        for sid in story_ids[:threads]:
            for p in await fetch_thread(client, sid):
                if p.source_id in seen:
                    continue
                seen.add(p.source_id)
                postings.append(p)
                if cap and len(postings) >= cap:
                    return postings
        return postings

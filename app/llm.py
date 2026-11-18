"""Thin wrapper over an OpenAI compatible endpoint.

Everything that touches a model goes through here so the provider can be
swapped with two env vars and so retries and concurrency live in one place.
"""

import asyncio
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from app.config import get_settings

settings = get_settings()

_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    timeout=60.0,
    max_retries=0,  # tenacity owns retries so the policy is visible
)

T = TypeVar("T", bound=BaseModel)


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=20))
async def embed_one(text: str) -> list[float]:
    resp = await _client.embeddings.create(
        model=settings.embedding_model,
        input=text[:8000],
    )
    return resp.data[0].embedding


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=20))
async def embed_batch(texts: list[str]) -> list[list[float]]:
    """One API call for up to ~100 inputs. Far cheaper than looping embed_one."""
    resp = await _client.embeddings.create(
        model=settings.embedding_model,
        input=[t[:8000] for t in texts],
    )
    # The API preserves input order but sort on index anyway rather than trust it.
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=20))
async def structured(
    model: str,
    system: str,
    user: str,
    schema: type[T],
    temperature: float = 0.0,
) -> T:
    """Force the model into a Pydantic shape. Raises if it cannot comply."""
    resp = await _client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError("model returned no parsable content")
    return parsed


async def gather_limited(coros, limit: int):
    """Run coroutines with a ceiling on concurrency.

    Returns results in input order. Exceptions come back as values rather than
    cancelling the whole batch, so one bad posting cannot kill an ingest run.
    """
    sem = asyncio.Semaphore(limit)

    async def _run(c):
        async with sem:
            try:
                return await c
            except Exception as exc:  # noqa: BLE001 - deliberate, see docstring
                return exc

    return await asyncio.gather(*[_run(c) for c in coros])

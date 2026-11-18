"""Ranking metrics. Pure functions, no database and no network.

Kept in their own module so they can be imported and tested without building a
database engine. Anything that scores a ranking belongs here.
"""

import math

K = 10
RELEVANT_AT = 2  # a grade of 2 or 3 counts as relevant


def ndcg_at_k(ranked_ids: list[int], grades: dict[int, int], k: int = K) -> float:
    """Normalised discounted cumulative gain.

    Uses the 2^grade - 1 gain so a grade 3 result is worth meaningfully more
    than a grade 2, which matters when comparing a reranker that is good at
    ordering against a retriever that merely finds the right set.
    Documents not in `grades` are treated as grade 0.
    """
    dcg = sum(
        (2 ** grades.get(jid, 0) - 1) / math.log2(i + 2)
        for i, jid in enumerate(ranked_ids[:k])
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[int], grades: dict[int, int], k: int = K) -> float:
    """Share of the judged-relevant documents that appear in the top k.

    Relevance is pooled, so this is recall against what any arm surfaced, not
    against the whole index. That is the standard caveat with pooled judging.
    """
    relevant = {jid for jid, g in grades.items() if g >= RELEVANT_AT}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked_ids[:k])) / len(relevant)

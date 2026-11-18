"""Logic tests that need no database and no API key.

Covers the two things most likely to be silently wrong: the fusion maths and
the ranking metrics. Run: python scripts/selftest.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.search import rrf, apply_filters, RRF_K
from app.sources import clean
from evals.metrics import ndcg_at_k, recall_at_k

ok = True
def check(name, cond):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: ok = False

def row(i, **kw):
    base = dict(id=i, title=f"job{i}", company="c", location="l", work_mode="remote",
                seniority="mid", stack=["go"], comp=None, summary="s",
                intern_friendly=False, posted_at=None, source_id=str(i), embed_text="e")
    base.update(kw); return base

print("\nRRF fusion")
d = [row(1), row(2), row(3)]
l = [row(3), row(4), row(1)]
f = rrf(d, l, k=10)
ids = [r["id"] for r in f]
# 1: dense rank1 + lex rank3 = 1/61 + 1/63 ; 3: dense rank3 + lex rank1 = 1/63 + 1/61
exp1 = 1/(RRF_K+1) + 1/(RRF_K+3)
check("documents in both lists outrank single-list documents", set(ids[:2]) == {1, 3})
check("score matches the RRF formula", abs(f[0]["score"] - exp1) < 1e-6)
check("union of both lists is kept", set(ids) == {1, 2, 3, 4})
check("ranks are recorded for explainability", f[0]["dense_rank"] is not None)
check("no duplicate ids after fusion", len(ids) == len(set(ids)))
check("scores descend", all(f[i]["score"] >= f[i+1]["score"] for i in range(len(f)-1)))
check("empty lexical list degrades to dense order", [r["id"] for r in rrf(d, [], k=10)] == [1,2,3])
check("empty both is empty, not a crash", rrf([], [], k=10) == [])

print("\nfilters")
rows = [row(1, work_mode="remote"), row(2, work_mode="onsite", intern_friendly=True)]
check("work_mode filter", [r["id"] for r in apply_filters(rows, work_mode="remote")] == [1])
check("intern filter", [r["id"] for r in apply_filters(rows, intern_only=True)] == [2])
check("no filter is a passthrough", len(apply_filters(rows)) == 2)

print("\nmetrics")
grades = {1:3, 2:0, 3:2, 4:0, 5:1}          # relevant (>=2) = {1,3}
# ideal order is by grade: 3,2,1,0,0 -> ids 1,3,5 then the zeros
check("ideal ranking gives nDCG 1.0", abs(ndcg_at_k([1,3,5,2,4], grades) - 1.0) < 1e-9)
check("demoting a relevant doc lowers nDCG", ndcg_at_k([1,3,2,4,5], grades) < 1.0)
check("reversed ranking scores lower", ndcg_at_k([5,4,2,3,1], grades) < ndcg_at_k([1,3,5,2,4], grades))
check("nDCG never exceeds 1", ndcg_at_k([1,3,5,2,4], grades) <= 1.0 + 1e-12)
check("recall counts only grade>=2", recall_at_k([1,3,9,9,9], grades) == 1.0)
check("half the relevant docs gives 0.5", recall_at_k([1,9,9,9,9], grades) == 0.5)
check("grade 1 does not count as relevant", recall_at_k([5,9,9,9,9], grades) == 0.0)
check("no relevant docs gives 0 not a crash", recall_at_k([1,2], {1:0,2:0}) == 0.0)
check("unjudged ids treated as grade 0", ndcg_at_k([99,98], grades) == 0.0)

print("\nHN html cleaning")
c = clean("Acme<p>We use <i>Go</i> &amp; Postgres.<p>Email <a href=\"x\">us</a>")
check("tags stripped", "<" not in c and ">" not in c)
check("entities decoded", "&" in c and "&amp;" not in c)
check("paragraphs become breaks", "\n" in c)
check("text preserved", "Go" in c and "Postgres" in c)

print("\n" + ("ALL TESTS PASSED" if ok else "FAILURES PRESENT"))
sys.exit(0 if ok else 1)

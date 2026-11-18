"""
Law-retrieval evaluation: before vs. after the review revision.

    python eval/eval_retrieval.py

Systems compared (top-5 law documents per query):
  before   text2vec-base-chinese, vector only, old index (law_embeddings.npy,
           勞基法 + 施行細則 only, old parent_id dedup) — the pre-review system
  vector   bge-m3, vector only, new index from backend/build_index.py
  hybrid   bge-m3 + jieba BM25 fused by RRF
  +terms   hybrid, with the intent call's search_terms appended to the query
           (what the app runs; terms cached by eval/gen_search_terms.py)

Test set: eval/taipei_faq_testset.json — questions from 台北市勞動局常見問答
(collected in backend/scraper2), expected articles = the statutes each
official answer cites (see eval/build_taipei_testset.py).

Metrics: Hit@5 (any expected article in top 5), Recall@5 (share of expected
articles found). "core" restricts to items whose expected articles are all
勞基法 / 施行細則, the only laws the old index covered.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Fronted"))
from hybrid_search import BM25Index, rrf_merge  # noqa: E402

EMB = os.path.join(ROOT, "backend", "embeddings")
LAB, RUL = "勞動基準法", "勞動基準法施行細則"
K = 5



def load_taipei():
    with open(os.path.join(ROOT, "eval", "taipei_faq_testset.json"), encoding="utf-8") as f:
        return [(x["question"], [(e["law"], e["article_no"]) for e in x["expected"]])
                for x in json.load(f)]


def load_terms():
    path = os.path.join(ROOT, "eval", "taipei_search_terms.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_index(prefix_suffix):
    emb = np.load(os.path.join(EMB, f"law_embeddings{prefix_suffix}.npy")).astype(np.float32)
    with open(os.path.join(EMB, f"law_docs{prefix_suffix}.json"), encoding="utf-8") as f:
        d = json.load(f)
    return emb, d["texts"], d["metas"]


def vector_top(q_emb, emb, metas, k, pool=30, key=lambda m: (m.get("source"), m.get("parent_id"))):
    """Mirror of app._search: best chunk per document, top-k documents."""
    scores = emb @ q_emb
    seen, out = set(), []
    for i in np.argsort(scores)[::-1][:pool]:
        kk = key(metas[i])
        if kk not in seen:
            seen.add(kk)
            out.append(i)
    return out[:k]


def score(results, expected):
    got = set(results)
    hits = sum(e in got for e in expected)
    return hits > 0, hits / len(expected)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer

    sets = {"taipei": load_taipei()}

    old_emb, _, old_metas = load_index("")
    new_emb, new_texts, new_metas = load_index("_bgem3")
    print(f"before: {len(old_metas)} chunks (text2vec)   after: {len(new_metas)} chunks (bge-m3)")
    old_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    new_model = SentenceTransformer("BAAI/bge-m3")
    bm25 = BM25Index(new_texts, new_metas)
    terms = load_terms()

    for name, items in sets.items():
        qs = [q for q, _ in items]
        old_q = old_model.encode(qs, normalize_embeddings=True)
        new_q = new_model.encode(qs, normalize_embeddings=True)
        # Same as app: law query = question + search terms; article-citation
        # boost still reads only the question.
        tqs = [" ".join([q, *terms.get(q, [])]) for q in qs]
        new_tq = new_model.encode(tqs, normalize_embeddings=True)
        rows = []
        for n, (q, exp) in enumerate(items):
            exp = [tuple(e) for e in exp]
            # Old index dedups by parent_id alone ("law_第 1 條"), as the old app did.
            before = [(old_metas[i]["source"], old_metas[i]["article_no"])
                      for i in vector_top(old_q[n], old_emb, old_metas, K, key=lambda m: m.get("parent_id"))]
            vec30 = [(new_texts[i], new_metas[i], 0.0) for i in vector_top(new_q[n], new_emb, new_metas, 30)]
            vector = [(m["source"], m["article_no"]) for _, m, _ in vec30[:K]]
            hyb = rrf_merge(vec30, bm25.search(q, 30), new_texts, new_metas, top_k=K, query=q)
            hybrid = [(m["source"], m["article_no"]) for _, m, _ in hyb]
            tvec30 = [(new_texts[i], new_metas[i], 0.0) for i in vector_top(new_tq[n], new_emb, new_metas, 30)]
            thyb = rrf_merge(tvec30, bm25.search(tqs[n], 30), new_texts, new_metas, top_k=K, query=q)
            plus = [(m["source"], m["article_no"]) for _, m, _ in thyb]
            core = all(l in (LAB, RUL) for l, _ in exp)
            rows.append((q, core, [score(r, exp) for r in (before, vector, hybrid, plus)]))

        print(f"\n=== {name}: {len(rows)} queries ===")
        print(f"{'':<16}{'before':>9}{'vector':>9}{'hybrid':>9}{'+terms':>9}")
        for label, sub in (("all", rows), ("core 勞基法+細則", [r for r in rows if r[1]])):
            if not sub:
                continue
            for mi, metric in enumerate(("Hit@5", "Recall@5")):
                vals = [sum(r[2][s][mi] for r in sub) / len(sub) for s in range(4)]
                print(f"{metric + ' ' + label:<16}" + "".join(f"{v:>9.0%}" for v in vals)
                      + (f"   (n={len(sub)})" if mi == 0 else ""))
        if not terms:
            print("(+terms = hybrid: eval/taipei_search_terms.json missing — run gen_search_terms.py)")
        misses = [q for q, _, s in rows if not s[3][0]]
        if misses:
            print(f"+terms misses ({len(misses)}): " + " / ".join(m[:24] for m in misses[:15]))


if __name__ == "__main__":
    main()

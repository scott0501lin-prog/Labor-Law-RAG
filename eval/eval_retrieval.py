"""
Law-retrieval evaluation: before vs. after the review revision.

    python eval/eval_retrieval.py

Systems compared (top-5 law documents per query):
  before   text2vec-base-chinese, vector only, old index (law_embeddings.npy,
           勞基法 + 施行細則 only, old parent_id dedup) — the pre-review system
  vector   bge-m3, vector only, new index from backend/build_index.py
  hybrid   bge-m3 + jieba BM25 fused by RRF
  +terms   hybrid, with the intent call's search_terms appended to the query
           (what the app runs; terms cached by eval/gen_search_terms.py).
           Terms are sampled, so this column is the mean over the cached runs;
           the per-run range is printed below the table.

Test set: eval/taipei_faq_testset.json — 台北市勞動局常見問答 (collected in
backend/scraper2); expected articles = the statutes each official answer
cites (see eval/build_taipei_testset.py). Only official citations are used
here: eval/bli_pension_testset.json (勞退新制) has Claude-chosen articles and
is kept for end-to-end evaluation, which grades against the official answer.
To score extra test-set files, add them to TESTSETS; the first table then
covers all of them and per-source tables follow.

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



TESTSETS = {"taipei": "taipei_faq_testset.json"}


def load_testset(filename):
    with open(os.path.join(ROOT, "eval", filename), encoding="utf-8") as f:
        return [(x["question"], [(e["law"], e["article_no"]) for e in x["expected"]])
                for x in json.load(f)]


def load_terms():
    """{question: [[term, ...] per run]} (old single-run files are wrapped)."""
    path = os.path.join(ROOT, "eval", "search_terms.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {q: (v if v and isinstance(v[0], list) else [v]) for q, v in data.items()}


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

    parts = {name: load_testset(fn) for name, fn in TESTSETS.items()}
    sets = {"all": [it for items in parts.values() for it in items]}

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
        # boost still reads only the question. One query list per run.
        n_runs = max((len(v) for v in terms.values()), default=0) or 1
        uncached = [q for q in qs if q not in terms]
        if terms and uncached:
            print(f"WARNING {name}: {len(uncached)} questions have no cached search terms; "
                  "their +terms score equals hybrid. Run: python eval/gen_search_terms.py --missing")
        run_tqs = [[" ".join([q, *(terms.get(q) or [[]] * n_runs)[r]]) for q in qs]
                   for r in range(n_runs)]
        run_tq = [new_model.encode(t, normalize_embeddings=True) for t in run_tqs]
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
            per_run = []
            for r in range(n_runs):
                tvec30 = [(new_texts[i], new_metas[i], 0.0)
                          for i in vector_top(run_tq[r][n], new_emb, new_metas, 30)]
                thyb = rrf_merge(tvec30, bm25.search(run_tqs[r][n], 30), new_texts, new_metas,
                                 top_k=K, query=q)
                per_run.append(score([(m["source"], m["article_no"]) for _, m, _ in thyb], exp))
            plus = (sum(h for h, _ in per_run) / n_runs, sum(rc for _, rc in per_run) / n_runs)
            core = all(l in (LAB, RUL) for l, _ in exp)
            rows.append((q, core, [score(r, exp) for r in (before, vector, hybrid)] + [plus],
                         per_run))

        def table(title, sub_rows):
            print(f"\n=== {title}: {len(sub_rows)} queries ===")
            print(f"{'':<16}{'before':>9}{'vector':>9}{'hybrid':>9}{'+terms':>9}")
            for label, sub in (("all", sub_rows), ("core 勞基法+細則", [r for r in sub_rows if r[1]])):
                if not sub:
                    continue
                for mi, metric in enumerate(("Hit@5", "Recall@5")):
                    vals = [sum(r[2][s][mi] for r in sub) / len(sub) for s in range(4)]
                    print(f"{metric + ' ' + label:<16}" + "".join(f"{v:>9.0%}" for v in vals)
                          + (f"   (n={len(sub)})" if mi == 0 else ""))

        table("test set (all sources)", rows)
        start = 0
        for part, part_items in (parts.items() if len(parts) > 1 else ()):
            table(f"  by source: {part}", rows[start:start + len(part_items)])
            start += len(part_items)
        print()
        if not terms:
            print("(+terms = hybrid: eval/search_terms.json missing — run gen_search_terms.py)")
        else:
            hits = [sum(r[3][k][0] for r in rows) / len(rows) for k in range(n_runs)]
            print(f"+terms Hit@5 per run ({n_runs} runs): "
                  + ", ".join(f"{h:.0%}" for h in hits)
                  + f"   range {min(hits):.0%}-{max(hits):.0%}")
        misses = [q for q, _, s, _ in rows if s[3][0] < 0.5]
        if misses:
            print(f"+terms misses in most runs ({len(misses)}): " + " / ".join(m[:24] for m in misses[:15]))


if __name__ == "__main__":
    main()

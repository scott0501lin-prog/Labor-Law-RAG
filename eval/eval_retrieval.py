"""
Law-retrieval evaluation: before vs. after the review revision.

    python eval/eval_retrieval.py                 # both test sets
    python eval/eval_retrieval.py --set taipei    # 台北市常見問答 only

Systems compared (top-5 law documents per query):
  before   text2vec-base-chinese, vector only, old index (law_embeddings.npy,
           勞基法 + 施行細則 only, old parent_id dedup) — the pre-review system
  vector   bge-m3, vector only, new index from backend/build_index.py
  hybrid   bge-m3 + jieba BM25 fused by RRF (what the app runs)

Test sets:
  taipei   eval/taipei_faq_testset.json (citations from 台北市勞動局常見問答)
  manual   hand-written queries below (explicit article numbers + colloquial)

Metrics: Hit@5 (any expected article in top 5), Recall@5 (share of expected
articles found). "core" restricts to items whose expected articles are all
勞基法 / 施行細則, the only laws the old index covered.
"""
import argparse
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

MANUAL = [
    ("勞基法第24條加班費怎麼算", [(LAB, "第 24 條")]),
    ("第三十二條之一補休規定", [(LAB, "第 32-1 條")]),
    ("勞基法 9-1 條競業禁止要件", [(LAB, "第 9-1 條")]),
    ("第38條特休天數", [(LAB, "第 38 條")]),
    ("勞基法第十一條資遣事由", [(LAB, "第 11 條")]),
    ("第12條不經預告終止契約", [(LAB, "第 12 條")]),
    ("施行細則第7條勞動契約應約定事項", [(RUL, "第 7 條")]),
    ("勞基法施行細則第24條之1特休", [(RUL, "第 24-1 條")]),
    ("勞基法第84條之1責任制", [(LAB, "第 84-1 條")]),
    ("第五十九條職災補償", [(LAB, "第 59 條")]),
    ("未休特休可以換錢嗎", [(LAB, "第 38 條")]),
    ("公司要資遣我，資遣費怎麼算", [(LAB, "第 17 條")]),
    ("預告期間多久", [(LAB, "第 16 條")]),
    ("謀職假可以請幾天", [(LAB, "第 16 條")]),
    ("平均工資的定義", [(LAB, "第 2 條")]),
    ("休息日加班費", [(LAB, "第 24 條")]),
    ("例假日可以叫員工上班嗎", [(LAB, "第 36 條"), (LAB, "第 40 條")]),
    ("國定假日出勤工資加倍", [(LAB, "第 39 條")]),
    ("工作滿幾年可以自請退休", [(LAB, "第 53 條")]),
    ("強制退休年齡", [(LAB, "第 54 條")]),
    ("派遣勞工被要派單位欠薪", [(LAB, "第 22-1 條")]),
    ("童工工作時間限制", [(LAB, "第 47 條")]),
    ("女性夜間工作", [(LAB, "第 49 條")]),
    ("工資清冊保存幾年", [(LAB, "第 23 條")]),
    ("最低服務年限約定", [(LAB, "第 15-1 條")]),
]


def load_taipei():
    with open(os.path.join(ROOT, "eval", "taipei_faq_testset.json"), encoding="utf-8") as f:
        return [(x["question"], [(e["law"], e["article_no"]) for e in x["expected"]])
                for x in json.load(f)]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["taipei", "manual"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer

    sets = {"taipei": load_taipei(), "manual": MANUAL}
    if args.set:
        sets = {args.set: sets[args.set]}

    old_emb, _, old_metas = load_index("")
    new_emb, new_texts, new_metas = load_index("_bgem3")
    print(f"before: {len(old_metas)} chunks (text2vec)   after: {len(new_metas)} chunks (bge-m3)")
    old_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    new_model = SentenceTransformer("BAAI/bge-m3")
    bm25 = BM25Index(new_texts, new_metas)

    for name, items in sets.items():
        qs = [q for q, _ in items]
        old_q = old_model.encode(qs, normalize_embeddings=True)
        new_q = new_model.encode(qs, normalize_embeddings=True)
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
            core = all(l in (LAB, RUL) for l, _ in exp)
            rows.append((q, core, [score(r, exp) for r in (before, vector, hybrid)]))

        print(f"\n=== {name}: {len(rows)} queries ===")
        print(f"{'':<16}{'before':>9}{'vector':>9}{'hybrid':>9}")
        for label, sub in (("all", rows), ("core 勞基法+細則", [r for r in rows if r[1]])):
            if not sub:
                continue
            for mi, metric in enumerate(("Hit@5", "Recall@5")):
                vals = [sum(r[2][s][mi] for r in sub) / len(sub) for s in range(3)]
                print(f"{metric + ' ' + label:<16}" + "".join(f"{v:>9.0%}" for v in vals)
                      + (f"   (n={len(sub)})" if mi == 0 else ""))
        misses = [q for q, _, s in rows if not s[2][0]]
        if misses:
            print(f"hybrid misses ({len(misses)}): " + " / ".join(m[:24] for m in misses[:15]))


if __name__ == "__main__":
    main()

"""
Compare vector-only vs. hybrid (BM25 + RRF) retrieval on the law corpus.

Usage:  python eval/eval_hybrid.py
Builds law chunks exactly like app._load_chroma_index (bge-m3, 500/50 split),
encodes them once (~1 min CPU), then reports Hit@5 / Recall@5 per query.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Fronted"))
from hybrid_search import BM25Index, rrf_merge  # noqa: E402

LAB = "勞動基準法"
RUL = "勞動基準法施行細則"
GEN = "性別平等工作法"

# (query, {(source, article_no), ...})  — hit if any expected doc is in top-5
CASES = [
    # explicit article numbers
    ("勞基法第24條加班費怎麼算", {(LAB, "第 24 條")}),
    ("第三十二條之一補休規定", {(LAB, "第 32-1 條")}),
    ("勞基法 9-1 條競業禁止要件", {(LAB, "第 9-1 條")}),
    ("第38條特休天數", {(LAB, "第 38 條")}),
    ("勞基法第十一條資遣事由", {(LAB, "第 11 條")}),
    ("第12條不經預告終止契約", {(LAB, "第 12 條")}),
    ("施行細則第7條勞動契約應約定事項", {(RUL, "第 7 條")}),
    ("勞基法施行細則第24條之1特休", {(RUL, "第 24-1 條")}),
    ("性平法第15條產假", {(GEN, "第 15 條")}),
    ("性別平等工作法第十六條育嬰留職停薪", {(GEN, "第 16 條")}),
    ("勞基法第84條之1責任制", {(LAB, "第 84-1 條")}),
    ("第五十九條職災補償", {(LAB, "第 59 條")}),
    # legal terms / colloquial phrasing
    ("未休特休可以換錢嗎", {(LAB, "第 38 條")}),
    ("公司要資遣我，資遣費怎麼算", {(LAB, "第 17 條")}),
    ("預告期間多久", {(LAB, "第 16 條")}),
    ("謀職假可以請幾天", {(LAB, "第 16 條")}),
    ("平均工資的定義", {(LAB, "第 2 條")}),
    ("休息日加班費", {(LAB, "第 24 條")}),
    ("例假日可以叫員工上班嗎", {(LAB, "第 36 條"), (LAB, "第 40 條")}),
    ("國定假日出勤工資加倍", {(LAB, "第 39 條")}),
    ("工作滿幾年可以自請退休", {(LAB, "第 53 條")}),
    ("強制退休年齡", {(LAB, "第 54 條")}),
    ("派遣勞工被要派單位欠薪", {(LAB, "第 22-1 條")}),
    ("哺乳時間", {(GEN, "第 18 條")}),
    ("陪產檢假幾天", {(GEN, "第 15 條")}),
    # 13 = 雇主防治義務, 27 = 雇主賠償責任 — both answer the question
    ("職場性騷擾雇主責任", {(GEN, "第 13 條"), (GEN, "第 27 條")}),
    ("童工工作時間限制", {(LAB, "第 47 條")}),
    ("女性夜間工作", {(LAB, "第 49 條")}),
    ("工資清冊保存幾年", {(LAB, "第 23 條")}),
    ("最低服務年限約定", {(LAB, "第 15-1 條")}),
]


def smart_split(text, max_chars, overlap):
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            for b in ["。", "；", "！", "？", "\n"]:
                pos = text.rfind(b, start + max_chars // 3, end)
                if pos != -1:
                    end = pos + 1
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_law():
    texts, metas = [], []
    for fn in ["labor_law_cleaned.json", "labor_law_rules_cleaned.json", "gender_equality_law.json"]:
        with open(os.path.join(ROOT, "backend", "data", fn), encoding="utf-8") as f:
            for item in json.load(f):
                art = item.get("article_no", "")
                for idx, ch in enumerate(smart_split(item.get("text", ""), 500, 50)[:50]):
                    texts.append(ch)
                    metas.append({"source": item.get("source", ""), "article_no": art,
                                  "parent_id": f"law_{art}", "chunk_idx": idx})
    return texts, metas


def vector_search(q_emb, emb, texts, metas, top_k, pool=30):
    scores = emb @ q_emb
    seen = {}
    for i in np.argsort(scores)[::-1][:pool]:
        key = (metas[i]["source"], metas[i]["parent_id"])
        if key not in seen:
            seen[key] = (texts[i], metas[i], float(scores[i]))
    return list(seen.values())[:top_k]


def main():
    from sentence_transformers import SentenceTransformer

    texts, metas = load_law()
    missing = [exp for _, s in CASES for exp in s
               if not any((m["source"], m["article_no"]) == exp for m in metas)]
    if missing:
        sys.exit(f"Expected articles not in corpus: {missing}")

    print(f"law chunks: {len(texts)} — encoding with bge-m3…", flush=True)
    model = SentenceTransformer("BAAI/bge-m3")
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True)
    bm25 = BM25Index(texts, metas)

    tot = {"vec": [0, 0.0], "bm25": [0, 0.0], "hyb": [0, 0.0]}
    print(f"\n{'query':<28} {'vec':>5} {'bm25':>5} {'hyb':>5}")
    for q, expected in CASES:
        q_emb = model.encode([q], normalize_embeddings=True)[0]
        vec5 = vector_search(q_emb, emb, texts, metas, 5)
        vec30 = vector_search(q_emb, emb, texts, metas, 30)
        bidx = bm25.search(q, 30)
        hyb5 = rrf_merge(vec30, bidx, texts, metas, top_k=5, query=q)
        bm5 = rrf_merge([], bidx, texts, metas, top_k=5)

        row = []
        for name, res in (("vec", vec5), ("bm25", bm5), ("hyb", hyb5)):
            got = {(m["source"], m["article_no"]) for _, m, _ in res}
            hits = len(got & expected)
            tot[name][0] += hits > 0
            tot[name][1] += hits / len(expected)
            row.append("✓" if hits else "✗")
        print(f"{q:<28} {row[0]:>5} {row[1]:>5} {row[2]:>5}")

    n = len(CASES)
    print(f"\n{'':<28} {'vec':>5} {'bm25':>5} {'hyb':>5}")
    print(f"{'Hit@5':<28} " + " ".join(f"{tot[k][0] / n:>5.0%}" for k in ("vec", "bm25", "hyb")))
    print(f"{'Recall@5':<28} " + " ".join(f"{tot[k][1] / n:>5.0%}" for k in ("vec", "bm25", "hyb")))


if __name__ == "__main__":
    main()

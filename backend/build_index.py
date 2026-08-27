"""
預先計算所有 embedding 並存成 numpy 檔案。
只需執行一次：python backend/build_index.py
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "shibing624/text2vec-base-chinese"
MAX_LEN    = 800

CASE_FILES = [
    "taipei_cases.json",
    "ntpc_cases.json",
    "mol_qa_cases.json",
    "mol_fint_cases.json",
    "judicial_cases.json",
    "derived_interpretations.json",
]


def load_json(filename):
    with open(os.path.join(BASE_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


def encode_and_save(model, texts, metas, prefix):
    print(f"  計算 embedding（共 {len(texts)} 筆）...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    np.save(os.path.join(OUT_DIR, f"{prefix}_embeddings.npy"), embeddings)
    with open(os.path.join(OUT_DIR, f"{prefix}_docs.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metas": metas}, f, ensure_ascii=False)
    print(f"  已儲存至 embeddings/{prefix}_*")


if __name__ == "__main__":
    print(f"載入模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ── 勞基法本法 + 施行細則條文 ───────────────────────────────
    print("\n[1/2] 勞基法本法 + 施行細則條文")
    laws  = load_json("labor_law_cleaned.json") + load_json("labor_law_rules_cleaned.json")
    texts = [item.get("text", "")[:MAX_LEN] for item in laws]
    metas = [{"article_no": item.get("article_no", ""), "source": item.get("source", "勞動基準法")} for item in laws]
    encode_and_save(model, texts, metas, "law")

    # ── 案例 / Q&A / 裁判書 ──────────────────────
    print("\n[2/2] 案例 / Q&A / 裁判書")
    texts, metas = [], []
    for fn in CASE_FILES:
        items = load_json(fn)
        print(f"  {fn}：{len(items)} 筆")
        for item in items:
            content = item.get("content", "").strip()
            if not content:
                continue
            title = item.get("title", "").strip()
            texts.append(f"{title}\n{content}"[:MAX_LEN])
            metas.append({
                "source"  : item.get("source", ""),
                "category": item.get("category", ""),
                "url"     : item.get("url", ""),
                "title"   : title,
            })
    encode_and_save(model, texts, metas, "cases")

    print("\n完成！可以啟動 streamlit run Fronted/app.py")

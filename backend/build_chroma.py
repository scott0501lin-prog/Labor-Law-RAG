"""
使用 Voyage AI API 建立向量索引並存入 ChromaDB。
執行一次即可：python backend/build_chroma.py

需先設定環境變數 VOYAGE_API_KEY（或在 .env 中設定）。

Collections：
  law_chunks_voyage   — 勞基法本法 + 施行細則 + 性別平等工作法
  cases_chunks_voyage — 案例 / Q&A / 裁判書
"""

import json
import os
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
CHROMA_DIR  = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "chroma_db"))
if not CHROMA_DIR.isascii():
    CHROMA_DIR = os.path.join(
        os.getenv("LOCALAPPDATA") or os.path.expanduser("~"), "law_rag", "chroma_db"
    )

VOYAGE_MODEL     = "voyage-multilingual-2"
COLLECTION_LAW   = "law_chunks_voyage"
COLLECTION_CASES = "cases_chunks_voyage"

LAW_MAX_CHARS      = 500
CASE_MAX_CHARS     = 400
CASE_OVERLAP       = 50
MAX_CHUNKS_PER_DOC = 50
EMBED_BATCH        = 128
CHROMA_BATCH       = 256

CASE_FILES = [
    "taipei_cases.json",
    "mol_qa_cases.json",
    "mol_fint_cases.json",
    "judicial_cases.json",
    "ntpc_cases.json",
    "derived_interpretations.json",
]


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠️  {filename} 不存在，略過")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def build_law_chunks(laws):
    texts, metas = [], []
    for item in laws:
        source     = item.get("source", "勞動基準法")
        article_no = item.get("article_no", "")
        full_text  = item.get("text", "")
        parent_id  = f"law_{article_no}"
        for idx, chunk in enumerate(smart_split(full_text, LAW_MAX_CHARS, 50)[:MAX_CHUNKS_PER_DOC]):
            texts.append(chunk)
            metas.append({
                "data_type" : "law",
                "source"    : source,
                "article_no": article_no,
                "parent_id" : parent_id,
                "chunk_idx" : idx,
                "category"  : "",
                "title"     : "",
                "url"       : "",
            })
    return texts, metas


def build_case_chunks(case_files):
    texts, metas = [], []
    for fn in case_files:
        items = load_json(fn)
        if not items:
            continue
        print(f"  {fn}：{len(items)} 筆")
        for doc_idx, item in enumerate(items):
            content  = item.get("content", "").strip()
            if not content:
                continue
            source    = item.get("source", "")
            category  = item.get("category", "")
            label     = f"【{source} {category}】"
            parent_id = f"{fn.replace('.json', '')}_{doc_idx}"
            for idx, chunk in enumerate(smart_split(content, CASE_MAX_CHARS, CASE_OVERLAP)[:MAX_CHUNKS_PER_DOC]):
                texts.append(f"{label}\n{chunk}")
                metas.append({
                    "data_type" : "case",
                    "source"    : source,
                    "category"  : category,
                    "title"     : item.get("title", "").strip(),
                    "url"       : item.get("url", ""),
                    "parent_id" : parent_id,
                    "chunk_idx" : idx,
                    "article_no": "",
                })
    return texts, metas


def embed_texts(voyage_client, texts, input_type="document"):
    embs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        res   = voyage_client.embed(batch, model=VOYAGE_MODEL, input_type=input_type)
        embs.extend(res.embeddings)
        done = min(i + EMBED_BATCH, len(texts))
        print(f"  {done}/{len(texts)} 塊已編碼…", end="\r", flush=True)
        if i + EMBED_BATCH < len(texts):
            time.sleep(0.05)
    print()
    return embs


def populate_collection(chroma_client, name, texts, metas, embeddings):
    try:
        chroma_client.delete_collection(name)
        print(f"  舊 collection '{name}' 已刪除")
    except Exception:
        pass
    col = chroma_client.create_collection(name, metadata={"hnsw:space": "cosine"})
    ids = [f"{name}_{i}" for i in range(len(texts))]
    for s in range(0, len(ids), CHROMA_BATCH):
        e = s + CHROMA_BATCH
        col.add(
            ids=ids[s:e],
            embeddings=embeddings[s:e],
            documents=texts[s:e],
            metadatas=metas[s:e],
        )
    print(f"  OK '{name}': {col.count()} chunks")
    return col


if __name__ == "__main__":
    import chromadb
    import voyageai

    voyage_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_key:
        raise SystemExit("❌ VOYAGE_API_KEY 未設定。請在 .env 中加入 VOYAGE_API_KEY=...")

    t_total       = time.perf_counter()
    voyage_client = voyageai.Client(api_key=voyage_key)
    print(f"Voyage AI 模型：{VOYAGE_MODEL}\n")

    print("[1/2] 勞基法本法 + 施行細則 + 性別平等工作法")
    laws = (
        load_json("labor_law_cleaned.json")
        + load_json("labor_law_rules_cleaned.json")
        + load_json("gender_equality_law.json")
    )
    law_texts, law_metas = build_law_chunks(laws)
    print(f"  chunk 數：{len(law_texts)}")
    t1 = time.perf_counter()
    law_embs = embed_texts(voyage_client, law_texts, input_type="document")
    print(f"  編碼：{time.perf_counter()-t1:.1f}s")

    print("\n[2/2] 案例 / Q&A / 裁判書")
    case_texts, case_metas = build_case_chunks(CASE_FILES)
    print(f"  chunk 數：{len(case_texts)}")
    t2 = time.perf_counter()
    case_embs = embed_texts(voyage_client, case_texts, input_type="document")
    print(f"  編碼：{time.perf_counter()-t2:.1f}s")

    print(f"\n存入 ChromaDB：{CHROMA_DIR}")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    print("\n寫入法條 collection…")
    populate_collection(chroma_client, COLLECTION_LAW, law_texts, law_metas, law_embs)

    print("\n寫入案例 collection…")
    populate_collection(chroma_client, COLLECTION_CASES, case_texts, case_metas, case_embs)

    elapsed = time.perf_counter() - t_total
    print(f"\n完成！總時間：{elapsed:.1f}s")
    print("重啟 Streamlit 即可使用新索引：streamlit run Fronted/app.py")

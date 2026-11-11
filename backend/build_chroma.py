"""
直接從 backend/data/ 計算 bge-m3 向量並存入 ChromaDB。
執行一次即可：python backend/build_chroma.py

Collections（與 app.py 一致）：
  law_chunks_bgem3   — 勞基法本法 + 施行細則 + 性別平等工作法（一條一塊）
  cases_chunks_bgem3 — 案例 / Q&A / 裁判書（約 400 字一塊）

metadata 欄位可用於 ChromaDB where 過濾：
  law   : source, article_no, parent_id, chunk_idx, data_type="law"
  cases : source, category, title, url, parent_id, chunk_idx, data_type="case"
"""

import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "chroma_db"))
# Same rule as app.py: Chroma's HNSW writer fails on non-ASCII paths on Windows
if not CHROMA_DIR.isascii():
    CHROMA_DIR = os.path.join(os.getenv("LOCALAPPDATA") or os.path.expanduser("~"), "law_rag", "chroma_db")

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_LAW   = "law_chunks_bgem3"
COLLECTION_CASES = "cases_chunks_bgem3"

LAW_MAX_CHARS      = 500
CASE_MAX_CHARS     = 400
CASE_OVERLAP       = 50
MAX_CHUNKS_PER_DOC = 50
BATCH_SIZE         = 256

CASE_FILES = [
    "taipei_cases.json",
    "mol_qa_cases.json",
    "mol_fint_cases.json",
    "judicial_cases.json",
    "ntpc_cases.json",
    "derived_interpretations.json",
]


# ── 工具 ─────────────────────────────────────────────────────────────────────

def load_json(filename):
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
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


# ── 切塊 ─────────────────────────────────────────────────────────────────────

def build_law_chunks(laws):
    texts, metas = [], []
    for item in laws:
        source     = item.get("source", "勞動基準法")
        article_no = item.get("article_no", "")
        full_text  = item.get("text", "")
        label      = f"【{source} {article_no}】"
        parent_id  = f"law_{article_no}"
        merged     = f"{label}\n{full_text}"
        for idx, chunk in enumerate(smart_split(merged, LAW_MAX_CHARS, 50)[:MAX_CHUNKS_PER_DOC]):
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
        try:
            items = load_json(fn)
        except FileNotFoundError:
            print(f"  ⚠️  {fn} 不存在，略過")
            continue
        print(f"  {fn}：{len(items)} 筆")
        for doc_idx, item in enumerate(items):
            source   = item.get("source", "")
            category = item.get("category", "")
            title    = item.get("title", "").strip()
            url      = item.get("url", "")
            content  = item.get("content", "").strip()
            if not content:
                continue
            label     = f"【{source} {category}】"
            parent_id = f"{fn.replace('.json', '')}_{doc_idx}"
            for idx, chunk in enumerate(smart_split(content, CASE_MAX_CHARS, CASE_OVERLAP)[:MAX_CHUNKS_PER_DOC]):
                merged = f"{label}\n{chunk}"
                texts.append(merged)
                metas.append({
                    "data_type" : "case",
                    "source"    : source,
                    "category"  : category,
                    "title"     : title,
                    "url"       : url,
                    "parent_id" : parent_id,
                    "chunk_idx" : idx,
                    "article_no": "",
                })
    return texts, metas


# ── ChromaDB 寫入 ─────────────────────────────────────────────────────────────

def populate_collection(client, name, texts, metas, embeddings):
    try:
        client.delete_collection(name)
        print(f"  舊 collection '{name}' 已刪除")
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    ids = [f"{name}_{i}" for i in range(len(texts))]
    for s in range(0, len(ids), BATCH_SIZE):
        e = s + BATCH_SIZE
        col.add(
            ids=ids[s:e],
            embeddings=embeddings[s:e].tolist(),
            documents=texts[s:e],
            metadatas=metas[s:e],
        )
    print(f"  OK '{name}': {col.count()} chunks")
    return col


# ── 主程式 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import psutil
    from sentence_transformers import SentenceTransformer
    import chromadb

    proc = psutil.Process()
    t_total = time.perf_counter()

    print(f"模型：{MODEL_NAME}")
    rss0 = proc.memory_info().rss / 1024 ** 2
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    load_time = time.perf_counter() - t0
    dim = model.encode(["test"], normalize_embeddings=True).shape[1]
    rss1 = proc.memory_info().rss / 1024 ** 2
    print(f"維度：{dim} | 載入：{load_time:.1f}s | 記憶體增加：{rss1 - rss0:.0f}MB\n")

    # ── 法條 ──────────────────────────────────────────────────────────────────
    print("[1/2] 勞基法本法 + 施行細則 + 性別平等工作法")
    laws = (load_json("labor_law_cleaned.json")
            + load_json("labor_law_rules_cleaned.json")
            + load_json("gender_equality_law.json"))
    law_texts, law_metas = build_law_chunks(laws)
    print(f"  chunk 數：{len(law_texts)}")
    t1 = time.perf_counter()
    law_emb = model.encode(law_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print(f"  編碼：{time.perf_counter()-t1:.1f}s")

    # ── 案例 ──────────────────────────────────────────────────────────────────
    print("\n[2/2] 案例 / Q&A / 裁判書")
    case_texts, case_metas = build_case_chunks(CASE_FILES)
    print(f"  chunk 數：{len(case_texts)}")
    t2 = time.perf_counter()
    case_emb = model.encode(case_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print(f"  編碼：{time.perf_counter()-t2:.1f}s")

    # ── 存入 ChromaDB ─────────────────────────────────────────────────────────
    print(f"\n存入 ChromaDB：{CHROMA_DIR}")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    print(f"\n寫入法條 collection…")
    populate_collection(chroma_client, COLLECTION_LAW, law_texts, law_metas, law_emb)

    print(f"\n寫入案例 collection…")
    populate_collection(chroma_client, COLLECTION_CASES, case_texts, case_metas, case_emb)

    elapsed = time.perf_counter() - t_total
    print(f"\n完成！總時間：{elapsed:.1f}s")
    print("重啟 Streamlit 即可使用新索引：streamlit run Fronted/app.py")

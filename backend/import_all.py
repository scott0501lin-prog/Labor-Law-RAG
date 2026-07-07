"""
把所有 JSON 資料匯入 ChromaDB
執行：python backend/import_all.py

建立兩個 collection：
  - labor_law_collection  ← 勞基法條文
  - cases_collection      ← 政府 Q&A、函釋、裁判書
"""

import json
import os
import time

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "law_db")

EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# 案例類 JSON 清單（格式相同，統一匯入 cases_collection）
CASE_FILES = [
    "taipei_cases.json",
    "ntpc_cases.json",
    "mol_qa_cases.json",
    "mol_fint_cases.json",
    "judicial_cases.json",
]

MAX_DOC_LEN = 800   # 每筆文件最多保留字數（避免 HNSW 索引失敗）


def load_json(filename: str) -> list[dict]:
    path = os.path.join(BASE_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


# ==========================================
# 1. 匯入勞基法條文 → labor_law_collection
# ==========================================
def import_law(client: chromadb.ClientAPI, emb_fn):
    print("\n========== 勞基法條文 ==========")

    try:
        client.delete_collection("labor_law_collection")
        print("已清除舊 collection")
    except Exception:
        pass

    col = client.create_collection(
        name="labor_law_collection",
        embedding_function=emb_fn,
        metadata={"hnsw:sync_threshold": 50, "hnsw:batch_size": 50},
    )

    data = load_json("labor_law_cleaned.json")
    print(f"載入 {len(data)} 筆")

    ids, docs, metas = [], [], []
    for i, item in enumerate(data):
        article_no = str(item.get("article_no", i)).strip()
        text       = item.get("text", "").strip()
        if not text:
            continue
        ids.append(f"law_{i}")
        docs.append(text)
        metas.append({
            "source"    : item.get("source", "勞動基準法"),
            "article_no": article_no,
            "type"      : "law",
        })

    _batch_add(col, ids, docs, metas)
    print(f"OK labor_law_collection 完成，共 {col.count()} 筆")


# ==========================================
# 2. 匯入案例資料 → cases_collection
# ==========================================
def import_cases(client: chromadb.ClientAPI, emb_fn):
    print("\n========== 案例 / Q&A / 裁判書 ==========")

    try:
        client.delete_collection("cases_collection")
        print("已清除舊 collection")
    except Exception:
        pass

    col = client.create_collection(
        name="cases_collection",
        embedding_function=emb_fn,
        metadata={"hnsw:sync_threshold": 50, "hnsw:batch_size": 50},
    )

    ids, docs, metas = [], [], []
    global_idx = 0

    for filename in CASE_FILES:
        data = load_json(filename)
        print(f"  {filename}：{len(data)} 筆")

        for item in data:
            title   = item.get("title", "").strip()
            content = item.get("content", "").strip()
            if not content:
                continue

            doc = f"{title}\n{content}" if title else content
            doc = doc[:MAX_DOC_LEN]   # 截斷超長文件

            ids.append(f"case_{global_idx}")
            docs.append(doc)
            metas.append({
                "source"  : item.get("source", ""),
                "category": item.get("category", ""),
                "url"     : item.get("url", ""),
                "date"    : item.get("date", ""),
                "title"   : title,
                "type"    : item.get("source_type", "gov"),
            })
            global_idx += 1

    print(f"總計 {len(ids)} 筆，開始寫入...")
    _batch_add(col, ids, docs, metas)
    print(f"OK cases_collection 完成，共 {col.count()} 筆")


# ==========================================
# 工具：分批寫入（避免一次太多 OOM）
# ==========================================
def _batch_add(col, ids, docs, metas, batch_size=50):
    total = len(ids)
    for i in range(0, total, batch_size):
        col.add(
            ids       = ids [i: i + batch_size],
            documents = docs[i: i + batch_size],
            metadatas = metas[i: i + batch_size],
        )
        print(f"  匯入 {min(i + batch_size, total)}/{total}")
        time.sleep(0.05)

    # 強制 HNSW 索引在程式結束前同步完成
    print("  等待 HNSW 索引建立...", end=" ", flush=True)
    for attempt in range(10):
        try:
            col.query(query_texts=["加班費"], n_results=1)
            print("OK")
            return
        except Exception:
            time.sleep(2)
    print("警告：HNSW 索引驗證逾時，請重新執行匯入")


# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    print(f"ChromaDB 路徑：{DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = get_embedding_fn()

    import_law(client, emb_fn)
    import_cases(client, emb_fn)

    # 強制停止內部系統，確保 HNSW 完整 flush 到磁碟後才結束程式
    print("\n正在 flush 索引到磁碟...", end=" ", flush=True)
    try:
        client._system.stop()
        print("OK")
    except Exception as e:
        print(f"略過（{e}）")

    print("\nOK 匯入完成！請執行下一步驗證：")
    print("  python backend/verify_db.py")

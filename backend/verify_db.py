"""用獨立 process 驗證 ChromaDB 是否正常"""
import chromadb
from chromadb.utils import embedding_functions
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "law_db")

client = chromadb.PersistentClient(path=DB_PATH)
emb    = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="shibing624/text2vec-base-chinese"
)

for name in ["labor_law_collection", "cases_collection"]:
    try:
        col = client.get_collection(name, embedding_function=emb)
        r   = col.query(query_texts=["加班費"], n_results=1)
        print(f"[OK] {name}：{col.count()} 筆")
    except Exception as e:
        print(f"[FAIL] {name}：{e}")

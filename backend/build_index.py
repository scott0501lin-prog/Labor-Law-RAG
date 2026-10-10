"""
預先計算所有 embedding 並存成 numpy 檔案。
只需執行一次：python backend/build_index.py

改動重點（v2）：
  1. embed_text = 來源標籤 + chunk（≤95字，約100 tokens，低於 max_seq_length=128）
  2. display_text 與 embed_text 分開：向量化用截斷，顯示用完整原文 / 主文+鄰近chunk
  3. 長文件切塊（CHUNK_SIZE=80字，重疊12字），每塊記錄 parent_id 和 chunk_idx
  4. 檢索後在 app.py 以 parent_id 合併，取回 display_text 給 Gemini
"""

import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME         = "shibing624/text2vec-base-chinese"
CHUNK_SIZE         = 80   # 內容字元數；加上標籤共約 95字 ≈ 100 tokens
CHUNK_OVERLAP      = 12   # 重疊字元數，讓跨塊語意不斷裂
LABEL_MAX          = 15   # 來源標籤最大字元數（含【】）
MAX_CHUNKS_PER_DOC = 50   # 單篇文件 chunk 上限，防止超大文件（如 15325 字）佔滿索引

CASE_FILES = [
    "taipei_cases.json",
    "ntpc_cases.json",
    "mol_qa_cases.json",
    "mol_fint_cases.json",
    "judicial_cases.json",
    "derived_interpretations.json",
]


# ==========================================
# 工具函式
# ==========================================

def load_json(filename):
    with open(os.path.join(BASE_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


def make_chunks(text):
    """將 text 切成 CHUNK_SIZE 字、重疊 CHUNK_OVERLAP 字的片段。
    回傳 (chunks, starts)：chunks 為字串列表，starts 為各塊起始位置。"""
    chunks, starts = [], []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        starts.append(start)
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks, starts


def make_label(source, extra=""):
    """製作來源標籤，截斷到 LABEL_MAX 字。"""
    raw = f"【{source}{' ' + extra if extra else ''}】"
    return raw[:LABEL_MAX]


def extract_main_text(content):
    """
    從判決書 content 中抽出主文（到「理由」段落前）。
    使用 negative lookbehind 排除「有/無/其/此/查」前綴的「理由」，
    並要求「理由」後接「：」「:」「\\n」「壹」才算段落分隔，
    避免「此部分無理由」「本院查理由」等誤判。
    """
    m = re.search(r'(?<![有無其此查])理由[：:\n壹]', content)
    if m:
        return content[:m.start()].strip()
    return ""   # 找不到則回傳空字串，display_text 只用鄰近 chunks


# ==========================================
# 法條索引建構
# ==========================================

def build_law_chunks(laws):
    """
    每條條文切塊。
    - embed_text  = 標籤 + chunk（向量化用）
    - display_text = 完整條文原文（Gemini 顯示用；法條不做主文/理由分割）
    """
    embed_texts, metas = [], []
    for item in laws:
        source     = item.get("source", "勞動基準法")
        article_no = item.get("article_no", "")
        full_text  = item.get("text", "")
        label      = make_label(source, article_no)
        parent_id  = f"law_{article_no}"

        raw_chunks, _ = make_chunks(full_text)
        raw_chunks = raw_chunks[:MAX_CHUNKS_PER_DOC]

        for idx, chunk in enumerate(raw_chunks):
            embed_text = label + chunk
            embed_texts.append(embed_text)
            metas.append({
                "article_no"  : article_no,
                "source"      : source,
                "parent_id"   : parent_id,
                "chunk_idx"   : idx,
                "display_text": full_text,   # 完整條文
            })
    return embed_texts, metas


# ==========================================
# 案例索引建構
# ==========================================

def build_case_chunks(case_files):
    """
    每筆案例切塊。
    - embed_text  = 標籤 + chunk
    - display_text = 主文（若找到）+ 命中 chunk 及其前後各一 chunk
    """
    embed_texts, metas = [], []
    for fn in case_files:
        items = load_json(fn)
        print(f"  {fn}：{len(items)} 筆")
        for doc_idx, item in enumerate(items):
            source   = item.get("source", "")
            category = item.get("category", "")
            title    = item.get("title", "").strip()
            url      = item.get("url", "")
            content  = item.get("content", "").strip()
            if not content:
                continue

            label     = make_label(f"{source[:7]}-{category[:6]}")
            parent_id = f"{fn.replace('.json', '')}_{doc_idx}"
            main_text = extract_main_text(content)

            raw_chunks, _ = make_chunks(content)
            raw_chunks = raw_chunks[:MAX_CHUNKS_PER_DOC]

            for idx, chunk in enumerate(raw_chunks):
                embed_text = label + chunk

                # display_text = 主文 + (前一塊 + 本塊 + 後一塊)
                window = raw_chunks[max(0, idx - 1) : idx + 2]
                context = "".join(window)
                if main_text and main_text not in context:
                    display_text = main_text + "\n" + context
                else:
                    display_text = context

                embed_texts.append(embed_text)
                metas.append({
                    "source"      : source,
                    "category"    : category,
                    "title"       : title,
                    "url"         : url,
                    "parent_id"   : parent_id,
                    "chunk_idx"   : idx,
                    "display_text": display_text,
                })
    return embed_texts, metas


# ==========================================
# 編碼 & 儲存
# ==========================================

def encode_and_save(model, texts, metas, prefix):
    print(f"  計算 embedding（共 {len(texts)} 筆）...")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    np.save(os.path.join(OUT_DIR, f"{prefix}_embeddings.npy"), embeddings)
    with open(os.path.join(OUT_DIR, f"{prefix}_docs.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metas": metas}, f, ensure_ascii=False)
    print(f"  已儲存：embeddings/{prefix}_*  shape={embeddings.shape}")


# ==========================================
# 主程式
# ==========================================

if __name__ == "__main__":
    print(f"載入模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("\n[1/2] 勞基法本法 + 施行細則")
    laws = load_json("labor_law_cleaned.json") + load_json("labor_law_rules_cleaned.json")
    law_texts, law_metas = build_law_chunks(laws)
    print(f"  法條 chunk 數：{len(law_texts)}")
    encode_and_save(model, law_texts, law_metas, "law")

    print("\n[2/2] 案例 / Q&A / 裁判書")
    case_texts, case_metas = build_case_chunks(CASE_FILES)
    print(f"  案例 chunk 數：{len(case_texts)}")
    encode_and_save(model, case_texts, case_metas, "cases")

    print("\n完成！請執行 streamlit run Fronted/app.py")

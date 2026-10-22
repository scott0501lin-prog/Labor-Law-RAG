"""
預先計算所有 embedding 並存成 numpy 檔案。
只需執行一次：python backend/build_index.py

模型選擇（環境變數 EMBED_MODEL，預設 bge-m3）：
  bge-m3   → BAAI/bge-m3（1024 dim）；輸出 *_bgem3.npy / *_docs_bgem3.json
  text2vec → shibing624/text2vec-base-chinese（768 dim）；輸出 *.npy / *_docs.json（原路徑）

切塊策略（bge-m3）：
  法條：一條一塊；超過 LAW_MAX_CHARS 才再切，保留條號與來源標籤。
  案例：約 CASE_MAX_CHARS 字一塊，CASE_OVERLAP 字重疊，在句號/段落處切割。
  embed_text = display_text（不再分開存），標籤前置。

切回 text2vec：
  EMBED_MODEL=text2vec python backend/build_index.py
  再重啟 Streamlit 並設 EMBED_MODEL=text2vec（app.py 自動選對應索引檔）。
"""

import json
import os
import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 模型選擇 ─────────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3").lower()
if "text2vec" in EMBED_MODEL:
    MODEL_NAME  = "shibing624/text2vec-base-chinese"
    FILE_SUFFIX = ""
else:
    MODEL_NAME  = "BAAI/bge-m3"
    FILE_SUFFIX = "_bgem3"

# ── 切塊參數 ─────────────────────────────────────────────────────────────────
LAW_MAX_CHARS      = 500   # 法條：超過此長度才切塊
CASE_MAX_CHARS     = 400   # 案例：目標每塊字元數
CASE_OVERLAP       = 50    # 案例：重疊字元數
MAX_CHUNKS_PER_DOC = 50    # 每份文件最多切幾塊

# ── 舊版切塊參數（text2vec 路徑用）─────────────────────────────────────────
CHUNK_SIZE_LEGACY    = 80
CHUNK_OVERLAP_LEGACY = 12
LABEL_MAX            = 15

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


def smart_split(text, max_chars, overlap):
    """在句號/段落邊界切割中文文字，每塊約 max_chars 字，重疊 overlap 字。"""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # 從後往前找最近的句子邊界
            for b in ['。', '；', '！', '？', '\n']:
                pos = text.rfind(b, start + max_chars // 3, end)
                if pos != -1:
                    end = pos + 1
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


# ── 舊版工具（text2vec） ──────────────────────────────────────────────────────

def _make_chunks_legacy(text):
    chunks, starts = [], []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE_LEGACY, len(text))
        chunks.append(text[start:end])
        starts.append(start)
        if end == len(text):
            break
        start += CHUNK_SIZE_LEGACY - CHUNK_OVERLAP_LEGACY
    return chunks, starts


def _make_label(source, extra=""):
    raw = f"【{source}{' ' + extra if extra else ''}】"
    return raw[:LABEL_MAX]


def _extract_main_text(content):
    m = re.search(r'(?<![有無其此查])理由[：:\n壹]', content)
    if m:
        return content[:m.start()].strip()
    return ""


# ==========================================
# 法條索引建構
# ==========================================

def build_law_chunks_legacy(laws):
    embed_texts, metas = [], []
    for item in laws:
        source     = item.get("source", "勞動基準法")
        article_no = item.get("article_no", "")
        full_text  = item.get("text", "")
        label      = _make_label(source, article_no)
        parent_id  = f"law_{article_no}"
        raw_chunks, _ = _make_chunks_legacy(full_text)
        for idx, chunk in enumerate(raw_chunks[:MAX_CHUNKS_PER_DOC]):
            embed_texts.append(label + chunk)
            metas.append({
                "article_no": article_no, "source": source,
                "parent_id": parent_id, "chunk_idx": idx,
                "display_text": full_text,
            })
    return embed_texts, metas


def build_law_chunks_bgem3(laws):
    """一條一塊；超過 LAW_MAX_CHARS 才再切，embed_text = display_text。"""
    embed_texts, metas = [], []
    for item in laws:
        source     = item.get("source", "勞動基準法")
        article_no = item.get("article_no", "")
        full_text  = item.get("text", "")
        label      = f"【{source} {article_no}】"
        parent_id  = f"law_{article_no}"
        merged     = f"{label}\n{full_text}"
        chunks     = smart_split(merged, LAW_MAX_CHARS, overlap=50)
        for idx, chunk in enumerate(chunks[:MAX_CHUNKS_PER_DOC]):
            embed_texts.append(chunk)
            metas.append({
                "article_no"  : article_no,
                "source"      : source,
                "parent_id"   : parent_id,
                "chunk_idx"   : idx,
                "display_text": chunk,
            })
    return embed_texts, metas


# ==========================================
# 案例索引建構
# ==========================================

def build_case_chunks_legacy(case_files):
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
            label     = _make_label(f"{source[:7]}-{category[:6]}")
            parent_id = f"{fn.replace('.json', '')}_{doc_idx}"
            main_text = _extract_main_text(content)
            raw_chunks, _ = _make_chunks_legacy(content)
            for idx, chunk in enumerate(raw_chunks[:MAX_CHUNKS_PER_DOC]):
                window = raw_chunks[max(0, idx - 1):idx + 2]
                context = "".join(window)
                display_text = (main_text + "\n" + context) if (main_text and main_text not in context) else context
                embed_texts.append(label + chunk)
                metas.append({
                    "source": source, "category": category, "title": title,
                    "url": url, "parent_id": parent_id, "chunk_idx": idx,
                    "display_text": display_text,
                })
    return embed_texts, metas


def build_case_chunks_bgem3(case_files):
    """約 CASE_MAX_CHARS 字一塊，CASE_OVERLAP 字重疊，在句號/段落邊界切割。
    embed_text = display_text（合併，不再分開）。"""
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
            label     = f"【{source} {category}】"
            parent_id = f"{fn.replace('.json', '')}_{doc_idx}"
            chunks    = smart_split(content, CASE_MAX_CHARS, CASE_OVERLAP)
            for idx, chunk in enumerate(chunks[:MAX_CHUNKS_PER_DOC]):
                merged = f"{label}\n{chunk}"
                embed_texts.append(merged)
                metas.append({
                    "source"      : source,
                    "category"    : category,
                    "title"       : title,
                    "url"         : url,
                    "parent_id"   : parent_id,
                    "chunk_idx"   : idx,
                    "display_text": merged,
                })
    return embed_texts, metas


# ==========================================
# 編碼 & 儲存
# ==========================================

def encode_and_save(model, texts, metas, prefix):
    t0 = time.perf_counter()
    print(f"  計算 embedding（共 {len(texts)} 筆）...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    np.save(os.path.join(OUT_DIR, f"{prefix}_embeddings{FILE_SUFFIX}.npy"), embeddings)
    with open(os.path.join(OUT_DIR, f"{prefix}_docs{FILE_SUFFIX}.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metas": metas}, f, ensure_ascii=False)
    print(f"  已儲存：embeddings/{prefix}_*{FILE_SUFFIX}  shape={embeddings.shape}  time={elapsed:.1f}s")
    return elapsed


# ==========================================
# 主程式
# ==========================================

if __name__ == "__main__":
    import psutil
    proc = psutil.Process()

    print(f"EMBED_MODEL : {EMBED_MODEL}")
    print(f"MODEL_NAME  : {MODEL_NAME}")
    print(f"FILE_SUFFIX : '{FILE_SUFFIX}'")

    rss_before = proc.memory_info().rss / 1024 ** 2
    t_load = time.perf_counter()
    print(f"\n載入模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    load_time = time.perf_counter() - t_load
    rss_after = proc.memory_info().rss / 1024 ** 2

    sample = model.encode(["test"], normalize_embeddings=True)
    dim = sample.shape[1]
    print(f"維度：{dim} | 載入時間：{load_time:.1f}s | 記憶體增加：{rss_after - rss_before:.0f}MB")

    t_enc = time.perf_counter()
    _ = model.encode(["休息日加班費怎麼計算"], normalize_embeddings=True)
    enc_ms = (time.perf_counter() - t_enc) * 1000
    print(f"單句編碼時間：{enc_ms:.1f}ms")

    t_total = time.perf_counter()

    print(f"\n[1/2] 勞基法本法 + 施行細則")
    laws = load_json("labor_law_cleaned.json") + load_json("labor_law_rules_cleaned.json")
    if "text2vec" in EMBED_MODEL:
        law_texts, law_metas = build_law_chunks_legacy(laws)
    else:
        law_texts, law_metas = build_law_chunks_bgem3(laws)
    print(f"  法條 chunk 數：{len(law_texts)}")
    law_elapsed = encode_and_save(model, law_texts, law_metas, "law")

    print(f"\n[2/2] 案例 / Q&A / 裁判書")
    if "text2vec" in EMBED_MODEL:
        case_texts, case_metas = build_case_chunks_legacy(CASE_FILES)
    else:
        case_texts, case_metas = build_case_chunks_bgem3(CASE_FILES)
    print(f"  案例 chunk 數：{len(case_texts)}")
    case_elapsed = encode_and_save(model, case_texts, case_metas, "cases")

    total_elapsed = time.perf_counter() - t_total
    print(f"\n完成！總建索引時間：{total_elapsed:.1f}s（法條：{law_elapsed:.1f}s，案例：{case_elapsed:.1f}s）")
    print("執行：streamlit run Fronted/app.py")

"""
離線建索引：把 scraper2 資料切塊、用 BAAI/bge-m3 編碼，存成向量檔。
資料更新後才需要重跑；app 啟動時直接載入結果，不再現場編碼。

    python backend/build_index.py              # 法規 + 案例
    python backend/build_index.py --only law   # 只重建法規（約 1 分鐘）

輸出（backend/embeddings/）：
    law_embeddings_bgem3.npy   / law_docs_bgem3.json
    cases_embeddings_bgem3.npy / cases_docs_bgem3.json
向量存成 float16（檔案減半；app 載入時轉回 float32），docs JSON 含
texts、metas 與 corpus_version（內容雜湊，app 用來判斷 Chroma 是否要重灌）。

切塊策略：
    法規：一條一塊；超過 LAW_MAX_CHARS 才切，每塊都帶【法規 條號】標頭。
          display_text 存整條條文，送進 Prompt 的永遠是完整條文。
    問答／函釋：約 CASE_MAX_CHARS 字一塊、重疊 CASE_OVERLAP 字，在句號處切。
    判決：依章節（主文、爭點、本院之判斷…）分段後再切塊，每塊帶
          【判決名｜案由｜章節】標頭；不截斷、不設每篇上限。
    parent_id 帶法規名稱，避免勞基法與施行細則同號條文被當成同一份。

舊的 text2vec 索引（law_embeddings.npy 等，無後綴）保留作為改善前的對照基準。
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
S2_DIR   = os.path.join(BASE_DIR, "scraper2", "data")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "embeddings")

MODEL_NAME  = "BAAI/bge-m3"
FILE_SUFFIX = "_bgem3"

LAW_MAX_CHARS  = 500
CASE_MAX_CHARS = 400
CASE_OVERLAP   = 50
MAX_SEQ_LEN    = 1024  # tokens; a few FAQ chunks with very long titles exceed 512

LAW_FILES = [
    "labor_standards_act.json",        # 勞動基準法
    "labor_standards_rules.json",      # 勞動基準法施行細則
    "labor_pension_act.json",          # 勞工退休金條例
    "occupational_accident_act.json",  # 勞工職業災害保險及保護法
    "minimum_wage_act.json",           # 最低工資法
    "leave_rules.json",                # 勞工請假規則
]
FAQ_FILES      = ["taipei_faq_cases.json", "mol_faq_cases.json"]
# 台北市 FAQ 中被選為測試題的題目不進索引，否則測試時系統會查到原題答案。
# 測試集由 eval/build_taipei_testset.py 從 taipei_faq_cases.json 產生。
TESTSET_FILE   = os.path.join(os.path.dirname(BASE_DIR), "eval", "taipei_faq_testset.json")
TESTSET_FAQ    = "taipei_faq_cases.json"
FINT_FILE      = "mol_fint_cases.json"
JUDICIAL_FILE  = "judicial_cases.json"
EXTRA_CASE_FILES = ["derived_interpretations.json"]   # backend/data，手動整理


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def smart_split(text, max_chars, overlap):
    """在句號／段落邊界切割，每塊約 max_chars 字，重疊 overlap 字。"""
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


# ── 法規 ─────────────────────────────────────────────────────────────────────
def build_law_chunks():
    texts, metas = [], []
    for fn in LAW_FILES:
        items = load_json(os.path.join(S2_DIR, fn))
        print(f"  {fn}: {len(items)} 條")
        for item in items:
            source, art = item["source"], item["article_no"]
            full = item["text"].strip()
            header = f"【{source} {art}】"
            body = full[len(header):].lstrip() if full.startswith(header) else full
            for idx, chunk in enumerate(smart_split(body, LAW_MAX_CHARS, 50)):
                texts.append(f"{header}\n{chunk}")
                metas.append({
                    "data_type": "law", "source": source, "article_no": art,
                    "chapter": item.get("chapter", ""), "law_date": item.get("law_date", ""),
                    "parent_id": f"{source}|{art}", "chunk_idx": idx,
                    "display_text": full,
                })
    return texts, metas


# ── 案例 ─────────────────────────────────────────────────────────────────────
def _case_meta(item, parent_id, idx, display, **extra):
    m = {
        "data_type": "case", "source": item.get("source", ""),
        "category": item.get("category", ""), "title": (item.get("title") or "").strip(),
        "url": item.get("url", ""), "date": item.get("date", ""),
        "parent_id": parent_id, "chunk_idx": idx, "display_text": display,
    }
    m.update(extra)
    return m


def load_testset_questions():
    return {x["question"].strip() for x in load_json(TESTSET_FILE)}


def build_faq_chunks(texts, metas):
    held_out = load_testset_questions()
    for fn in FAQ_FILES:
        items = load_json(os.path.join(S2_DIR, fn))
        skipped = 0
        for i, item in enumerate(items):
            title, content = item["title"].strip(), item["content"].strip()
            if fn == TESTSET_FAQ and title in held_out:
                skipped += 1
                continue
            if not content:
                continue
            header = f"【{item['source']} {item.get('category', '')}】問：{title}"
            display = f"問：{title}\n答：{content}"
            for idx, chunk in enumerate(smart_split(content, CASE_MAX_CHARS, CASE_OVERLAP)):
                texts.append(f"{header}\n答：{chunk}")
                metas.append(_case_meta(item, f"{fn[:-5]}_{i}", idx, display))
        print(f"  {fn}: {len(items)} 題" + (f"（排除測試集 {skipped} 題）" if skipped else ""))
        if fn == TESTSET_FAQ and skipped != len(held_out):
            raise SystemExit(f"測試集有 {len(held_out)} 題，但只在 {fn} 找到 {skipped} 題；"
                             "請重跑 eval/build_taipei_testset.py")


def build_fint_chunks(texts, metas):
    items = load_json(os.path.join(S2_DIR, FINT_FILE))
    print(f"  {FINT_FILE}: {len(items)} 則")
    for i, item in enumerate(items):
        content = item["content"].strip()
        if not content:
            continue
        laws = "、".join(item.get("from_articles") or item.get("related_laws") or [])
        header = f"【勞動部函釋 {item['title']}】{item.get('summary', '')}"
        for idx, chunk in enumerate(smart_split(content, CASE_MAX_CHARS, CASE_OVERLAP)):
            texts.append(f"{header}\n{chunk}")
            metas.append(_case_meta(item, f"mol_fint_{i}", idx, content, cited_laws=laws))


def build_judicial_chunks(texts, metas):
    items = load_json(os.path.join(S2_DIR, JUDICIAL_FILE))
    print(f"  {JUDICIAL_FILE}: {len(items)} 篇")
    for i, item in enumerate(items):
        # Consecutive entries with the same section label form one segment.
        segments = []
        for s in item.get("sections", []):
            if segments and segments[-1][0] == s["section"]:
                segments[-1][1].append(s["text"])
            else:
                segments.append((s["section"], [s["text"]]))
        extra = {
            "case_title": item.get("case_title", ""), "court": item.get("court", ""),
            "result": item.get("result", ""), "cited_laws": "、".join(item.get("cited_laws") or []),
        }
        idx = 0
        for section, paras in segments:
            header = f"【判決 {item['title']}｜{item.get('case_title', '')}｜{section}】"
            for chunk in smart_split("\n".join(paras), CASE_MAX_CHARS, CASE_OVERLAP):
                texts.append(f"{header}\n{chunk}")
                metas.append(_case_meta(item, f"judicial_{i}", idx,
                                        f"{item.get('brief', '')}\n【{section}】{chunk}",
                                        section=section, **extra))
                idx += 1


def build_extra_chunks(texts, metas):
    for fn in EXTRA_CASE_FILES:
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            continue
        items = load_json(path)
        print(f"  {fn}: {len(items)} 筆")
        for i, item in enumerate(items):
            content = (item.get("content") or "").strip()
            if not content:
                continue
            header = f"【{item.get('source', '')} {item.get('category', '')}】{item.get('title', '')}"
            for idx, chunk in enumerate(smart_split(content, CASE_MAX_CHARS, CASE_OVERLAP)):
                texts.append(f"{header}\n{chunk}")
                metas.append(_case_meta(item, f"{fn[:-5]}_{i}", idx, content))


def build_case_chunks():
    texts, metas = [], []
    build_faq_chunks(texts, metas)
    build_fint_chunks(texts, metas)
    build_judicial_chunks(texts, metas)
    build_extra_chunks(texts, metas)
    return texts, metas


# ── 編碼與儲存 ───────────────────────────────────────────────────────────────
def corpus_version(texts, metas) -> str:
    h = hashlib.sha1()
    h.update(MODEL_NAME.encode())
    h.update(json.dumps([texts, metas], ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()[:16]


def encode_and_save(model, texts, metas, prefix):
    t0 = time.perf_counter()
    print(f"  編碼 {len(texts)} 塊…", flush=True)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    np.save(os.path.join(OUT_DIR, f"{prefix}_embeddings{FILE_SUFFIX}.npy"), emb.astype(np.float16))
    with open(os.path.join(OUT_DIR, f"{prefix}_docs{FILE_SUFFIX}.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "corpus_version": corpus_version(texts, metas),
                   "texts": texts, "metas": metas}, f, ensure_ascii=False)
    print(f"  已儲存 {prefix}: shape={emb.shape}  {time.perf_counter() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["law", "cases"])
    ap.add_argument("--dry-run", action="store_true", help="只切塊、印統計，不編碼")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = []
    if args.only in (None, "law"):
        print("[法規]")
        jobs.append(("law", *build_law_chunks()))
    if args.only in (None, "cases"):
        print("[案例]")
        jobs.append(("cases", *build_case_chunks()))
    for prefix, texts, _ in jobs:
        lens = sorted(len(t) for t in texts)
        print(f"  {prefix}: {len(texts)} 塊，長度中位數 {lens[len(lens) // 2]}，最長 {lens[-1]}")
    if args.dry_run:
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LEN
    for prefix, texts, metas in jobs:
        encode_and_save(model, texts, metas, prefix)
    print("完成。")


if __name__ == "__main__":
    main()

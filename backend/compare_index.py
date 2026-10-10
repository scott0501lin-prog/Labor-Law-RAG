"""
比較舊索引（embeddings_v1_backup）與新索引（embeddings）的檢索結果。
用法：
  python backend/compare_index.py              # 跑全部 20 題測試題庫
  python backend/compare_index.py "加班費怎麼算"  # 單題
"""

import os, sys, json
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "shibing624/text2vec-base-chinese"
TOP_K = 5
TOP_K_CHUNKS = 20   # 合併前候選 chunk 數

# 20 題測試題庫（來自成果評估測試包）
TEST_QUESTIONS = [
    # 中文-一般
    "加班費要怎麼算？",
    "我做滿一年被資遣，可以領多少資遣費？",
    "公司要資遣我，需要提前多久通知？",
    "我在工作時受傷，公司要怎麼補償我？",
    "特別休假是怎麼計算天數的？",
    # 非中文
    "How much is overtime pay if I work on a rest day?",
    "Nếu công ty sa thải tôi trái luật, tôi phải làm gì?",
    "Berapa uang pesangon yang saya dapatkan jika di-PHK?",
    "Ano ang gagawin ko kung nasaktan ako sa trabaho?",
    "특별휴가는 며칠 받을 수 있나요？",
    # 長句/細節多
    "我在一家小公司做了兩年多，上個月主管突然跟我說因為業績不好要精簡人力，叫我下個月就不用來了，也沒有給我加班費，我想知道這樣算不算違法解僱，還有可以拿到多少資遣費",
    "我是做輪班的，這個月有兩次被排在國定假日上班，主管說會補休但一直沒排給我，我想知道國定假日上班沒有補休到底該怎麼辦，工資有沒有算加倍",
    "我上禮拜在工作的時候不小心被機台夾到手，現在還在醫院治療沒辦法上班，公司說治療期間沒有上班就不會有薪水，這樣對嗎，醫藥費是不是也要公司出",
    "我懷孕八個月了，醫生說要安胎休息，但主管一直打電話叫我回去上班，說請假太久要扣全勤獎金，這樣合法嗎",
    "我做滿三年準備要離職，但主管說特休沒休完不能折現要作廢，我想知道特休沒休完到底該怎麼處理",
    # 加班關鍵字觸發
    "假日加班費怎麼算？",
    "我休息日被叫去加班，工資怎麼算？",
    "平日加班超過兩小時的加班費倍率是多少？",
    "國定假日加班算不算加班費？",
    "加班費可以用補休代替嗎？",
]


def load_index(emb_dir, model):
    def _load(prefix):
        emb = np.load(os.path.join(emb_dir, f"{prefix}_embeddings.npy"))
        with open(os.path.join(emb_dir, f"{prefix}_docs.json"), encoding="utf-8") as f:
            data = json.load(f)
        return emb, data["texts"], data["metas"]
    law_emb,  law_t,  law_m  = _load("law")
    case_emb, case_t, case_m = _load("cases")
    return law_emb, law_t, law_m, case_emb, case_t, case_m


def search_with_merge(model, query, embeddings, texts, metas, top_k=TOP_K):
    q_emb  = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:TOP_K_CHUNKS]

    seen = {}
    for i in top_idx:
        pid = metas[i].get("parent_id", str(i))
        if pid not in seen or scores[i] > seen[pid][1]:
            seen[pid] = (i, float(scores[i]))

    merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(texts[i], metas[i], score) for i, score in merged]


def fmt_result(meta, score, is_law):
    if is_law:
        return f"  {meta.get('source','')} {meta.get('article_no','')}  score={score:.3f}"
    else:
        title = meta.get("title", "")[:30]
        cat   = meta.get("category", "")[:15]
        return f"  [{cat}] {title}  score={score:.3f}"


def compare_one(q, old_law, old_lt, old_lm, old_ce, old_ct, old_cm,
                        new_law, new_lt, new_lm, new_ce, new_ct, new_cm, model):
    old_l = search_with_merge(model, q, old_law, old_lt, old_lm)
    old_c = search_with_merge(model, q, old_ce,  old_ct,  old_cm)
    new_l = search_with_merge(model, q, new_law, new_lt, new_lm)
    new_c = search_with_merge(model, q, new_ce,  new_ct,  new_cm)

    print(f"\n問：{q[:60]}")
    print(f"{'舊索引 法條':<40}  {'新索引 法條'}")
    old_law_ids = set()
    new_law_ids = set()
    for (_, om, os_), (_, nm, ns_) in zip(old_l, new_l):
        old_id = f"{om.get('source','')} {om.get('article_no','')}".strip()
        new_id = f"{nm.get('source','')} {nm.get('article_no','')}".strip()
        old_law_ids.add(old_id)
        new_law_ids.add(new_id)
        mark = "✓" if old_id == new_id else " "
        print(f"  {old_id:<35} {mark}  {new_id}")

    gained = new_law_ids - old_law_ids
    lost   = old_law_ids - new_law_ids
    if gained: print(f"  ++ 新增命中: {gained}")
    if lost:   print(f"  -- 消失命中: {lost}")

    print(f"{'舊索引 案例':<40}  {'新索引 案例'}")
    for (_, om, os_), (_, nm, ns_) in zip(old_c, new_c):
        old_cat = om.get("category", "")[:20]
        new_cat = nm.get("category", "")[:20]
        print(f"  [{old_cat}]  →  [{new_cat}]  score {os_:.3f}→{ns_:.3f}")


def main():
    old_dir = os.path.join(BASE_DIR, "embeddings_v1_backup")
    new_dir = os.path.join(BASE_DIR, "embeddings")

    if not os.path.isdir(old_dir):
        print(f"找不到舊索引備份：{old_dir}")
        print("請先備份：xcopy /E /I backend\\embeddings backend\\embeddings_v1_backup")
        sys.exit(1)

    print(f"載入模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("載入舊索引...")
    old_law, old_lt, old_lm, old_ce, old_ct, old_cm = load_index(old_dir, model)
    print(f"  舊：law={old_law.shape}  case={old_ce.shape}")

    print("載入新索引...")
    new_law, new_lt, new_lm, new_ce, new_ct, new_cm = load_index(new_dir, model)
    print(f"  新：law={new_law.shape}  case={new_ce.shape}")

    questions = sys.argv[1:] if len(sys.argv) > 1 else TEST_QUESTIONS

    for q in questions:
        compare_one(q,
                    old_law, old_lt, old_lm, old_ce, old_ct, old_cm,
                    new_law, new_lt, new_lm, new_ce, new_ct, new_cm,
                    model)

    print("\n" + "=" * 60)
    print(f"共比較 {len(questions)} 題")


if __name__ == "__main__":
    main()

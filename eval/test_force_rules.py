"""
eval/test_force_rules.py  ─  Force-include regression suite
=====================================================================
讀取 eval/force_rule_cases.csv，對每一句呼叫真實的 force-include 邏輯
（不呼叫 Gemini 或向量檢索），輸出每句的預期／實際／通過或失敗，
最後印出總通過率，並把結果存成 CSV（檔名含日期 + git commit hash）。

用法（從專案根目錄執行）：
    python eval/test_force_rules.py
    python eval/test_force_rules.py --csv eval/force_rule_cases.csv
"""

import csv
import datetime
import io
import json
import os
import subprocess
import sys
from argparse import ArgumentParser

# ── stdout UTF-8（Windows CP950 保護）─────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 讓 Fronted/force_rules.py 可被 import（不觸發 Streamlit init）──────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "Fronted"))

from force_rules import (  # noqa: E402
    FORCE_ARTICLE_RULES,
    MAX_FORCED_LAW_RESULTS,
    _force_include_articles,
    run_force_include,
)


# ── 載入法條索引（供 run_force_include 使用）─────────────────────────────
def _load_law_docs():
    path = os.path.join(_project_root, "backend", "embeddings", "law_docs.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw["texts"], raw["metas"]
    # 舊格式（list of dicts）
    texts = [d["text"] for d in raw]
    metas = [d["meta"] for d in raw]
    return texts, metas


# ── git commit hash ───────────────────────────────────────────────────────
def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_project_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ── 比較預期與實際條號 ────────────────────────────────────────────────────
def _parse_expected(raw: str) -> set[str]:
    """空字串 → None（未填），'（無）' → 空集合，其他 → 以 | 分割的集合。"""
    raw = raw.strip()
    if not raw:
        return None  # type: ignore[return-value]
    if raw in {"（無）", "(無)", "none", "None", ""}:
        return set()
    return {a.strip() for a in raw.split("|") if a.strip()}


def _actual_articles(forced) -> set[str]:
    return {m.get("article_no", "") for _, m, _ in forced if m.get("article_no")}


# ── 主流程 ────────────────────────────────────────────────────────────────
def main(csv_path: str):
    law_texts, law_metas = _load_law_docs()

    with open(csv_path, encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))

    results = []
    pass_count = checked = 0

    print("=" * 70)
    print(f"  Force-include eval  ─  {len(cases)} 句")
    print("=" * 70)

    for row in cases:
        q        = row["question"].strip()
        expected = _parse_expected(row.get("expected_articles", ""))
        group    = row.get("group", "").strip()
        kind     = row.get("type", "").strip()

        forced  = run_force_include(q, law_texts, law_metas)
        actual  = _actual_articles(forced)

        if expected is None:
            # expected_articles 尚未填寫 → 只顯示觸發結果，不計分
            status = "─"
            passed_flag = None
            extra = missing = set()
        else:
            extra   = actual - expected
            missing = expected - actual
            passed_flag = (not extra) and (not missing)
            status  = "PASS" if passed_flag else "FAIL"
            checked += 1
            if passed_flag:
                pass_count += 1

        # ── 印出 ──────────────────────────────────────────────────────────
        actual_str  = " | ".join(sorted(actual))  if actual  else "（無）"
        exp_str     = " | ".join(sorted(expected)) if expected is not None else "（未填）"
        print(f"\n[{status}] [{kind}] {q}")
        print(f"  group    : {group}")
        print(f"  expected : {exp_str}")
        print(f"  actual   : {actual_str}")
        if extra:
            print(f"  多出的條號: {' | '.join(sorted(extra))}")
        if missing:
            print(f"  缺少的條號: {' | '.join(sorted(missing))}")

        results.append({
            "question":          q,
            "group":             group,
            "type":              kind,
            "expected_articles": row.get("expected_articles", ""),
            "actual_articles":   " | ".join(sorted(actual)),
            "status":            status,
            "extra_articles":    " | ".join(sorted(extra))   if extra   else "",
            "missing_articles":  " | ".join(sorted(missing)) if missing else "",
        })

    # ── 總結 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if checked:
        pct = pass_count / checked * 100
        print(f"  結果：{pass_count} / {checked} 通過（{pct:.1f}%）"
              f"  [共 {len(cases)} 句，{len(cases)-checked} 句尚未填寫預期條號]")
    else:
        print(f"  共 {len(cases)} 句，所有預期條號尚未填寫 — 僅顯示觸發結果。")
    print("=" * 70)

    # ── 儲存結果 CSV ──────────────────────────────────────────────────────
    today    = datetime.date.today().strftime("%Y%m%d")
    git_hash = _git_hash()
    out_dir  = os.path.dirname(os.path.abspath(csv_path))
    out_path = os.path.join(out_dir, f"results_{today}_{git_hash}.csv")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  結果已儲存至：{out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Force-include regression test")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "force_rule_cases.csv"),
        help="測試案例 CSV 路徑（預設：eval/force_rule_cases.csv）",
    )
    args = parser.parse_args()
    main(args.csv)

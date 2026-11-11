"""
eval/benchmark_models.py — 比較 text2vec-base-chinese 與 BAAI/bge-m3 的資源使用

前置需求：
    pip install psutil

BAAI/bge-m3 首次執行時會自動下載模型（約 2.3 GB）。
在 CPU-only 環境，bge-m3 全量編碼可能需要 10–20 分鐘；
加 --skip-corpus 可跳過全量編碼（只量載入與單句速度）。

用法（從專案根目錄）：
    python eval/benchmark_models.py
    python eval/benchmark_models.py --skip-corpus
"""

import argparse
import io
import json
import os
import sys
import time

# ── stdout UTF-8（Windows CP950 保護）─────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 套件檢查 ──────────────────────────────────────────────────────────────
try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    sys.exit(
        "[錯誤] 缺少 psutil。請先安裝：\n"
        "    pip install psutil\n"
        "安裝後重新執行此腳本。"
    )

# ── 資料路徑（不修改任何現有檔案，只讀取）────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EMB_DIR = os.path.join(_PROJECT_ROOT, "backend", "embeddings")

TEST_QUERY = "休息日加班費怎麼計算"
ENCODE_REPEATS = 20


def _mem_rss_mb() -> float:
    """回傳目前 process RSS（實體記憶體）使用量，單位 MB。"""
    return _PROC.memory_info().rss / 1024 ** 2


def load_all_texts() -> list[str]:
    """從現有索引讀取所有文字（不覆蓋，read-only）。"""
    texts = []
    for prefix in ["law", "cases"]:
        path = os.path.join(_EMB_DIR, f"{prefix}_docs.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        chunk = d.get("texts", d) if isinstance(d, dict) else d
        texts.extend(chunk)
    return texts


def benchmark_model(model_name: str, all_texts: list[str], skip_corpus: bool) -> dict:
    """載入模型並量測各項指標，回傳結果 dict。"""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")

    # ── 載入時間與記憶體 ──────────────────────────────────────────────────
    mem_before = _mem_rss_mb()
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_time = time.perf_counter() - t0
    mem_after = _mem_rss_mb()
    mem_delta = mem_after - mem_before
    print(f"  載入完成：{load_time:.2f} 秒，記憶體增加 {mem_delta:.1f} MB")

    # ── 向量維度 ──────────────────────────────────────────────────────────
    sample_emb = model.encode([TEST_QUERY], normalize_embeddings=True)
    dim = sample_emb.shape[1]
    print(f"  向量維度：{dim}")

    # ── 單句編碼（warmup + 20 次平均）────────────────────────────────────
    model.encode([TEST_QUERY], normalize_embeddings=True)  # warmup
    single_times = []
    for _ in range(ENCODE_REPEATS):
        t0 = time.perf_counter()
        model.encode([TEST_QUERY], normalize_embeddings=True)
        single_times.append(time.perf_counter() - t0)
    single_avg_ms = sum(single_times) / len(single_times) * 1000
    print(f"  單句編碼平均（×{ENCODE_REPEATS}）：{single_avg_ms:.1f} ms")

    # ── 全量編碼 ──────────────────────────────────────────────────────────
    corpus_time = None
    if skip_corpus:
        print(f"  全量編碼：已跳過（--skip-corpus）")
    else:
        n = len(all_texts)
        print(f"  全量編碼中（{n} 筆，batch_size=64）…")
        t0 = time.perf_counter()
        model.encode(all_texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=True)
        corpus_time = time.perf_counter() - t0
        print(f"  全量編碼完成：{corpus_time:.1f} 秒")

    return {
        "model":          model_name.split("/")[-1],   # 短名稱，供表格顯示
        "mem_delta_mb":   mem_delta,
        "load_time_s":    load_time,
        "single_avg_ms":  single_avg_ms,
        "corpus_time_s":  corpus_time,
        "dim":            dim,
    }


def print_table(results: list[dict], n_texts: int, skip_corpus: bool) -> None:
    """印出對比表格。"""
    col_labels = [r["model"] for r in results]
    rows = [
        ("載入後記憶體增加 (MB)",
         [f"{r['mem_delta_mb']:.1f}" for r in results]),
        ("載入時間 (秒)",
         [f"{r['load_time_s']:.2f}" for r in results]),
        (f"單句編碼平均 (ms)  ×{ENCODE_REPEATS}",
         [f"{r['single_avg_ms']:.1f}" for r in results]),
        (f"全量編碼  ({n_texts} 筆) (秒)",
         ["已跳過" if r["corpus_time_s"] is None else f"{r['corpus_time_s']:.1f}"
          for r in results]),
        ("向量維度",
         [str(r["dim"]) for r in results]),
    ]

    w0 = max(len(rows[i][0]) for i in range(len(rows))) + 2
    wc = max(max(len(v) for v in r[1]) for r in rows)
    wc = max(wc, max(len(c) for c in col_labels)) + 2

    sep = "+" + f"{'-'*(w0+2)}+" + (f"{'-'*(wc+2)}+" * len(results))

    def _row(label, vals):
        cells = f"| {label:<{w0}} |"
        for v in vals:
            cells += f" {v:^{wc}} |"
        return cells

    print("\n\n" + "=" * (w0 + (wc + 3) * len(results) + 4))
    print("  量測結果對比")
    print("=" * (w0 + (wc + 3) * len(results) + 4))
    print(sep)
    print(_row("指標", col_labels))
    print(sep)
    for label, vals in rows:
        print(_row(label, vals))
    print(sep)

    if skip_corpus:
        print("\n  ※ 加 --skip-corpus 旗標跳過了全量編碼。")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark text2vec-base-chinese vs BAAI/bge-m3"
    )
    parser.add_argument(
        "--skip-corpus", action="store_true",
        help="跳過全量 6000+ 筆編碼（CPU 上 bge-m3 可能需 10–20 分鐘）"
    )
    args = parser.parse_args()

    print("讀取現有切塊資料…")
    all_texts = load_all_texts()
    print(f"  法條 + 案例合計：{len(all_texts)} 筆")

    models = [
        "shibing624/text2vec-base-chinese",
        "BAAI/bge-m3",
    ]

    results = []
    for name in models:
        r = benchmark_model(name, all_texts, args.skip_corpus)
        results.append(r)

    print_table(results, len(all_texts), args.skip_corpus)


if __name__ == "__main__":
    main()

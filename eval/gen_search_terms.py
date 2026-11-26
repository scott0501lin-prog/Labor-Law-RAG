"""
Run the app's intent prompt over the test-set questions and save the
search_terms it returns, so eval_retrieval.py can score the "+terms" system
without calling Claude.

    python eval/gen_search_terms.py              # all questions, 5 runs each
    python eval/gen_search_terms.py --runs 3
    python eval/gen_search_terms.py --missing    # only questions not cached yet,
                                                 # same run count as the cache

Input:  the test sets listed in eval_retrieval.TESTSETS
Output: eval/search_terms.json  {question: [[term, ...] per run]}
Rerun (without --missing) after changing INTENT_ANALYSIS_PROMPT
(Fronted/intent_config.py).

No longer runnable as is: the app dropped search_terms from the intent
prompt (they made the cited articles differ on every ask for +2 points of
Hit@5). eval/search_terms.json keeps the 10 runs from the last prompt that
produced them (commit 70d09c7); check that commit out to regenerate.

Several runs because the terms are sampled (the API takes no temperature on
current models): a single run's Hit@5 ranged 72%-80% on the same prompt, so
eval_retrieval.py reports the mean and range over runs.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Fronted"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from intent_config import INTENT_ANALYSIS_PROMPT  # noqa: E402
from eval_retrieval import TESTSETS, load_testset  # noqa: E402

MODEL = "claude-haiku-5-5"  # same as app.CLAUDE_MODEL
OUT = os.path.join(ROOT, "eval", "search_terms.json")


def main():
    if '"search_terms"' not in INTENT_ANALYSIS_PROMPT:
        sys.exit("INTENT_ANALYSIS_PROMPT no longer asks for search_terms (removed from the app); "
                 "eval/search_terms.json holds the cached runs from commit 70d09c7.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--missing", action="store_true",
                    help="keep the cache; generate only questions it lacks")
    args = ap.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    client = anthropic.Anthropic()

    qs = list(dict.fromkeys(q for fn in TESTSETS.values() for q, _ in load_testset(fn)))
    cache = {}
    if args.missing and os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            cache = json.load(f)
        n_runs = max((len(v) for v in cache.values()), default=args.runs)
        qs = [q for q in qs if q not in cache]
        if not qs:
            print("nothing missing")
            return
    else:
        n_runs = args.runs
    print(f"{len(qs)} questions x {n_runs} runs")

    def terms(q):
        prompt = INTENT_ANALYSIS_PROMPT.format(history="（無）", user_message=q, ui_lang="繁體中文", clarify_allowed="可以")
        for _ in range(3):
            try:
                r = client.messages.create(model=MODEL, max_tokens=1024, thinking={"type": "disabled"},
                                           messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if b.type == "text")
                return json.loads(re.search(r"\{.*\}", text, re.S).group()).get("search_terms") or []
            except Exception as e:  # noqa: BLE001
                err = e
        print(f"failed: {q[:30]} ({err})")
        return []

    runs = []
    with ThreadPoolExecutor(8) as ex:
        for r in range(n_runs):
            runs.append(list(ex.map(terms, qs)))
            print(f"run {r + 1}/{n_runs} done")
    cache.update({q: [run[i] for run in runs] for i, q in enumerate(qs)})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT} ({len(cache)} questions)")


if __name__ == "__main__":
    main()

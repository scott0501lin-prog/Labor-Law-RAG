"""
Run the app's intent prompt over the 台北 FAQ test questions and save the
search_terms it returns, so eval_retrieval.py can score the "+terms" system
without calling Claude.

    python eval/gen_search_terms.py      # ~93 short Haiku calls

Output: eval/taipei_search_terms.json  {question: [term, ...]}
Rerun after changing INTENT_ANALYSIS_PROMPT (Fronted/intent_config.py).
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Fronted"))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from intent_config import INTENT_ANALYSIS_PROMPT  # noqa: E402

MODEL = "claude-haiku-5-5"  # same as app.CLAUDE_MODEL
OUT = os.path.join(ROOT, "eval", "taipei_search_terms.json")


def main():
    load_dotenv(os.path.join(ROOT, ".env"))
    client = anthropic.Anthropic()
    with open(os.path.join(ROOT, "eval", "taipei_faq_testset.json"), encoding="utf-8") as f:
        qs = [x["question"] for x in json.load(f)]

    def terms(q):
        prompt = INTENT_ANALYSIS_PROMPT.format(known_info_json="{}", user_message=q, ui_lang="繁體中文")
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

    with ThreadPoolExecutor(8) as ex:
        out = dict(zip(qs, ex.map(terms, qs)))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT} ({len(out)} questions)")


if __name__ == "__main__":
    main()

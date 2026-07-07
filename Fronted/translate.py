"""
translate.py
使用 Gemini 將文字翻譯成指定語言。
"""

import os
from dotenv import load_dotenv

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_base_dir, ".env"))

LANGUAGES = {
    "繁體中文": "Traditional Chinese",
    "English":  "English",
    "日本語":   "Japanese",
    "한국어":   "Korean",
}


def translate(text: str, target_lang: str, api_key: str = "") -> str:
    """
    把 text 翻譯成 target_lang（使用 LANGUAGES 的 key）。
    回傳翻譯後的文字；失敗時回傳帶錯誤訊息的字串。
    """
    if not text or not text.strip():
        return ""

    lang_name = LANGUAGES.get(target_lang)
    if not lang_name:
        return text

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        return "🚨 找不到 GEMINI_API_KEY"

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)

        model  = genai.GenerativeModel(model_name="gemini-2.5-flash")
        prompt = (
            f"請將以下文字翻譯成 {lang_name}。\n"
            "要求：\n"
            "- 保留所有 Markdown 格式（標題、粗體、引用區塊等）\n"
            "- 法條編號（如「第 14 條」）保持原樣，不要翻譯條號本身\n"
            "- 直接輸出翻譯結果，不要加任何前言或解釋\n\n"
            f"{text}"
        )
        resp = model.generate_content(prompt)
        return resp.text

    except Exception as e:
        return f"🚨 翻譯失敗：{e}"

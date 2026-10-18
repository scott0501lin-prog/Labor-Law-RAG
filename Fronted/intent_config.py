"""
Configurable parameters and prompts for intent detection + follow-up system.
Edit this file to tune triggers, required fields, and Gemini prompts
without touching app.py.
"""
import os

# ── Feature toggle ────────────────────────────────────────────────────────────
# Set INTENT_DETECTION=false in .env or Streamlit Secrets to revert to the
# original single-call RAG flow.
ENABLE_INTENT_DETECTION = os.getenv("INTENT_DETECTION", "true").lower() == "true"

# ── Follow-up limits ──────────────────────────────────────────────────────────
MAX_FOLLOWUP_ROUNDS = 2

# Skip condition: reply length <= SKIP_MAX_LEN AND contains a trigger keyword.
# Replies longer than SKIP_MAX_LEN are never treated as a skip request even if
# they happen to contain a trigger word (e.g. "我不知道休息日加班費怎麼算").
SKIP_FOLLOWUP_TRIGGERS = ["不知道", "不清楚", "直接回答", "算了", "隨便", "先不說", "跳過", "skip"]
SKIP_MAX_LEN = 10

# ── Required fields for overtime_pay scenario ─────────────────────────────────
# Missing any of these triggers a follow-up question.
# compensation_type is OPTIONAL — never prompt for it.
OVERTIME_PAY_REQUIRED_FIELDS = ["day_type", "hours", "wage_amount"]

# ── Human-readable labels for missing-field disclaimer ────────────────────────
MISSING_FIELDS_LABELS = {
    "day_type":          "加班日類型（平日／休息日／例假日／國定假日）",
    "hours":             "加班時數",
    "wage_type":         "薪資類型（月薪／時薪）",
    "wage_amount":       "薪資金額",
    "compensation_type": "給付方式（加班費／補休）",
}

# ── Prompts ───────────────────────────────────────────────────────────────────
# Placeholders (filled at runtime):
#   {user_message}     — the raw user input
#   {known_info_json}  — JSON of fields collected so far (empty dict on first call)
#   {ui_lang}          — interface language (e.g. "繁體中文", "English")

INTENT_ANALYSIS_PROMPT = """\
你是一個勞動法諮詢助理的前置分析模組。分析使用者訊息並以 JSON 格式回傳結果。
除 JSON 本身外，請勿輸出任何其他文字（包括 ```json 標記、說明文字等）。

## 目前已知資訊（空 {{}} 表示首次提問）
{known_info_json}

## 使用者訊息
{user_message}

## 介面語言（follow_up 欄位須使用此語言）
{ui_lang}

## 回傳格式（嚴格 JSON）
{{
  "translated_query": "若原文非繁體中文請翻譯，否則原樣輸出",
  "intent": "solve_problem 或 find_cases 或 law_lookup 或 out_of_scope",
  "scenario": "overtime_pay 或 other",
  "known_info": {{
    "day_type": "平日 或 休息日 或 例假日 或 國定假日 或 null",
    "hours": 數字或 null,
    "wage_type": "月薪 或 時薪 或 null",
    "wage_amount": 數字或 null,
    "compensation_type": "加班費 或 補休 或 null"
  }},
  "missing_fields": ["缺少的必填欄位名稱（僅在 overtime_pay + solve_problem 時填寫）"],
  "follow_up": "追問句（使用 {ui_lang}，若無需追問則為空字串）"
}}

## 說明
- intent 定義：solve_problem = 有具體勞動問題要解決；find_cases = 想找判決案例；
  law_lookup = 查法條內容；out_of_scope = 與勞動法無關
- scenario overtime_pay：問題涉及加班費計算或補休換算
- known_info：從訊息中提取已知值，未知設 null
- missing_fields 必填欄位：day_type、hours、wage_amount（compensation_type 不追問）
- follow_up：一次最多問兩個問題，語氣親切；資訊已足夠或非 overtime_pay+solve_problem 則輸出空字串
"""

OVERTIME_EXTRACTION_PROMPT = """\
你是一個欄位提取模組。使用者剛剛回覆了一個關於加班費的追問。
請從使用者的回覆中提取欄位值，並判斷這是否是一個全新的問題（與加班費無關）。
除 JSON 本身外，請勿輸出任何其他文字（包括 ```json 標記、說明文字等）。

## 目前已知資訊
{known_info_json}

## 使用者回覆
{user_message}

## 介面語言（follow_up 欄位須使用此語言）
{ui_lang}

## 回傳格式（嚴格 JSON）
{{
  "is_new_question": true 或 false,
  "known_info": {{
    "day_type": "平日 或 休息日 或 例假日 或 國定假日 或 null",
    "hours": 數字或 null,
    "wage_type": "月薪 或 時薪 或 null",
    "wage_amount": 數字或 null,
    "compensation_type": "加班費 或 補休 或 null"
  }},
  "missing_fields": ["缺少的必填欄位（day_type、hours、wage_amount）"],
  "follow_up": "追問句（使用 {ui_lang}，若無需追問則為空字串）"
}}

## 說明
- is_new_question：若回覆明顯是一個全新問題（與加班費計算無關）設為 true，否則 false
- known_info：保留已知資訊，並從此次回覆補充新的欄位值（null 表示仍未知）
- missing_fields：必填欄位 day_type, hours, wage_amount 缺任一即列入
- follow_up：一次最多問兩個問題，語氣親切；資訊已足夠則輸出空字串
"""

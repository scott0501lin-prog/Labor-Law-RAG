"""
Configurable parameters and prompts for intent detection + follow-up system.
Edit this file to tune triggers, required fields, and Gemini prompts
without touching app.py.
"""
import os

# ── Feature toggle ────────────────────────────────────────────────────────────
# On by default: the intent decides which corpora are searched and which
# answer format is used (see intent_router.py). The same call also does the
# translation, so non-Chinese questions need no extra round-trip.
# Set INTENT_DETECTION=false to fall back to keyword heuristics.
ENABLE_INTENT_DETECTION = os.getenv("INTENT_DETECTION", "true").lower() == "true"

# ── Clarifying questions ──────────────────────────────────────────────────────
# When the answer depends on facts the user hasn't given (still employed? how
# many hours? company closed?), the app asks first instead of listing every
# possibility. At most MAX_FOLLOWUP_ROUNDS rounds; the user can always skip
# and get a general answer.
MAX_FOLLOWUP_ROUNDS = 2
MAX_CLARIFY_QUESTIONS = 3

# Skip condition: reply length <= SKIP_MAX_LEN AND contains a trigger keyword.
# Replies longer than SKIP_MAX_LEN are never treated as a skip request even if
# they happen to contain a trigger word (e.g. "我不知道休息日加班費怎麼算").
SKIP_FOLLOWUP_TRIGGERS = ["不知道", "不清楚", "直接回答", "算了", "隨便", "先不說", "跳過", "skip"]
SKIP_MAX_LEN = 10

# ── Prompts ───────────────────────────────────────────────────────────────────
# Placeholders (filled at runtime):
#   {history}          — recent turns as text, 「（無）」 on the first question
#   {user_message}     — the raw user input
#   {ui_lang}          — interface language (e.g. "繁體中文", "English")
#   {clarify_allowed}  — 「可以」/「不可以」: whether another clarifying round is allowed

INTENT_ANALYSIS_PROMPT = """\
你是一個勞動法諮詢助理的前置分析模組。分析使用者訊息並以 JSON 格式回傳結果。
除 JSON 本身外，請勿輸出任何其他文字（包括 ```json 標記、說明文字等）。

## 先前對話（最近幾輪，舊→新；「（無）」表示這是第一個問題）
{history}

## 使用者訊息
{user_message}

## 介面語言（clarify 的問題與選項須使用此語言）
{ui_lang}

## 這次是否還可以反問使用者
{clarify_allowed}

## 回傳格式（嚴格 JSON）
{{
  "translated_query": "這次問題的完整繁體中文版本（規則見說明）",
  "is_followup": true 或 false,
  "intent": "solve_problem 或 find_cases 或 law_lookup 或 out_of_scope",
  "laws": ["相關法規，依關聯度排序，只能從：勞動基準法、勞動基準法施行細則、勞工退休金條例、勞工職業災害保險及保護法、最低工資法、勞工請假規則"],
  "search_terms": ["3 到 6 個法規條文會使用的詞語，用來檢索法條"],
  "violation": true 或 false,
  "clarify": {{
    "needed": true 或 false,
    "questions": [{{"q": "問題", "options": ["選項1", "選項2"]}}]
  }}
}}

## 說明
- translated_query：用來檢索法條與案例，必須不看先前對話也能理解。
  - 先判斷使用者訊息能不能單獨看懂：
    - 能單獨看懂（有自己的主題，例如「預告工資以何薪資為基準？」）→ 原樣輸出，
      即使和先前對話的主題相近，也不得加入先前對話的任何內容。
    - 不能單獨看懂（用「那」「這個」「這樣」「呢」「如果…呢」指涉前文、只補充條件，
      或是在回答助理的反問，例如「那休息日呢？」「如果年資三年呢？」「已離職；公司還在營運」）
      → 把先前對話的主題與使用者提供過的所有條件補回來，改寫成完整問題。
    - 訊息看得懂但缺少情境（例如前文在談女性勞工夜間工作，這次問「公司沒經過工會同意可以嗎？」）
      → 只補上前文的情境（「女性勞工夜間工作」），問句其餘部分保持原樣。
  - 原文非繁體中文時一併翻譯成繁體中文。
- is_followup：translated_query 有用到先前對話的內容時為 true（包括回答助理的反問），否則 false
- intent、laws、search_terms、violation、clarify 都以改寫後的完整問題判斷。
- intent 定義：
  - solve_problem：描述自己的處境，想知道權益或怎麼處理（勞資爭議諮詢）
  - find_cases：想看案例、判決、函釋、問答或實際處理情形（例：「資遣的判決」「加班費的函釋」）
  - law_lookup：想知道某條法條或某項制度的規定內容（例：「勞基法第 30 條」「特休怎麼規定」）
  - out_of_scope：與勞動、職場無關，或明確只涉及資料庫以外的法規：性別平等工作法事項（性騷擾、性別歧視、
    育嬰留職停薪、陪產檢及陪產假、生理假）、勞工保險條例、就業服務法、民法、刑法、稅法、公司法
  - 不確定時選 solve_problem，不要選 out_of_scope
- laws：退休金提繳與個人專戶把「勞工退休金條例」排第一；職災保險給付把「勞工職業災害保險及保護法」排第一；
  最低工資把「最低工資法」排第一；婚喪病事公假把「勞工請假規則」排第一；其餘勞動條件把「勞動基準法」排第一；
  問施行細則才把施行細則排第一。out_of_scope 時給空陣列
- search_terms：把口語問題換成法規條文裡的用字，涵蓋判斷這個問題需要的法律概念，
  不只是問題表面的字（例：「懷孕了公司還排我上夜班」→ 女工、妊娠期間、午後十時至翌晨六時、工作時間；
  「週末被叫回公司值班」→ 休息日、例假、延長工作時間、工資加給；
  「主管逼我簽自願離職」→ 終止勞動契約、資遣費、預告期間）。
  不要寫條號。out_of_scope 時給空陣列
- violation：問題涉及雇主（或投保單位）可能違反規定（例如不給加班費、未提繳退休金、
  未依法預告就解僱），或直接詢問罰則、罰鍰、檢舉時為 true；單純詢問制度內容或計算方式為 false
- clarify：本系統要像顧問一樣先弄清楚狀況再給建議，不要把各種可能情況全部列給使用者。
  - needed 為 true 的條件（全部成立）：intent 為 solve_problem；使用者（含先前對話）沒有提供某些事實，
    而這些事實會改變適用的規定、處理方式或計算結果；「這次是否還可以反問使用者」為「可以」。
  - 例：「被公司調職到外縣市怎麼辦？」→ 問「勞動契約或工作規則有沒有約定可以調動工作地點？」
    〔有約定／沒有約定／不確定〕、「調動後薪資或職務有沒有變差？」〔沒有變差／有變差〕；
    「加班費怎麼算？」→ 問「是哪一天加班？」〔平日／休息日／例假日／國定假日〕、
    「加班幾小時？」、「月薪或時薪多少？」（數字題 options 給空陣列）。
  - 問題已經具體、事實足以回答、或只是問制度規定（law_lookup）、找案例（find_cases）時 needed 為 false。
  - questions 最多 3 題，只問會影響答案的事，先問最關鍵的；能用選項回答的給 2 到 4 個簡短選項，
    數字或自由回答的 options 給空陣列；不要重問使用者已回答或已選擇跳過的事。
  - needed 為 false 時 questions 給空陣列。
"""

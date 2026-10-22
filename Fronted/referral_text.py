# Fixed referral / disclaimer text appended after every RAG answer.
# TODO items: verify all phone numbers and URLs on official websites before publishing.

REFERRAL_BLOCK = """

---
> ⚠️ **本回答僅供參考，不構成法律意見。**
>
> 如需進一步諮詢，可洽以下管道：
> - **1955 勞工諮詢專線**：📞 **1955**（TODO：確認服務時段與費率）
> - **勞動部**：🌐 TODO：填入官方網址
> - **各縣市勞工局／勞動局**：🌐 TODO：填入各縣市網址
> - **法律扶助基金會**：📞 TODO：電話 ／ 🌐 TODO：官方網址
> - **當地律師公會**：🌐 TODO：各縣市律師公會網址
"""

# Prepended before REFERRAL_BLOCK when the topic falls outside labor law scope.
OUT_OF_SCOPE_NOTE = "\n> ⚠️ **此問題超出本系統範圍**（本系統僅涵蓋台灣勞動基準法及相關施行細則）。\n"

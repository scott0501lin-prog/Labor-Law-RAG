# Fixed referral / disclaimer text appended after every RAG answer.
# TODO items: verify all phone numbers and URLs on official websites before publishing.

REFERRAL_BLOCK = """

---
> ⚠️ **本回答僅供參考，不構成法律意見。**
>
> 如需進一步諮詢，可洽以下管道：
> - **1955 勞工諮詢專線**：📞 **1955**（TODO：24小時全年無休，完全免費）
> - **勞動部**：🌐 TODO：https://www.mol.gov.tw/
> - **各縣市勞工局／勞動局**：🌐 TODO：台北勞動局 : https://bola.gov.taipei/
新北市勞工局 : https://www.ntpc.gov.tw/ch/home.jsp?id=1cc8be79ad8046ae
> - **法律扶助基金會**：📞 TODO：電話 0809090999 🌐 TODO：https://0809090999.com.tw/
"""

# Prepended before REFERRAL_BLOCK when the topic falls outside labor law scope.
OUT_OF_SCOPE_NOTE = "\n> ⚠️ **此問題超出本系統範圍**（本系統僅涵蓋台灣勞動基準法及相關施行細則）。\n"

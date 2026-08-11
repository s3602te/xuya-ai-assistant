# LINE 官方 AI 客服系統demo (RAG 雙軌架構)

此專案為解決企業既有知識庫稀疏、傳統客服機器人回覆不精準的痛點，主導規劃並從零開發具備 RAG (檢索增強生成) 能力的 LINE AI 客服系統。

## 系統架構圖
![系統架構圖](my-aics/src/assets/創群LINE架構.png)

## 核心技術與亮點

* **雙軌 RAG 檢索架構：** 整合 Ollama 與 ChromaDB，設計「高精準 A 軌」與「模糊擴展 B 軌」檢索機制，動態調整 L2 距離閾值，確保回答極高準確率。
* **Vision 多模態防幻覺：** 實作圖片 Base64 轉換，採用「先解析、後客服」的兩段式 Prompt Engineering，有效防止 AI 產生幻覺。
* **高併發與狀態管理：** 開發多執行緒 (Threading) 訊息緩衝機制，解決使用者「碎語」造成的 API 頻繁呼叫問題；並實作 SQLite 持久化記憶與超時狀態機 (Daemon Thread)。
* **業務防護網 (Guardrails)：** 導入 Regex 嚴格攔截財務與金流敏感字眼，確保企業服務合規與安全。

## 使用到的技術

* **後端與架構：** Python (Flask), SQLite, Threading (多執行緒狀態機)
* **AI 應用：** Ollama (LLM & Vision), ChromaDB (Vector DB), SentenceTransformers
* **系統整合：** LINE Messaging API (Webhook), RESTful API

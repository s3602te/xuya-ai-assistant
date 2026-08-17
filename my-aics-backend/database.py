# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
# 1. 引入 sqlite3 處理關聯式資料庫，引入 chromadb 處理向量資料庫
import sqlite3
import chromadb

# 2. 引入剛剛建立好的 config 模組，來取得資料庫的實體儲存路徑
from config import CHAT_DB_PATH, DB_PATH
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# SQLite 資料庫初始化與連線開始
# ============================
def init_chat_db():
    # 1. 建立或連線至指定的 SQLite 資料庫檔案
    conn = sqlite3.connect(CHAT_DB_PATH)
    c = conn.cursor()
    
    # 2. 建立 sessions 資料表 (記錄對話的 ID、使用者 ID、標題與最後更新時間)
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, updated_at DATETIME)''')
                 
    # 3. 建立 messages 資料表 (記錄每一則對話的詳細內容、角色與建立時間)
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
                 
    # 4. 提交變更並關閉資料庫連線
    conn.commit()
    conn.close()
    print("[系統] ✅ SQLite 歷史對話資料庫初始化完成！")

# 5. 當模組被載入時自動執行初始化函式，確保資料表存在
init_chat_db()

def get_db_connection():
    # 6. 提供一個取得資料庫連線的工具函式，供 app.py 或其他模組呼叫
    conn = sqlite3.connect(CHAT_DB_PATH)
    
    # 7. 設定 row_factory，讓查詢出來的結果可以像字典 (Dict) 一樣透過欄位名稱存取
    conn.row_factory = sqlite3.Row
    return conn
# ============================
# SQLite 資料庫初始化與連線結束
# ============================

# ============================
# ChromaDB 向量資料庫連線開始
# ============================
# 1. 宣告全域變數，供其他模組 (如 ai_core) 引入使用
collection_manual = None
collection_auto = None

try:
    # 2. 嘗試連線至本地端的 ChromaDB 實體儲存路徑
    print("[系統] 正在連接 ChromaDB 保險箱...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 3. 獲取高精準手動資料庫 (A 軌) 的抽屜 (Collection)
    collection_manual = chroma_client.get_collection(name="xuya_qa_manual")
    
    # 4. 獲取自動擴展資料庫 (B 軌) 的抽屜 (Collection)
    collection_auto = chroma_client.get_collection(name="xuya_qa_auto")
    
    print("[系統] ✅ ChromaDB 雙軌知識庫連接成功！")
except Exception as e:
    # 5. 若連線或獲取資料庫失敗，捕捉例外並印出錯誤訊息防呆
    print(f"[錯誤] 無法連接 ChromaDB: {e}")
# ============================
# ChromaDB 向量資料庫連線結束
# ============================
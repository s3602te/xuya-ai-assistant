# -*- coding: utf-8 -*-
import sqlite3
import chromadb

# 引入剛剛建立好的 config 模組來取得路徑
from config import CHAT_DB_PATH, DB_PATH

# ============================
# SQLite 資料庫初始化與連線
# ============================
def init_chat_db():
    conn = sqlite3.connect(CHAT_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, updated_at DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
    conn.commit()
    conn.close()
    print("[系統] ✅ SQLite 歷史對話資料庫初始化完成！")

# 模組載入時自動執行初始化
init_chat_db()

def get_db_connection():
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============================
# ChromaDB 向量資料庫連線
# ============================
# 宣告全域變數，供其他模組引入
collection_manual = None
collection_auto = None

try:
    print("[系統] 正在連接 ChromaDB 保險箱...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    collection_manual = chroma_client.get_collection(name="xuya_qa_manual")
    collection_auto = chroma_client.get_collection(name="xuya_qa_auto")
    
    print("[系統] ✅ ChromaDB 雙軌知識庫連接成功！")
except Exception as e:
    print(f"[錯誤] 無法連接 ChromaDB: {e}")
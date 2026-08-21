# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
# 1. 引入 sqlite3 處理關聯式資料庫，引入 chromadb 處理向量資料庫
import sqlite3
import chromadb

# 2. 引入 config 模組，取得資料庫的實體儲存路徑
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

    # 4. 【SA v2.3 新增】為常用查詢建立索引。
    # app.py 的 get_session_messages / get_chat_sessions 都會用 session_id 過濾再依時間排序，
    # 資料量小的時候感覺不出來，但歷史紀錄累積到幾千筆後全表掃描會明顯變慢。
    c.execute('''CREATE INDEX IF NOT EXISTS idx_messages_session
                 ON messages (session_id, created_at)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_sessions_user
                 ON sessions (user_id, updated_at)''')

    # 5. 提交變更並關閉資料庫連線
    conn.commit()
    conn.close()
    print("[系統] ✅ SQLite 歷史對話資料庫初始化完成！")

# 6. 當模組被載入時自動執行初始化函式，確保資料表存在
init_chat_db()

def get_db_connection():
    # 7. 提供一個取得資料庫連線的工具函式，供 app.py 或其他模組呼叫
    conn = sqlite3.connect(CHAT_DB_PATH)

    # 8. 設定 row_factory，讓查詢結果可以像字典 (Dict) 一樣透過欄位名稱存取
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

# 【SA v2.3 新增】：記錄兩軌各自的距離度量與筆數，供 ai_core / 校準腳本 / 健康檢查查詢。
# 為什麼需要這個？
# ChromaDB 的 hnsw:space 是建立 collection 當下就寫死的，事後無法修改，
# 而且對已存在的 collection 傳 metadata 不會生效【也不會報錯】——
# 你會以為換成 cosine 了，實際上還是舊的 l2，然後怎麼調門檻都對不上。
# 這種「靜默失敗」最難查，所以乾脆在服務啟動時就把實際度量印出來。
COLLECTION_INFO = {
    "manual": {"name": "xuya_qa_manual", "space": None, "count": 0, "ok": False},
    "auto":   {"name": "xuya_qa_auto",   "space": None, "count": 0, "ok": False},
}

MANUAL_COLLECTION_NAME = "xuya_qa_manual"
AUTO_COLLECTION_NAME = "xuya_qa_auto"

chroma_client = None

try:
    # 2. 連線至本地端的 ChromaDB 實體儲存路徑
    print("[系統] 正在連接 ChromaDB 保險箱...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
except Exception as e:
    print(f"[錯誤] 無法連接 ChromaDB 實體路徑 ({DB_PATH})：{e}")


def _load_collection(track_key: str):
    """
    【SA v2.3 新增】獨立載入單一軌道，並記錄它的距離度量與筆數。

    舊版把兩軌寫在同一個 try 區塊裡：
        collection_manual = client.get_collection("xuya_qa_manual")
        collection_auto   = client.get_collection("xuya_qa_auto")
    只要第二行失敗，例外就會跳過，但第一行其實已經成功了 ——
    結果是「手動軌明明是好的，卻因為自動軌壞掉而一起被當成沒連上」。
    這裡拆成獨立載入，一軌壞掉不會拖累另一軌，而且錯誤訊息會指名道姓。
    """
    info = COLLECTION_INFO[track_key]
    name = info["name"]
    if chroma_client is None:
        return None
    try:
        col = chroma_client.get_collection(name=name)
        info["space"] = (col.metadata or {}).get("hnsw:space", "l2")
        info["count"] = col.count()
        info["ok"] = True
        return col
    except Exception as e:
        print(f"[警告] 無法取得 collection「{name}」：{e}")
        print(f"        → 若尚未建立，請執行 xuya_vdb 底下對應的 ingest 腳本。")
        return None


collection_manual = _load_collection("manual")
collection_auto = _load_collection("auto")


def describe_vector_db() -> str:
    """把兩軌的狀態整理成一行字，供啟動訊息與 /api/health 使用。"""
    parts = []
    for key, label in (("manual", "手動精準軌"), ("auto", "自動擴展軌")):
        i = COLLECTION_INFO[key]
        if i["ok"]:
            parts.append(f"{label}={i['count']}筆/{i['space']}")
        else:
            parts.append(f"{label}=未連線")
    return " | ".join(parts)


# 3. 啟動時把實際狀態印出來，並主動檢查兩軌度量是否一致
if collection_manual is not None or collection_auto is not None:
    print(f"[系統] ✅ ChromaDB 連接完成：{describe_vector_db()}")

    m_space = COLLECTION_INFO["manual"]["space"]
    a_space = COLLECTION_INFO["auto"]["space"]

    # 【SA v2.3 檢查 1】：兩軌度量不一致 → 分數尺度不同，門檻無法互相參照
    if m_space and a_space and m_space != a_space:
        print(f"[系統] ⚠️ 兩軌的距離度量不一致（手動={m_space} / 自動={a_space}）。")
        print(f"        兩邊的分數尺度不同，門檻無法互相比較。")
        print(f"        建議兩軌都重建成 cosine：")
        print(f"          python xuya_vdb/ingest_manual.py --rebuild")
        print(f"          python xuya_vdb/ingest_automatic.py --rebuild")

    # 【SA v2.3 檢查 2】：還停在預設的 l2 → 實測證明 l2 分不開該命中與不該命中
    stale = [f"{k}({v['space']})" for k, v in COLLECTION_INFO.items()
             if v["ok"] and v["space"] == "l2"]
    if stale:
        print(f"[系統] ⚠️ 以下軌道仍使用 ChromaDB 預設的 l2 距離：{', '.join(stale)}")
        print(f"        實測顯示 l2 在目前的 embedding 模型上無法把「該命中」與")
        print(f"        「不該命中」的問題分開，建議改用 cosine 重建。")
else:
    print("[錯誤] ChromaDB 兩軌都無法連線，RAG 功能將完全停用。")
# ============================
# ChromaDB 向量資料庫連線結束
# ============================
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# ============================
# 動態路徑與環境變數初始化
# ============================
# 1. 載入 .env 檔案中的機密環境變數
load_dotenv() 

# 2. 【動態路徑核心】取得目前 config.py 所在的資料夾路徑，定義系統根目錄絕對路徑
# 這樣寫死就不會因為執行路徑不同而找不到資料庫
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================
# 系統參數與路徑設定
# ============================
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
IIS_SEND_URL = os.getenv("IIS_SEND_URL", "")
IIS_API_USER_ID = os.getenv("IIS_API_USER_ID", "")
LINE_INTERNAL_GROUP_ID = os.getenv("LINE_INTERNAL_GROUP_ID", "")

# 資料庫存放路徑 (結合 BASE_DIR 動態生成)
DB_PATH = os.path.join(BASE_DIR, "xuya_vdb", "chroma_storage")           
CHAT_DB_PATH = os.path.join(BASE_DIR, "xuya_vdb", "chat_history.db")     

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 6

# ============================
# 業務邏輯規則與防護網設定
# ============================
HANDOFF_TIMEOUT_SECONDS = 600       
TAXID_COLLECTION_TIMEOUT_SECONDS = 30 
TAX_ID_PATTERN = r"^[0-9]{8}$"

HANDOFF_PATTERNS = [r"真人客服", r"人工客服", r"轉.*?人工", r"轉.*?客服", r"不要\s*ai", r"我要真人", r"找真人", r"人類", r"不要\s*機器人"]
CONFIRM_YES_PATTERNS = [r"^是$", r"^要$", r"^好$", r"^確定$", r"^ok$", r"^yes$", r"^要轉接$", r"^轉接$", r"^真人客服$", r"^人工客服$"]
CONFIRM_NO_PATTERNS = [r"^否$", r"^不用$", r"^不要$", r"^先不用$", r"^取消$", r"^no$"]
HANDOFF_CLEAR_PATTERN = r"(\#?真人接手|\#?humanon|\#?humaninoff)"
AI_RESTART_PATTERNS_FROM_CLIENT = [r"解決完畢", r"以上內容", r"ai\s*開", r"ai\s*on", r"重啟\s*ai", r"好了", r"謝謝客服", r"!ai啟動", r"ai啟動"]

# 財務防護網
BILLING_PATTERNS = [
    r"匯款", r"轉帳", r"付錢", r"付款", r"結帳", r"扣款", r"查收", r"匯給", r"退款", r"匯過去",
    r"多少錢", r"帳戶", r"帳號", r"尾款", r"訂金", r"發票金額", r"報價", r"收費", r"存摺",
    r"\d+\s*(元|塊|千|萬)" 
]

# 上班時間與假日設定
WORKING_HOURS_START = 9
WORKING_HOURS_END = 18
WORKING_DAYS = [0, 1, 2, 3, 4] 
HOLIDAYS = [
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-28", "2026-04-03", "2026-04-06", "2026-05-01", "2026-06-19",
    "2026-09-25", "2026-09-28", "2026-10-09", "2026-12-25",
]
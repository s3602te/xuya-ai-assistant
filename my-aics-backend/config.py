# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# ============================
# 動態路徑與環境變數初始化開始
# ============================
# 1. 載入 .env 檔案中的機密環境變數
load_dotenv()

# 2. 【動態路徑核心】取得目前 config.py 所在的資料夾路徑，定義系統根目錄絕對路徑
# 這樣寫死就不會因為執行路徑不同而找不到資料庫
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================
# 動態路徑與環境變數初始化結束
# ============================

# ============================
# 系統參數與路徑設定開始
# ============================
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")

# 【拔除 IIS】改為官方 LINE Bot SDK 需要的金鑰 (請於 .env 中設定)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")  # 這邊之後讓ai提醒我要去line official account更新token去.env
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")  # 這邊之後讓ai提醒我要去line official account更新token去.env
LINE_INTERNAL_GROUP_ID = os.getenv("LINE_INTERNAL_GROUP_ID", "")  # 這邊之後讓ai提醒我要去line official account更新token去.env

# 外部工具 MCP 金鑰設定 (從 .env 讀取，若無則給空字串)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

# 資料庫存放路徑 (結合 BASE_DIR 動態生成)
DB_PATH = os.path.join(BASE_DIR, "xuya_vdb", "chroma_storage")
CHAT_DB_PATH = os.path.join(BASE_DIR, "xuya_vdb", "chat_history.db")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 6
# ============================
# 系統參數與路徑設定結束
# ============================

# ============================
# 【SA v2 新增】航空母艦戰鬥群參數設定開始
# ============================
# 為什麼把這些搬到 config？
# 原本模型名稱、次數上限、逾時秒數全部寫死在 graph_core.py 裡面。
# 那是「核心邏輯檔」，每次只是想換一顆模型試試看就要動核心程式碼，
# 很容易改到一半不小心動壞別的東西。搬到這裡之後，調參數不需要碰 graph_core。

# 1. 模型分工設定
#    MAIN_MODEL_NAME   -> Supervisor 保底判斷 / Planner 規劃官 / Final_Answer 統整回覆
#    VERIFY_MODEL_NAME -> 關鍵字抽取、數值萃取、算式抽取這類窄任務
MAIN_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "XUYA:latest")
VERIFY_MODEL_NAME = os.getenv("VERIFY_MODEL_NAME", "gemma3:4b")

# 2. 單輪次的工具呼叫上限 (物理煞車，防止死迴圈燒 GPU 與 API 額度)
MAX_SEARCH_CALLS_PER_TURN = 6   # 雙實體比較題(2 次查詢 + 各自最多 2 次重試)剛好夠用
MAX_MATH_CALLS_PER_TURN = 4
MAX_RETRY = 2                   # 單一節點連續失敗的重試上限
MAX_STEP_ATTEMPTS = 3           # 同一個任務清單項目最多嘗試幾次，超過就標記失敗並跳過

# 3. LLM 單次呼叫的逾時保護 (秒)
#    本地 8GB 顯卡建議先抓 90 秒觀察，如果常常看到「LLM 呼叫超過 N 秒」就往上調
LLM_TIMEOUT_SECONDS = 90

# 4. 搜尋結果快取存活時間 (秒)
#    靜態知識(建築高度、歷史事件)可以拉長到 3600；
#    時效性資料(股價、天氣)建議縮短到 60~120
SEARCH_CACHE_TTL_SECONDS = 600

# 5. LangGraph 單輪最大步數 (第二道防線，真正的把關是上面的次數上限)
GRAPH_RECURSION_LIMIT = 30

# 6. RAG 高精準區的 L2 距離門檻
#    原本這個數字寫死在 ai_core.py 的 search_knowledge() 裡面，一併搬過來統一管理
RAG_HIGH_PRECISION_THRESHOLD = 0.50
# ============================
# 【SA v2 新增】航空母艦戰鬥群參數設定結束
# ============================

# ============================
# 業務邏輯規則與防護網設定開始
# ============================
HANDOFF_TIMEOUT_SECONDS = 30 #600秒=10分鐘後切換回AI
TAXID_COLLECTION_TIMEOUT_SECONDS = 30
TAX_ID_PATTERN = r"^[0-9]{8}$"

# 【SA v2 重要修正】：轉真人的觸發正則有兩顆會誤爆的地雷，實測結果如下：
#   r"人類"        → 「請問人類的平均壽命是多少？」會被判定成要找真人客服 ❌
#                    改成 r"(找|要|轉|給我).{0,4}人類" 才是真的在表達「我要跟人講話」
#   r"轉.*?客服"   → 「.*?」中間可以隔任意長度，「我想了解怎麼轉換客服系統」也會中。
#                    改成限制中間最多 4 個字。
# 這種誤判在客服場景很致命 —— 客人只是問個知識題，AI 卻突然說要幫他轉真人。
HANDOFF_PATTERNS = [
    r"真人客服", r"人工客服",
    r"轉.{0,4}人工", r"轉.{0,4}客服", r"轉.{0,4}真人",
    r"不要\s*ai", r"我要真人", r"找真人",
    r"(找|要|給我|轉).{0,4}(真人|專員|服務人員)",
    r"不要\s*機器人"
]

CONFIRM_YES_PATTERNS = [r"^是$", r"^要$", r"^好$", r"^確定$", r"^ok$", r"^yes$", r"^要轉接$", r"^轉接$", r"^真人客服$", r"^人工客服$"]
CONFIRM_NO_PATTERNS = [r"^否$", r"^不用$", r"^不要$", r"^先不用$", r"^取消$", r"^no$"]
HANDOFF_CLEAR_PATTERN = r"(\#?真人接手|\#?humanon|\#?humaninoff)"

# 【SA v2 重要修正】：重啟 AI 的觸發正則地雷更多，實測誤爆三條：
#   r"好了"      → 「好了，我想問一下貴公司的產品規格」會被當成「真人服務結束」❌
#   r"以上內容"  → 「以上內容我看不太懂」也會誤觸 ❌
#   r"ai\s*開"   → 「AI 開發流程大概是怎樣？」直接命中 ❌ (這條最容易中，因為客人很愛問 AI 相關問題)
# 這些指令的本質是「短促的結束語」，所以全部加上 ^...$ 錨點，只有整句就是這幾個字才算數。
AI_RESTART_PATTERNS_FROM_CLIENT = [
    r"^解決完畢$", r"^以上$", r"^好了$", r"^好了[，,。!！]?$",
    r"^ai\s*開$", r"^ai\s*on$", r"^重啟\s*ai$", r"^ai\s*啟動$", r"^!ai啟動$",
    r"^謝謝客服$", r"^謝謝[，,。!！]?$"
]

# 財務防護網
BILLING_PATTERNS = [
    r"匯款", r"轉帳", r"付錢", r"付款", r"結帳", r"扣款", r"查收", r"匯給", r"退款", r"匯過去",
    r"多少錢", r"帳戶", r"帳號", r"尾款", r"訂金", r"發票金額", r"報價", r"收費", r"存摺",
    r"\d+\s*(元|塊|千|萬)"
]
# 【SA v2 提醒】：這份清單目前在 state_manager.py 裡是註解掉的狀態。
# 如果之後要打開，請特別注意最後一條 r"\d+\s*(元|塊|千|萬)" ——
# 它會把「台灣高鐵資本額 562 億元」「這棟樓 508 公尺」這類單純的數字查詢題全部擋掉，
# 導致你的多智能體再也不能回答任何含金額/數量的知識題。
# 建議打開前先把最後一條拿掉，或改成只在同時出現「匯/付/退」等動詞時才成立。

# 上班時間與假日設定
WORKING_HOURS_START = 9
WORKING_HOURS_END = 18
WORKING_DAYS = [0, 1, 2, 3, 4]
HOLIDAYS = [
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-28", "2026-04-03", "2026-04-06", "2026-05-01", "2026-06-19",
    "2026-09-25", "2026-09-28", "2026-10-09", "2026-12-25",
]
# ============================
# 業務邏輯規則與防護網設定結束
# ============================
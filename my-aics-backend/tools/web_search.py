# -*- coding: utf-8 -*-
# tools/web_search.py
# ============================
# 核心模組與套件引入開始
# ============================
import requests
import sys
import os
import time
import re

# 1. 由於此腳本位於 tools/ 資料夾內，需將上層目錄加入系統路徑，方能成功載入 config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BRAVE_API_KEY
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# 【SA v2 新增】搜尋結果快取區塊開始
# ============================
# 為什麼要做快取？
# 上一版 log 顯示「台北 101 建築總高度 公尺」被連續打了 3 次 Brave API，
# 三次的回傳內容當然一模一樣 —— 那兩次純粹是白白燒掉的免費額度。
# graph_core 那邊已經加了「同一句不准查第二次」的硬去重，
# 但那只擋得住同一輪；如果不同客人問同一題、或同一客人重問，還是會重打。
# 這裡再補一層行程內(in-process)快取，同樣的關鍵字在 TTL 內只會真正上網一次。
#
# 【注意】：這是記憶體快取，重啟服務就會清空；
# 若之後要做多台機器部署，再把這裡換成 Redis 即可，介面不用動。
_SEARCH_CACHE = {}
CACHE_TTL_SECONDS = 600   # 10 分鐘。時效性高的題目(股價/天氣)建議調短，靜態知識可調長
CACHE_MAX_ENTRIES = 200   # 避免長時間執行把記憶體吃光


def _cache_key(query: str, count: int) -> str:
    # 正規化：忽略大小寫與多餘空白，讓「台北101  高度」和「台北101 高度」視為同一句
    # 【SA 注意】：正規化結果先存成變數再組字串，
    # 因為 Python 3.11(含)以前的 f-string 大括號內不允許出現反斜線
    normalized = re.sub(r'\s+', '', (query or '')).lower()
    return f"{normalized}::{count}"


def _cache_get(key: str):
    item = _SEARCH_CACHE.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > CACHE_TTL_SECONDS:
        _SEARCH_CACHE.pop(key, None)
        return None
    return item["payload"]


def _cache_put(key: str, payload: dict):
    if len(_SEARCH_CACHE) >= CACHE_MAX_ENTRIES:
        # 最簡單的淘汰策略：丟掉最舊的一筆
        oldest = min(_SEARCH_CACHE.items(), key=lambda kv: kv[1]["ts"])[0]
        _SEARCH_CACHE.pop(oldest, None)
    _SEARCH_CACHE[key] = {"ts": time.time(), "payload": payload}
# ============================
# 【SA v2 新增】搜尋結果快取區塊結束
# ============================


# ============================
# MCP 外部工具：網頁搜尋模組開始
# ============================
def search_web_ex(query, count=3) -> dict:
    """
    【SA v2 新增】結構化版本的搜尋函式，這是給 graph_core 呼叫的主要入口。

    為什麼要多做這一個？
    舊版 search_web() 不管成功或失敗都回傳「一串中文字」，
    上游只能用「內文有沒有出現『錯誤』兩個字」來猜成敗 ——
    但網頁摘要裡本來就常常出現「錯誤」，這就是品管員誤判的根源。
    改成回傳 dict，成敗由 ok 旗標明確表示，上游再也不用猜。

    回傳格式：
      {"ok": True,  "text": "整理好的搜尋結果文字", "message": "", "cached": False}
      {"ok": False, "text": "",                      "message": "失敗原因",  "cached": False}
    """
    # 1. 防呆檢查：若未設定金鑰則直接回傳錯誤
    if not BRAVE_API_KEY:
        return {"ok": False, "text": "", "message": "找不到 BRAVE_API_KEY，請確認 .env 檔案是否設定正確。", "cached": False}

    # 2. 先查快取，命中就直接回，完全不動用 API 額度
    key = _cache_key(query, count)
    cached = _cache_get(key)
    if cached is not None:
        print(f"[搜尋工具] 💾 命中快取，未消耗 API 額度：{query}")
        return {**cached, "cached": True}

    url = "https://api.search.brave.com/res/v1/web/search"

    # 3. 封裝 API 驗證標頭 (Headers)
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }

    # 4. 封裝搜尋條件 (Parameters)
    # 【SA 防禦性修改】：拔除容易引發 422 錯誤的選填參數，讓 Brave 自動依據關鍵字判斷語系
    params = {
        "q": query  # 只傳送最單純的搜尋關鍵字，保證 100% 被伺服器接受
    }

    try:
        # 5. 發送 GET 請求，並設定 10 秒逾時保護
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 6. 萃取 JSON 中的網頁搜尋結果陣列
        # 【SA 防禦性修改】：不在 API 參數限制筆數，而是拿回資料後用 Python 切割前 count 筆
        results = data.get("web", {}).get("results", [])[:count]

        if not results:
            payload = {"ok": False, "text": "", "message": f"搜尋引擎沒有回傳任何關於「{query}」的結果。"}
            _cache_put(key, payload)   # 沒結果也快取，避免同一句沒用的關鍵字被反覆重試
            return {**payload, "cached": False}

        # 7. 迴圈遍歷結果，整理成容易讓 AI 閱讀的純文字格式
        formatted_results = []
        for i, res in enumerate(results):
            title = res.get("title", "無標題")
            snippet = res.get("description", "無摘要")
            link = res.get("url", "")
            formatted_results.append(f"[{i+1}] 標題: {title}\n摘要: {snippet}\n來源: {link}\n")

        payload = {"ok": True, "text": "\n".join(formatted_results), "message": ""}
        _cache_put(key, payload)
        return {**payload, "cached": False}

    except requests.exceptions.HTTPError as e:
        # 【SA v2 新增】：把 HTTP 狀態碼獨立出來，401/429 這種「金鑰或額度問題」
        # 跟「查無資料」是完全不同的病，log 分清楚才好 debug
        status = getattr(e.response, "status_code", "未知")
        hint = {401: "API 金鑰無效或過期", 429: "API 額度已用盡或請求過於頻繁"}.get(status, "")
        return {"ok": False, "text": "", "message": f"搜尋 API 回應 HTTP {status} {hint}".strip(), "cached": False}
    except Exception as e:
        return {"ok": False, "text": "", "message": f"上網搜尋時發生錯誤: {e}", "cached": False}


def search_web(query, count=3) -> str:
    """
    【SA v2 保留】舊介面的相容包裝，回傳純文字。
    如果專案其他地方(例如 ai_core.py)還有呼叫這個函式，不用改就能繼續跑。
    新的程式碼請一律改用 search_web_ex()。
    """
    payload = search_web_ex(query, count)
    if payload["ok"]:
        return payload["text"]
    return f"搜尋未取得結果：{payload['message']}"
# ============================
# MCP 外部工具：網頁搜尋模組結束
# ============================


# ============================
# 單元測試區塊開始 (僅直接執行此檔案時觸發)
# ============================
if __name__ == "__main__":
    print("🔍 正在啟動上網工具測試...\n")

    test_query = "台北101 建築總高度 公尺"
    print(f"發送搜尋關鍵字：「{test_query}」\n")

    print("========== 第一次搜尋(會真的打 API) ==========")
    r1 = search_web_ex(test_query)
    print(f"ok={r1['ok']} / cached={r1['cached']} / message={r1['message']}")
    print(r1["text"][:500])

    print("\n========== 第二次搜尋(應該命中快取，不消耗額度) ==========")
    r2 = search_web_ex(test_query)
    print(f"ok={r2['ok']} / cached={r2['cached']}")
    print("==============================")
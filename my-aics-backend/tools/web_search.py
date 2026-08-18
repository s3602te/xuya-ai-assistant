# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import requests
import sys
import os

# 1. 由於此腳本位於 tools/ 資料夾內，需將上層目錄加入系統路徑，方能成功載入 config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BRAVE_API_KEY
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# MCP 外部工具：網頁搜尋模組開始
# ============================
def search_web(query, count=3):
    """
    這就是我們的「上網 USB 隨身碟」。
    傳入 query (關鍵字)，它會回傳前 3 筆最相關的搜尋結果摘要。
    """
    # 1. 防呆檢查：若未設定金鑰則直接回傳錯誤
    if not BRAVE_API_KEY:
        return "錯誤：找不到 BRAVE_API_KEY，請確認 .env 檔案是否設定正確。"

    url = "https://api.search.brave.com/res/v1/web/search"
    
    # 2. 封裝 API 驗證標頭 (Headers)
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    # 3. 封裝搜尋條件 (Parameters)
    # 【SA 防禦性修改】：拔除容易引發 422 錯誤的選填參數，讓 Brave 自動依據關鍵字判斷語系
    params = {
        "q": query  # 只傳送最單純的搜尋關鍵字，保證 100% 被伺服器接受
    }
    try:
        # 4. 發送 GET 請求，並設定 10 秒逾時保護
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        # 若發生 HTTP 錯誤 (如 401 未授權、429 額度用盡)，將直接拋出例外中斷執行
        response.raise_for_status() 
        data = response.json()

        # 5. 萃取 JSON 中的網頁搜尋結果陣列
        # 【SA 防禦性修改】：我們不在 API 參數限制筆數，而是拿回資料後，用 Python 切割前 count 筆
        results = data.get("web", {}).get("results", [])[:count] 

        if not results:
            return f"找不到關於「{query}」的相關結果。"

        # 6. 迴圈遍歷結果，整理成容易讓 AI 閱讀的純文字格式
        formatted_results = []
        for i, res in enumerate(results):
            title = res.get("title", "無標題")
            snippet = res.get("description", "無摘要") # snippet 是網頁的文字片段
            link = res.get("url", "")
            
            # 把資料整理成容易讓 AI 閱讀的格式
            formatted_results.append(f"[{i+1}] 標題: {title}\n摘要: {snippet}\n來源: {link}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"上網搜尋時發生錯誤: {e}"
# ============================
# MCP 外部工具：網頁搜尋模組結束
# ============================


# ============================
# 單元測試區塊開始 (僅直接執行此檔案時觸發)
# ============================
if __name__ == "__main__":
    print("🔍 正在啟動上網工具測試...\n")
    
    # 設定測試關鍵字：搜尋昨天 (8/16) 的新聞
    test_query = "2026年8月16日 台灣新聞"
    print(f"發送搜尋關鍵字：「{test_query}」\n")
    
    # 執行搜尋函式並印出結果
    result = search_web(test_query)
    
    print("========== 搜尋結果 ==========")
    print(result)
    print("==============================")
# ============================
# 單元測試區塊結束
# ============================
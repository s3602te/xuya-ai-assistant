# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
# 1. 引入系統與標準函式庫，處理 JSON、多執行緒、離開事件、唯一識別碼與時間
import json
import threading
import atexit
import uuid
import time  # 【SA v2.1 調整】：原本這行藏在 admin_reply() 函式內部才 import，搬到最上面統一管理
import requests  # 【SA v2.1 新增】：健康檢查端點需要用它去 ping Ollama
from datetime import datetime

# 2. 引入 Flask 框架與 CORS 套件，處理 API 路由與跨域請求
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# 模組化引入區塊開始
# ============================
# 1. 引入系統設定參數與資料庫連線實體
from config import *
from database import get_db_connection

# 2. 引入 AI 核心檢索與生成模組
# 【SA v2.1 說明】：get_ollama_response 是航空母艦上線前的舊版單體流程，
# 目前實際不會被呼叫(state_manager 走的是 graph_core.app_graph)，
# 保留 import 是為了萬一多智能體出狀況時可以快速切回來救急。
from ai_core import search_knowledge, get_ollama_response

# 3. 引入狀態機，負責處理訊息邏輯、緩衝機制與超時監控
from state_manager import handle_message_logic_with_buffer, check_timeouts, timeout_checker_stop

# 4. 引入 WebSocket 廣播中心，負責即時雙向通訊
from websocket_manager import socketio
# ============================
# 模組化引入區塊結束
# ============================

# ============================
# 環境變數與全域初始化開始
# ============================
# 1. 建立 Flask 應用程式實體，並設定靜態檔案與打包好的前端 (dist) 路徑
app = Flask(__name__, static_folder='dist', static_url_path='')

# 2. 啟用 CORS，允許跨網域請求 (開發階段與前後端分離必備)
CORS(app)

# 3. 將 Flask 應用程式實體綁定至 WebSocket 引擎，啟動雙向通訊支援
socketio.init_app(app)
# ============================
# 環境變數與全域初始化結束
# ============================

# ============================
# 【SA v2.1 新增】系統健康檢查區塊開始
# ============================
# 為什麼要加這支 API？
# 多智能體最痛苦的除錯情境是：客人送出問題之後等了 90 秒，最後只拿到一句
# 「AI 系統處理您的請求時發生錯誤」，然後你完全不知道到底是哪一環壞了 ——
# 是 Ollama 沒開？模型名稱打錯？Brave 金鑰過期？還是 ChromaDB 是空的？
#
# 這支 API 一次把所有外部相依性掃過一遍，直接告訴你哪一項是紅燈。
# 啟動伺服器後在瀏覽器打開 http://localhost:5000/api/health 就能看到。
@app.route("/api/health", methods=['GET'])
def health_check():
    report = {"ok": True, "checks": {}}

    def _mark(name, ok, detail=""):
        report["checks"][name] = {"ok": ok, "detail": detail}
        if not ok:
            report["ok"] = False

    # 1. Ollama 服務是否活著、需要的兩顆模型是否都 pull 好了
    try:
        r = requests.get(f"{OLLAMA_API_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        installed = [m.get("name", "") for m in r.json().get("models", [])]
        _mark("ollama_service", True, f"連線正常，已安裝 {len(installed)} 顆模型")

        # 【SA 注意】：Ollama 的模型名稱有時會帶 :latest 後綴，這裡做寬鬆比對
        def _has(name):
            base = name.split(":")[0]
            return any(m == name or m.split(":")[0] == base for m in installed)

        _mark("main_model", _has(MAIN_MODEL_NAME),
              f"主模型 {MAIN_MODEL_NAME}" + ("" if _has(MAIN_MODEL_NAME) else f" 不在清單中，目前有：{installed}"))
        _mark("verify_model", _has(VERIFY_MODEL_NAME),
              f"驗證模型 {VERIFY_MODEL_NAME}" + ("" if _has(VERIFY_MODEL_NAME) else f" 不在清單中，目前有：{installed}"))
    except Exception as e:
        _mark("ollama_service", False, f"無法連線到 {OLLAMA_API_BASE_URL}：{e}")
        _mark("main_model", False, "因 Ollama 無法連線而略過")
        _mark("verify_model", False, "因 Ollama 無法連線而略過")

    # 2. Brave 搜尋金鑰是否設定
    _mark("brave_api_key", bool(BRAVE_API_KEY),
          "已設定" if BRAVE_API_KEY else "未設定，Search_Agent 將完全無法運作(請檢查 .env)")

    # 3. RAG 知識庫是否有資料
    try:
        docs = search_knowledge("測試檢索")
        _mark("rag_knowledge_base", True, f"檢索正常，本次回傳 {len(docs)} 筆參考資料")
    except Exception as e:
        _mark("rag_knowledge_base", False, f"檢索異常：{e}")

    # 4. 聊天紀錄資料庫是否可寫
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        _mark("chat_database", True, "連線正常")
    except Exception as e:
        _mark("chat_database", False, f"連線異常：{e}")

    # 5. 多智能體地圖是否成功編譯
    try:
        from graph_core import app_graph  # noqa: F401
        _mark("agent_graph", True, "航空母艦地圖編譯成功")
    except Exception as e:
        _mark("agent_graph", False, f"地圖編譯失敗：{e}")

    return jsonify(report), (200 if report["ok"] else 503)
# ============================
# 【SA v2.1 新增】系統健康檢查區塊結束
# ============================

# ============================
# 真人客服後台 API 區塊開始
# ============================
@app.route("/api/admin_reply", methods=['POST'])
def admin_reply():
    """
    這支 API 讓你模擬「真人客服在後台打字」。
    你可以用 Postman 打這支 API，文字就會瞬間出現在前端網頁上！
    """
    # 1. 取得前端或 Postman 傳入的 JSON 負載資料
    data = request.json
    user_id = data.get("user_id")
    message = data.get("message")
    action = data.get("action", "reply") # reply: 傳送訊息, end_human: 結束真人模式

    # 2. 參數防呆檢驗
    if not user_id:
        return jsonify({"error": "缺少 user_id"}), 400

    if action == "reply":
        # 3. 透過 WebSocket 直接廣播客服訊息給指定用戶 (標記 role 為 admin 讓前台辨識)
        socketio.emit('chat_reply', {'session_id': user_id, 'reply': message, 'role': 'admin'})

        # 4. 建立資料庫連線，將真人回覆永久寫入 SQLite，且不再和 ai 混淆
        conn = get_db_connection()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 5. 查找該用戶最新的一筆對話 Session
        session_row = conn.execute("SELECT id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1", (user_id,)).fetchone()
        if session_row:
            latest_session_id = session_row['id']
            # 6. 將回覆訊息以 admin 角色存入 messages 資料表
            conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (latest_session_id, 'admin', message, current_time))
            conn.commit()
        conn.close()

        # 7. 優化核心：客服一旦回覆，立刻將閒置計時器歸零重新計算
        # 【SA v2.1 調整】：time 已改在檔案最上方 import，這裡不再重複 import
        from state_manager import handoff_start_times
        if user_id in handoff_start_times and handoff_start_times[user_id].get("handoff_start_time"):
            handoff_start_times[user_id]["handoff_start_time"] = time.time()

        return jsonify({"status": "success", "msg": "已傳送真人回覆"})

    elif action == "end_human":
        # 8. 結束真人模式：引入狀態機與廣播模組，準備重置狀態與切換前端 UI
        from state_manager import human_handoff, handoff_collect_taxid, handoff_start_times, human_lock
        from websocket_manager import broadcast_state_change

        # 9. 強制清除該用戶在狀態機內的所有鎖定與計時標記
        human_handoff.pop(user_id, None)
        handoff_collect_taxid.pop(user_id, None)
        handoff_start_times.pop(user_id, None)
        human_lock.pop(user_id, None)

        # 10. 將「真人服務結束」的系統通知寫入資料庫，確保歷史紀錄完整
        conn = get_db_connection()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_row = conn.execute("SELECT id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1", (user_id,)).fetchone()
        if session_row:
            latest_session_id = session_row['id']
            conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (latest_session_id, 'ai', "【系統通知】真人服務結束，AI 已重新上線。", current_time))
            conn.commit()
        conn.close()

        # 11. 透過 WebSocket 推播系統通知，並廣播狀態切換讓前端 UI 解除鎖定
        socketio.emit('chat_reply', {'session_id': user_id, 'reply': "【系統通知】真人服務結束，AI 已重新上線。"})
        broadcast_state_change(user_id, 'ai')
        return jsonify({"status": "success", "msg": "已切換回 AI 模式"})

    # 12. 【SA v2.1 修正】：補上未知 action 的回傳。
    # 舊版只有 reply / end_human 兩個分支，如果前端不小心送了第三種 action，
    # 函式會走到底而回傳 None，Flask 會直接丟出
    # 「The view function did not return a valid response」的 500 錯誤，
    # 而且訊息很難聯想到是 action 打錯字造成的。
    return jsonify({"error": f"未知的 action：{action}（目前只支援 reply / end_human）"}), 400
# ============================
# 真人客服後台 API 區塊結束
# ============================

# ============================
# Web 前端 React 專屬 API 路由開始
# ============================
@app.route("/api/chat_sessions", methods=['GET'])
def get_chat_sessions():
    # 1. 取得前端傳遞的使用者身分標籤 (user_id)
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "缺少 user_id 參數"}), 400

    try:
        # 2. 建立資料庫連線
        conn = get_db_connection()

        # 3. 權限分流：判斷是否為後台管理員上帝視角 (admin 撈取全部，user 撈取個人)
        if user_id == "admin":
            sessions_rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        else:
            sessions_rows = conn.execute("SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()

        # 4. 資料整理與紅點標記判斷
        result = []
        for s in sessions_rows:
            s_dict = dict(s)

            # 5. 擴增判斷條件：撈取最後一筆與狀態切換相關的系統訊息，涵蓋所有的「轉回AI」情境
            last_status = conn.execute(
                "SELECT content FROM messages WHERE session_id = ? AND (content LIKE '%正在為您轉接真人客服%' OR content LIKE '%真人服務結束%' OR content LIKE '%AI 客服將重新為您服務%' OR content LIKE '%已自動切回 AI%' OR content LIKE '%AI 繼續為您服務%') ORDER BY created_at DESC LIMIT 1",
                (s_dict['id'],)
            ).fetchone()

            # 6. 如果最後一筆狀態是「正在轉接」，代表客服還沒按下結束，需於後台標記為待處理紅點
            s_dict['needs_attention'] = True if last_status and "正在為您轉接真人客服" in last_status['content'] else False
            result.append(s_dict)

        conn.close()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat_sessions/<session_id>", methods=['GET'])
def get_session_messages(session_id):
    try:
        # 1. 建立資料庫連線，獲取指定對話的歷史訊息
        conn = get_db_connection()

        # 2. 將角色、文字與資料庫寫入的時間 (created_at) 一併交給前端，作為絕對計時標準
        messages = conn.execute("SELECT role, content as text, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
        conn.close()

        # 3. 組合預設的歡迎詞與查詢結果，回傳給前端渲染
        default_msg = [{"role": "ai", "text": "你好！我是這位求職者的專屬 AI 助理。您可以問我任何關於他專案、技術或開發過程的問題！"}]
        return jsonify(default_msg + [dict(m) for m in messages]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat_sessions/<session_id>", methods=['DELETE'])
def delete_session(session_id):
    try:
        # 1. 建立資料庫連線
        conn = get_db_connection()

        # 2. 執行 DELETE 語法，同步刪除 sessions 總表與對應的 messages 明細
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat_sessions/<session_id>", methods=['PUT'])
def rename_session(session_id):
    # 1. 擷取前端 PUT 傳遞的新標題名稱
    data = request.json
    new_title = data.get('title')
    if not new_title:
        return jsonify({"error": "缺少 title 參數"}), 400
    try:
        # 2. 建立資料庫連線並執行 UPDATE 語法更新標題
        conn = get_db_connection()
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/web_chat", methods=['POST'])
def web_chat():
    # 1. 擷取前端 POST 發送的對話參數
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message', '')
    user_id = data.get('user_id')

    # 2. 執行空白防呆與缺漏檢查
    if not user_message or not user_id:
        return jsonify({"error": "缺少參數"}), 400

    try:
        conn = get_db_connection()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3. 判斷是否為全新對話：若前端無傳入 session_id，則動態配發一組 UUID
        if not session_id:
            session_id = str(uuid.uuid4())
            # 取訊息前 12 個字作為自動命名標題
            title = user_message[:12] + "..." if len(user_message) > 12 else user_message
            conn.execute("INSERT INTO sessions (id, user_id, title, updated_at) VALUES (?, ?, ?, ?)", (session_id, user_id, title, current_time))
        else:
            # 4. 若為舊有對話，更新最後異動時間，並將原有標題拉出
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (current_time, session_id))
            row = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
            title = row['title'] if row else "未命名對話"

        # 5. 寫入資料庫：將使用者的提問先行存入 messages 資料表
        conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'user', user_message, current_time))
        conn.commit()
        conn.close()

        # 6. 將資料送入核心大腦：把 session_id 與訊息傳入狀態機，處理緩衝機制，並確保後續 AI 的回覆可正確寫入
        handle_message_logic_with_buffer(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            user_image=None
        )

        # 7. 立即回傳 queued 狀態，讓前端不必等待 AI 運算，避免 HTTP 連線逾時阻塞
        return jsonify({
            "status": "queued",
            "session_id": session_id,
            "title": title
        }), 200

    except Exception as e:
        print(f"[Web API 錯誤] {e}")
        return jsonify({"error": str(e)}), 500
# ============================
# Web 前端 React 專屬 API 路由結束
# ============================

# ============================
# 靜態檔案路由與伺服器啟動區塊開始
# ============================
@app.route('/')
def serve():
    # 1. 根目錄存取時，預設導向 React 打包好的 index.html
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    # 2. 捕捉 404 錯誤：因 React 屬於單頁應用 (SPA)，將所有未定義路由導回首頁，交由前端 Router 處理
    #
    # 【SA v2.1 修正】：舊版把「所有」404 都導回 index.html，包括 /api/ 開頭的路徑。
    # 後果是：如果你把 API 網址打錯字(例如 /api/helth)，Postman 或前端拿到的
    # 會是一坨 HTML 網頁原始碼，然後 JSON 解析失敗，你會以為是後端壞掉，
    # 實際上只是路徑打錯 —— 這種 bug 非常浪費時間。
    # 現在 /api/ 底下的 404 一律回傳明確的 JSON 錯誤訊息。
    if request.path.startswith('/api/'):
        return jsonify({"error": f"找不到這個 API 路徑：{request.path}"}), 404
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # 3. 啟動背景執行緒 (Daemon Thread) 執行狀態機超時監控
    timeout_thread = threading.Thread(target=check_timeouts)
    timeout_thread.daemon = True
    timeout_thread.start()

    # 4. 註冊伺服器關閉時的安全清理動作：觸發 Event 旗標，終止背景監控迴圈
    atexit.register(lambda: timeout_checker_stop.set())

    # 5. 啟動伺服器：改用 socketio.run 封裝 app，提供對 WebSocket 的協定支援
    print("\n[系統] 🚀 啟動支援 WebSocket 的 Flask 伺服器...")
    print(f"[系統] 🩺 啟動後請先打開 http://localhost:5000/api/health 確認所有相依項目都是綠燈\n")
    # 【SA v2.1 提醒】：debug=True 會開啟 Werkzeug 除錯器，正式對外部署前務必改成 False，
    # 否則一旦程式出錯，任何人都能透過瀏覽器在你的機器上執行任意 Python 程式碼。
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
# ============================
# 靜態檔案路由與伺服器啟動區塊結束
# ============================
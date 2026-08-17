# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
# 1. 引入系統與標準函式庫，處理 JSON、多執行緒、離開事件、唯一識別碼與時間
import json
import threading 
import atexit 
import uuid
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
        import time
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
            # 統一變數名稱為 sessions_rows 避免報錯
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
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # 3. 啟動背景執行緒 (Daemon Thread) 執行狀態機超時監控
    timeout_thread = threading.Thread(target=check_timeouts)
    timeout_thread.daemon = True 
    timeout_thread.start()
    
    # 4. 註冊伺服器關閉時的安全清理動作：觸發 Event 旗標，終止背景監控迴圈
    atexit.register(lambda: timeout_checker_stop.set())

    # 5. 啟動伺服器：改用 socketio.run 封裝 app，提供對 WebSocket 的協定支援
    print("\n[系統] 🚀 啟動支援 WebSocket 的 Flask 伺服器...\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
# ============================
# 靜態檔案路由與伺服器啟動區塊結束
# ============================
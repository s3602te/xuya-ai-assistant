# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import json
import threading 
import atexit 
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# 模組化引入區塊
# ============================
from config import *
from database import get_db_connection
from ai_core import search_knowledge, get_ollama_response

# 4. 引入狀態機與通訊樞紐
from state_manager import handle_message_logic_with_buffer, check_timeouts, timeout_checker_stop

# ============================
# 環境變數與全域初始化開始
# ============================
app = Flask(__name__, static_folder='dist', static_url_path='')
CORS(app) 
# ============================
# 環境變數與全域初始化結束
# ============================

# ============================
# IIS Server API 接收路由開始
# ============================
@app.route("/api/iis_forward", methods=['POST'])
def iis_forward():
    def make_response_json(success, msg=""):
        return {
            "Response": {
                "Header": {
                    "ServiceResult": "Y" if success else "N",
                    "StatusCode": "200" if success else "400",
                    "StatusDesc": "Success" if success else msg,
                    "ResponseTime": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                }
            }
        }

    try:
        raw_data = request.json or json.loads(request.get_data(as_text=True))
        debug_str = json.dumps(raw_data, ensure_ascii=False)
        if len(debug_str) > 500:
            print(f"\n[IIS 接收] 收到資料 (內容過長已省略): {debug_str[:500]} ... [Base64 圖片資料隱藏] ...\n")
        else:
            print(f"\n[IIS 接收] 收到的資料: {debug_str}\n")
            
        if 'Request' in raw_data and 'Data' in raw_data['Request']:
            data_node = raw_data['Request']['Data']
        elif 'Data' in raw_data: 
            data_node = raw_data['Data']
        else:
            data_node = raw_data

        channel_id = data_node.get('channelId')
        
        to_value = data_node.get('to') 
        if not to_value:
            if data_node.get('groupId'):
                to_value = data_node.get('groupId')
            elif data_node.get('userId'):
                to_value = data_node.get('userId')
        
        group_id = None
        user_id = None
        if to_value:
            if to_value.startswith("C") or to_value.startswith("G") or to_value.startswith("R"):
                group_id = to_value 
            else:
                user_id = to_value  

        reply_token = data_node.get('replyToken')
        user_message = data_node.get('message') or ""
        user_image = data_node.get('image')

    except Exception as e:
        return jsonify(make_response_json(False, f"JSON Error: {e}")), 200

    if not user_message and not user_image:
        return jsonify(make_response_json(False, "Missing message and image")), 200

    try:
        # 將訊息轉交給 state_manager 處理
        handle_message_logic_with_buffer(channel_id, user_id, reply_token, user_message, group_id, user_image)
        return jsonify(make_response_json(True)), 200
    except Exception as e:
        print(f"[IIS] Error: {e}")
        return jsonify(make_response_json(False, str(e))), 500
# ============================
# IIS Server API 接收路由結束
# ============================

# ============================
# Web 前端 React 專屬 API 路由開始
# ============================
@app.route("/api/chat_sessions", methods=['GET'])
def get_chat_sessions():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "缺少 user_id 參數"}), 400

    try:
        conn = get_db_connection()
        sessions = conn.execute("SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
        conn.close()
        return jsonify([dict(s) for s in sessions]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat_sessions/<session_id>", methods=['GET'])
def get_session_messages(session_id):
    try:
        conn = get_db_connection()
        messages = conn.execute("SELECT role, content as text FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
        conn.close()
        default_msg = [{"role": "ai", "text": "你好！我是這位求職者的專屬 AI 助理。您可以問我任何關於他專案、技術或開發過程的問題！"}]
        return jsonify(default_msg + [dict(m) for m in messages]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/chat_sessions/<session_id>", methods=['DELETE'])
def delete_session(session_id):
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat_sessions/<session_id>", methods=['PUT'])
def rename_session(session_id):
    data = request.json
    new_title = data.get('title')
    if not new_title:
        return jsonify({"error": "缺少 title 參數"}), 400
    try:
        conn = get_db_connection()
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/web_chat", methods=['POST'])
def web_chat():
    data = request.json
    session_id = data.get('session_id') 
    user_message = data.get('message', '')
    user_id = data.get('user_id') 

    if not user_message:
        return jsonify({"error": "缺少 message 參數"}), 400
    if not user_id:
        return jsonify({"error": "缺少 user_id 參數"}), 400

    try:
        conn = get_db_connection()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not session_id:
            session_id = str(uuid.uuid4())
            title = user_message[:12] + "..." if len(user_message) > 12 else user_message
            conn.execute("INSERT INTO sessions (id, user_id, title, updated_at) VALUES (?, ?, ?, ?)", (session_id, user_id, title, current_time))
        else:
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (current_time, session_id))
            row = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
            title = row['title'] if row else "未命名對話"

        conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'user', user_message, current_time))
        conn.commit()

        msgs = conn.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
        
        history_str = ""
        if len(msgs) > 1:
            history_str = "【先前對話紀錄 (Context History)】\n"
            for m in msgs[:-1]: 
                role_name = "面試官" if m['role'] == 'user' else "張序亞助理"
                history_str += f"{role_name}：{m['content']}\n"
            history_str += "------------------\n"

        relevant = search_knowledge(user_message)
        context_str = "\n".join(relevant) if relevant else "無相關資料。"
        
        print(f"\n===== [Web 介面記憶優化] 🔍 餵給 AI 的參考資料 =====\n{context_str}\n===========================================\n")

        prompt = (
            f"你是一位專屬的面試 AI 助理，代表軟體工程師 張序亞 (Steven)。\n"
            f"你的主要職責是專業、自信且友善地向面試官介紹張序亞的技術能力、專案經驗與人格特質。\n\n"
            f"{history_str}"
            f"【ChromaDB 知識庫檢索資訊】：\n"
            f"{context_str}\n\n"
            f"【面試官的最新提問】：{user_message}\n\n"
            f"【核心回答守則】：\n"
            f"1. 請結合上述的「先前對話紀錄」與「知識庫檢索資訊」來回答。如果面試官是用代名詞追問（例如『那這個怎麼解決？』），請根據上下文判斷他在問哪個專案或技術。\n"
            f"2. 誠實且專業，絕對不要自行捏造經歷。\n"
            f"3. 請使用繁體中文，條理分明，直接給出回答，絕對不要印出系統提示或括號備註。\n\n"
            f"請直接給出回答："
        )

        ollama_resp = get_ollama_response(prompt, None, "XUYA:latest")
        if not ollama_resp or ollama_resp == "AI_IMAGE_ERROR":
            ollama_resp = "抱歉，AI 暫時無法回應。"

        conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'ai', ollama_resp, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        return jsonify({
            "reply": ollama_resp,
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
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # 啟動背景執行緒 (Daemon Thread) 執行超時監控
    timeout_thread = threading.Thread(target=check_timeouts)
    timeout_thread.daemon = True 
    timeout_thread.start()
    
    atexit.register(lambda: timeout_checker_stop.set())

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
# ============================
# 靜態檔案路由與伺服器啟動區塊結束
# ============================
# -*- coding: utf-8 -*-
# ============================
# WebSocket 核心套件引入開始
# ============================
# 1. 引入 flask_socketio 套件中的 SocketIO 模組，用於建立 WebSocket 伺服器
from flask_socketio import SocketIO
# ============================
# WebSocket 核心套件引入結束
# ============================

# ============================
# WebSocket 全域物件初始化開始
# ============================
# 1. 建立 SocketIO 全域實體
# 2. 設定 cors_allowed_origins="*"，允許所有網域跨網域連線 (CORS)，確保 React 前端能順利建立雙向通訊
socketio = SocketIO(cors_allowed_origins="*")
# ============================
# WebSocket 全域物件初始化結束
# ============================

# ============================
# WebSocket 連線與事件路由開始
# ============================
@socketio.on('connect')
def handle_connect():
    # 1. 監聽 'connect' 事件：當前端客戶端成功建立 WebSocket 連線時觸發並印出提示
    print("\n[WebSocket] 🟢 一個新的前端客戶端已成功連線！\n")

@socketio.on('disconnect')
def handle_disconnect():
    # 2. 監聽 'disconnect' 事件：當前端客戶端斷開 WebSocket 連線時觸發並印出提示
    print("\n[WebSocket] 🔴 前端客戶端已斷線。\n")

def broadcast_state_change(session_id, state):
    """
    這是一個模塊化接口。
    未來狀態機 (state_manager) 或是 D 模組 (指揮官) 只要呼叫這個函式，
    就能瞬間把狀態推播給 React 前端。
    """
    # 4. 透過 emit 發送 'state_update' 事件，夾帶指定的 session_id 與切換狀態 (如 'human' 或 'ai')
    socketio.emit('state_update', {'session_id': session_id, 'state': state})
    
    # 5. 於伺服器控制台印出廣播成功的日誌紀錄，方便開發與追蹤除錯
    print(f"[WebSocket] 📡 已廣播狀態更新 -> Session: {session_id}, State: {state}")
# ============================
# WebSocket 連線與事件路由結束
# ============================
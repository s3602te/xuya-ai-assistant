# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
# 1. 引入 Python 內建標準套件 (正則表達式、時間、執行緒控制與日期時間)
import re
import time
import threading 
from datetime import datetime

# 2. 引入自訂模組與設定 (包含全域變數、資料庫連線、AI 核心檢索與生成模組)
from config import *
from database import get_db_connection
# 【SA 航空母艦升級】：保留 RAG 檢索，但把生成大腦替換為 graph_core 的多智能體地圖
from ai_core import search_knowledge, needs_contact_footer
from graph_core import app_graph
# 我們還要引入 langchain 的 Message 格式來裝進背包
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# 3. 引入 WebSocket 廣播大聲公，用於前後端即時狀態切換與通訊
from websocket_manager import broadcast_state_change, socketio
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# 記憶體狀態與未解耦之狀態機開始
# ============================
# 1. 記憶體狀態變數 (用於記錄每個使用者的轉接進度、鎖定狀態與各式計時器)
human_handoff = {}       
handoff_pending = {}     
handoff_collect_taxid = {} 
handoff_context = {}     
human_lock = {}          
handoff_start_times = {} 
handoff_pending_times = {} # 記錄等待選擇是/否的起始時間
timeout_checker_stop = threading.Event()

# 2. 短期記憶緩衝區配置 (限制回溯回合數與防碎語的訊息截流秒數)
MAX_HISTORY_TURNS = 2
conversation_memory = {} 

BUFFER_SECONDS = 5 
IMAGE_BUFFER_SECONDS = 10 
message_buffer = {} 
# ============================
# 記憶體狀態與未解耦之狀態機結束
# ============================

# ============================
# 核心工具函式區塊開始
# ============================
# 1. 定義取得對話唯一識別碼的工具，網頁版統一使用 user_id 進行綁定
def get_conv_key(user_id, group_id=None):
    return user_id # 網頁版統一使用 user_id 作為唯一識別碼

# 2. 判斷是否為上班時間 (目前測試模式強制回傳 True，允許下班時間進行測試)
def is_working_hours() -> bool:
    # 測試模式：強制回傳 True 允許下班時間測試
    return True

# 3. 核心訊息發送與紀錄：將 AI 與系統的回覆寫入 SQLite 資料庫，並同步透過 WebSocket 推播
def send_websocket_reply(user_id, session_id, message, options=None):
    if session_id:
        try:
            conn = get_db_connection()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'ai', message, current_time))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Error] {e}")

    payload = {'session_id': user_id, 'reply': message}
    if options:
        payload['options'] = options
    socketio.emit('chat_reply', payload)
    print(f"[WebSocket] 💬 已回覆客戶 {user_id}: {message[:20]}...")
# ============================
# 核心工具函式區塊結束
# ============================

# ============================
# 共用大腦：核心對話邏輯分發區塊開始
# ============================
def process_actual_logic(conv_key):
    if conv_key not in message_buffer: return
    
    # 1. 提取緩衝區內的用戶訊息、圖片與 Meta 資料 (user_id, session_id)
    buf = message_buffer.pop(conv_key)
    user_message = (buf.get("text") or "").strip()
    user_image = buf.get("image")
    meta = buf["meta"]
    user_id = meta["user_id"]
    session_id = meta.get("session_id") 

    if not user_message and user_image:
        user_message = "請幫我看看這張圖片，並說明內容或解決其中的問題。"

    # 因為新增計算mcp之後再看看要不要開啟
    ##############################################################################################################
    # 2. 財務防護網：比對是否包含帳務等敏感關鍵字，若有則直接阻擋並建議電話聯絡
    # if any(re.search(p, user_message, re.IGNORECASE) for p in BILLING_PATTERNS):
    #     send_websocket_reply(user_id, session_id, "涉及到帳務、金額與匯款確認，為保障您的權益，AI 無法處理此類問題。請您於上班時間撥打總公司電話，將有專人為您服務。")
    #     return
    ##############################################################################################################


    # 3. 客戶主動請求重啟 AI：清除所有轉接與鎖定狀態，寫入通知並廣播前端切換為 AI 模式
    if any(re.search(p, user_message, re.IGNORECASE) for p in AI_RESTART_PATTERNS_FROM_CLIENT) or user_message == "取消轉接重啟AI":
        human_handoff[conv_key] = False
        handoff_pending.pop(conv_key, None)
        handoff_pending_times.pop(conv_key, None)
        handoff_collect_taxid.pop(conv_key, None)
        handoff_start_times.pop(conv_key, None)
        human_lock.pop(conv_key, None)
        conversation_memory.pop(conv_key, None) 
        
        msg = "【系統通知】真人服務結束，AI 已重新上線。"
        
        if user_message == "取消轉接重啟AI":
            if session_id:
                try:
                    conn = get_db_connection()
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'ai', msg, current_time))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"[DB Error] {e}")
        else:
            # 如果是客戶手動打字 (例如打: 好了)，則正常發送到畫面上
            send_websocket_reply(user_id, session_id, msg)
            
        broadcast_state_change(user_id, 'ai') 
        return

    # 4. 判斷是否處於真人模式 (靜音 AI)：若正在真人服務中，將閒置計時器歸零，並阻擋 AI 回應
    is_silence_mode = human_handoff.get(conv_key) or human_lock.get(conv_key)
    if is_silence_mode and not handoff_collect_taxid.get(conv_key):
        # 只要客人在真人模式下有傳送訊息，就將閒置計時器歸零
        if conv_key in handoff_start_times and handoff_start_times[conv_key].get("handoff_start_time"):
            handoff_start_times[conv_key]["handoff_start_time"] = time.time()
        return

    # 5. 等待統編階段：檢查輸入格式是否正確，錯誤則重置計時器並提示，正確則進入轉接等候
    if handoff_collect_taxid.get(conv_key):
        tax_id_input = user_message.strip()
        if not re.match(TAX_ID_PATTERN, tax_id_input):
            # 其實原本就有重新計時！現在加上明確的文字回饋，讓使用者知道時間重置了。
            if conv_key in handoff_start_times:
                handoff_start_times[conv_key]["taxid_start_time"] = time.time()
            send_websocket_reply(user_id, session_id, "【系統通知】統編格式錯誤，請輸入 8 碼數字 (已為您重新計時 30 秒)：")
            return
        
        handoff_collect_taxid.pop(conv_key, None)
        handoff_context[conv_key]["tax_id"] = tax_id_input
        handoff_start_times[conv_key] = {
            "handoff_start_time": time.time(),
            "taxid_start_time": None,
            "session_id": session_id
        }
        
        send_websocket_reply(user_id, session_id, f"【系統通知】已收到統編「{tax_id_input}」，正在為您轉接真人客服，請稍候...")
        broadcast_state_change(user_id, 'human')  
        return

    # 6. 轉接確認階段：處理使用者對「是否轉接」的回答，若回答其他問題則取消轉接意圖繼續 AI 流程
    if handoff_pending.get(conv_key):
        handoff_pending_times.pop(conv_key, None) # 只要有回應就取消超時計時
        
        if any(re.search(p, user_message, re.IGNORECASE) for p in CONFIRM_YES_PATTERNS):
            handoff_pending.pop(conv_key, None)
            human_handoff[conv_key] = True
            handoff_collect_taxid[conv_key] = True
            handoff_start_times[conv_key] = {
                "taxid_start_time": time.time(),
                "handoff_start_time": None,
                "session_id": session_id
            }
            send_websocket_reply(user_id, session_id, f"【系統通知】好的，請在 {TAXID_COLLECTION_TIMEOUT_SECONDS} 秒內輸入貴公司統編：")
            return
        elif any(re.search(p, user_message, re.IGNORECASE) for p in CONFIRM_NO_PATTERNS):
            handoff_pending.pop(conv_key, None)
            send_websocket_reply(user_id, session_id, "【系統通知】好的，AI 繼續為您服務。")
            return
        else:
            # 如果輸入的不是是/否，代表問了新問題。取消轉接狀態，讓程式碼繼續往下走到 AI 流程！
            handoff_pending.pop(conv_key, None)

    # 7. 觸發轉接意圖：比對使用者是否主動表達找真人的意圖，若有則跳出確認選項與計時
    if any(re.search(p, user_message, re.IGNORECASE) for p in HANDOFF_PATTERNS):
        handoff_pending[conv_key] = True
        handoff_context[conv_key] = {"trigger": user_message, "user_id": user_id}
        handoff_pending_times[conv_key] = {"time": time.time(), "session_id": session_id}
        send_websocket_reply(user_id, session_id, "【系統通知】是否轉接真人客服？", options=["是", "否"])
        return

    # 8. 正常 AI 流程：動態組裝標準 Multi-Agent 格式的訊息陣列
    relevant = search_knowledge(user_message)
    context_str = "\n".join(relevant) if relevant else "無相關資料。"
    now = datetime.now()
    current_time_str = now.strftime("%Y年%m月%d日 %H點%M分 (星期%w)")

    # ========================================================
    # 🚀 航空母艦 (LangGraph) 派工作業開始
    # ========================================================
    # 【SA 資料隔離升級】：不再把 history 和當前問題塞進同一包 messages，
    # 而是拆成兩個獨立艙室送進背包：
    #   - chat_history：過去乾淨的 Q&A，只給 Supervisor / Final_Answer 做語意連貫判斷
    #   - messages：這一輪全新的工作記憶，只有「動態背景 + 當前問題」，
    #               Search_Agent / Math_Agent 只看得到這裡，
    #               physically 不會被上一題殘留的數字污染

    # (A) 組裝過去乾淨對話歷史 (只有 User 問題 + 最終回答，不含任何內部工具紀錄)
    history_messages = []
    history = conversation_memory.get(conv_key, [])
    if history:
        for turn in history:
            history_messages.append(HumanMessage(content=turn['user']))
            history_messages.append(AIMessage(content=turn['ai']))

    # (B) 組裝這一輪全新的工作記憶：動態背景 (時間/RAG) + 當前問題
    system_context = (
        f"【動態系統變數】現在時間：{current_time_str}\n\n"
        f"【目前檢索到的參考知識庫】：\n{context_str}\n\n"
    )
    current_turn_messages = [
        SystemMessage(content=system_context),
        HumanMessage(content=user_message)  # (目前暫不支援圖片傳遞給 LangGraph，純文字處理)
    ]
    print(f"\n===== 🔍 餵給 AI 的參考資料 (Top-{TOP_K}) =====\n{context_str}\n===========================================\n")

    # (C) 將拆分好的兩個艙室裝入初始背包
    # 【SA 新增】：明確初始化 search_calls / math_calls 為 0，
    # 雙重保險，確保這兩個計數器從第一步就有明確的初始值，
    # 避免跨節點延續時發生任何預設值判讀上的疑慮。
    #
    # 【SA v2 新增】：補上任務清單版新增的四個欄位。
    #   plan             -> 由 Planner 節點在圖的第一步產生，這裡先給空清單
    #   facts            -> 事實帳本，每一輪都必須是全新的空字典(這是跨輪次資料隔離的關鍵)
    #   searched_queries -> 本輪已打過 API 的關鍵字，同樣每輪清空
    #   current_step     -> 主管指定的當前任務 id，-1 表示尚未指定
    # 雖然 Planner 節點內部也會再初始化一次，但這裡明確給值是雙重保險，
    # 避免哪天 Planner 拋例外時，下游節點讀到 None 而炸掉。
    initial_state = {
        "chat_history": history_messages,
        "messages": current_turn_messages,
        "retry_count": 0,
        "search_calls": 0,
        "math_calls": 0,
        "plan": [],
        "facts": {},
        "searched_queries": [],
        "current_step": -1
    }
    ollama_resp = ""

    print(f"\n[狀態機] 🎒 背包打包完成 (歷史 {len(history_messages)} 則 / 本輪 {len(current_turn_messages)} 則)，送入航空母艦...")
    
    try:
        # (D) 啟動 LangGraph 地圖！
        # 【SA 調整】：recursion_limit 從 15 提高到 25 當作第二道防線。
        # 真正防止失控的關鍵是 search_calls/math_calls 的總次數上限(已修復)，
        # 這裡只是多留一點緩衝空間，避免正常的多實體查詢(例如查兩家公司)
        # 因為步數剛好卡在邊界而被誤傷。
        for output in app_graph.stream(initial_state, {"recursion_limit": 25}): #  限制步數 25 步
            for key, value in output.items():
                # 我們只關心最後一個房間 (Final_Answer) 的結果
                if key == "Final_Answer":
                    ollama_resp = value['messages'][-1].content
    except Exception as e:
        print(f"[系統錯誤] 航空母艦運作異常: {e}")
        ollama_resp = "【系統通知】抱歉，AI 系統處理您的請求時發生錯誤。"
    
    # ========================================================
    # 🚀 航空母艦 (LangGraph) 派工作業結束
    # ========================================================
    
    # 9. 後處理與防呆：清理 AI 偷漏出來的內部標籤字眼，並寫入短期記憶
    if ollama_resp == "AI_IMAGE_ERROR":
        ollama_resp = "【系統通知】抱歉，圖片讀取異常，請重新傳送，或者直接用文字向我描述遇到的問題呢？"
    elif not ollama_resp: 
        ollama_resp = "【系統通知】抱歉，AI 暫時無法回應。"
    else:
        # 清除 AI 偷漏出來的標籤，例如: （【資料分類：XXX】） 或 【參考來源：XXX】
        ollama_resp = re.sub(r'[（\(]?【資料分類.*?】[）\)]?', '', ollama_resp)
        ollama_resp = re.sub(r'[（\(]?【參考來源.*?】[）\)]?', '', ollama_resp)
        # 濾除類似 "這個回答屬於【資料分類...】" 的多餘解釋碎語
        ollama_resp = re.sub(r'這個回答屬於.*?。?', '', ollama_resp)
        ollama_resp = ollama_resp.strip()

    # 【SA 說明】：conversation_memory 只存「乾淨的」使用者問題 + 最終回答，
    # 不含任何 Search_Agent / Math_Agent 的內部工作紀錄，
    # 這是能讓 chat_history 保持乾淨、不會二次污染下一輪 messages 的關鍵。
    if conv_key not in conversation_memory: 
        conversation_memory[conv_key] = []
    conversation_memory[conv_key].append({"user": user_message, "ai": ollama_resp})
    if len(conversation_memory[conv_key]) > MAX_HISTORY_TURNS:
        conversation_memory[conv_key].pop(0) 

    # 10. AI 信心評估：若知識庫不足或 AI 含有不確定字眼，主動於句尾補上詢問轉接真人的選項
    if needs_contact_footer(relevant, ollama_resp):
        handoff_pending[conv_key] = True
        handoff_context[conv_key] = {"trigger": user_message, "user_id": user_id}
        handoff_pending_times[conv_key] = {"time": time.time(), "session_id": session_id}
        send_websocket_reply(user_id, session_id, ollama_resp + "\n\n(資訊不足，是否轉接真人？)", options=["是", "否"])
        return
    
    send_websocket_reply(user_id, session_id, ollama_resp)
# ============================
# 共用大腦：核心對話邏輯分發區塊結束
# ============================

# ============================
# 訊息佇列與緩衝處理區塊開始
# ============================
def handle_message_logic_with_buffer(user_id, session_id, user_message, user_image=None):
    conv_key = get_conv_key(user_id)
    
    # 1. 檢查對話是否已在緩衝區，若有則取消舊計時器，並將新訊息與舊訊息進行拼接
    if conv_key in message_buffer:
        message_buffer[conv_key]["timer"].cancel()
        if user_message:
            message_buffer[conv_key]["text"] = message_buffer[conv_key].get("text", "") + "，" + user_message 
        if user_image:
            message_buffer[conv_key]["image"] = user_image
    else:
        # 2. 若為全新訊息，則建立緩衝區字典並記錄 user_id 與 session_id 等 Meta 資料
        message_buffer[conv_key] = {"text": user_message or "", "image": user_image, "meta": { "user_id": user_id, "session_id": session_id }}
    
    # 3. 設定計時器等待時間並啟動背景執行緒 (純文字 0.5 秒，圖片 10 秒)
    wait_time = IMAGE_BUFFER_SECONDS if message_buffer[conv_key].get("image") else 0.5
    timer = threading.Timer(wait_time, process_actual_logic, args=[conv_key])
    message_buffer[conv_key]["timer"] = timer
    timer.start()
# ============================
# 訊息佇列與緩衝處理區塊結束
# ============================

# ============================
# 背景超時檢查執行緒開始
# ============================
def check_timeouts():
    # 1. 建立無窮迴圈，持續監控直到收到系統停止的 Event 旗標為止
    while not timeout_checker_stop.is_set():
        # 2. 檢查「統編輸入」與「真人對話閒置」是否超時，若超時則強制解除狀態並切回 AI 模式
        keys_to_check = list(handoff_start_times.keys())
        for conv_key in keys_to_check:
            try:
                data = handoff_start_times.get(conv_key)
                if not data or human_lock.get(conv_key): continue 

                session_id = data.get("session_id")
                
                if handoff_collect_taxid.get(conv_key) and data.get("taxid_start_time"):
                    if time.time() - data["taxid_start_time"] > TAXID_COLLECTION_TIMEOUT_SECONDS:
                        human_handoff[conv_key] = False
                        handoff_collect_taxid.pop(conv_key, None)
                        handoff_start_times.pop(conv_key, None)
                        send_websocket_reply(conv_key, session_id, "【系統通知】統編輸入超時，已自動切回 AI。")
                        broadcast_state_change(conv_key, 'ai')

                elif human_handoff.get(conv_key) and data.get("handoff_start_time"):
                    if time.time() - data["handoff_start_time"] > HANDOFF_TIMEOUT_SECONDS:
                        human_handoff[conv_key] = False
                        handoff_start_times.pop(conv_key, None)
                        send_websocket_reply(conv_key, session_id, "【系統通知】客服忙線中，已自動切回 AI。")
                        broadcast_state_change(conv_key, 'ai')
            except: pass
                
        # 3. 檢查「選項等待 (是否轉接真人)」是否超時 (固定 30 秒)，若未回應則自動取消轉接意圖
        keys_to_check_pending = list(handoff_pending_times.keys())
        for conv_key in keys_to_check_pending:
            try:
                data = handoff_pending_times.get(conv_key)
                if data and time.time() - data["time"] > 30:
                    session_id = data.get("session_id")
                    handoff_pending.pop(conv_key, None)
                    handoff_pending_times.pop(conv_key, None)
                    send_websocket_reply(conv_key, session_id, "【系統通知】選擇超時，AI 繼續為您服務。")
            except: pass
                
        # 4. 迴圈休眠 1 秒以降低 CPU 負載
        time.sleep(1)
# ============================
# 背景超時檢查執行緒結束
# ============================
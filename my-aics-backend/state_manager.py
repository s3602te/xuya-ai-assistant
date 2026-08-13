# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import re
import json
import requests
import time
import threading 
from datetime import datetime

# 引入自訂模組
from config import *
from ai_core import search_knowledge, get_ollama_response, needs_contact_footer
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# 記憶體狀態與未解耦之狀態機開始
# ============================
# 1. 記憶體狀態變數 (用於記錄每個使用者的轉接進度與對話鎖定)
human_handoff = {}       
handoff_pending = {}     
handoff_collect_taxid = {} 
handoff_context = {}     
human_lock = {}          
handoff_start_times = {} 
timeout_checker_stop = threading.Event()

# 2. 短期記憶緩衝區配置 (限制回溯回合數與訊息截流秒數)
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
def get_conv_key(user_id, group_id=None):
    if group_id and group_id.startswith(("C", "G", "R")): 
        return f"{group_id}:{user_id}"
    return user_id

def is_working_hours() -> bool:
    local_time = time.localtime()
    current_date_str = time.strftime("%Y-%m-%d", local_time)
    
    if current_date_str in HOLIDAYS:
        print(f"[TimeCheck] 今日是國定假日 ({current_date_str})，暫停真人服務。")
        return False

    current_hour = local_time.tm_hour
    current_day = local_time.tm_wday 
    
    is_day = current_day in WORKING_DAYS
    is_hour = WORKING_HOURS_START <= current_hour < WORKING_HOURS_END
    return is_day and is_hour

def send_via_iis(channel_id, user_id, group_id, reply_token, message, buttons=None):
    service_name = "Reply" if reply_token else "Push"
    to_target = group_id if group_id else user_id

    data_payload = {
        "channelId": channel_id or "",
        "to": to_target or "",
        "message": message or ""
    }
    
    if reply_token:
        data_payload["replyToken"] = reply_token

    if buttons and isinstance(buttons, list):
        data_payload["buttons"] = buttons

    payload = {
        "Request": {
            "Header": {
                "Version": "1.0",
                "ApiUserId": IIS_API_USER_ID,
                "ServiceName": service_name 
            },
            "Data": data_payload
        }
    }

    print(f"[Send] 發送 ({service_name}) To:{to_target} Msg:{message[:10]}... Buttons:{bool(buttons)}")
    
    try:
        r = requests.post(IIS_SEND_URL, json=payload, timeout=10, verify=False)
        r.raise_for_status()
        resp = r.json().get("Response", {}).get("Header", {})
        if resp.get("ServiceResult") == "N":
            print(f"[Send] 失敗: {resp.get('StatusDesc')} (Code: {resp.get('StatusCode')})")
        else:
            print(f"[Send] 成功！")
    except Exception as e:
        print(f"[Send] 連線錯誤: {e}")

def get_yes_no_buttons():
    return [
        {"label": "是，轉接真人", "text": "是"},
        {"label": "否，繼續AI", "text": "否"}
    ]

def get_user_profile(user_id):
    return {"displayName": user_id} 

def format_support_notification(display_name, user_id, tax_id=None):
    dn = display_name or "用戶"
    uid = user_id or "無ID"
    return (f"【轉人工通知】\n用戶：{dn} ({uid})\n統編：{tax_id or '未提供'}\n"
            f"指令：\n!停止AI {uid}\n!轉回AI {uid}")
# ============================
# 核心工具函式區塊結束
# ============================

# ============================
# 共用大腦：核心對話邏輯分發區塊開始
# ============================
def process_actual_logic(conv_key):
    if conv_key not in message_buffer: return
    
    buf = message_buffer.pop(conv_key)
    user_message = (buf.get("text") or "").strip()
    user_image = buf.get("image")
    meta = buf["meta"]
    
    channel_id = meta["channel_id"]
    user_id = meta["user_id"]
    reply_token = meta["reply_token"] 
    group_id = meta["group_id"]
    push_target_id = group_id if group_id else user_id

    if not user_message and user_image:
        user_message = "請幫我看看這張圖片，並說明內容或解決其中的問題。"

    if any(re.search(p, user_message, re.IGNORECASE) for p in BILLING_PATTERNS):
        print(f"[{conv_key}] 觸發財務防護網，已攔截。")
        send_via_iis(channel_id, user_id, group_id, "", "涉及到帳務、金額與匯款確認，為保障您的權益，AI 無法處理此類問題。請您於上班時間撥打總公司電話 (02)8511-1288，將有專人為您服務。")
        return

    is_ai_restart = any(re.search(p, user_message, re.IGNORECASE) for p in AI_RESTART_PATTERNS_FROM_CLIENT)
    if is_ai_restart:
        human_handoff[conv_key] = False
        handoff_pending.pop(conv_key, None)
        handoff_collect_taxid.pop(conv_key, None)
        handoff_context.pop(conv_key, None)
        handoff_start_times.pop(conv_key, None)
        human_lock.pop(conv_key, None)
        conversation_memory.pop(conv_key, None) 
        send_via_iis(channel_id, user_id, group_id, reply_token, "【系統通知】AI 客服將重新為您服務。")
        return

    if re.search(HANDOFF_CLEAR_PATTERN, user_message, re.IGNORECASE):
        if human_handoff.get(conv_key) and conv_key in handoff_start_times:
            handoff_start_times[conv_key]["handoff_start_time"] = None
        return

    if LINE_INTERNAL_GROUP_ID and group_id == LINE_INTERNAL_GROUP_ID:
        match_cmd = re.match(r"!(轉回AI|停止AI)\s+(U[0-9a-fA-F]{32})", user_message.strip(), re.IGNORECASE)
        if match_cmd:
            cmd = match_cmd.group(1).upper()
            target_uid = match_cmd.group(2)
            
            found_key = None
            active_keys = list(set(list(human_handoff.keys()) + list(handoff_collect_taxid.keys())))
            for k in active_keys:
                if k == target_uid or k.endswith(f":{target_uid}"):
                    found_key = k
                    break
            if not found_key: found_key = target_uid 
            
            target_channel_id = handoff_start_times.get(found_key, {}).get("channel_id", channel_id)

            if cmd == "轉回AI":
                human_handoff[found_key] = False
                handoff_pending.pop(found_key, None)
                handoff_collect_taxid.pop(found_key, None)
                human_lock.pop(found_key, None)
                handoff_start_times.pop(found_key, None)
                conversation_memory.pop(found_key, None)
                
                send_via_iis(target_channel_id, target_uid, "", "", "✅ 真人服務結束，AI 已重新上線。")
                send_via_iis(channel_id, user_id, group_id, reply_token, f"✅ 已將 {target_uid} 轉回 AI。")
            
            elif cmd == "停止AI":
                human_lock[found_key] = True
                if found_key in handoff_start_times:
                    handoff_start_times[found_key]["handoff_start_time"] = None
                    handoff_start_times[found_key]["taxid_start_time"] = None
                handoff_collect_taxid.pop(found_key, None)
                send_via_iis(channel_id, user_id, group_id, reply_token, f"✅ 已對 {target_uid} 設定永久接手鎖定。")
            return

    is_silence_mode = human_handoff.get(conv_key) or human_lock.get(conv_key)
    if is_silence_mode and not handoff_collect_taxid.get(conv_key):
        print(f"[{conv_key}] AI 靜音中 (Handoff:{human_handoff.get(conv_key)}, Lock:{human_lock.get(conv_key)})")
        return

    if handoff_collect_taxid.get(conv_key):
        tax_id_input = user_message.strip()
        if not re.match(TAX_ID_PATTERN, tax_id_input):
            send_via_iis(channel_id, user_id, group_id, reply_token, "統編格式錯誤，請輸入 8 碼數字：")
            return
        
        handoff_collect_taxid.pop(conv_key, None)
        handoff_context[conv_key]["tax_id"] = tax_id_input
        handoff_start_times[conv_key] = {
            "push_target_id": push_target_id,
            "channel_id": channel_id, 
            "handoff_start_time": time.time(),
            "taxid_start_time": None
        }

        if LINE_INTERNAL_GROUP_ID:
            prof = get_user_profile(user_id)
            notify = format_support_notification(prof.get("displayName"), user_id, tax_id_input)
            send_via_iis(channel_id, "", LINE_INTERNAL_GROUP_ID, "", notify)
        
        msg = f"已收到統編「{tax_id_input}」，正在為您轉接真人客服。AI 已靜音。"
        send_via_iis(channel_id, user_id, group_id, "", msg)
        return

    if handoff_pending.get(conv_key):
        if any(re.search(p, user_message, re.IGNORECASE) for p in CONFIRM_YES_PATTERNS):
            if not is_working_hours():
                handoff_pending.pop(conv_key, None)
                send_via_iis(channel_id, user_id, group_id, reply_token, "抱歉，目前是非上班時間，無法轉接真人。")
                return

            handoff_pending.pop(conv_key, None)
            human_handoff[conv_key] = True
            handoff_collect_taxid[conv_key] = True
            handoff_start_times[conv_key] = {
                "push_target_id": push_target_id,
                "channel_id": channel_id,
                "taxid_start_time": time.time(),
                "handoff_start_time": None
            }
            send_via_iis(channel_id, user_id, group_id, reply_token, f"好的，請在 {TAXID_COLLECTION_TIMEOUT_SECONDS} 秒內輸入貴公司統編：")
            return
        elif any(re.search(p, user_message, re.IGNORECASE) for p in CONFIRM_NO_PATTERNS):
            handoff_pending.pop(conv_key, None)
            send_via_iis(channel_id, user_id, group_id, reply_token, "好的，AI 繼續為您服務。")
            return
        else:
            send_via_iis(channel_id, user_id, group_id, reply_token, "請選擇是否轉接真人？", buttons=get_yes_no_buttons())
            return

    if any(re.search(p, user_message, re.IGNORECASE) for p in HANDOFF_PATTERNS):
        if not is_working_hours():
            send_via_iis(channel_id, user_id, group_id, reply_token, "抱歉，目前是非上班時間，無法轉接真人。")
            return
        handoff_pending[conv_key] = True
        handoff_context[conv_key] = {"trigger": user_message, "user_id": user_id}
        send_via_iis(channel_id, user_id, group_id, reply_token, "是否轉接真人客服？", buttons=get_yes_no_buttons())
        return

    relevant = search_knowledge(user_message)
    context_str = "\n".join(relevant) if relevant else "無相關資料。"
    
    now = datetime.now()
    current_time_str = now.strftime("%Y年%m月%d日 %H點%M分 (星期%w)")

    history = conversation_memory.get(conv_key, [])
    history_str = ""
    if history:
        history_str = "【前情提要】\n"
        for idx, turn in enumerate(history):
            history_str += f"[回合 {idx+1}]\n用戶問：{turn['user']}\nAI答：{turn['ai']}\n"
        history_str += "------------------\n"

    print(f"\n===== 🔍 餵給 AI 的參考資料 (Top-{TOP_K}) =====\n{context_str}\n===========================================\n")

    if user_image:
        prompt = (
            f"【參考知識庫】（若知識庫內有提及解法，請嚴格按照其中步驟；若無，請絕對不要自行發明步驟）：\n"
            f"{context_str}\n\n"
            f"【用戶問題】：{user_message}\n\n"
            f"請嚴格按照以下兩段式格式回覆：\n"
            f"🔍 圖片解析：(先印出你肉眼真實看到的內容，絕不可瞎掰。若模糊難辨，請誠實說明並要求提供更清楚的照片)\n"
            f"💡 客服回應：(根據解析結果與參考知識庫，給予用戶專業且安全的建議)"
        )
        ollama_resp = get_ollama_response(prompt, user_image, "XUYA:latest")
    else:
        prompt = (
            f"{history_str}現在時間：{current_time_str}\n\n"
            f"【系統強制指令】\n"
            f"1. 請務必使用「繁體中文 (Traditional Chinese)」進行回答。\n"
            f"2. 參考知識庫中帶有【資料分類：XXX】的標籤。請『優先針對用戶提問中提及的分類或機型』進行回答。\n"
            f"3. 若參考知識庫中明確包含用戶所詢問類別的資訊，請直接給出答案，不要再反問。\n"
            f"4. 【極重要防護網】當參考知識庫內包含「多個不同分類（或機型）」的解答，且用戶先前的對話與本次問題中都【沒有提到任何分類特徵】時，你絕對不可以瞎猜或給出綜合答案！你必須且只能回覆：「請問您詢問的是哪一種服務或機型呢（例如：客服退換貨、V12收銀機）？」\n"
            f"5. 嚴格依據參考知識庫回答，絕對不可自行編造步驟。\n\n"
            f"【參考知識庫】：\n{context_str}\n\n"
            f"請回答最新問題：{user_message}"
        )
        ollama_resp = get_ollama_response(prompt, None, "XUYA:latest")
    
    if ollama_resp == "AI_IMAGE_ERROR":
        ollama_resp = "【系統通知】\n抱歉，我剛剛無法順利讀取您傳送的圖片（可能是檔案格式不支援或連線異常）。\n能不能請您再重新傳送一次清晰的圖片，或者直接用文字向我描述遇到的問題呢？"
    elif not ollama_resp: 
        ollama_resp = "抱歉，AI 暫時無法回應。"
    else:
        ollama_resp += "\n\n※ 本回覆由 AI 自動產生，僅供參考，實際操作請以產品手冊或專人指導為準。"

    if conv_key not in conversation_memory: 
        conversation_memory[conv_key] = []
    conversation_memory[conv_key].append({"user": user_message, "ai": ollama_resp})
    if len(conversation_memory[conv_key]) > MAX_HISTORY_TURNS:
        conversation_memory[conv_key].pop(0) 

    if needs_contact_footer(relevant, ollama_resp):
        if not is_working_hours():
            send_via_iis(channel_id, user_id, group_id, reply_token, ollama_resp + "\n\n(目前非上班時間，無法轉接真人)")
            return

        handoff_pending[conv_key] = True
        handoff_context[conv_key] = {"trigger": user_message, "user_id": user_id}
        full_reply = ollama_resp + "\n\n(資訊不足，是否轉接真人？)"
        send_via_iis(channel_id, user_id, group_id, reply_token, full_reply, buttons=get_yes_no_buttons())
        return
    
    send_via_iis(channel_id, user_id, group_id, reply_token, ollama_resp)
# ============================
# 共用大腦：核心對話邏輯分發區塊結束
# ============================

# ============================
# 訊息佇列與緩衝處理區塊開始
# ============================
def handle_message_logic_with_buffer(channel_id, user_id, reply_token, user_message, group_id=None, user_image=None):
    conv_key = get_conv_key(user_id, group_id)
    
    if LINE_INTERNAL_GROUP_ID and group_id == LINE_INTERNAL_GROUP_ID and user_message and user_message.startswith("!"):
        message_buffer[conv_key] = {
            "text": user_message, "image": None,
            "meta": {"channel_id": channel_id, "user_id": user_id, "reply_token": reply_token, "group_id": group_id}
        }
        process_actual_logic(conv_key)
        return

    if conv_key in message_buffer:
        message_buffer[conv_key]["timer"].cancel()
        if user_message:
            if message_buffer[conv_key].get("text"):
                message_buffer[conv_key]["text"] += "，" + user_message 
            else:
                message_buffer[conv_key]["text"] = user_message
        if user_image:
            message_buffer[conv_key]["image"] = user_image
    else:
        message_buffer[conv_key] = {
            "text": user_message or "",
            "image": user_image,
            "meta": {
                "channel_id": channel_id, "user_id": user_id, 
                "reply_token": reply_token, "group_id": group_id
            }
        }
    
    wait_time = IMAGE_BUFFER_SECONDS if message_buffer[conv_key].get("image") else BUFFER_SECONDS
    print(f"[{conv_key}] 訊息/圖片收容中...等待 {wait_time} 秒")
    
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
    while not timeout_checker_stop.is_set():
        keys_to_check = list(handoff_start_times.keys())
        for conv_key in keys_to_check:
            try:
                data = handoff_start_times.get(conv_key)
                if not data: continue
                if human_lock.get(conv_key): continue 

                push_target_id = data.get("push_target_id")
                saved_channel_id = data.get("channel_id", "") 

                if handoff_collect_taxid.get(conv_key) and data.get("taxid_start_time"):
                    if time.time() - data["taxid_start_time"] > TAXID_COLLECTION_TIMEOUT_SECONDS:
                        human_handoff[conv_key] = False
                        handoff_collect_taxid.pop(conv_key, None)
                        handoff_context.pop(conv_key, None)
                        handoff_start_times.pop(conv_key, None)
                        human_lock.pop(conv_key, None)
                        send_via_iis(saved_channel_id, push_target_id, "", "", "【系統通知】統編輸入超時，已自動切回 AI。")

                elif human_handoff.get(conv_key) and data.get("handoff_start_time"):
                    if time.time() - data["handoff_start_time"] > HANDOFF_TIMEOUT_SECONDS:
                        human_handoff[conv_key] = False
                        handoff_collect_taxid.pop(conv_key, None)
                        handoff_context.pop(conv_key, None)
                        handoff_start_times.pop(conv_key, None)
                        human_lock.pop(conv_key, None)
                        send_via_iis(saved_channel_id, push_target_id, "", "", "【系統通知】客服忙線中，已自動切回 AI。")
            except Exception as e:
                print(f"[Timeout] Error: {e}")
        time.sleep(1)
# ============================
# 背景超時檢查執行緒結束
# ============================
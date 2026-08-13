# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import os
import re
import json
import requests
import time
import threading 
import atexit 
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
# ============================
# 核心模組與套件引入結束
# ============================

# ============================
# 模組化引入區塊 (取代原本冗長的設定與資料庫)
# ============================
from config import *
from database import get_db_connection, collection_manual, collection_auto

# ============================
# 環境變數與全域初始化開始
# ============================
def pick_device():
    try:
        if torch.cuda.is_available():
            _ = torch.randn(1, device='cuda') * 2
            torch.cuda.synchronize()
            print("[Device] Using CUDA")
            return 'cuda'
    except Exception as e:
        print(f"[Device] CUDA 不可用，改用 CPU：{e}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[Device] Using CPU")
    return 'cpu'

DEVICE = pick_device()

app = Flask(__name__, static_folder='dist', static_url_path='')
CORS(app) 
# ============================
# 環境變數與全域初始化結束
# ============================

# ============================
# 記憶體狀態與未解耦之模型連線開始
# ============================
# 1. 記憶體狀態變數 (用於記錄每個使用者的轉接進度與對話鎖定)
human_handoff = {}       
handoff_pending = {}     
handoff_collect_taxid = {} 
handoff_context = {}     
human_lock = {}          
handoff_start_times = {} 
timeout_checker_stop = threading.Event()

# 2. 載入 Embedding 模型用於自然語言向量化
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

# (ChromaDB 連線已移至 database.py)

# 3. 短期記憶緩衝區配置 (限制回溯回合數與訊息截流秒數)
MAX_HISTORY_TURNS = 2
conversation_memory = {} 

BUFFER_SECONDS = 5 
IMAGE_BUFFER_SECONDS = 10 
message_buffer = {} 
# ============================
# 記憶體狀態與未解耦之模型連線結束
# ============================

# ============================
# 核心工具函式區塊開始
# ============================
def get_conv_key(user_id, group_id=None):
    # 若來自群組，組合群組 ID 與使用者 ID，當作 Key，確保狀態唯一性
    if group_id and group_id.startswith(("C", "G", "R")): 
        return f"{group_id}:{user_id}"
    return user_id

def is_working_hours() -> bool:
    # 1. 取得目前系統時間並格式化
    local_time = time.localtime()
    
    # 先抓出今天的日期字串 (這是正常的程式碼，測試時註解)
    current_date_str = time.strftime("%Y-%m-%d", local_time)
    
    # [測試模式區塊保留]
    # print("[測試模式] 強制將今天偽裝成 2026-02-16") 
    # current_date_str = "2026-02-16" 
    
    # 2. 優先驗證國定假日清單，直接判定為休息 (False)
    if current_date_str in HOLIDAYS:
        print(f"[TimeCheck] 今日是國定假日 ({current_date_str})，暫停真人服務。")
        return False

    # 3. 如果不是國定假日，才檢查是不是平日的上班時間
    current_hour = local_time.tm_hour
    current_day = local_time.tm_wday 
    # 0=週一, 6=週日
    
    is_day = current_day in WORKING_DAYS
    is_hour = WORKING_HOURS_START <= current_hour < WORKING_HOURS_END
    return is_day and is_hour

def send_via_iis(channel_id, user_id, group_id, reply_token, message, buttons=None):
    # 1. 決定服務類型 (有 reply_token 即為 Reply，否則為主動 Push)
    """
    channel_id: 分店代號
    user_id/group_id: 用來決定 'to'
    reply_token: 決定是 Reply 還是 Push
    message: 文字內容
    buttons: (List) [{"label":"顯示文字", "text":"傳送文字"}, ...]
    """
    service_name = "Reply" if reply_token else "Push"

    # 2. 決定接收對象 (優先使用群組 ID)
    to_target = group_id if group_id else user_id

    # 3. 建構 Data (前輩的新格式)
    data_payload = {
        "channelId": channel_id or "",
        "to": to_target or "",      # ★ 這裡改成了 to
        "message": message or ""
    }
    
    # 如果是 Reply，還是要帶 Token
    if reply_token:
        data_payload["replyToken"] = reply_token

    # ★ 如果有按鈕，就加進去
    if buttons and isinstance(buttons, list):
        data_payload["buttons"] = buttons

    # 4. 組合最終信封，封裝最終傳輸的 API Request 格式
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
    print(f"[Debug] 傳送內容: {json.dumps(payload, ensure_ascii=False)}")
    
    # 5. 執行 HTTP POST 請求至 IIS 伺服器
    try:
        r = requests.post(IIS_SEND_URL, json=payload, timeout=10, verify=False)
        r.raise_for_status()
        
        # 檢查回傳
        resp = r.json().get("Response", {}).get("Header", {})
        if resp.get("ServiceResult") == "N":
            print(f"[Send] 失敗: {resp.get('StatusDesc')} (Code: {resp.get('StatusCode')})")
        else:
            print(f"[Send] 成功！")
    except Exception as e:
        print(f"[Send] 連線錯誤: {e}")

# 產生標準的二選一確認按鈕 (支援 LINE UI) 產生按鈕清單的工具 (配合前輩格式)
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
# RAG 知識庫檢索與 AI 模型互動區塊開始
# ============================
# def load_knowledge():
#     global knowledge_data, faiss_index
#     try:
#         if not os.path.exists(KNOWLEDGE_FILE_PATH): return
#         knowledge_data = pd.read_csv(KNOWLEDGE_FILE_PATH)
#         texts = knowledge_data['question'].tolist() + knowledge_data['answer'].tolist()
#         embeddings = embedding_model.encode(texts, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
#         faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#         faiss_index.add(embeddings)
#         print("知識庫載入完成。")
#     except Exception as e:
#         print(f"知識庫載入失敗: {e}")

def search_knowledge(query, top_k=TOP_K):
    if collection_manual is None or collection_auto is None:
        return []
        
    try:
        # 1. 將使用者的問題轉成向量
        qv = embedding_model.encode([query]).tolist()
        res = []
        
        # 2. Stage 1: 優先向 A 軌 (高精準手動資料庫) 進行嚴格檢索
        results_manual = collection_manual.query(
            query_embeddings=qv,
            n_results=1 # 高精準區通常取最像的 1 筆即可
        )
        
        # 檢查 A 軌是否有資料，且 L2 距離是否小於我們設定的嚴格閾值
        if results_manual['distances'] and len(results_manual['distances'][0]) > 0:
            best_dist = results_manual['distances'][0][0]
            print(f"[檢索路由] 查找高精準區，最佳距離分數為: {best_dist:.3f}")
            
            # 收緊判定閾值確保極高精準度，避免答非所問
            # 因為我們現在只比對「短問題」，分數會極低且精準
            # 將門檻收緊到 1.5 或 2.0 左右，確保超高精準度，避免誤判
            ROUTING_THRESHOLD = 2.0 
            
            if best_dist < ROUTING_THRESHOLD:
                # 🎯 命中高精準區！從 metadata 中直接提取標準答案
                matched_q = results_manual['documents'][0][0]
                matched_a = results_manual['metadatas'][0][0].get('answer', '無對應解答')                
                # 組合給 AI 參考的標準格式
                formatted_ans = f"【標準問題】{matched_q}\n【標準解答】{matched_a}"
                res.append(formatted_ans)
                
                print("[檢索路由] 🎯 命中高精準區，直接回傳標準答案。")
                return res

        # 3. Stage 2: Fallback 機制 - 轉向 B 軌 (自動擴展資料庫) 進行模糊搜尋
        # 只有當 A 軌找不到，或是距離分數太差 (大於 ROUTING_THRESHOLD) 時，才會走到這裡
        print("[檢索路由] ⚠️ 高精準區查無結果，啟動 Fallback 翻閱參考說明書...")
        results_auto = collection_auto.query(
            query_embeddings=qv,
            n_results=top_k
        )
        
        if results_auto['documents'] and len(results_auto['documents'][0]) > 0:
            for doc, meta in zip(results_auto['documents'][0], results_auto['metadatas'][0]):
                source = meta.get("source", "未知說明書")
                res.append(f"【參考來源：{source}】\n{doc}")
                
        return res
    except Exception as e:
        print(f"[搜尋錯誤] {e}")
        return []
    
# def search_knowledge(query, top_k=TOP_K):
#     if knowledge_data is None or faiss_index is None: return []
#     try:
#         qv = embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
#         D, I = faiss_index.search(qv, top_k)
#         res = []
#         nq = len(knowledge_data['question'])
#         for i in I[0]:
#             if i < nq: res.append(f"問: {knowledge_data['question'].iloc[i]}\n答: {knowledge_data['answer'].iloc[i]}")
#             elif i >= nq: res.append(f"答: {knowledge_data['answer'].iloc[i-nq]}")
#         return res
#     except: return []

def needs_contact_footer(relevant_knowledge, ai_text: str) -> bool:
    # 判斷 AI 回覆內容是否具有不確定性，以決定是否觸發真人轉接按鈕
    if not relevant_knowledge: return True
    markers = ["抱歉", "無法提供", "不知道", "不清楚"]
    return any(m in ai_text for m in markers)

# ==========================================
# 雙腦外掛修改區：動態切換大腦模型
# ==========================================
def get_ollama_response(prompt, image_b64=None, model_name="XUYA:latest"):
    try:
        # 1. 關閉 stream 模式，確保 AI 可以看完整張圖並深思熟慮再回答
        payload = {
            "prompt": prompt, 
            "model": model_name, 
            "stream": False,
            "options": {
                "num_predict": 1024  # 強制允許它生成更多的字，避免講到一半斷掉
            }
        }
        
        # 2. 判斷是否有夾帶圖片，執行多模態 (Multimodal) 處理
        if image_b64:
            payload["images"] = [image_b64]
            
        r = requests.post(f"{OLLAMA_API_BASE_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get('response', '').strip()

    except Exception as e:
        print(f"[Ollama 錯誤] {e}")
        # 3. 容錯機制：如果是帶圖片時發生錯誤(圖片處理異常時)，回傳特定錯誤碼
        if image_b64:
            return "AI_IMAGE_ERROR"
        return "AI 通訊錯誤。"
# ============================
# RAG 知識庫檢索與 AI 模型互動區塊結束
# ============================


# ============================
# 共用大腦：核心對話邏輯分發區塊(開始更新按鈕邏輯 + 視覺邏輯)
# ============================
def process_actual_logic(conv_key):
    if conv_key not in message_buffer: return
    
    # 1. 提取緩衝區 (Buffer) 文字與圖片內的統整資料，並做防呆處理 (避免 null crash)
    buf = message_buffer.pop(conv_key)
    user_message = (buf.get("text") or "").strip()
    user_image = buf.get("image")
    meta = buf["meta"]
    
    channel_id = meta["channel_id"]
    user_id = meta["user_id"]
    reply_token = meta["reply_token"] 
    group_id = meta["group_id"]
    push_target_id = group_id if group_id else user_id

    # 2. 視覺防呆：如果只有傳送圖片而無文字，自動補齊系統提示詞
    if not user_message and user_image:
        user_message = "請幫我看看這張圖片，並說明內容或解決其中的問題。"

    # 3. 【財務防護網】攔截高風險關鍵字
    if any(re.search(p, user_message, re.IGNORECASE) for p in BILLING_PATTERNS):
        print(f"[{conv_key}] 觸發財務防護網，已攔截。")
        send_via_iis(channel_id, user_id, group_id, "", "涉及到帳務、金額與匯款確認，為保障您的權益，AI 無法處理此類問題。請您於上班時間撥打總公司電話 (02)8511-1288，將有專人為您服務。")
        return

    # 4. 偵測使用者主動請求重啟 AI
    is_ai_restart = any(re.search(p, user_message, re.IGNORECASE) for p in AI_RESTART_PATTERNS_FROM_CLIENT)
    if is_ai_restart:
        human_handoff[conv_key] = False
        handoff_pending.pop(conv_key, None)
        handoff_collect_taxid.pop(conv_key, None)
        handoff_context.pop(conv_key, None)
        handoff_start_times.pop(conv_key, None)
        human_lock.pop(conv_key, None)
        conversation_memory.pop(conv_key, None) # 清空記憶
        send_via_iis(channel_id, user_id, group_id, reply_token, "【系統通知】AI 客服將重新為您服務。")
        return

    # 5. 真人客服宣告接手 (清除計時器)
    if re.search(HANDOFF_CLEAR_PATTERN, user_message, re.IGNORECASE):
        if human_handoff.get(conv_key) and conv_key in handoff_start_times:
            handoff_start_times[conv_key]["handoff_start_time"] = None
        return

    # 6. 內部群組專用管理指令 (限定內部群組強制切換 AI 狀態)
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
                conversation_memory.pop(found_key, None) # 清空記憶
                
                # 通知客戶 (使用 to)
                send_via_iis(target_channel_id, target_uid, "", "", "✅ 真人服務結束，AI 已重新上線。")
                # 回覆群組
                send_via_iis(channel_id, user_id, group_id, reply_token, f"✅ 已將 {target_uid} 轉回 AI。")
            
            elif cmd == "停止AI":
                human_lock[found_key] = True
                if found_key in handoff_start_times:
                    handoff_start_times[found_key]["handoff_start_time"] = None
                    handoff_start_times[found_key]["taxid_start_time"] = None
                handoff_collect_taxid.pop(found_key, None)
                send_via_iis(channel_id, user_id, group_id, reply_token, f"✅ 已對 {target_uid} 設定永久接手鎖定。")
            return

    # 7. 判斷人工模式AI 靜音邏輯 (等待統編狀態例外放行)
    # 修改：只要「在人工模式」或是「被永久鎖定(停止AI)」，AI 都要閉嘴
    # 唯一的例外是「正在等待統編」，那時候還是要讓程式處理統編邏輯
    is_silence_mode = human_handoff.get(conv_key) or human_lock.get(conv_key)
    if is_silence_mode and not handoff_collect_taxid.get(conv_key):
        # 這裡可以選擇要不要印 Log，方便除錯
        print(f"[{conv_key}] AI 靜音中 (Handoff:{human_handoff.get(conv_key)}, Lock:{human_lock.get(conv_key)})")
        return

    # 8. 等待收集統編階段邏輯
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
            # 推播到內部群組 (Group ID 放在 to 的位置)
            send_via_iis(channel_id, "", LINE_INTERNAL_GROUP_ID, "", notify)
        
        msg = f"已收到統編「{tax_id_input}」，正在為您轉接真人客服。AI 已靜音。"
        send_via_iis(channel_id, user_id, group_id, "", msg)
        return

    # 9. 處理轉接確認階段 (利用按鈕選項)
    if handoff_pending.get(conv_key):
        if any(re.search(p, user_message, re.IGNORECASE) for p in CONFIRM_YES_PATTERNS):
            
            # 雙重保險：就算按了是，也要檢查是否上班時間
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
            # 這裡加上按鈕
            send_via_iis(channel_id, user_id, group_id, reply_token, 
                         "請選擇是否轉接真人？", buttons=get_yes_no_buttons())
            return

    # 10. 偵測是否觸發轉接意圖 (使用按鈕)
    if any(re.search(p, user_message, re.IGNORECASE) for p in HANDOFF_PATTERNS):
        if not is_working_hours():
            send_via_iis(channel_id, user_id, group_id, reply_token, "抱歉，目前是非上班時間，無法轉接真人。")
            return
        handoff_pending[conv_key] = True
        handoff_context[conv_key] = {"trigger": user_message, "user_id": user_id}
        # 這裡加上按鈕
        send_via_iis(channel_id, user_id, group_id, reply_token, 
                     "是否轉接真人客服？", buttons=get_yes_no_buttons())
        return

    # 11. 啟動 RAG 檢索與包含短期記憶與視覺強化組裝 (使用按鈕)
    relevant = search_knowledge(user_message)
    context_str = "\n".join(relevant) if relevant else "無相關資料。"
    
    # 抓取現在時間，並格式化成人類看得懂的樣子
    now = datetime.now()
    current_time_str = now.strftime("%Y年%m月%d日 %H點%M分 (星期%w)")

    # 提取短期記憶
    history = conversation_memory.get(conv_key, [])
    history_str = ""
    if history:
        history_str = "【前情提要】\n"
        for idx, turn in enumerate(history):
            history_str += f"[回合 {idx+1}]\n用戶問：{turn['user']}\nAI答：{turn['ai']}\n"
        history_str += "------------------\n"

    # 加入透視鏡：把要餵給 AI 的參考資料印在終端機上
    print(f"\n===== 🔍 餵給 AI 的參考資料 (Top-{TOP_K}) =====\n{context_str}\n===========================================\n")

    # 12. 雙腦模型分流機制：處理視覺模型或純文字模型
    if user_image:
        # 有圖時：呼叫自訂的視覺防呆大腦，且將知識庫傳入給它嚴格比對
        # 將看圖與解答以嚴謹的兩段式結構進行，防止幻覺污染
        prompt = (
            f"【參考知識庫】（若知識庫內有提及解法，請嚴格按照其中步驟；若無，請絕對不要自行發明步驟）：\n"
            f"{context_str}\n\n"
            f"【用戶問題】：{user_message}\n\n"
            f"請嚴格按照以下兩段式格式回覆：\n"
            f"🔍 圖片解析：(先印出你肉眼真實看到的內容，絕不可瞎掰。若模糊難辨，請誠實說明並要求提供更清楚的照片)\n"
            f"💡 客服回應：(根據解析結果與參考知識庫，給予用戶專業且安全的建議)"
        )
        # 如果未來有專屬視覺模型可以改名，暫時先用 XUYA:latest
        ollama_resp = get_ollama_response(prompt, user_image, "XUYA:latest")
    else:
        # 無圖時：退回純文字模式
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
    
    # 13. 後處理：圖片錯誤容錯與免責聲明附加
    if ollama_resp == "AI_IMAGE_ERROR":
        ollama_resp = "【系統通知】\n抱歉，我剛剛無法順利讀取您傳送的圖片（可能是檔案格式不支援或連線異常）。\n能不能請您再重新傳送一次清晰的圖片，或者直接用文字向我描述遇到的問題呢？"
    elif not ollama_resp: 
        ollama_resp = "抱歉，AI 暫時無法回應。"
    else:
        # 只要是正常回應，一律在最後補上免責聲明
        ollama_resp += "\n\n※ 本回覆由 AI 自動產生，僅供參考，實際操作請以產品手冊或專人指導為準。"

    # 14. 存入短期記憶 (只存文字，不存圖片如 Base64，保護記憶體消耗)
    if conv_key not in conversation_memory: 
        conversation_memory[conv_key] = []
    conversation_memory[conv_key].append({"user": user_message, "ai": ollama_resp})
    if len(conversation_memory[conv_key]) > MAX_HISTORY_TURNS:
        conversation_memory[conv_key].pop(0) # 保持在 MAX 筆內

    # 15. 判斷是否需要推播轉接提示
    if needs_contact_footer(relevant, ollama_resp):
        # 新增修補漏洞：如果是假日，就不跳出轉接按鈕，只回傳 AI 內容
        if not is_working_hours():
            send_via_iis(channel_id, user_id, group_id, reply_token, ollama_resp + "\n\n(目前非上班時間，無法轉接真人)")
            return

        handoff_pending[conv_key] = True
        handoff_context[conv_key] = {"trigger": user_message, "user_id": user_id}
        full_reply = ollama_resp + "\n\n(資訊不足，是否轉接真人？)"
        # 這裡加上按鈕
        send_via_iis(channel_id, user_id, group_id, reply_token, full_reply, buttons=get_yes_no_buttons())
        return
    
    send_via_iis(channel_id, user_id, group_id, reply_token, ollama_resp)
# ============================
# 共用大腦：核心對話邏輯分發區塊結束
# ============================
# ==========================================
# 訊息佇列與緩衝處理區塊開始 (負責攔截碎語與圖片)
# ==========================================
def handle_message_logic_with_buffer(channel_id, user_id, reply_token, user_message, group_id=None, user_image=None):
    conv_key = get_conv_key(user_id, group_id)
    
    # 1. 內部群組管理指令特權 (不進緩衝佇列，立即執行因為需要秒回)
    if LINE_INTERNAL_GROUP_ID and group_id == LINE_INTERNAL_GROUP_ID and user_message and user_message.startswith("!"):
        message_buffer[conv_key] = {
            "text": user_message, "image": None,
            "meta": {"channel_id": channel_id, "user_id": user_id, "reply_token": reply_token, "group_id": group_id}
        }
        process_actual_logic(conv_key)
        return

    # 2. 緩衝區文字與圖片疊加處理 (防碎語機制)
    if conv_key in message_buffer:
        # 如果水桶裡已經有東西，把新話接在後面，並【取消舊的計時器】
        message_buffer[conv_key]["timer"].cancel()
        # 累積文字
        if user_message:
            if message_buffer[conv_key].get("text"):
                message_buffer[conv_key]["text"] += "，" + user_message 
            else:
                message_buffer[conv_key]["text"] = user_message
        # 更新圖片 (保留最新的一張)
        if user_image:
            message_buffer[conv_key]["image"] = user_image
    else:
        # 建立新的獨立佇列容器
        message_buffer[conv_key] = {
            "text": user_message or "",
            "image": user_image,
            "meta": {
                "channel_id": channel_id, "user_id": user_id, 
                "reply_token": reply_token, "group_id": group_id
            }
        }
    
    # 3. 動態決定秒數：只要水桶裡有圖片，倒數計時就拉長為 10 秒
    wait_time = IMAGE_BUFFER_SECONDS if message_buffer[conv_key].get("image") else BUFFER_SECONDS
    
    print(f"[{conv_key}] 訊息/圖片收容中...等待 {wait_time} 秒")
    # 啟動非同步執行緒計時器，時間到後執行 process_actual_logic
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

                # 1. 偵測統編收集狀態是否超時
                if handoff_collect_taxid.get(conv_key) and data.get("taxid_start_time"):
                    if time.time() - data["taxid_start_time"] > TAXID_COLLECTION_TIMEOUT_SECONDS:
                        human_handoff[conv_key] = False
                        handoff_collect_taxid.pop(conv_key, None)
                        handoff_context.pop(conv_key, None)
                        handoff_start_times.pop(conv_key, None)
                        human_lock.pop(conv_key, None)
                        send_via_iis(saved_channel_id, push_target_id, "", "", "【系統通知】統編輸入超時，已自動切回 AI。")

                # 2. 偵測真人客服對話是否超時未回應
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


# ============================
# IIS Server API 接收路由開始，IIS 轉發入口 (超級相容版：同時支援 to / userId)
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
        # 1. 攔截並解析 IIS 傳入的 JSON 格式
        raw_data = request.json or json.loads(request.get_data(as_text=True))

        # ★★★ 新增這行：把前輩傳來的原始 JSON 印在終端機上 ★★★
        # print(f"\n[IIS 接收] 收到前輩的資料: {json.dumps(raw_data, ensure_ascii=False)}\n")

        # ★★★ 修正：把前輩傳來的原始 JSON 印出來，但如果太長（包含圖片 Base64）就縮寫，避免終端機當機 ★★★
        debug_str = json.dumps(raw_data, ensure_ascii=False)
        if len(debug_str) > 500:
            print(f"\n[IIS 接收] 收到資料 (內容過長已省略): {debug_str[:500]} ... [Base64 圖片資料隱藏] ...\n")
        else:
            print(f"\n[IIS 接收] 收到的圖片資料: {debug_str}\n")
            
        # 2. 處理資料層級封裝結構的相容性問題，兼容 Request/Data 結構或直接傳 Data
        if 'Request' in raw_data and 'Data' in raw_data['Request']:
            data_node = raw_data['Request']['Data']
        elif 'Data' in raw_data: 
            data_node = raw_data['Data']
        else:
            data_node = raw_data

        channel_id = data_node.get('channelId')
        
        # 3. 識別使用者或群組身分 (優先抓取 to 欄位) 先試試看有沒有新的 'to'
        to_value = data_node.get('to') 
        # 3-1. 如果沒有 'to'，就去抓舊的 'userId' 或 'groupId'
        if not to_value:
            if data_node.get('groupId'):
                to_value = data_node.get('groupId')
            elif data_node.get('userId'):
                to_value = data_node.get('userId')
        
        # 3-2. 分配給變數
        group_id = None
        user_id = None
        if to_value:
            if to_value.startswith("C") or to_value.startswith("G") or to_value.startswith("R"):
                group_id = to_value # 是群組
            else:
                user_id = to_value  # 是個人
        # 修正結束

        reply_token = data_node.get('replyToken')
        
        # 防呆提取，如果沒文字給空字串，並提取 image 欄位
        user_message = data_node.get('message') or ""
        user_image = data_node.get('image')

    except Exception as e:
        return jsonify(make_response_json(False, f"JSON Error: {e}")), 200

    # 4. 驗證通過後，將訊息推進緩衝佇列模組，避開空值陷阱：只要文字或圖片「其中一個」有內容就放行
    if not user_message and not user_image:
        return jsonify(make_response_json(False, "Missing message and image")), 200

    try:
        # 這裡改成呼叫水桶，把 user_image 也傳進去而不是直接處理
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
# 1. [GET] 取得所有歷史對話 Session 清單 (側邊欄專用)
@app.route("/api/chat_sessions", methods=['GET'])
def get_chat_sessions():
    # 1-1. 接收前端傳來的訪客身分證
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "缺少 user_id 參數"}), 400

    try:
        conn = get_db_connection()
        # 1-2. 僅提取綁定該 user_id 訪客的 Session 紀錄，改成只撈取該訪客專屬的對話紀錄！
        sessions = conn.execute("SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
        conn.close()
        return jsonify([dict(s) for s in sessions]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. [GET] 取得特定 Session 的詳細對話紀錄 (點擊側邊欄項目時載入)
@app.route("/api/chat_sessions/<session_id>", methods=['GET'])
def get_session_messages(session_id):
    try:
        conn = get_db_connection()
        messages = conn.execute("SELECT role, content as text FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
        conn.close()
        # 前端需要的第一句話是預設的歡迎詞
        default_msg = [{"role": "ai", "text": "你好！我是這位求職者的專屬 AI 助理。您可以問我任何關於他專案、技術或開發過程的問題！"}]
        return jsonify(default_msg + [dict(m) for m in messages]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# 3. [DELETE] 刪除特定對話紀錄與其內容
@app.route("/api/chat_sessions/<session_id>", methods=['DELETE'])
def delete_session(session_id):
    try:
        conn = get_db_connection()
        # 聯集刪除：同時清空 Session 標題表與對話明細表(同時刪除標題與底下的所有對話內容)
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 4. [PUT] 重新命名對話標題
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
    
# 5. [POST] 網頁版即時聊天入口 (處理前端的提問)
@app.route("/api/web_chat", methods=['POST'])
def web_chat():
    data = request.json
    session_id = data.get('session_id') # 前端如果沒有傳，代表是全新的對話
    user_message = data.get('message', '')
    user_id = data.get('user_id') # 接住前端傳來的訪客身分證

    if not user_message:
        return jsonify({"error": "缺少 message 參數"}), 400
    if not user_id:
        return jsonify({"error": "缺少 user_id 參數"}), 400

    try:
        conn = get_db_connection()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Session 驗證與生成，如果沒有 session_id，代表這是全新的對話
        if not session_id:
            session_id = str(uuid.uuid4())
            # 取第一句話的前 12 個字當作側邊欄標題
            title = user_message[:12] + "..." if len(user_message) > 12 else user_message
            # 建立新對話時，把 user_id 一起寫入資料庫綁定！
            conn.execute("INSERT INTO sessions (id, user_id, title, updated_at) VALUES (?, ?, ?, ?)", (session_id, user_id, title, current_time))
        else:
            # 更新對話時間，確保它在側邊欄排在最上面
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (current_time, session_id))
            row = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
            title = row['title'] if row else "未命名對話"

        # 2. 將使用者的提問存入持久化資料庫
        conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'user', user_message, current_time))
        conn.commit()

        # 3. 從資料庫撈出過去的歷史紀錄 (Context Memory) 提供 AI 上下文參考
        msgs = conn.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
        
        history_str = ""
        # 如果歷史紀錄大於 1 (扣掉剛剛存進去的那句)，代表有前情提要
        if len(msgs) > 1:
            history_str = "【先前對話紀錄 (Context History)】\n"
            for m in msgs[:-1]: 
                role_name = "面試官" if m['role'] == 'user' else "張序亞助理"
                history_str += f"{role_name}：{m['content']}\n"
            history_str += "------------------\n"

        # 4. 呼叫 ChromaDB 知識庫檢索
        relevant = search_knowledge(user_message)
        context_str = "\n".join(relevant) if relevant else "無相關資料。"
        
        print(f"\n===== [Web 介面記憶優化] 🔍 餵給 AI 的參考資料 =====\n{context_str}\n===========================================\n")

        # 5. 組合最終 Prompt 並呼叫 Ollama 模型
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

        # 6. 將生成的 AI 回答寫入資料庫
        conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'ai', ollama_resp, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        # 回傳給 React 前端，包含最新產生的 session_id 與 title
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
# 靜態檔案路由與伺服器啟動區塊開始，讓 Flask 負責提供 React 靜態網頁
# ============================
# 負責處理 React 前端編譯後的靜態檔案 (Single Page Application 行為)
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

# 針對前端路由的 SPA Fallback 處理 (如果你未來在 React 加了換頁功能，未註冊的路由皆導向 index.html 這能避免重整時 404 錯誤)
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

# ==========================================
# # 主程式進入點啟動區
# ==========================================

if __name__ == '__main__':
    # 註解備用區：舊版知識庫手動載入邏輯
    # load_knowledge()
    
    # 1. 建立並啟動背景執行緒 (Daemon Thread) 執行超時監控
    timeout_thread = threading.Thread(target=check_timeouts)
    timeout_thread.daemon = True 
    timeout_thread.start()
    
    # 2. 註冊退出事件，確保主程式結束時能安全停止監控執行緒
    atexit.register(lambda: timeout_checker_stop.set())

    # 註解備用區：掛載憑證的 SSL 模式啟動邏輯 (憑證釘選)
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, ssl_context=('cert.pem', 'key.pem'))

    # 3. 啟動 Flask 伺服器 ((改成這行使用無 SSL 憑證的純淨 HTTP 模式測試）)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
# ============================
# 靜態檔案路由與伺服器啟動區塊結束
# ============================
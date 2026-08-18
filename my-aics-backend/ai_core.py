# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
# 1. 引入系統操作、網路請求、深度學習框架等標準套件
import os
import json
import requests
import re  # 【新增】：引入正則表達式，用於捕捉漏氣的 JSON
import torch
from sentence_transformers import SentenceTransformer

# 2. 引入自訂模組，包含全域設定參數與 ChromaDB 雙軌資料庫實體
from config import *
from database import collection_manual, collection_auto

# 【MCP 外部工具擴充】：引入網頁搜尋隨身碟 (未來有新工具直接在此 import)
from tools.web_search import search_web
# 【MCP 外部工具擴充】：引入精準數學計算機
from tools.calculator import calculate_math
# ============================
# 核心模組與套件引入結束
# ============================


# ============================
# 裝置硬體偵測與模型初始化開始
# ============================
def pick_device():
    try:
        # 1. 嘗試偵測並初始化 NVIDIA CUDA 繪圖核心加速
        if torch.cuda.is_available():
            _ = torch.randn(1, device='cuda') * 2
            torch.cuda.synchronize()
            print("[Device] Using CUDA")
            return 'cuda'
    except Exception as e:
        print(f"[Device] CUDA 不可用，改用 CPU：{e}")
    # 2. 若 CUDA 無法使用，強制清空環境變數並降級使用 CPU 進行計算
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[Device] Using CPU")
    return 'cpu'

# 3. 執行硬體偵測函式，決定並儲存全域運算裝置
DEVICE = pick_device()

# 4. 初始化並將 Embedding 模型載入至記憶體，用於後續自然語言的向量化處理
print(f"[系統] 正在載入 Embedding 模型 ({DEVICE})...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
# ============================
# 裝置硬體偵測與模型初始化結束
# ============================


# ============================
# RAG 知識庫檢索模組開始
# ============================
def search_knowledge(query, top_k=TOP_K):
    # 1. 安全性檢查：若資料庫實體尚未建立，直接回傳空陣列防呆
    if collection_manual is None or collection_auto is None:
        return []
        
    try:
        # 2. 語意向量化：將使用者的文字問題轉換為高維度向量陣列
        qv = embedding_model.encode([query]).tolist()
        res = []
        
        # 3. Stage 1 (A 軌)：優先向高精準手動資料庫進行嚴格檢索
        results_manual = collection_manual.query(
            query_embeddings=qv,
            n_results=1
        )
        
        # 4. 檢驗 A 軌是否成功命中，並取得最佳距離分數
        if results_manual['distances'] and len(results_manual['distances'][0]) > 0:
            best_dist = results_manual['distances'][0][0]
            print(f"[檢索路由] 查找高精準區，最佳距離分數為: {best_dist:.3f}")
            
            # 5. 設定 L2 距離門檻 (小於 2.0 代表高度相關)
            ROUTING_THRESHOLD = 2.0 
            
            if best_dist < ROUTING_THRESHOLD:
                # 6. 命中高精準區：提取標準問題與預設解答，格式化後直接回傳並中斷後續檢索
                matched_q = results_manual['documents'][0][0]
                matched_a = results_manual['metadatas'][0][0].get('answer', '無對應解答')                
                formatted_ans = f"【標準問題】{matched_q}\n【標準解答】{matched_a}"
                res.append(formatted_ans)
                
                print("[檢索路由] 🎯 命中高精準區，直接回傳標準答案。")
                return res

        # 7. Stage 2 (B 軌)：若 A 軌未命中或分數過大，啟動 Fallback 機制向自動擴展庫檢索 Top-K 參考資料
        print("[檢索路由] ⚠️ 高精準區查無結果，啟動 Fallback 翻閱參考說明書...")
        results_auto = collection_auto.query(
            query_embeddings=qv,
            n_results=top_k
        )
        
        # 8. 組合參考文件：將檢索到的文件段落與來源名稱整併後，回傳給 AI 作為生成上下文
        if results_auto['documents'] and len(results_auto['documents'][0]) > 0:
            for doc, meta in zip(results_auto['documents'][0], results_auto['metadatas'][0]):
                source = meta.get("source", "未知說明書")
                res.append(f"【參考來源：{source}】\n{doc}")
                
        return res
    except Exception as e:
        # 9. 錯誤捕捉：檢索過程發生異常時，印出錯誤並安全回傳空陣列
        print(f"[搜尋錯誤] {e}")
        return []

def needs_contact_footer(relevant_knowledge, ai_text: str) -> bool:
    # 1. 判斷防呆條件：若完全沒有參考知識，預設需要補上真人客服轉接選項
    if not relevant_knowledge: return True
    # 2. 定義不確定性的關鍵字清單
    markers = ["抱歉", "無法提供", "不知道", "不清楚"]
    # 3. 掃描 AI 的回覆內容，若包含上述關鍵字，則觸發真人轉接機制
    return any(m in ai_text for m in markers)
# ============================
# RAG 知識庫檢索模組結束
# ============================


# ============================
# MCP 工具設定檔 (Tool Schema) 開始
# ============================
# 這裡就是 AI 的「工具清單」。未來如果要加新工具，直接在此 JSON 陣列擴充定義即可。
mcp_tools = [
    # ------------------------------------------------------------------
    # 【MCP 工具 1】：網頁搜尋引擎 (search_web)
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "網頁搜尋引擎。用於補充【參考知識庫】中完全缺乏的外部最新資訊。",
            "parameters": {
                "type": "object",
                "properties": {
                    # 【SA 獨家設計】：利用內心獨白 (Chain of Thought) 逼迫 AI 審視 RAG 知識庫
                    "thought_process": {
                        "type": "string",
                        "description": "在搜尋前，請先仔細閱讀使用者提供的【參考知識庫】。並在這裡用一句話說明：知識庫裡面是否『已經有』足夠的資訊來回答這個問題？"
                    },
                    "need_internet_search": {
                        "type": "boolean",
                        "description": "如果知識庫已有答案，請務必填寫 false。只有當知識庫完全找不到任何相關資料時，才准許填寫 true。"
                    },
                    "query": {
                        "type": "string",
                        "description": "需要上網搜尋的精準關鍵字。若不需搜尋請填寫 'None'。"
                    }
                },
                "required": ["thought_process", "need_internet_search", "query"]
            }
        }
    },
    # ============================
    # 網頁搜尋 MCP 工具區塊結束
    # ============================

    # ============================
    # 數學計算機 MCP 工具區塊開始
    # ============================
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "精準數學計算機。當問題涉及任何數值運算、薪資預算、日期天數計算或複雜算式時，『必須』呼叫此工具，絕對禁止自己心算以避免幻覺。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought_process": {
                        "type": "string",
                        "description": "說明為什麼需要進行這道數學計算？"
                    },
                    "expression": {
                        "type": "string",
                        "description": "要執行的純數學算式，例如：'1250000 / (80000 * 1.15)' 或 '2026 - 1995'。禁止包含中文字。"
                    }
                },
                "required": ["thought_process", "expression"]
            }
        }
    }
    # ============================
    # 數學計算機 MCP 工具區塊結束
    # ============================
]
# ============================
# MCP 工具設定檔 (Tool Schema) 結束
# ============================


# ============================
# Ollama 多模態與 MCP 生成模組開始
# ============================
# 【SA 結構優化】：原本接收純字串 prompt，現在改為接收已經整理好的 messages_list 陣列
def get_ollama_response(messages_list, image_b64=None, model_name="XUYA:latest"):
    try:
        # 1. 最高權限防火牆 (System Guardrail) 保持不變，作為陣列的最開頭
        system_guardrail = (
            "你是專業的 AI 面試助理。你的任務是精準回答問題。\n"
            "【嚴格規定】：請優先整理並依靠你收到的【參考知識庫】來回答問題，絕對禁止為了偷懶而上網搜尋已經存在的履歷或專案資訊！"
        )

        # 2. 將最高指令與 state_manager 整理好的對話清單組合起來
        messages = [{"role": "system", "content": system_guardrail}] + messages_list
        
        # 3. 圖片處理：將圖片外掛到陣列中「最後一個使用者 (user)」的對話框裡
        if image_b64:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    msg["images"] = [image_b64]
                    break

        payload = {
            "model": model_name, 
            "messages": messages,
            "stream": False,
            "tools": mcp_tools,  # 將可用工具清單注入給 AI
            "options": {
                "num_predict": 1024
            }
        }
        
        print(f"[AI 引擎] 🧠 正在思考並評估是否需要使用外部工具...")
        r = requests.post(f"{OLLAMA_API_BASE_URL}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        response_message = r.json().get("message", {})

        # ==========================================
        # 🛡️ 【SA 防漏氣攔截網】：捕捉 Ollama 引擎漏接的 JSON 工具呼叫
        # ==========================================
        tool_calls = response_message.get("tool_calls", [])
        content_str = response_message.get("content", "").strip()

        # 這裡也加強了防漏氣，同時捕捉 search_web 和 calculate_math
        if not tool_calls and ('"name": "search_web"' in content_str or '"name": "calculate_math"' in content_str):
            print("[SA 防護網] ⚠️ 偵測到模型原生 JSON 漏氣，啟動強制解析！")
            try:
                # 尋找任何包含 "name": "工具名稱" 的 JSON 區塊
                match = re.search(r'\{.*"name":\s*"(search_web|calculate_math)".*\}', content_str, re.DOTALL)
                if match:
                    leaked_json = json.loads(match.group(0))
                    # 手動把它轉回標準的 tool_calls 陣列
                    tool_calls = [{
                        "function": {
                            "name": leaked_json.get("name"),
                            "arguments": leaked_json.get("parameters", {})
                        }
                    }]
                    # 清空 content，避免髒資料干擾歷史對話
                    response_message["content"] = ""
            except Exception as parse_err:
                print(f"[SA 防護網] JSON 解析失敗: {parse_err}")
        # ==========================================

        if tool_calls:
            print(f"[AI 引擎] 🛠️ 嘗試呼叫外部工具...")
            
            # 如果是我們手動救援的，也要把 tool_calls 加進 response_message 裡
            response_message["tool_calls"] = tool_calls
            messages.append(response_message)
            
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                
                # ============================
                # 網頁搜尋執行區塊開始
                # ============================
                if func_name == "search_web":
                    # 【擷取 AI 的內心獨白與決策】
                    thought = arguments.get("thought_process", "未提供理由")
                    # 容錯處理：有時 AI 會傳字串的 "true"/"false"，統一轉為布林值
                    need_search_val = arguments.get("need_internet_search", True)
                    need_search = str(need_search_val).lower() == "true"
                    search_query = arguments.get("query", "")

                    print(f"\n[AI 思考過程] 💭 {thought}")
                    
                    if need_search is False or search_query == "None":
                        print(f"[MCP 防火牆] 🛑 AI 判定知識庫已有解答，攔截網路請求，成功保護 RAG 與 API 額度！")
                        tool_result = "【系統防護】：你已判斷不需要上網搜尋。請立刻停止使用工具，直接根據【參考知識庫】的內容給出完美的解答！"
                    else:
                        # 真正遭遇外部知識，才放行呼叫 Brave API
                        print(f"[MCP 執行] 🌐 放行！正在上網搜尋：「{search_query}」...")
                        tool_result = search_web(search_query)  
                    
                    messages.append({
                        "role": "tool",
                        "content": tool_result
                    })
                # ============================
                # 網頁搜尋執行區塊結束
                # ============================

                # ============================
                # 數學計算機執行區塊開始
                # ============================
                elif func_name == "calculate_math":
                    thought = arguments.get("thought_process", "未提供計算理由")
                    expression = arguments.get("expression", "")

                    print(f"\n[AI 思考過程] 💭 {thought}")
                    print(f"[MCP 執行] 🧮 啟動計算機，正在計算算式：「{expression}」...")
                    
                    tool_result = calculate_math(expression)
                    print(f"[MCP 結果] ✅ {tool_result}")
                    
                    messages.append({
                        "role": "tool",
                        "content": tool_result
                    })
                # ============================
                # 數學計算機執行區塊結束
                # ============================
            
            # 5. 第二階段請求：讓 AI 參考搜尋回傳的內容，進行最終語言統整
            print(f"[AI 引擎] 🧠 獲取外部資料完畢，正在統整最終回覆...")
            payload["messages"] = messages
            r_final = requests.post(f"{OLLAMA_API_BASE_URL}/api/chat", json=payload, timeout=300)
            r_final.raise_for_status()
            return r_final.json().get('message', {}).get('content', '').strip()

        else:
            print(f"[AI 引擎] 💬 判斷不需使用工具，直接回答。")
            return content_str

    except requests.exceptions.HTTPError as e:
        # ==========================================
        # 【SA 進階除錯區塊】：抓取 HTTP 狀態碼與 Ollama 具體報錯訊息
        # ==========================================
        error_details = e.response.text if e.response is not None else str(e)
        status_code = e.response.status_code if e.response is not None else "未知"
        print(f"\n[Ollama HTTP 錯誤] 狀態碼: {status_code}")
        print(f"[Ollama 錯誤細節] {error_details}\n")
        if image_b64: return "AI_IMAGE_ERROR"
        return f"【系統提示】AI 通訊錯誤 (HTTP {status_code})。"

    except Exception as e:
        print(f"\n[Ollama 系統錯誤] {e}\n")
        if image_b64: return "AI_IMAGE_ERROR"
        return "【系統提示】AI 通訊發生未知錯誤。"
# ============================
# Ollama 多模態與 MCP 生成模組結束
# ============================
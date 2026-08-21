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
def search_knowledge_ex(query, top_k=TOP_K) -> dict:
    """
    【SA v2 新增】結構化版本的 RAG 檢索，這是給 state_manager / 多智能體使用的主要入口。

    為什麼要多做這一個？
    舊版 search_knowledge() 不管是「高精準區命中標準答案」還是「Fallback 撈了 6 篇
    可能完全不相關的參考文件」，回傳的都是一個長得一模一樣的字串陣列。
    上游拿到之後根本分不清楚「這是精準答案」還是「這只是勉強撈到的雜訊」。

    這件事在 v2 特別重要，因為規劃官(Planner)需要先判斷：
      「這題知識庫已經有答案了嗎？有的話就不要浪費一次 Brave API 上網查。」
    分不清楚就沒辦法做這個判斷。

    回傳格式：
      {
        "docs": ["【標準問題】…", …],   # 給模型看的文字陣列
        "hit_type": "manual" | "auto" | "none",
        "best_distance": float | None
      }
      hit_type 意義：
        manual -> 命中高精準人工問答庫，內容就是標準答案，可信度最高
        auto   -> 只是從自動擴展庫撈了 Top-K 參考段落，可能完全不相關
        none   -> 什麼都沒撈到
    """
    if collection_manual is None or collection_auto is None:
        return {"docs": [], "hit_type": "none", "best_distance": None}

    try:
        # 1. 語意向量化：將使用者的文字問題轉換為高維度向量陣列
        qv = embedding_model.encode([query]).tolist()

        # 2. Stage 1 (A 軌)：優先向高精準手動資料庫進行嚴格檢索
        results_manual = collection_manual.query(query_embeddings=qv, n_results=1)

        best_dist = None
        if results_manual['distances'] and len(results_manual['distances'][0]) > 0:
            best_dist = results_manual['distances'][0][0]
            print(f"[檢索路由] 查找高精準區，最佳距離分數為: {best_dist:.3f}")

            # 3. 門檻判斷 (【SA v2 調整】：門檻值改由 config.py 統一管理)
            if best_dist < RAG_HIGH_PRECISION_THRESHOLD:
                matched_q = results_manual['documents'][0][0]
                matched_a = results_manual['metadatas'][0][0].get('answer', '無對應解答')
                formatted_ans = f"【標準問題】{matched_q}\n【標準解答】{matched_a}"
                print("[檢索路由] 🎯 命中高精準區，直接回傳標準答案。")
                return {"docs": [formatted_ans], "hit_type": "manual", "best_distance": best_dist}

        # 4. Stage 2 (B 軌)：Fallback 向自動擴展庫檢索 Top-K 參考資料
        print("[檢索路由] ⚠️ 高精準區查無結果，啟動 Fallback 翻閱參考說明書...")
        results_auto = collection_auto.query(query_embeddings=qv, n_results=top_k)

        docs = []
        if results_auto['documents'] and len(results_auto['documents'][0]) > 0:
            for doc, meta in zip(results_auto['documents'][0], results_auto['metadatas'][0]):
                source = meta.get("source", "未知說明書")
                docs.append(f"【參考來源：{source}】\n{doc}")

        return {
            "docs": docs,
            "hit_type": "auto" if docs else "none",
            "best_distance": best_dist
        }

    except Exception as e:
        print(f"[搜尋錯誤] {e}")
        return {"docs": [], "hit_type": "none", "best_distance": None}


def search_knowledge(query, top_k=TOP_K):
    """
    【SA v2 保留】舊介面的相容包裝，回傳純字串陣列。
    app.py 等舊有呼叫端不用改就能繼續跑。新程式請改用 search_knowledge_ex()。
    """
    return search_knowledge_ex(query, top_k)["docs"]


def needs_contact_footer(relevant_knowledge, ai_text: str,
                         rag_hit_type: str = "auto",
                         tools_all_succeeded: bool = True) -> bool:
    """
    判斷是否要在回覆末端附上「是否轉接真人客服」的選項。

    【SA v2 大幅改寫】舊版有兩個會讓客人很困擾的問題：

    問題 1：`if not relevant_knowledge: return True`
      collection_auto.query(n_results=6) 幾乎一定會撈回 6 筆東西(不管相不相關)，
      所以這條在正常情況下永遠不會成立 —— 看似有防護，其實是空的。
      反過來，一旦 ChromaDB 是空的或掛掉，就變成「每一句回覆都問要不要轉真人」。
      改成看 hit_type：只有真的什麼都沒撈到(none)才算知識庫沒東西。

    問題 2：markers 裡有一個裸的「抱歉」
      這是最大的誤判來源。任何禮貌性用語都會中：
        「抱歉讓您久等了，台北 101 的高度是 508 公尺」→ 明明答得好好的，卻跳出轉真人。
      而 v2 的 Final_Answer 在查詢失敗時本來就會誠實說「抱歉，目前查不到…」，
      這種情況才是真的該轉真人。所以改成「完整語句片語」比對，而不是單一個「抱歉」兩字。

    【SA v2 新增參數】：
      rag_hit_type        -> 由 search_knowledge_ex() 提供，區分精準命中/勉強撈到/完全沒有
      tools_all_succeeded -> 由多智能體的任務清單提供。只要有任何一項查詢/計算失敗，
                             就代表這次的回答是不完整的，主動提供真人管道才合理。
    """
    # 1. 多智能體明確回報有任務失敗 → 這次回答不完整，主動提供真人管道
    if not tools_all_succeeded:
        return True

    # 2. 知識庫完全沒撈到任何東西 → 沒有任何依據可以回答
    if rag_hit_type == "none" and not relevant_knowledge:
        return True

    # 3. 掃描 AI 回覆中「真正表達無能為力」的完整語句
    #    注意：這裡刻意不使用單獨的「抱歉」「不清楚」等兩字詞，避免禮貌用語誤觸
    uncertain_patterns = [
        r"查詢失敗", r"查不到", r"找不到相關", r"沒有查到", r"無法查詢",
        r"無法提供", r"無法回答", r"無法確認", r"資料不足", r"資訊不足",
        r"超出我的處理能力", r"我不知道", r"無法完成計算", r"計算步驟未能完成",
        r"目前沒有(這|該|相關)",
    ]
    return any(re.search(p, ai_text) for p in uncertain_patterns)
# ============================
# RAG 知識庫檢索模組結束
# ============================


# ============================
# MCP 工具設定檔 (Tool Schema) 開始
# ============================
# 【SA v2 重要說明】：
# 從下面這一段開始到檔案結尾的 mcp_tools + get_ollama_response()，
# 是「航空母艦(LangGraph)上線之前」的舊版單體 Tool Calling 流程。
# 目前 state_manager.py 走的是 graph_core.app_graph，這段其實已經不會被執行到，
# 只有 app.py 還 import 著 get_ollama_response(但也沒有呼叫)。
#
# 保留不刪的理由：
#   1. 之後如果要做 A/B 對照(單體 vs 多智能體)，這是現成的對照組
#   2. 萬一多智能體出大問題，可以快速切回來救急
# 如果你確定不再需要，可以整段刪掉，並把 app.py 的 import 一起拿掉。
# ============================
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
# Ollama 多模態與 MCP 生成模組開始 (舊版單體流程，目前未使用)
# ============================
# 【SA 結構優化】：原本接收純字串 prompt，現在改為接收已經整理好的 messages_list 陣列
def get_ollama_response(messages_list, image_b64=None, model_name=None):
    # 【SA v2 調整】：預設模型改為讀 config.MAIN_MODEL_NAME，避免這裡又寫死一次模型名稱
    if model_name is None:
        model_name = MAIN_MODEL_NAME
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

        if not tool_calls and ('"name": "search_web"' in content_str or '"name": "calculate_math"' in content_str):
            print("[SA 防護網] ⚠️ 偵測到模型原生 JSON 漏氣，啟動強制解析！")
            try:
                match = re.search(r'\{.*"name":\s*"(search_web|calculate_math)".*\}', content_str, re.DOTALL)
                if match:
                    leaked_json = json.loads(match.group(0))
                    tool_calls = [{
                        "function": {
                            "name": leaked_json.get("name"),
                            "arguments": leaked_json.get("parameters", {})
                        }
                    }]
                    response_message["content"] = ""
            except Exception as parse_err:
                print(f"[SA 防護網] JSON 解析失敗: {parse_err}")
        # ==========================================

        if tool_calls:
            print(f"[AI 引擎] 🛠️ 嘗試呼叫外部工具...")
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

                    messages.append({"role": "tool", "content": tool_result})
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
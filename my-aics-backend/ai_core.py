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
# ============================
# 🏹 RAG 弓箭手 (RAG_Archer) — 智慧檢索器
# ============================
# 【SA v2.8 全新】這一段取代舊版的 search_knowledge_ex。
#
# 為什麼要換掉舊版？舊版有兩個結構性缺陷，剛好造成了實測看到的兩種錯誤回答：
#
#   缺陷 1：manual 軌只取 n_results=1（只看第一名）
#           「rag 雙軌為什麼這樣做」和「你懂 Transformer 嗎」這兩題語意很接近，
#           分數只差一點點。誰險勝就整碗端走，補進去的正確素材永遠排第二、看不到。
#
#   缺陷 2：auto 軌完全沒有相關性門檻（撈回 6 筆就無條件全塞）
#           問「rag 雙軌」時，auto 軌把 6 段不相關的履歷 PDF 切塊全丟給模型，
#           模型就從裡面拼出「結合 Transformer 和矩陣運算的方法」這種融合幻覺。
#
# 弓箭手的四支箭：
#   箭 1｜多取幾名再挑：manual 軌改取 top 5，看前幾名而不是只看第一名。
#   箭 2｜主題投票：前幾名如果集中在同一個 source 分類，可信度更高；
#         用主題一致性當作「這批結果到底可不可信」的第二個訊號。
#   箭 3｜auto 軌加門檻：距離超過門檻的段落直接丟掉，不夠像就不塞給模型，
#         寧可讓模型誠實說「這部分建議面試時再聊」，也不要餵雜訊逼它幻覺。
#   箭 4｜回傳決策而非一堆文字：明確告訴上游「這是精準答案 / 這是模糊參考 / 什麼都沒有」，
#         讓盜賊客服知道該把這批資料當標準答案用，還是只當背景參考。
#
# 【方案 A】：弓箭手仍然是「檢索在前」——每題一進來先射一箭，
#   結果放進背包給規劃官與盜賊客服參考，不佔用規劃官的派工判斷。

# 【SA v2.8】auto 軌的相關性門檻。
# manual 軌用 RAG_HIGH_PRECISION_THRESHOLD（config，目前 0.50）判斷「精準命中」；
# auto 軌則用這個較寬鬆的門檻判斷「這段到底沾不沾得上邊」。
# 兩軌都是 cosine 距離（越小越相似），所以數值可以直接比較。
# 這個值可以之後也搬進 config，先放這裡方便你調。
RAG_AUTO_RELEVANCE_THRESHOLD = 0.85   # 距離 > 0.85 的 auto 段落視為不相關，直接丟棄
RAG_MANUAL_TOP_K = 5                    # manual 軌一次看前 5 名做主題投票


def _archer_query(collection, query_vec, n):
    """對單一 collection 射一箭，回傳 [(距離, 文件, metadata), ...]，由近到遠。"""
    try:
        res = collection.query(query_embeddings=query_vec, n_results=n)
    except Exception as e:
        print(f"[RAG弓箭手] ⚠️ 檢索某一軌時發生錯誤：{e}")
        return []
    out = []
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    for d, doc, m in zip(dists, docs, metas):
        out.append((d, doc, m or {}))
    return out


def search_knowledge_ex(query, top_k=TOP_K) -> dict:
    """
    【SA v2.8 RAG 弓箭手】結構化 RAG 檢索，給 state_manager／多智能體使用的主要入口。

    回傳格式（與舊版相容，另外多帶幾個欄位供除錯）：
      {
        "docs": [...],                         # 給模型看的文字陣列
        "hit_type": "manual" | "auto" | "none",
        "best_distance": float | None,
        "topic": str | None,                   # 【新增】命中的主題分類
        "candidates": [(dist, source, q), ...] # 【新增】前幾名候選，方便 log 檢視
      }
    """
    if collection_manual is None or collection_auto is None:
        return {"docs": [], "hit_type": "none", "best_distance": None, "topic": None, "candidates": []}

    try:
        qv = embedding_model.encode([query]).tolist()

        # ---- 箭 1：manual 軌一次取前 K 名，不再只看第一名 ----
        manual_hits = _archer_query(collection_manual, qv, RAG_MANUAL_TOP_K)

        candidates = []
        for d, doc, m in manual_hits:
            candidates.append((round(d, 3), m.get("source", "?"), m.get("question_raw", doc.split("\n")[-1])[:20]))

        if manual_hits:
            best_dist, best_doc, best_meta = manual_hits[0]
            best_topic = best_meta.get("source", "?")
            print(f"[RAG弓箭手] 🎯 手動精準軌最佳距離 {best_dist:.3f}（主題：{best_topic}）")
            print(f"[RAG弓箭手] 📋 前 {len(candidates)} 名候選：")
            for dist, src, q in candidates:
                print(f"           {dist}  [{src}] {q}")

            # ---- 箭 2：主題投票。前幾名裡跟第一名同主題的數量，是可信度的第二訊號 ----
            same_topic = [h for h in manual_hits if h[2].get("source") == best_topic]
            topic_votes = len(same_topic)

            # 命中條件：第一名夠近（低於精準門檻）
            if best_dist < RAG_HIGH_PRECISION_THRESHOLD:
                # 【SA v3.2 改良】：不再只回傳第一名，而是把「同主題且夠近」的條目一起帶回（最多 3 筆）。
                #
                # 實測慘案：面試官問「柬埔寨專案用了什麼工具？為什麼要用？」
                # 弓箭手命中 0.290、主題正確，但第一名條目是「柬埔寨專案具體幹嘛的？」，
                # 內容只講專案背景、完全沒提工具。模型手上沒素材，就自己編了一個
                # 「ArcGIS 地理空間分析」出來 —— 而知識庫裡明明有「銀行家捨入法」
                # 「設備心跳監控」這些真正的答案，只是排在第二、第三名沒被拿出來。
                #
                # 多給幾筆同主題素材，模型就不必無中生有。這比在提示裡多寫一條
                # 「不准編造」有效得多 —— 缺料才是編造的根因。
                bundle = []
                seen_answers = []
                for d, doc, m in manual_hits[:RAG_MANUAL_TOP_K]:
                    if m.get("source") != best_topic:
                        continue
                    if d > RAG_HIGH_PRECISION_THRESHOLD + 0.15:
                        continue
                    q = m.get("question_raw", doc.split("\n")[-1])
                    a = (m.get("answer") or "").strip()
                    if not a:
                        continue
                    # 【SA v3.3 新增】：內容高度重複的條目只留一筆。
                    #
                    # v3.2 開始帶回同主題前 3 筆，本意是給模型足夠素材別亂編，
                    # 但實測「談談你的 DevOps 與 CI/CD 經驗」翻車了 ——
                    # 那個主題底下三題的答案幾乎一模一樣（都在講 Docker + GitHub Actions
                    # + Docker Hub），模型讀到三段複製貼上的文字就卡進複讀迴圈，
                    # 同一句話連續輸出了幾十遍。
                    #
                    # 判斷方式很土但有效：比較前 30 個字，重疊就當成同一段。
                    head = re.sub(r'\s+', '', a)[:30]
                    if any(head[:20] in s or s[:20] in head for s in seen_answers):
                        continue
                    seen_answers.append(head)
                    bundle.append(f"【參考問題】{q}\n【參考答案】{a}")
                    if len(bundle) >= 3:
                        break
                if not bundle:
                    matched_q = best_meta.get("question_raw", best_doc.split("\n")[-1])
                    matched_a = best_meta.get("answer", "無對應解答")
                    bundle = [f"【參考問題】{matched_q}\n【參考答案】{matched_a}"]

                print(f"[RAG弓箭手] ✅ 命中精準軌（主題 {best_topic}，同主題票數 {topic_votes}/{len(manual_hits)}），帶回 {len(bundle)} 筆同主題素材。")
                return {
                    "docs": bundle,
                    "hit_type": "manual",
                    "best_distance": best_dist,
                    "topic": best_topic,
                    "candidates": candidates,
                }

            # ---- 邊界救援：第一名沒過門檻，但第一名的「主題自洽」 ----
            # 【SA v2.9 改良】：舊規則要求「前5名有≥3筆同主題」，太嚴。
            # 實測「rag雙軌為什麼」這題：第一名 0.534（門檻 0.50，只差 0.034）、
            # 主題正確是「RAG與向量資料庫」、同主題在前5名出現 2 次 —— 卻因為湊不滿3票被放掉，
            # 掉到 auto 軌拼出幻覺。
            #
            # 新規則：第一名略高於門檻一點點（margin 內）、且第一名的主題【不是孤例】
            # （在前 5 名重複出現至少 2 次），就採用第一名。
            # 為什麼這樣安全？因為它要求「第一名主題自洽」——
            # 像「台積電股價」那種前 5 名主題散亂（基本資料/技術能力/工作經歷各1）的情況，
            # 同主題數不會 ≥2，所以不會被誤救，不破壞已經調好的分離度。
            RESCUE_MARGIN = 0.08
            if best_dist < RAG_HIGH_PRECISION_THRESHOLD + RESCUE_MARGIN and topic_votes >= 2:
                matched_q = best_meta.get("question_raw", best_doc.split("\n")[-1])
                matched_a = best_meta.get("answer", "無對應解答")
                print(f"[RAG弓箭手] 🩹 第一名 {best_dist:.3f} 略高於門檻，但主題「{best_topic}」在前 {len(manual_hits)} 名出現 {topic_votes} 次（主題自洽），採用第一名。")
                return {
                    "docs": [f"【參考問題】{matched_q}\n【參考答案】{matched_a}"],
                    "hit_type": "manual",
                    "best_distance": best_dist,
                    "topic": best_topic,
                    "candidates": candidates,
                }

        # ---- 箭 3：auto 軌加相關性門檻，不夠像的段落直接丟棄 ----
        print("[RAG弓箭手] ⚠️ 未命中精準軌，改射自動擴展軌（會套用相關性門檻）...")
        auto_hits = _archer_query(collection_auto, qv, top_k)

        kept = []
        for d, doc, m in auto_hits:
            if d <= RAG_AUTO_RELEVANCE_THRESHOLD:
                src = m.get("source", "未知說明書")
                kept.append(f"【參考來源：{src}（相關度距離 {d:.2f}）】\n{doc}")
            else:
                # 印出被丟掉的，方便你確認門檻鬆緊
                print(f"[RAG弓箭手] 🗑️ 丟棄不相關段落（距離 {d:.2f} > {RAG_AUTO_RELEVANCE_THRESHOLD}）")

        if kept:
            print(f"[RAG弓箭手] 📚 自動軌保留 {len(kept)} 段相關參考（原始 {len(auto_hits)} 段）。")
            best_auto = auto_hits[0][0] if auto_hits else None
            return {
                "docs": kept,
                "hit_type": "auto",
                "best_distance": best_auto,
                "topic": None,
                "candidates": candidates,
            }

        # ---- 箭 4：兩軌都沒有夠格的結果 → 誠實回報「沒有」 ----
        # 這一步很重要：與其硬塞不相關的東西逼模型幻覺，
        # 不如明確回 none，讓盜賊客服照鐵則 10 說「這部分建議面試時再聊」。
        print("[RAG弓箭手] 🚫 兩軌都沒有足夠相關的內容，回報 none（讓模型誠實說不知道）。")
        return {
            "docs": [],
            "hit_type": "none",
            "best_distance": (manual_hits[0][0] if manual_hits else None),
            "topic": None,
            "candidates": candidates,
        }

    except Exception as e:
        print(f"[RAG弓箭手] ❌ 檢索發生未預期錯誤：{e}")
        return {"docs": [], "hit_type": "none", "best_distance": None, "topic": None, "candidates": []}


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

    【SA v3.2 修正】：純數學題成功答完卻跳出「資訊不足，是否轉接真人？」
    實測情境：「10顆糖果分給6個人」—— 弓箭手正確回報 rag_hit_type="none"
    （這本來就不是履歷題，知識庫查無內容是【正確】結果），
    計算也順利完成，結果舊版第二條規則直接把「知識庫沒東西」當成「資訊不足」，
    害使用者在拿到正確答案之後莫名其妙被問要不要轉真人。

    正解：知識庫沒東西不等於答不出來 —— 只有在「知識庫沒東西」
    【而且】「也沒有靠工具查到任何東西」的時候，才算真的資訊不足。
    """
    # 1. 多智能體明確回報有任務失敗 → 這次回答不完整，主動提供真人管道
    if not tools_all_succeeded:
        return True

    # 2. 知識庫沒撈到，而且回覆本身也沒有實質內容 → 才算資訊不足
    #    （工具成功答完的數學題／搜尋題不該落在這裡）
    if rag_hit_type == "none" and not relevant_knowledge:
        # 回覆裡有數字或有一定長度，代表工具其實有交出東西，不必轉真人
        has_substance = bool(re.search(r'\d', ai_text or "")) or len((ai_text or "").strip()) > 40
        if not has_substance:
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

                    messages.append({"role": "tool", "content": tool_result})
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
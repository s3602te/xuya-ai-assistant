# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
# 1. 引入系統操作、網路請求、深度學習框架等標準套件
import os
import requests
import torch
from sentence_transformers import SentenceTransformer

# 2. 引入自訂模組，包含全域設定參數與 ChromaDB 雙軌資料庫實體
from config import *
from database import collection_manual, collection_auto
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
# Ollama 多模態生成模組開始
# ============================
def get_ollama_response(prompt, image_b64=None, model_name="XUYA:latest"):
    try:
        # 1. 建立請求酬載 (Payload)：設定模型名稱、關閉串流模式以獲得完整回答，並限制最大生成長度
        payload = {
            "prompt": prompt, 
            "model": model_name, 
            "stream": False,
            "options": {
                "num_predict": 1024
            }
        }
        
        # 2. 多模態 (Multimodal) 處理：若有傳入 Base64 圖片編碼，則將其附加至影像陣列中
        if image_b64:
            payload["images"] = [image_b64]
            
        # 3. 發送 POST 請求至本地端的 Ollama API，並設定逾時保護為 300 秒
        r = requests.post(f"{OLLAMA_API_BASE_URL}/api/generate", json=payload, timeout=300)
        
        # 4. 檢查 HTTP 狀態碼，若非 200 則拋出例外
        r.raise_for_status()
        
        # 5. 解析 JSON 回應，濾除頭尾空白後回傳純文字結果
        return r.json().get('response', '').strip()

    except Exception as e:
        # 6. 例外處理：若通訊異常或模型崩潰，根據是否為圖片處理模式回傳對應錯誤代碼
        print(f"[Ollama 錯誤] {e}")
        if image_b64:
            return "AI_IMAGE_ERROR"
        return "【系統提示】AI 通訊錯誤。"
# ============================
# Ollama 多模態生成模組結束
# ============================
# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import os
import requests
import torch
from sentence_transformers import SentenceTransformer

# 引入自訂模組
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

# 載入 Embedding 模型用於自然語言向量化 (初始化即載入至記憶體)
print(f"[系統] 正在載入 Embedding 模型 ({DEVICE})...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
# ============================
# 裝置硬體偵測與模型初始化結束
# ============================

# ============================
# RAG 知識庫檢索模組開始
# ============================
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
            n_results=1
        )
        
        # 檢查 A 軌是否有資料
        if results_manual['distances'] and len(results_manual['distances'][0]) > 0:
            best_dist = results_manual['distances'][0][0]
            print(f"[檢索路由] 查找高精準區，最佳距離分數為: {best_dist:.3f}")
            
            ROUTING_THRESHOLD = 2.0 
            
            if best_dist < ROUTING_THRESHOLD:
                # 🎯 命中高精準區
                matched_q = results_manual['documents'][0][0]
                matched_a = results_manual['metadatas'][0][0].get('answer', '無對應解答')                
                formatted_ans = f"【標準問題】{matched_q}\n【標準解答】{matched_a}"
                res.append(formatted_ans)
                
                print("[檢索路由] 🎯 命中高精準區，直接回傳標準答案。")
                return res

        # 3. Stage 2: Fallback 機制 - 轉向 B 軌 (自動擴展資料庫)
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

def needs_contact_footer(relevant_knowledge, ai_text: str) -> bool:
    # 判斷 AI 回覆內容是否具有不確定性，以決定是否觸發真人轉接按鈕
    if not relevant_knowledge: return True
    markers = ["抱歉", "無法提供", "不知道", "不清楚"]
    return any(m in ai_text for m in markers)
# ============================
# RAG 知識庫檢索模組結束
# ============================

# ============================
# Ollama 多模態生成模組開始
# ============================
def get_ollama_response(prompt, image_b64=None, model_name="XUYA:latest"):
    try:
        # 1. 關閉 stream 模式，確保 AI 可以深思熟慮
        payload = {
            "prompt": prompt, 
            "model": model_name, 
            "stream": False,
            "options": {
                "num_predict": 1024
            }
        }
        
        # 2. 多模態 (Multimodal) 圖片處理
        if image_b64:
            payload["images"] = [image_b64]
            
        r = requests.post(f"{OLLAMA_API_BASE_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get('response', '').strip()

    except Exception as e:
        print(f"[Ollama 錯誤] {e}")
        if image_b64:
            return "AI_IMAGE_ERROR"
        return "AI 通訊錯誤。"
# ============================
# Ollama 多模態生成模組結束
# ============================
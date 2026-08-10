# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import os
import glob
import json
import re
import fitz  # 引入 PyMuPDF，用於高效解析 PDF 文件
import requests
import pandas as pd
# ============================
# 核心模組與套件引入結束
# ============================


# ============================
# 系統路徑與全域參數設定開始
# ============================
# 1. 動態取得當前腳本所在的絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定義待處理 PDF 資料夾 (Inbox) 與最終輸出的 CSV 總表路徑
PDF_FOLDER = os.path.join(BASE_DIR, "pdf_inbox")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "0731ai問答總表.csv")

# 3. 定義 Ollama 本地端 API 節點與使用的核心模型 (設定為客製化模型 XUYA:latest)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "XUYA:latest"  

# 4. 文本分塊 (Chunking) 參數設定：稍微拉大段落長度，確保 AI 有足夠上下文產生完整 QA
CHUNK_SIZE = 800  
OVERLAP = 100
# ============================
# 系統路徑與全域參數設定結束
# ============================


# ============================
# 文本分塊 (Text Chunking) 邏輯開始
# ============================
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_length = len(text)
    
    # 1. 透過滑動視窗演算法，依據指定的 Size 與 Overlap 進行文本切割
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # 2. 指針前進時扣除 overlap，保留上下文重疊區域
        start += (chunk_size - overlap)
    return chunks
# ============================
# 文本分塊 (Text Chunking) 邏輯結束
# ============================


# ============================
# 大語言模型 (LLM) Q&A 萃取邏輯開始
# ============================
def ask_ollama_for_qa(chunk_text, filename):
    # 1. 根據檔名動態生成資料分類標籤，作為 RAG 系統中高權重的識別依據
    # 取消限定機型，改用完整檔名作為分類 (去掉 .pdf)
    category_name = filename.replace(".pdf", "")
    
    # 2. 構建 Prompt (提示詞工程)：嚴格限制輸出格式為標準 JSON 陣列
    prompt = f"""
你是一個專業的客服知識庫建置專家。請閱讀以下來自《{filename}》的片段，將其內容提煉成 2 到 4 組「使用者最常問的問題與解答 (Q&A)」。

【文件片段】：
{chunk_text}

【嚴格要求】：
1. 所有的問題 (question) 開頭都「必須」加上【資料分類：{category_name}】，例如：「【資料分類：{category_name}】如何進行退貨操作？」。
2. 答案 (answer) 必須簡短精確，完全基於內文。
3. 一律使用「繁體中文」回答。
4. 必須『嚴格』僅輸出標準 JSON 陣列格式，不要包含任何 markdown 標記或額外解釋，格式如下：
[
  {{"question": "【資料分類：{category_name}】問題1", "answer": "答案1"}},
  {{"question": "【資料分類：{category_name}】問題2", "answer": "答案2"}}
]
"""
    # 3. 封裝 API 請求載體 (Payload)，並將溫度 (temperature) 設低以確保輸出穩定性
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    try:
        # 4. 發送 HTTP POST 請求至 Ollama (設定超時為 120 秒防呆)
        r = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        r.raise_for_status()
        raw_response = r.json().get("response", "").strip()
        
        # 5. 清洗回傳字串：透過正規表達式剝離可能存在的 Markdown 標記 (如 ```json ... ```)
        clean_json_str = re.sub(r'```json\s*|\s*```', '', raw_response).strip()
        qa_list = json.loads(clean_json_str)
        
        # 6. 回傳解析後的 JSON 物件 (若非陣列格式則回傳空陣列以策安全)
        return qa_list if isinstance(qa_list, list) else []
    except Exception as e:
        print(f"  └─ ⚠️ Ollama 轉換 QA 發生錯誤: {e}")
        return []
# ============================
# 大語言模型 (LLM) Q&A 萃取邏輯結束
# ============================


# ============================
# 批次處理與 CSV 總表寫入主程式開始
# ============================
def main():
    # 1. 掃描輸入目錄下的所有 PDF 檔案
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_files:
        print(f"[提醒] 在 {PDF_FOLDER} 找不到 PDF 檔案。")
        return

    # 2. 建立快取機制 (Cache)：檢查並載入已處理的檔案紀錄，避免重複生成消耗算力
    # 檢查並載入已處理的檔案紀錄，避免重複生成
    processed_files = set()
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV_PATH, encoding='utf-8-sig')
            # 若總表內具備 source 欄位，提取為已處理清單
            if 'source' in df_existing.columns:
                processed_files = set(df_existing['source'].unique())
                print(f"[系統] 發現總表，已載入 {len(processed_files)} 筆已處理檔案紀錄。")
        except Exception as e:
            print(f"[警告] 無法讀取總表: {e}")

    all_qa_pairs = []

    # 3. 開始迭代處理每個 PDF 檔案
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        
        # 4. 驗證檔案是否已在快取清單中，若是則略過 (Idempotency)跳過已經處理過的檔案
        if filename in processed_files:
            print(f"⏩ [跳過] {filename} 先前已經處理過。")
            continue

        print(f"\n[處理中] 開始讀取 PDF: {filename}")
        
        # 5. 使用 PyMuPDF (fitz) 解析並提取 PDF 文字
        full_text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text:
                    # 使用正則表達式濾除連續符號，確保文本品質
                    full_text += re.sub(r'\.{4,}|-{4,}|_{4,}', ' ', text) + "\n"
            doc.close()
        except Exception as e:
            print(f"❌ 讀取 {filename} 失敗: {e}")
            continue

        if not full_text.strip(): continue

        # 6. 執行文本分塊，並針對每個 Chunk 向 LLM 請求 QA 萃取
        chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)
        for idx, chunk in enumerate(chunks):
            print(f"  ├─ 正在處理第 {idx+1}/{len(chunks)} 段...", end="\r")
            
            qa_pairs = ask_ollama_for_qa(chunk, filename)
            
            # 將生成的問答對注入 source 欄位並收集
            for item in qa_pairs:
                if "question" in item and "answer" in item:
                    all_qa_pairs.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "source": filename
                    })
        print(f"\n  └─ ✅ {filename} 處理完畢！已累積生成 {len(all_qa_pairs)} 組 QA。")

    # 7. 最終資料匯整與寫入 CSV 階段
    if all_qa_pairs:
        df_new = pd.DataFrame(all_qa_pairs)
        
        # 8. 判斷如果舊的 CSV 已存在採取追加寫入 (Append) 或新建檔案 (Create) 模式
        if os.path.exists(OUTPUT_CSV_PATH):
            df_old = pd.read_csv(OUTPUT_CSV_PATH, encoding='utf-8-sig')
            # 將新舊資料合併，並針對 question 欄位進行去重 (Drop Duplicates) 確保資料唯一性
            df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['question'])
            df_combined.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
            print(f"\n🎉 成功追加 QA 至總表：{OUTPUT_CSV_PATH} (共 {len(df_combined)} 筆)")
        else:
            df_new.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
            print(f"\n🎉 成功建立新總表：{OUTPUT_CSV_PATH} (共 {len(df_new)} 筆)")

if __name__ == "__main__":
    main()
# ============================
# 批次處理與 CSV 總表寫入主程式結束
# ============================
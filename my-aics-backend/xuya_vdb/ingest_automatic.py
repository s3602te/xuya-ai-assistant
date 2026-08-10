# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import os
import glob
import fitz  # 引入 PyMuPDF，用於高效解析與擷取 PDF 文件文字
import chromadb
from sentence_transformers import SentenceTransformer
import re
# ============================
# 核心模組與套件引入結束
# ============================


# ============================
# 系統路徑與全域參數設定開始
# ============================
# 1. 取得當前執行腳本的絕對路徑 (指向 xuya_vdb 目錄)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定位專案根目錄 (向上一層，指向 my-aics-backend)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 3. 設定待處理 PDF 檔案的輸入目錄 (pdf_inbox)
PDF_FOLDER = os.path.join(PROJECT_ROOT, "pdf_inbox")

# 4. 設定 ChromaDB 向量資料庫的實體儲存路徑
DB_PATH = os.path.join(CURRENT_DIR, "chroma_storage")

# 5. 指定文本向量化嵌入模型 (Embedding) 的核心模型 (須與系統主架構 app.py 保持一致)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# 6. 設定文本分塊 (Chunking) 切碎設定參數：每段 400 字，保留 50 字重疊區段以維持上下文語意連貫，避免一句話被硬生生切斷
CHUNK_SIZE = 400
OVERLAP = 50
# ============================
# 系統路徑與全域參數設定結束
# ============================


# ============================
# AI 模型與資料庫初始化開始
# ============================
print("[系統] 正在載入 Embedding 模型 (這可能需要幾秒鐘)...")
# 1. 實例化 SentenceTransformer 模型，準備進行向量運算
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("[系統] 正在連接 ChromaDB 資料庫...")
# 2. 建立 ChromaDB 持久化客戶端 (Persistent Client)，確保向量資料落地寫入硬碟
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# 3. 連接或建立名為 "xuya_qa_auto" 的資料集合 (Collection) 作為 B 軌自動擴展區
collection = chroma_client.get_or_create_collection(name="xuya_qa_auto")
# ============================
# AI 模型與資料庫初始化結束
# ============================


# ============================
# 文本分塊 (Text Chunking) 邏輯開始
# ============================
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_length = len(text)
    
    # 1. 使用滑動視窗 (Sliding Window) 演算法切割長文本
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # 2. 扣除重疊字數 (Overlap)，推進下一個切割起點
        start += (chunk_size - overlap)  # 往後移動，但保留 overlap 重疊
    return chunks
# ============================
# 文本分塊 (Text Chunking) 邏輯結束
# ============================


# ============================
# 自動化寫入 (Data Ingestion) 主程式開始 (讀取 PDF 並存入資料庫)
# ============================
def main():
    # 1. 掃描輸入目錄下所有的 PDF 檔案
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_files:
        print(f"[提醒] 找不到任何 PDF 檔案。")
        return

    # 2. 遍歷並處理每一個 PDF 檔案
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n[處理中] 開始讀取: {filename}")
        
        # 3. 讀取並清洗 PDF 文本資料
        full_text = ""
        try:
            # 使用 PyMuPDF (fitz) 高效解析 PDF 內容
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text:
                    # 透過正規表達式濾除連續的無意義標點符號 (如目錄虛線)，優化資料品質
                    full_text += re.sub(r'\.{4,}|-{4,}|_{4,}', ' ', text) + "\n"
            doc.close()
        except Exception as e:
            print(f"[錯誤] 讀取 {filename} 失敗: {e}")
            continue

        # 排除完全空白的無效文件
        if not full_text.strip(): continue

        # 4. 執行文本分塊將文字切碎
        print(f"  -> 正在執行文本分塊 (Chunking)切碎文字...")
        chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)
        print(f"  -> 共分出 {len(chunks)} 個段落。")

        # 5. 建立 ChromaDB 寫入所需的資料載體結構 (Vectors & Metadata) 準備存入資料庫容器
        documents = []
        embeddings = []
        metadatas = []
        ids = []

        print(f"  -> 正在將文字轉為向量 (Embedding) 並存入資料庫...")
        for i, chunk in enumerate(chunks):
            # 6. 強化文本語意：將檔名轉為分類標籤，並注入段落開頭以提高檢索的精準度與權重 (取消限定機型，改用通用分類)
            category_name = filename.replace(".pdf", "")
            enhanced_chunk = f"【資料分類：{category_name}】\n這段內容屬於 {category_name}。{chunk}"
            
            # 7. 將強化後的文本存入文檔列表
            documents.append(enhanced_chunk)
            
            # 8. 將文本轉換為高維度向量矩陣 (轉為 list 以符合 ChromaDB 寫入格式，向量化數字，這裡只能算一次，且必須算 enhanced_chunk))
            vector = embedding_model.encode(enhanced_chunk).tolist()
            embeddings.append(vector)
            
            # 9. 綁定元數據 (Metadata) 供後續精準刪除 (如 delete_pdf.py) 或條件過濾使用
            metadatas.append({"source": filename, "chunk_index": i})
            
            # 10. 生成該段落的唯一識別碼 (Unique ID)
            ids.append(f"{filename}_chunk_{i}")

        # 11. 執行批次寫入 (Batch Insert) 至 ChromaDB
        try:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ [成功] {filename} 已永久寫入 ChromaDB 向量資料庫！")
        except Exception:
            print(f"⚠️ [注意] {filename} 的內容可能已經存在於資料庫中 (ID 衝突)。")
# ============================
# 自動化寫入 (Data Ingestion) 主程式結束
# ============================


# ============================
# 程式執行入口開始
# ============================
if __name__ == "__main__":
    main()
# ============================
# 程式執行入口結束
# ============================
# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
# ============================
# 核心模組與套件引入結束
# ============================


# ============================
# 系統路徑與全域參數設定開始
# ============================
# 1. 取得當前執行腳本的絕對路徑 (指向 xuya_vdb 目錄)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 定位專案根目錄 (向上一層，指向 my-aics-backend)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 3. 設定來源資料檔案路徑 (指向根目錄下的 CSV 高精準問答總表，CSV 在外層 (BASE_DIR)，資料庫就建在當前資料夾 (CURRENT_DIR))
CSV_FILE_PATH = os.path.join(BASE_DIR, "0731ai問答總表.csv")

# 4. 設定 ChromaDB 向量資料庫的實體儲存路徑 (與自動區共用同一個實體庫(保險箱)，但存放於不同抽屜 Collection)
DB_PATH = os.path.join(BASE_DIR, "xuya_vdb", "chroma_storage")

# 5. 指定文本向量化嵌入模型 (Embedding) 的核心模型 (必須與自動區完全一致)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
# ============================
# 系統路徑與全域參數設定結束
# ============================


# ============================
# AI 模型與資料庫初始化開始
# ============================
print("[系統] 正在載入 Embedding 模型...")
# 1. 實例化 SentenceTransformer 模型，準備進行向量運算
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("[系統] 正在連接 ChromaDB 資料庫...")
# 2. 建立 ChromaDB 持久化客戶端 (Persistent Client)
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# 3. 核心差異：建立或獲取專屬的高精準資料集合 (Collection)
# 此為 A 軌 (手動高精準區)，命名為 xuya_qa_manual
collection = chroma_client.get_or_create_collection(name="xuya_qa_manual")
# ============================
# AI 模型與資料庫初始化結束
# ============================


# ============================
# 高精準資料寫入 (Manual Ingestion) 主程式開始
# ============================
def main():
    # 1. 檢查來源 CSV 檔案是否存在
    if not os.path.exists(CSV_FILE_PATH):
        print(f"[錯誤] 找不到 CSV 總表：{CSV_FILE_PATH}")
        return

    print(f"\n[處理中] 開始讀取高精準小本本 (CSV): {os.path.basename(CSV_FILE_PATH)}")
    
    try:
        # 2. 讀取 CSV 資料表 (使用 utf-8-sig 編碼以防中文 BOM 亂碼問題)
        df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig') 
        
        # 3. 提取對應的欄位資料 (確保 CSV 具備 'question' 與 'answer' 欄位)
        # ⚠️ 注意：這裡假設你的 CSV 有 'question' 和 'answer' 兩個欄位
        # 如果你的欄位名稱叫 '問題' 和 '解答'，請把中括號內的字改掉
        questions = df['question'].astype(str).tolist()
        answers = df['answer'].astype(str).tolist()
        
    except Exception as e:
        print(f"[錯誤] 讀取 CSV 失敗: {e}")
        return

    # 4. 建立 ChromaDB 寫入所需的資料載體結構 (Vectors & Metadata)
    documents = []
    embeddings = []
    metadatas = []
    ids = []

    print(f"  -> 正在將 Q&A 轉為向量並存入資料庫 (共 {len(questions)} 筆)...")
    
    # 5. 遍歷每一筆問答資料進行處理
    for i in range(len(questions)):
        q = questions[i].strip()
        a = answers[i].strip()
        
        # 6. 取消限定機型，直接提取來源分類 (Source 欄位
        source_name = df['source'].iloc[i] if 'source' in df.columns else ""
        category_name = str(source_name).strip() if pd.notna(source_name) and str(source_name).strip() else "通用分類"
        
        # 排除空值資料
        if not q or not a or q == 'nan' or a == 'nan':
            continue

        # 7. 強化提問語意：將問題 (Question) 加上通用分類標籤，提升比對精準度
        enhanced_q = f"【資料分類：{category_name}】\n{q}"

        # 8. 將強化後的問題存入文檔列表，並轉換為高維度向量矩陣
        documents.append(enhanced_q)
        embeddings.append(embedding_model.encode(enhanced_q).tolist())
        
        # 9. 封裝元數據 (Metadata)：將「標準答案 (Answer)」作為屬性隱藏於此，供檢索時精準提取
        # 把「答案」當作屬性，塞進 Metadata 貼紙裡隱藏起來
        metadatas.append({
            "source": str(source_name), 
            "type": "explicit_rule",
            "answer": a  # 將真實解答放在這邊，存放於 Metadata 中
        })
        
        # 10. 生成該筆問答的唯一識別碼 (Unique ID)
        ids.append(f"qa_manual_{i}")

    # 11. 執行覆寫寫入 (Upsert) 至 ChromaDB
    try:
        # 採用 upsert (Update or Insert) 模式更新或插入。這樣以後更新 CSV 重新執行時，它會自動覆蓋舊資料，不會發生重複寫入的錯誤。
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ [成功] 高精準 Q&A 已成功寫入 ChromaDB 向量資料庫的「xuya_qa_manual」集合中！")
        #向量資料庫=保險箱 抽屜=集合
    except Exception as e:
        print(f"❌ [寫入失敗] {e}")
# ============================
# 高精準資料寫入 (Manual Ingestion) 主程式結束
# ============================


# ============================
# 程式執行入口開始
# ============================
if __name__ == "__main__":
    main()
# ============================
# 程式執行入口結束
# ============================
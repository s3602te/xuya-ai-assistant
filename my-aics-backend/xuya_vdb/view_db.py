# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import os
import chromadb
import pandas as pd
# ============================
# 核心模組與套件引入結束
# ============================


# ============================
# 系統路徑與全域參數設定開始
# ============================
# 1. 動態取得當前執行腳本的絕對路徑 (指向 xuya_vdb 目錄)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 設定 ChromaDB 向量資料庫的實體儲存路徑 (與當前腳本位於同層目錄)
DB_PATH = os.path.join(CURRENT_DIR, "chroma_storage")
# ============================
# 系統路徑與全域參數設定結束
# ============================


# ============================
# 終端機互動與執行參數配置開始
# ============================
# 1. 提供終端機互動介面 (CLI)，提示使用者選擇目標資料集合
print("請問你要檢視哪個向量資料庫集合？") #向量資料庫=保險箱 抽屜=集合
print("1. 自動擴展區 (PDF 檔案 / B軌)")
print("2. 手動高精準區 (CSV 檔案 / A軌)")
choice = input("請輸入 1 或 2: ")

# 2. 根據使用者輸入，指派對應的集合 Collection 名稱與預期匯出的 CSV 檔名
if choice == "1":
    col_name = "xuya_qa_auto"
    out_file = "db_content_auto.csv"
elif choice == "2":
    col_name = "xuya_qa_manual"
    out_file = "db_content_manual.csv"
else:
    # 例外防呆：輸入無效選項則終止程式
    print("輸入錯誤，程式結束。")
    exit()
# ============================
# 終端機互動與執行參數配置結束
# ============================


# ============================
# 向量資料庫連線與資料提取開始
# ============================
print(f"[系統] 正在連接 ChromaDB 向量資料庫的【{col_name}】集合...") #向量資料庫=保險箱 抽屜=集合
try:
    # 1. 建立 ChromaDB 持久化客戶端 (Persistent Client)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    # 2. 嘗試獲取指定的資料集合 (Collection)
    collection = chroma_client.get_collection(name=col_name)
except Exception as e:
    # 若集合不存在，提示使用者需先執行 ingest 寫入腳本
    print(f"[錯誤] 無法連接該資料集合，請確認是否已經執行過資料寫入 (Ingest) 程式: {e}")
    exit()

print("[系統] 正在取出所有資料...")

# 3. 執行無條件查詢 (get)，提取該集合內所有的文檔內容 (documents) 與元數據 (metadatas) 把所有文字和標籤拿出來
data = collection.get(include=["documents", "metadatas"])

# 4. 驗證資料庫是否為空
if not data['documents']:
    print("⚠️ 資料庫目前無任何文檔紀錄！")
else:
    # 5. 宣告陣列容器，整理成表格格式用於儲存結構化後的資料列
    rows = []
    
    # 6. 遍歷提取的資料集合，進行結構化整理
    for i in range(len(data['documents'])):
        doc = data['documents'][i]
        meta = data['metadatas'][i]
        source = meta.get('source', '未知來源')
        # 7. 嘗試提取隱藏於元數據 (Metadata) 中的解答欄位 (適用於 A 軌高精準 QA 集合)
        # 把隱藏在 metadata 的 answer 抓出來
        answer = meta.get('answer', '') 
        
        # 8. 建立基礎資料基本欄位映射結構 (Dictionary)
        row_data = {
            "來源檔案": source, 
            "向量比對文字 (問題或段落)": doc
        }
        
        # 9. 若存在解答欄位，則動態擴充資料結構
        # 如果這筆資料有 answer (代表是 A 軌的高精準 QA)，就多加一個欄位顯示
        if answer:
            row_data["解答 (隱藏標籤)"] = answer
            
        rows.append(row_data)

    # 10. 利用 Pandas 將結構化陣列轉換為資料表物件 (DataFrame)存成 CSV 檔案
    df = pd.DataFrame(rows)
    
    # 11. 指定匯出路徑 (儲存於當前的 xuya_vdb 目錄下)
    output_path = os.path.join(CURRENT_DIR, out_file)
    
    # 12. 寫入 CSV 檔案 (採用 utf-8-sig 編碼以確保 Excel 開啟時不會產生中文 BOM 亂碼)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 成功！總共提取出 {len(rows)} 筆文檔段落。")
    print(f"檔案已匯出至：{output_path}")
# ============================
# 向量資料庫連線與資料提取結束
# ============================
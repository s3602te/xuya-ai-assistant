# -*- coding: utf-8 -*-
# ============================
# 核心模組與環境設定開始
# ============================
import os
import chromadb

# 1. 動態取得當前腳本的絕對路徑，確保跨環境執行不報錯
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 組合出 ChromaDB 向量資料庫的實體儲存路徑
DB_PATH = os.path.join(BASE_DIR, "chroma_storage")   
# ============================
# 核心模組與環境設定結束
# ============================


# ============================
# 向量資料庫刪除邏輯區塊開始
# ============================
def delete_by_filename(filename, collection_name):
    # 1. 建立連接 ChromaDB 資料庫
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        # 2. 嘗試獲取指定的資料集合 (Collection / 即原本俗稱的抽屜)
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        # 若集合不存在，代表尚未執行過任何資料寫入 (Ingest)
        print(f"❌ 找不到該資料庫抽屜 ({collection_name})，請確認是否已經執行過 ingest 程式。")
        return

    # 3. 執行精準條件刪除 (Conditional Deletion)
    # 透過 metadata 過濾器 (where 子句)，將來源名稱為特定檔名的向量資料批次移除
    # 告訴 ChromaDB：請把 metadata 裡面 "source" 等於某個檔名的資料通通刪掉
    try:
        collection.delete(where={"source": filename})
        print(f"✅ 已成功將「{filename}」從保險箱的【{collection_name}】中徹底移除！")
    except Exception as e:
        print(f"❌ 刪除失敗或找不到檔案: {e}")
# ============================
# 向量資料庫刪除邏輯區塊結束
# ============================


# ============================
# 終端機互動與執行入口開始
# ============================
if __name__ == "__main__":
    # 1. 提供終端機互動介面 (CLI)，讓使用者選擇目標資料集合 (A軌或B軌)
    print("請問你要刪除哪個保險箱裡的檔案？")
    print("1. 自動擴展區 (PDF 檔案)")
    print("2. 手動高精準區 (CSV 檔案)")
    choice = input("請輸入 1 或 2: ")
    
    # 2. 根據使用者輸入，指派對應的 Collection 名稱
    if choice == "1":
        col_name = "xuya_qa_auto"
    elif choice == "2":
        col_name = "xuya_qa_manual"
    else:
        # 例外防呆：輸入無效選項則終止程式
        print("輸入錯誤，程式結束。")
        exit()

    # 3. 請求輸入欲刪除的來源檔案名稱 (作為刪除時的比對條件)
    target = input("請輸入想要刪除的完整檔名 (例如: V12說明書.pdf 或 manual_csv): ")
    
    # 4. 驗證輸入不為空後，呼叫核心刪除函式執行清除作業
    if target:
        delete_by_filename(target, col_name)
# ============================
# 終端機互動與執行入口結束
# ============================
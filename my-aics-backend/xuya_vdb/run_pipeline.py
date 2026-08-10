# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import os
import shutil
import subprocess # 引入子程序模組，用於在系統層級呼叫其他 Python 腳本
# ============================
# 核心模組與套件引入結束
# ============================


# ============================
# 系統路徑與全域參數設定開始
# ============================
# 1. 動態取得當前執行腳本的絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 設定資料管線的輸入端 (Inbox) 與封存端 (Archive) 路徑
PDF_INBOX = os.path.join(BASE_DIR, "pdf_inbox")
PDF_ARCHIVE = os.path.join(BASE_DIR, "pdf_archive")

# 3. 明確指定子程序執行時的工作目錄 (Working Directory)，確保各腳本的相對路徑能正確對應
SCRIPT_DIR = os.path.join(BASE_DIR, "xuya_vdb")
# ============================
# 系統路徑與全域參數設定結束
# ============================


# ============================
# 自動化管線 (Data Pipeline) 核心執行邏輯開始
# ============================
def run_pipeline():
    # 1. 執行前置檢查：掃描收件匣內是否存在待處理的 PDF 檔案 檢查是否有檔案需要處理
    pdf_files = [f for f in os.listdir(PDF_INBOX) if f.endswith('.pdf')]
    if not pdf_files:
        print("收件匣沒有 PDF，提早收工。")
        return

    # 確保存檔目錄存在，若無則自動建立 (Idempotent Directory Creation 機制) 確保備份資料夾存在
    if not os.path.exists(PDF_ARCHIVE):
        os.makedirs(PDF_ARCHIVE)

    try:
        # 3. 啟動 A 軌 (精準問答) 階段一：透過 LLM 逆向工程萃取 Q&A
        print("\n=== [步驟一] 啟動 A 軌前半段：由 AI 閱讀 PDF 並生成 QA (pdf_to_qa_csv.py) ===")
        # 將 cwd 更改為 SCRIPT_DIR，確保子腳本能在獨立的工作路徑下安全執行
        subprocess.run(["python", "pdf_to_qa_csv.py"], check=True, cwd=SCRIPT_DIR)
        
        # 4. 啟動 A 軌 (精準問答) 階段二：將萃取出的 Q&A 寫入高精準向量資料庫
        print("\n=== [步驟二] 啟動 A 軌後半段：將生成的 CSV 寫入資料庫 (ingest_manual.py) ===")
        subprocess.run(["python", "ingest_manual.py"], check=True, cwd=SCRIPT_DIR)

        # 5. 啟動 B 軌 (自動擴展) 階段：將原始 PDF 文本分塊並寫入備用向量資料庫
        print("\n=== [步驟三] 啟動 B 軌：切碎原始 PDF 並寫入備用資料庫 (ingest_automatic.py) ===")
        subprocess.run(["python", "ingest_automatic.py"], check=True, cwd=SCRIPT_DIR)

        # 6. 管線收尾作業 (Housekeeping)：將處理完畢的 PDF 移至歸檔區，保持收件匣整潔
        print("\n=== [步驟四] 打掃戰場：將已處理的 PDF 移至 pdf_archive ===")
        for filename in pdf_files:
            source_path = os.path.join(PDF_INBOX, filename)
            dest_path = os.path.join(PDF_ARCHIVE, filename)
            # 執行檔案移動 (若目標路徑已存在同名檔案將自動覆寫)
            shutil.move(source_path, dest_path)
            print(f"  -> 已封存: {filename}")
            
        print("\n🎉 所有管線任務順利完成！雙軌資料庫已更新！")

    except subprocess.CalledProcessError as e:
        # 捕捉子程序執行失敗的例外狀況，防止主程式無預警崩潰，並輸出錯誤代碼以供除錯
        print(f"\n❌ 發生錯誤，管線中斷！錯誤代碼: {e.returncode}")
# ============================
# 自動化管線 (Data Pipeline) 核心執行邏輯結束
# ============================


# ============================
# 程式執行入口開始
# ============================
if __name__ == "__main__":
    run_pipeline()
# ============================
# 程式執行入口結束
# ============================
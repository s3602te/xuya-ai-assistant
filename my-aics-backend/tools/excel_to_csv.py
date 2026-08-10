# -*- coding: utf-8 -*-
# ============================
# 核心模組與套件引入開始
# ============================
import pandas as pd # 1. 引入資料處理庫，用於高效率的 DataFrame 矩陣操作
import csv          # 2. 引入 CSV 處理庫，確保匯出時的引號編碼安全
import sys          # 3. 引入系統模組，用於觸發系統層級的致命錯誤中斷
# ============================
# 核心模組與套件引入結束
# ============================


# ============================
# 系統參數與靜態變數設定開始
# ============================
# 1. 定義資料轉化管線 (ETL Pipeline) 的輸入來源檔案
INPUT_EXCEL = '0731ai問答總表.xlsx'

# 2. 定義資料轉化管線 (ETL Pipeline) 的輸出目標檔案
OUTPUT_CSV = 'qa_dataset_output.csv'
# ============================
# 系統參數與靜態變數設定結束
# ============================


# ============================
# 資料轉化管線 (ETL Pipeline) 主程式開始
# ============================
def main():
    print(f"開始處理檔案: {INPUT_EXCEL}")
    
    # 1. 執行資料萃取 (Data Extraction) 階段
    try:
        # 呼叫 openpyxl 引擎解析 Excel 檔案至 Pandas DataFrame 記憶體中
        df = pd.read_excel(INPUT_EXCEL, engine='openpyxl')
    except FileNotFoundError:
        # 例外防呆：若輸入來源遺失，觸發致命錯誤並中斷管線
        print(f"❌ 找不到檔案！請確認 '{INPUT_EXCEL}' 是否與 excel_to_csv.py 放在同一個資料夾內。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 讀取 Excel 失敗: {e}")
        sys.exit(1)


    # 2. 執行欄位自動映射 (Fuzzy Column Mapping) 階段
    # 主動巡覽表頭，利用模糊比對鎖定包含「問題」及「答案」的目標欄位，提升對前端 Excel 格式的容錯率
    cols = df.columns.astype(str).tolist()
    q_col = next((c for c in cols if '問題' in c), None)
    a_col = next((c for c in cols if '答案' in c or '回覆' in c), None)

    if not q_col or not a_col:
        # 缺乏必要欄位時終止轉換管線，防止產生髒資料
        print(f"❌ 找不到目標欄位！目前讀取到的 Excel 標題列有: {cols}")
        sys.exit(1)
        
    print(f"✅ 成功鎖定欄位: 將 Excel 的 [{q_col}] -> 轉為 question，[{a_col}] -> 轉為 answer")


    # 3. 資料萃取與重命名 執行資料變換與架構正規化 (Schema Normalization) 階段
    # 僅提取鎖定的目標欄位，並透過 copy() 脫離原 DataFrame 參照以防記憶體污染
    df_qa = df[[q_col, a_col]].copy()
    
    # 強制將欄位名稱統一正規化為 RAG 系統標準鍵值 (question / answer)
    df_qa.columns = ['question', 'answer']


    # 4. 執行資料淨化 (Data Sanitization) 階段
    # 強制將欄位轉型為字串型態，並清除字串前後潛藏的空白鍵與換行符號 (Trim)
    df_qa['question'] = df_qa['question'].astype(str).str.strip()
    df_qa['answer'] = df_qa['answer'].astype(str).str.strip()

    # 透過布林索引 (Boolean Indexing) 濾除各類型的空值與無效紀錄
    # 包含被 pandas 誤轉為字串的 'nan'，以及只剩下空白的字串
    df_qa = df_qa[
        (df_qa['question'] != 'nan') & 
        (df_qa['answer'] != 'nan') & 
        (df_qa['question'] != '') & 
        (df_qa['answer'] != '')
    ]


    # 5. 執行資料寫入 (Data Load) 階段
    try:
        # 將 DataFrame 持久化為 CSV 格式
        # 啟用 quoting=csv.QUOTE_MINIMAL 確保內容中若含有逗號，會自動以雙引號包覆，嚴防 CSV 結構崩壞
        df_qa.to_csv(
            OUTPUT_CSV, 
            index=False, 
            encoding='utf-8-sig', # 採用 utf-8-sig 以相容 Windows Excel 中文編碼
            quoting=csv.QUOTE_MINIMAL
        )
        print(f"🎉 轉換成功！共產出 {len(df_qa)} 筆乾淨的問答資料，已儲存至 {OUTPUT_CSV}")
    except Exception as e:
        print(f"❌ 寫入 CSV 失敗: {e}")
# ============================
# 資料轉化管線 (ETL Pipeline) 主程式結束
# ============================


# ============================
# 程式執行入口開始
# ============================
if __name__ == "__main__":
    main()
# ============================
# 程式執行入口結束
# ============================
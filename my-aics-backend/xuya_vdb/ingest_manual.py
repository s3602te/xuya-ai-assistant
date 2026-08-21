# -*- coding: utf-8 -*-
# xuya_vdb/ingest_manual.py
# ============================
# 核心模組與套件引入開始
# ============================
import os
import sys
import glob
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

# 3. 設定來源資料檔案路徑
# 【SA v2.5 擴充】：現在支援「多份 CSV」。
#
# 原本只讀根目錄那一份 0731ai問答總表.csv，但知識庫一直長大之後，
# 全部擠在同一個檔案裡會很難維護（改一題履歷要在幾百行裡找）。
# 現在的規則是：
#   (1) 根目錄的 0731ai問答總表.csv    ← 主檔，一定會讀（向下相容，不用搬動）
#   (2) knowledge/ 資料夾底下所有 *.csv ← 選用，有就一起讀
#
# 所以你可以這樣分檔管理：
#   my-aics-backend/
#   ├── 0731ai問答總表.csv          ← 主檔
#   └── knowledge/
#       ├── 專案_電子發票.csv
#       ├── 人格特質_補充.csv
#       └── 面試被問倒的題目.csv
#
# 每個檔案的欄位規則完全一樣（source, question, answer），
# ID 會自動帶上檔名前綴避免衝突，所以不同檔案之間不會互相覆蓋。
LEGACY_CSV_PATH = os.path.join(BASE_DIR, "0731ai問答總表.csv")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

# 4. 設定 ChromaDB 向量資料庫的實體儲存路徑 (與自動區共用同一個實體庫，但存放於不同 Collection)
DB_PATH = os.path.join(BASE_DIR, "xuya_vdb", "chroma_storage")

# 5. 指定文本向量化嵌入模型 (Embedding) 的核心模型 (必須與自動區完全一致)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

COLLECTION_NAME = "xuya_qa_manual"

# ============================
# 【SA v2.3 重大變更】距離度量改用餘弦相似度
# ============================
# 為什麼要改？rag_calibrate.py 的實測數據說話：
#
#   把「面試官會問、答案確實在 CSV 裡」的題目，和「完全無關、該走搜尋工具」的題目
#   各測一輪，比較兩組的分數有沒有辦法用一條線切開：
#
#     L2 距離（ChromaDB 預設）  分離度 = -0.615  ❌ 兩組重疊，怎麼調門檻都會誤判
#     餘弦相似度                分離度 = +0.096  ✅ 兩組分得開
#
#   原因在於 paraphrase-multilingual-mpnet-base-v2 輸出的向量【沒有正規化】，
#   而 L2 距離會被向量長度主導，語意相似度反而被稀釋掉。
#   餘弦相似度會先除掉長度，只比方向，正好避開這個問題。
#
# 【重要】：hnsw:space 是 collection 建立當下就固定的，事後無法修改。
#          所以這支腳本預設會【先刪除舊 collection 再重建】。
#          資料本身不會遺失 —— 全部來源都在 CSV 裡，重跑一次就回來了。
DISTANCE_SPACE = "cosine"
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
# Collection 重建邏輯開始
# ============================
def prepare_collection(force_rebuild: bool):
    """
    【SA v2.3 新增】取得 collection，必要時先刪除重建以套用新的距離度量。

    ChromaDB 的 hnsw:space 只能在建立 collection 的當下指定，
    對已存在的 collection 呼叫 get_or_create_collection(metadata=...) 並不會改變它，
    而且不會報錯 —— 這是最容易踩到的陷阱：你以為改好了，其實還是舊的 L2。
    所以這裡明確檢查現況，需要換度量就整個砍掉重建。
    """
    existing = None
    try:
        existing = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        existing = None

    if existing is not None:
        current_space = (existing.metadata or {}).get("hnsw:space", "l2")
        count = existing.count()
        print(f"[系統] 發現既有 collection「{COLLECTION_NAME}」：{count} 筆，距離度量 = {current_space}")

        if current_space == DISTANCE_SPACE and not force_rebuild:
            print(f"[系統] 距離度量已經是 {DISTANCE_SPACE}，直接沿用（採 upsert 更新資料）。")
            return existing

        reason = "使用者指定強制重建" if force_rebuild else f"距離度量需要從 {current_space} 換成 {DISTANCE_SPACE}"
        print(f"\n⚠️  即將【刪除並重建】collection「{COLLECTION_NAME}」")
        print(f"    原因：{reason}")
        print(f"    影響：{count} 筆向量會被清空，但資料來源都在 CSV，這支腳本會立刻重新寫回。")
        print(f"    注意：另一軌 xuya_qa_auto（PDF 自動擴展區）完全不受影響。")
        answer = input("\n    確定要重建嗎？輸入 y 繼續，其他任意鍵取消：").strip().lower()
        if answer != "y":
            print("[系統] 已取消，未做任何變更。")
            sys.exit(0)

        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"[系統] 舊 collection 已刪除。")

    col = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_SPACE}
    )
    print(f"[系統] ✅ 已建立新 collection「{COLLECTION_NAME}」，距離度量 = {DISTANCE_SPACE}")
    return col
# ============================
# Collection 重建邏輯結束
# ============================


def collect_csv_files():
    """
    【SA v2.5 新增】蒐集所有要匯入的 CSV 檔案。
    主檔 + knowledge/ 底下的所有 csv，主檔排最前面。
    """
    files = []
    if os.path.exists(LEGACY_CSV_PATH):
        files.append(LEGACY_CSV_PATH)
    if os.path.isdir(KNOWLEDGE_DIR):
        extra = sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.csv")))
        files.extend(extra)
    return files


def load_rows(csv_path):
    """讀取單一 CSV，回傳 [(source, question, answer), ...]，並做欄位檢查。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for need in ("question", "answer"):
        if need not in df.columns:
            print(f"  ❌ {os.path.basename(csv_path)} 缺少必要欄位「{need}」，跳過。目前欄位：{list(df.columns)}")
            return None
    if "source" not in df.columns:
        print(f"  ⚠️ {os.path.basename(csv_path)} 沒有 source 欄位，全部條目會被標成「通用分類」。")

    rows = []
    for i in range(len(df)):
        q = str(df["question"].iloc[i]).strip()
        a = str(df["answer"].iloc[i]).strip()
        if not q or not a or q == "nan" or a == "nan":
            continue
        src = df["source"].iloc[i] if "source" in df.columns else ""
        cat = str(src).strip() if pd.notna(src) and str(src).strip() else "通用分類"
        rows.append((cat, q, a))
    return rows


# ============================
# 高精準資料寫入 (Manual Ingestion) 主程式開始
# ============================
def main():
    force_rebuild = "--rebuild" in sys.argv

    # 1. 蒐集所有來源 CSV
    csv_files = collect_csv_files()
    if not csv_files:
        print(f"[錯誤] 找不到任何來源 CSV。")
        print(f"       預期位置：{LEGACY_CSV_PATH}")
        print(f"       或是：{KNOWLEDGE_DIR}/*.csv")
        return

    print(f"\n[處理中] 找到 {len(csv_files)} 份知識檔：")
    for f in csv_files:
        print(f"  - {os.path.relpath(f, BASE_DIR)}")

    # 2. 準備 collection（必要時重建）
    collection = prepare_collection(force_rebuild)

    # 5. 建立 ChromaDB 寫入所需的資料載體結構
    documents, embeddings, metadatas, ids = [], [], [], []
    seen_questions = {}
    dup_count = 0

    # 3. 逐檔讀取並向量化
    for csv_path in csv_files:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        rows = load_rows(csv_path)
        if rows is None:
            continue
        print(f"\n  -> 正在處理 {os.path.basename(csv_path)}（{len(rows)} 筆）...")

        for i, (category_name, q, a) in enumerate(rows):
            # 【SA v2.5】：跨檔案的重複問題會互相搶答，這裡先示警再以先出現的為準
            if q in seen_questions:
                dup_count += 1
                print(f"     ⚠️ 問題重複，已略過：「{q[:24]}」（先前出現於 {seen_questions[q]}）")
                continue
            seen_questions[q] = os.path.basename(csv_path)

            # 4. 強化提問語意：把分類標籤加在問題前面一起向量化
            enhanced_q = f"【資料分類：{category_name}】\n{q}"

            documents.append(enhanced_q)
            embeddings.append(embedding_model.encode(enhanced_q).tolist())
            metadatas.append({
                "source": category_name,
                "type": "explicit_rule",
                "question_raw": q,
                "answer": a,
                "from_file": os.path.basename(csv_path),   # 【SA v2.5】方便追查某筆答案來自哪個檔案
            })
            # 【SA v2.5】：ID 帶上檔名前綴，不同檔案之間才不會互相覆蓋
            ids.append(f"qa_{stem}_{i}")

    if not documents:
        print("[錯誤] 沒有任何有效資料可寫入。")
        return

    # 5. 執行寫入
    try:
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"\n✅ [成功] 已寫入 {len(documents)} 筆到「{COLLECTION_NAME}」"
              + (f"（跨檔重複略過 {dup_count} 筆）" if dup_count else ""))

        # 6. 寫入後驗證，確認距離度量真的是我們要的那一種
        final = chroma_client.get_collection(name=COLLECTION_NAME)
        space = (final.metadata or {}).get("hnsw:space", "l2")
        print(f"   目前收錄筆數：{final.count()}")
        print(f"   距離度量：{space}")
        if space != DISTANCE_SPACE:
            print(f"   ⚠️ 度量不符預期！請加上 --rebuild 參數重跑：python ingest_manual.py --rebuild")
        else:
            print(f"\n   分類分佈：")
            counts = {}
            for m in metadatas:
                counts[m["source"]] = counts.get(m["source"], 0) + 1
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"     {k:<18} {v} 題")
            print(f"\n   下一步：回到專案根目錄執行 python rag_calibrate.py 量出新的門檻。")

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
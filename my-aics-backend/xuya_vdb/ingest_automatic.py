# -*- coding: utf-8 -*-
# xuya_vdb/ingest_automatic.py
# ============================
# 核心模組與套件引入開始
# ============================
import os
import sys
import glob
import re
import fitz  # PyMuPDF，用於高效解析與擷取 PDF 文件文字
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
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 3. 設定待處理 PDF 檔案的輸入目錄 (pdf_inbox)
PDF_FOLDER = os.path.join(PROJECT_ROOT, "pdf_inbox")

# 4. 設定 ChromaDB 向量資料庫的實體儲存路徑
DB_PATH = os.path.join(CURRENT_DIR, "chroma_storage")

# 5. 指定文本向量化嵌入模型 (須與手動軌完全一致)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

COLLECTION_NAME = "xuya_qa_auto"

# ============================
# 【SA v2.3 變更 1】距離度量改用 cosine，與手動軌保持一致
# ============================
# 為什麼自動軌也要改？
# 兩軌如果度量不同，分數尺度就完全不同（l2 大約落在 4~10，cosine 落在 0~2），
# 這樣「手動軌不夠準就退到自動軌」這個判斷根本沒有共同基準可以比較。
# 更重要的是：自動軌目前【完全沒有相關性門檻】，
# ai_core 只要手動軌沒命中就無條件撈 Top-6 塞給模型 ——
# 所以問「台北101有多高」也會拿到六段張序亞的履歷。
# 要替自動軌加上門檻，前提是兩軌的分數可以互相參照。
DISTANCE_SPACE = "cosine"

# ============================
# 【SA v2.3 變更 2】切塊改為「句子邊界感知」
# ============================
# 舊版是固定 400 字的滑動視窗，完全不管句子在哪裡結束，
# 實際跑出來的段落長這樣（真實 log）：
#     「這段內容屬於 張序亞 個人履歷與技術亮點。浮點數累加所產生的誤差。他對 AI 的理解...」
# 開頭那句「浮點數累加所產生的誤差。」是上一段被硬生生切斷的殘尾，
# 這種碎片不但沒有資訊量，還會稀釋整個段落的語意向量，讓檢索變差。
#
# 新版先依中文標點切成句子，再把句子累積成接近目標長度的段落，
# 保證每個段落都是從完整句子開始、到完整句子結束。
CHUNK_SIZE = 400        # 每段目標字數（累積到接近這個數字就切）
OVERLAP_SENTENCES = 1   # 段落之間重疊幾個句子（維持上下文連貫）
MIN_CHUNK_CHARS = 40    # 太短的段落直接丟棄，避免產生只有標題的無意義向量
# ============================
# 系統路徑與全域參數設定結束
# ============================


# ============================
# AI 模型與資料庫初始化開始
# ============================
print("[系統] 正在載入 Embedding 模型 (這可能需要幾秒鐘)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("[系統] 正在連接 ChromaDB 資料庫...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
# ============================
# AI 模型與資料庫初始化結束
# ============================


# ============================
# Collection 重建邏輯開始
# ============================
def prepare_collection(force_rebuild: bool):
    """
    取得 collection，必要時先刪除重建以套用新的距離度量。

    【重要】hnsw:space 只能在建立 collection 的當下指定。
    對已存在的 collection 呼叫 get_or_create_collection(metadata=...) 不會改變它，
    而且【不會報錯】—— 這是最容易踩到的靜默失敗。
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
            print(f"[系統] 距離度量已經是 {DISTANCE_SPACE}，直接沿用（新 PDF 會以 upsert 加入）。")
            return existing

        reason = "使用者指定強制重建" if force_rebuild else f"距離度量需要從 {current_space} 換成 {DISTANCE_SPACE}"
        print(f"\n⚠️  即將【刪除並重建】collection「{COLLECTION_NAME}」")
        print(f"    原因：{reason}")
        print(f"    影響：{count} 筆向量會被清空。")
        print(f"    ⚠️ 注意：自動軌的來源是 PDF，如果原始 PDF 已經被 run_pipeline.py")
        print(f"       搬到 pdf_archive/，請先把它們搬回 pdf_inbox/，否則重建後會是空的。")
        print(f"    另一軌 xuya_qa_manual（手動精準區）完全不受影響。")
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


# ============================
# 文本分塊 (Text Chunking) 邏輯開始
# ============================
def split_sentences(text: str):
    """
    【SA v2.3 新增】依中文與英文的句末標點切句。

    切完之後保留標點（用 lookbehind），這樣重新組回段落時讀起來還是通順的。
    換行也視為切點，因為 PDF 抽出來的文字常常用換行代表段落結束。
    """
    # 先把 PDF 常見的多餘換行壓成單一換行，避免切出一堆空句
    text = re.sub(r'\n{2,}', '\n', text)
    # 在句末標點之後切開（標點留在前一句）
    pieces = re.split(r'(?<=[。！？；!?;])\s*|\n', text)
    return [p.strip() for p in pieces if p and p.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap_sentences: int = OVERLAP_SENTENCES):
    """
    【SA v2.3 改寫】句子邊界感知的分塊。

    作法：把句子一句一句累加，長度接近 chunk_size 就收成一段，
    下一段從「前一段的最後 N 句」開始，維持上下文重疊。
    這樣每一段一定是完整句子的組合，不會再出現半截殘句。
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    buf = []
    buf_len = 0

    for s in sentences:
        # 單句就超過目標長度（例如沒有標點的長列表），直接自成一段
        if len(s) >= chunk_size:
            if buf:
                chunks.append("".join(buf))
                buf, buf_len = [], 0
            chunks.append(s)
            continue

        if buf_len + len(s) > chunk_size and buf:
            chunks.append("".join(buf))
            # 保留最後幾句當作下一段的重疊區
            tail = buf[-overlap_sentences:] if overlap_sentences > 0 else []
            buf = list(tail)
            buf_len = sum(len(x) for x in buf)

        buf.append(s)
        buf_len += len(s)

    if buf:
        chunks.append("".join(buf))

    # 過濾掉過短的碎片（通常是標題或頁碼殘留）
    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]
# ============================
# 文本分塊 (Text Chunking) 邏輯結束
# ============================


# ============================
# 自動化寫入 (Data Ingestion) 主程式開始
# ============================
def main():
    force_rebuild = "--rebuild" in sys.argv

    # 1. 掃描輸入目錄下所有的 PDF 檔案
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_files:
        print(f"[提醒] 在 {PDF_FOLDER} 找不到任何 PDF 檔案。")
        if force_rebuild:
            print("        你指定了 --rebuild 但收件匣是空的，重建後自動軌會沒有任何內容。")
            print("        如果 PDF 已被搬到 pdf_archive/，請先搬回 pdf_inbox/ 再執行。")
        return

    # 2. 準備 collection（必要時重建）
    collection = prepare_collection(force_rebuild)

    total_chunks = 0

    # 3. 遍歷並處理每一個 PDF 檔案
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n[處理中] 開始讀取: {filename}")

        # 4. 讀取並清洗 PDF 文本資料
        full_text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text:
                    # 濾除連續的無意義標點符號 (如目錄虛線)
                    full_text += re.sub(r'\.{4,}|-{4,}|_{4,}', ' ', text) + "\n"
            doc.close()
        except Exception as e:
            print(f"[錯誤] 讀取 {filename} 失敗: {e}")
            continue

        if not full_text.strip():
            print(f"[略過] {filename} 沒有可擷取的文字（可能是掃描檔，需要先做 OCR）。")
            continue

        # 5. 執行句子邊界感知的分塊
        print(f"  -> 正在依句子邊界切塊...")
        chunks = chunk_text(full_text)
        print(f"  -> 共分出 {len(chunks)} 個段落（平均 {sum(len(c) for c in chunks)//max(len(chunks),1)} 字）。")

        # 【SA v2.3 新增】：印出第一段供你目視確認切得對不對。
        # 舊版切壞的時候完全沒有徵兆，要等到查詢時看到殘句才發現。
        if chunks:
            preview = chunks[0][:80].replace("\n", " ")
            print(f"  -> 首段預覽：{preview}...")

        # 6. 建立 ChromaDB 寫入所需的資料載體結構
        documents, embeddings, metadatas, ids = [], [], [], []

        print(f"  -> 正在將文字轉為向量並存入資料庫...")
        category_name = filename.replace(".pdf", "")

        for i, chunk in enumerate(chunks):
            # 7. 強化文本語意：把檔名轉為分類標籤注入段落開頭，提高檢索精準度
            enhanced_chunk = f"【資料分類：{category_name}】\n這段內容屬於 {category_name}。{chunk}"

            documents.append(enhanced_chunk)
            embeddings.append(embedding_model.encode(enhanced_chunk).tolist())

            # 8. 綁定元數據，供 delete_pdf.py 精準刪除或條件過濾使用
            metadatas.append({
                "source": filename,
                "chunk_index": i,
                "chunk_chars": len(chunk),   # 【SA v2.3 新增】方便日後檢查切塊品質
            })

            ids.append(f"{filename}_chunk_{i}")

        # 9. 執行批次寫入
        # 【SA v2.3 修正】：舊版用 collection.add()，同一份 PDF 重跑會因為 ID 衝突而整批失敗，
        # 而且錯誤訊息只說「可能已經存在」，看不出到底寫進去了沒。
        # 改用 upsert()：同一個 ID 直接覆蓋，重跑就是單純的更新，不會再有 ID 衝突。
        try:
            collection.upsert(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            total_chunks += len(documents)
            print(f"✅ [成功] {filename} 已寫入 {len(documents)} 個段落。")
        except Exception as e:
            print(f"❌ [寫入失敗] {filename}: {e}")

    # 10. 寫入後驗證
    try:
        final = chroma_client.get_collection(name=COLLECTION_NAME)
        space = (final.metadata or {}).get("hnsw:space", "l2")
        print(f"\n[系統] 自動擴展軌目前狀態：{final.count()} 筆 / 距離度量 = {space}")
        if space != DISTANCE_SPACE:
            print(f"       ⚠️ 度量不符預期！請加上 --rebuild 重跑：python ingest_automatic.py --rebuild")
        elif total_chunks:
            print(f"       本次寫入 {total_chunks} 個段落。")
            print(f"       下一步：回專案根目錄執行 python rag_calibrate.py 確認兩軌分數尺度一致。")
    except Exception as e:
        print(f"[警告] 無法驗證最終狀態：{e}")
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
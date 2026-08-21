# -*- coding: utf-8 -*-
# rag_calibrate.py  ── v2
# ==========================================
# 【SA v2.3】RAG 高精準區門檻校準工具
# ==========================================
# 放在專案根目錄 (my-aics-backend/)，執行：python rag_calibrate.py
#
# 【v1 的判讀缺陷 —— 這一版修掉的東西】
# v1 只比較「兩組的極值有沒有交叉」，一個離群值就能把整組結論帶歪。
# 實測時「請自我介紹一下」這一題把 L2 拉到 6.190，
# 導致腳本結論變成「餘弦也重疊，兩種度量都沒救」——
# 但那題之所以分數爛，是因為【CSV 裡根本沒有自我介紹這個條目】，
# 是內容缺口，不是度量問題。把它排除後餘弦分離度立刻變成 +0.096，完全分得開。
#
# 所以 v2 改成：
#   1. 逐題標記，直接指出「是哪幾題卡住分離」，而不是只丟一個總分
#   2. 自動辨識 collection 的距離度量（l2 / cosine / ip），並用正確方向解讀分數
#   3. 會自動嘗試「排除幾題離群值後能不能分開」，藉此區分
#      「這是內容缺口」還是「這是度量問題」—— 兩者的解法完全不同
#   4. 命中判定不只看距離，也檢查「撈回來的條目主題對不對」
# ==========================================
import os
import sys
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 路徑與參數設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "xuya_vdb", "chroma_storage")
CSV_PATH = os.path.join(BASE_DIR, "0731ai問答總表.csv")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
MANUAL_COLLECTION = "xuya_qa_manual"

# ==========================================
# 測試題組
# ==========================================
# 【應該命中】：(問題, 期望命中的 source 分類)
# 刻意全部用「跟 CSV 原文不同的講法」，因為面試官不會照著你的 CSV 打字。
# 期望分類是用來檢查「撈回來的是不是對的主題」—— 距離夠近但撈錯主題，一樣是失敗。
SHOULD_HIT = [
    ("請自我介紹一下",               "自我介紹"),
    ("可以簡單說明你是誰嗎",          "自我介紹"),
    ("這個 AI 是誰做的？",            "產品介紹"),
    ("這套系統是用什麼技術寫的",       "產品介紹"),
    ("你今年幾歲",                   "基本資料"),
    ("你住在哪裡？通勤方便嗎",        "基本資料"),
    ("兵役處理完了嗎",               "基本資料"),
    ("你大學念什麼的",               "求學與轉職"),
    ("你是怎麼轉職當工程師的",        "求學與轉職"),
    ("轉職前你在做什麼",             "求學與轉職"),
    ("你想找什麼樣的職位",            "職涯規劃"),
    ("最快什麼時候可以來上班",        "職涯規劃"),
    ("你的優缺點是什麼",             "人格特質"),
    ("跟同事吵架的時候你怎麼處理",     "人格特質"),
    ("舉一個你跟人意見不合的例子",     "人格特質"),
    ("你現在在哪裡上班",              "工作經歷"),
    ("這些專案是你一個人做的嗎",       "工作經歷"),
    ("LINE 客服那個專案是你做的嗎",    "LINE客服專案"),
    ("客服系統怎麼避免 AI 亂講話",     "LINE客服專案"),
    ("柬埔寨那個案子在做什麼",        "柬埔寨稅務專案"),
    ("加油站設備怎麼監控",            "柬埔寨稅務專案"),
    ("電子發票那個模組你怎麼做的",     "電子發票專案"),
    ("你會哪些技術？",               "技術能力"),
    ("你的顯卡跑得動 AI 嗎",          "技術能力"),
    ("你懂 Transformer 嗎",          "AI與LLM"),
    ("RAG 雙軌架構怎麼運作？",        "RAG與向量資料庫"),
    ("檢索門檻你是怎麼調出來的",       "RAG與向量資料庫"),
    ("你做過 AI Agent 的專案嗎",      "多智能體架構"),
    ("你怎麼避免 Agent 無限迴圈",      "多智能體架構"),
    ("談談你的 DevOps 與 CI/CD 經驗",  "DevOps與CICD"),
    ("你有處理過 SSL 憑證嗎",         "DevOps與CICD"),
    ("使用者一直連續傳訊息怎麼辦",     "系統防護"),
    ("外部 API 掛掉的時候怎麼辦",      "系統防護"),
]

# 【不該命中】：跟履歷完全無關，應該落到搜尋或計算工具的問題
SHOULD_MISS = [
    "台北101有多高",
    "從10個人中選3個有幾種組合",
    "請上網查詢台灣本島的土地面積",
    "今天天氣如何",
    "台積電股價多少",
    "日本首相是誰",
    "幫我算 128 乘以 47",
    "明天要不要帶傘",
]


def main():
    print("\n" + "=" * 76)
    print("🎯 RAG 高精準區門檻校準工具 v2")
    print("=" * 76 + "\n")

    if not os.path.exists(DB_PATH):
        print(f"❌ 找不到向量資料庫：{DB_PATH}")
        return

    print(f"[系統] 正在載入 Embedding 模型 {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        col = client.get_collection(name=MANUAL_COLLECTION)
    except Exception as e:
        print(f"❌ 找不到 collection「{MANUAL_COLLECTION}」：{e}")
        print("   → 請先執行 python xuya_vdb/ingest_manual.py")
        return

    # ---- 辨識距離度量，決定分數要往哪個方向解讀 ----
    space = (col.metadata or {}).get("hnsw:space", "l2")
    smaller_is_closer = space in ("l2", "cosine")   # ip(內積) 才是越大越相似
    print(f"[系統] collection「{MANUAL_COLLECTION}」距離度量 = {space}"
          f"（{'越小越相似' if smaller_is_closer else '越大越相似'}）")
    print(f"[系統] 目前收錄筆數：{col.count()}")

    if space == "l2":
        print("       ⚠️ 目前仍是 ChromaDB 預設的 l2。實測顯示 l2 在這個 embedding 模型上")
        print("          無法把「該命中」與「不該命中」分開，建議改用 cosine 重建：")
        print("          python xuya_vdb/ingest_manual.py --rebuild")

    # ---- CSV 概況 ----
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
            has_src = "source" in df.columns
            print(f"[系統] 手動總表共 {len(df)} 筆，{'有' if has_src else '【沒有】'} source 欄位")
            if has_src:
                vc = df["source"].value_counts()
                print("       分類分佈：" + "、".join(f"{k}({v})" for k, v in vc.items()))
        except Exception as e:
            print(f"[警告] 讀取 CSV 失敗：{e}")
    print()

    def query_one(q):
        qv = model.encode(q)
        res = col.query(query_embeddings=[qv.tolist()], n_results=1)
        dist = res["distances"][0][0]
        meta = res["metadatas"][0][0] if res["metadatas"] and res["metadatas"][0] else {}
        doc = res["documents"][0][0] if res["documents"] and res["documents"][0] else ""
        matched_q = meta.get("question_raw") or doc.split("\n")[-1]
        return dist, meta.get("source", "?"), matched_q

    # ---- 應該命中組 ----
    print("-" * 76)
    print("【應該命中】(改寫過的問法，模擬面試官真實提問)")
    print("-" * 76)
    print(f"{'問題':<26}{'分數':>8}  {'撈到主題':<12}{'期望主題':<12} 撈到的條目")
    hit_rows = []
    for q, want_src in SHOULD_HIT:
        dist, got_src, matched_q = query_one(q)
        topic_ok = (got_src == want_src)
        mark = "✅" if topic_ok else "⚠️"
        print(f"{q[:24]:<26}{dist:>8.3f}  {got_src[:10]:<12}{want_src[:10]:<12}{mark} {matched_q[:20]}")
        hit_rows.append({"q": q, "dist": dist, "topic_ok": topic_ok})
    print()

    # ---- 不該命中組 ----
    print("-" * 76)
    print("【不該命中】(應該走搜尋 / 計算工具)")
    print("-" * 76)
    print(f"{'問題':<26}{'分數':>8}  最接近的條目")
    miss_rows = []
    for q in SHOULD_MISS:
        dist, got_src, matched_q = query_one(q)
        print(f"{q[:24]:<26}{dist:>8.3f}  {matched_q[:34]}")
        miss_rows.append({"q": q, "dist": dist})
    print()

    # ---- 分離度分析 ----
    print("=" * 76)
    print("📊 分離度分析")
    print("=" * 76)

    def worst_hit(rows):
        """命中組裡「最不像命中」的那一題"""
        return max(rows, key=lambda r: r["dist"]) if smaller_is_closer else min(rows, key=lambda r: r["dist"])

    def worst_miss(rows):
        """誤判組裡「最像命中」的那一題"""
        return min(rows, key=lambda r: r["dist"]) if smaller_is_closer else max(rows, key=lambda r: r["dist"])

    def separation(hrows, mrows):
        wh, wm = worst_hit(hrows), worst_miss(mrows)
        s = (wm["dist"] - wh["dist"]) if smaller_is_closer else (wh["dist"] - wm["dist"])
        return s, wh, wm

    sep, wh, wm = separation(hit_rows, miss_rows)
    hv = [r["dist"] for r in hit_rows]
    mv = [r["dist"] for r in miss_rows]
    print(f"\n  應該命中   最好={min(hv):.3f}  最差={max(hv):.3f}  平均={np.mean(hv):.3f}")
    print(f"  不該命中   最好={min(mv):.3f}  最差={max(mv):.3f}  平均={np.mean(mv):.3f}")
    print(f"\n  分離度 = {sep:+.3f}")
    print(f"    卡住的兩題：")
    print(f"      命中組最差 → 「{wh['q']}」  {wh['dist']:.3f}")
    print(f"      誤判組最近 → 「{wm['q']}」  {wm['dist']:.3f}")

    if sep > 0:
        cut = (wh["dist"] + wm["dist"]) / 2
        print(f"\n  ✅ 兩組完全分得開！")
        print(f"     建議把 config.py 的 RAG_HIGH_PRECISION_THRESHOLD 設為 {cut:.2f}")
        print(f"     （檢索距離小於這個值 → 直接採用手動精準軌的標準答案）")
    else:
        # 【SA v2.3 重點】：不要只說「重疊」，要區分是內容缺口還是度量問題
        print(f"\n  ❌ 兩組重疊 {abs(sep):.3f}，先判斷是少數幾題拖累、還是整體度量有問題。")
        blockers = sorted(hit_rows, key=(lambda r: -r["dist"]) if smaller_is_closer else (lambda r: r["dist"]))[:3]
        print(f"\n     命中組裡表現最差的三題：")
        for r in blockers:
            note = "" if r["topic_ok"] else "   ← 連主題都撈錯了"
            print(f"       {r['dist']:.3f}  「{r['q']}」{note}")

        # 逐一排除最差的題目，看要拿掉幾題才分得開
        remain = list(hit_rows)
        removed = []
        ok = False
        while len(remain) > 3:
            s, _, _ = separation(remain, miss_rows)
            if s > 0:
                ok = True
                break
            w = worst_hit(remain)
            remain.remove(w)
            removed.append(w)

        if ok and removed:
            s2, wh2, wm2 = separation(remain, miss_rows)
            cut = (wh2["dist"] + wm2["dist"]) / 2
            print(f"\n     排除這 {len(removed)} 題之後，分離度變成 {s2:+.3f}（分得開）：")
            for r in removed:
                print(f"       - 「{r['q']}」")
            print(f"\n     👉 這代表問題出在【CSV 內容缺口】，不是門檻也不是度量。")
            print(f"        請在 0731ai問答總表.csv 補上這幾題對應的條目，補完再跑一次。")
            print(f"        補完之後門檻大約會落在 {cut:.2f}。")
        else:
            print(f"\n     👉 拿掉離群題也分不開，代表是【距離度量】的問題。")
            if space != "cosine":
                print(f"        請改用 cosine 重建：python xuya_vdb/ingest_manual.py --rebuild")
            else:
                print(f"        已經是 cosine 仍分不開。建議：")
                print(f"        (1) 每個主題再多寫兩三種問法（實測改寫問法會掉 0.2 分）")
                print(f"        (2) 或改用中文檢索表現更好的 embedding 模型")

    # ---- 主題正確率 ----
    topic_ok_n = sum(1 for r in hit_rows if r["topic_ok"])
    print(f"\n  主題命中率：{topic_ok_n}/{len(hit_rows)}（撈回來的條目分類是否符合預期）")
    if topic_ok_n < len(hit_rows):
        print("     沒對上的題目代表該主題的問法覆蓋不足，建議在 CSV 多寫幾種說法。")

    print("\n" + "=" * 76 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中斷。")
        sys.exit(1)
# -*- coding: utf-8 -*-
# preflight_check.py
# ==========================================
# 【SA v2.1 新增】航空母艦啟動前置檢查
# ==========================================
# 用法：在專案根目錄執行  python preflight_check.py
#
# 為什麼要有這支？
# 多智能體最痛苦的除錯情境是「等了 90 秒才拿到一句『系統錯誤』」，
# 而且完全不知道是哪一環壞掉。這支腳本會在你動 app.py 之前，
# 把每一個外部相依項目「單獨」測一次，紅燈直接指到問題點。
#
# 特別重要的是【檢查 5】：
# 規劃官 Planner 需要主模型輸出巢狀 JSON (list of object)。
# 這對小模型有難度，如果 XUYA 底層撐不住，這裡會直接告訴你，
# 你就知道系統會自動走第二層(兩段式)或第三層(正則)保底，不用猜。
# ==========================================
import sys
import time
import traceback

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []


def record(name, ok, detail="", warn_only=False):
    icon = PASS if ok else (WARN if warn_only else FAIL)
    results.append((icon, name, detail))
    print(f"{icon} {name}")
    if detail:
        for line in str(detail).split("\n"):
            print(f"    {line}")
    print()


def main():
    print("\n" + "=" * 60)
    print("🩺 航空母艦戰鬥群 v2.1 - 啟動前置檢查")
    print("=" * 60 + "\n")

    # ------------------------------------------------
    # 檢查 1：config.py 與 .env
    # ------------------------------------------------
    try:
        from config import (
            OLLAMA_API_BASE_URL, BRAVE_API_KEY,
            MAIN_MODEL_NAME, VERIFY_MODEL_NAME,
            MAX_SEARCH_CALLS_PER_TURN, LLM_TIMEOUT_SECONDS,
        )
        record("檢查 1 - config.py 載入", True,
               f"Ollama 位址：{OLLAMA_API_BASE_URL}\n"
               f"主模型：{MAIN_MODEL_NAME}\n"
               f"驗證模型：{VERIFY_MODEL_NAME}\n"
               f"搜尋次數上限：{MAX_SEARCH_CALLS_PER_TURN} / LLM 逾時：{LLM_TIMEOUT_SECONDS}s")
    except ImportError as e:
        record("檢查 1 - config.py 載入", False,
               f"{e}\n→ 你可能還沒把新版 config.py 覆蓋上去（缺少 v2 參數區塊）")
        print("config 載入失敗，後續檢查無法進行。請先處理這一項。\n")
        return

    if not BRAVE_API_KEY:
        record("檢查 1b - BRAVE_API_KEY", False,
               "未設定！Search_Agent 會 100% 失敗。請檢查 .env 裡的 BRAVE_API_KEY。")
    else:
        record("檢查 1b - BRAVE_API_KEY", True, f"已設定 (開頭 {BRAVE_API_KEY[:6]}...)")

    # ------------------------------------------------
    # 檢查 2：Ollama 服務與模型清單
    # ------------------------------------------------
    installed = []
    try:
        import requests
        r = requests.get(f"{OLLAMA_API_BASE_URL}/api/tags", timeout=8)
        r.raise_for_status()
        installed = [m.get("name", "") for m in r.json().get("models", [])]
        record("檢查 2 - Ollama 服務", True, f"連線正常，已安裝 {len(installed)} 顆模型：\n" + "\n".join(installed))
    except Exception as e:
        record("檢查 2 - Ollama 服務", False,
               f"無法連線到 {OLLAMA_API_BASE_URL}\n{e}\n→ 請確認 Ollama 已啟動 (ollama serve)")
        return

    def has_model(name):
        base = name.split(":")[0]
        return any(m == name or m.split(":")[0] == base for m in installed)

    record(f"檢查 2b - 主模型 {MAIN_MODEL_NAME}", has_model(MAIN_MODEL_NAME),
           "" if has_model(MAIN_MODEL_NAME) else f"不在已安裝清單中！請執行 ollama pull {MAIN_MODEL_NAME}")
    record(f"檢查 2c - 驗證模型 {VERIFY_MODEL_NAME}", has_model(VERIFY_MODEL_NAME),
           "" if has_model(VERIFY_MODEL_NAME) else f"不在已安裝清單中！請執行 ollama pull {VERIFY_MODEL_NAME}")

    # ------------------------------------------------
    # 檢查 3：計算機工具
    # ------------------------------------------------
    try:
        from tools.calculator import calculate_math
        out = calculate_math("634 - 508")
        ok = "126" in out
        record("檢查 3 - 計算機工具", ok, out)
    except Exception as e:
        record("檢查 3 - 計算機工具", False, f"{e}\n{traceback.format_exc(limit=2)}")

    # ------------------------------------------------
    # 檢查 4：Brave 搜尋工具 (會實際消耗 1 次 API 額度)
    # ------------------------------------------------
    try:
        from tools.web_search import search_web_ex
        t0 = time.time()
        payload = search_web_ex("台北101 建築總高度")
        elapsed = time.time() - t0
        record("檢查 4 - Brave 搜尋工具", payload["ok"],
               f"耗時 {elapsed:.1f}s / cached={payload['cached']}\n"
               + (payload["text"][:200] if payload["ok"] else payload["message"]))

        # 順便驗證快取有沒有生效(第二次不應該再打 API)
        payload2 = search_web_ex("台北101 建築總高度")
        record("檢查 4b - 搜尋快取", payload2.get("cached", False),
               "快取命中，重複查詢不會再消耗 API 額度"
               if payload2.get("cached") else "快取未命中，請檢查 web_search.py 是否為新版",
               warn_only=True)
    except ImportError as e:
        record("檢查 4 - Brave 搜尋工具", False,
               f"{e}\n→ 找不到 search_web_ex，你可能還沒把新版 tools/web_search.py 覆蓋上去")
    except Exception as e:
        record("檢查 4 - Brave 搜尋工具", False, f"{e}")

    # ------------------------------------------------
    # 檢查 5：【最關鍵】主模型的巢狀結構化輸出能力
    # ------------------------------------------------
    # 這一項決定了 Planner 走第一層還是要退到第二/第三層保底。
    # 不管結果如何系統都能跑，但你會知道自己站在哪一層。
    planner_layer = "未知"
    try:
        from graph_core import planner_structured_llm, TaskPlan  # noqa: F401
        from langchain_core.messages import HumanMessage, SystemMessage

        t0 = time.time()
        res = planner_structured_llm.invoke([
            SystemMessage(content=(
                "你是任務拆解專家，只做拆解，禁止回答問題。"
                "比較兩個對象時必須拆成兩個獨立的 search 步驟，一個步驟只能有一個對象。"
            )),
            HumanMessage(content="請拆解這個問題：請查台北101與東京晴空塔的總高度，並算出誰高多少公尺？")
        ])
        elapsed = time.time() - t0
        steps = res.steps or []
        detail = f"耗時 {elapsed:.1f}s，拆出 {len(steps)} 個步驟：\n"
        for i, s in enumerate(steps):
            detail += f"  [{i}] {s.step_type} / {s.target}\n"
        search_count = sum(1 for s in steps if s.step_type == "search")
        ok = len(steps) >= 2 and search_count >= 2
        if ok:
            planner_layer = "第一層（巢狀結構化）"
            detail += "→ 主模型撐得住巢狀結構化輸出，Planner 會走第一層。"
        else:
            planner_layer = "需要保底"
            detail += "→ 拆解結果不理想(搜尋步驟少於 2 個)，系統會自動退到第二層兩段式拆解。"
        record("檢查 5 - 主模型巢狀結構化輸出", ok, detail, warn_only=True)
    except Exception as e:
        planner_layer = "需要保底"
        record("檢查 5 - 主模型巢狀結構化輸出", False,
               f"{e}\n→ 主模型無法穩定產生巢狀 JSON。這不是致命問題：\n"
               f"  系統會自動退到第二層(兩段式拆解，用 {VERIFY_MODEL_NAME})，\n"
               f"  再不行還有第三層純 Python 正則。但你應該考慮換一顆結構化輸出較強的主模型。",
               warn_only=True)

    # ------------------------------------------------
    # 檢查 6：第二層保底(小模型的純字串清單輸出)
    # ------------------------------------------------
    try:
        from graph_core import verify_llm, SearchTargets
        from langchain_core.messages import HumanMessage, SystemMessage

        t0 = time.time()
        res = verify_llm.with_structured_output(SearchTargets).invoke([
            SystemMessage(content="請把問題拆成搜尋關鍵字清單，一個字串只能有一個查詢對象，不要輸出解釋。"),
            HumanMessage(content="請查台北101與東京晴空塔的總高度")
        ])
        targets = res.targets or []
        ok = len(targets) >= 2
        record("檢查 6 - 第二層保底(小模型字串清單)", ok,
               f"耗時 {time.time()-t0:.1f}s，拆出：{targets}"
               + ("" if ok else "\n→ 連第二層都不穩，系統會退到第三層純 Python 正則"),
               warn_only=True)
    except Exception as e:
        record("檢查 6 - 第二層保底(小模型字串清單)", False, f"{e}", warn_only=True)

    # ------------------------------------------------
    # 檢查 7：第三層保底(純 Python 正則，不依賴任何模型)
    # ------------------------------------------------
    try:
        from graph_core import _heuristic_plan
        plan = _heuristic_plan("請幫我分別查詢台北 101 與日本東京晴空塔的建築總高度（公尺），並計算晴空塔和台北 101 誰比誰高多少公尺？")
        ok = len(plan) >= 2
        detail = "\n".join(f"  [{s['id']}] {s['type']} / {s['target']}" for s in plan)
        record("檢查 7 - 第三層保底(純 Python 正則)", ok, detail or "拆解失敗")
    except Exception as e:
        record("檢查 7 - 第三層保底(純 Python 正則)", False, f"{e}")

    # ------------------------------------------------
    # 檢查 8：RAG 知識庫與聊天資料庫
    # ------------------------------------------------
    try:
        from ai_core import search_knowledge_ex
        rag = search_knowledge_ex("公司提供哪些服務？")
        record("檢查 8 - RAG 知識庫", True,
               f"檢索正常，可信度={rag['hit_type']}，回傳 {len(rag['docs'])} 筆")
    except ImportError as e:
        record("檢查 8 - RAG 知識庫", False,
               f"{e}\n→ 找不到 search_knowledge_ex，你可能還沒把新版 ai_core.py 覆蓋上去")
    except Exception as e:
        record("檢查 8 - RAG 知識庫", False, f"{e}")

    try:
        from database import get_db_connection
        conn = get_db_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        record("檢查 9 - 聊天紀錄資料庫", True, "連線正常")
    except Exception as e:
        record("檢查 9 - 聊天紀錄資料庫", False, f"{e}")

    # ------------------------------------------------
    # 總結
    # ------------------------------------------------
    print("=" * 60)
    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"檢查完成：{len(results)} 項，{len(fails)} 項失敗，{len(warns)} 項警告")
    print(f"規劃官目前會走：{planner_layer}")
    if fails:
        print("\n必須先修好這些才能測試：")
        for _, name, _ in fails:
            print(f"  {FAIL} {name}")
    else:
        print("\n🎉 沒有致命問題，可以執行 python app.py 了！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中斷。")
        sys.exit(1)
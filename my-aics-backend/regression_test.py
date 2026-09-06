# -*- coding: utf-8 -*-
# regression_test.py
# ============================================================
# 【P0】多智能體回歸測試腳本
# ============================================================
# 為什麼需要這支腳本？
#
# 踩坑筆記第 16 條記錄過一件事：同一份提示詞、同一個問題，
# 小模型不保證每次輸出一樣（組合數題有時輸出 comb(...)，有時輸出 C(...)）。
# 手動測 1~2 次、看 log 正不正常，沒辦法分辨「這次真的修好了」
# 還是「這次剛好運氣好」。
#
# 這支腳本做的事：把踩過的坑全部變成可以重複執行的測試案例，
# 每個案例檢查的不是「答案文字長什麼樣子」，而是「不變條件」——
# 那些不管模型怎麼回答，正確答案都必須滿足的性質
# （例如：純數學題不該打任何一次搜尋 API、答案裡的數字必須全部有來源）。
# 這跟踩坑筆記第 3 條的「校準腳本」是同一種哲學，只是這次套用在
# Planner / Search_Agent / Math_Agent 這條主流程，而不只是 RAG 門檻。
#
# 這支腳本直接 import 你正式在跑的 graph_core.app_graph，
# 不另外複製一份判斷邏輯 —— 確保測試永遠對照「現在真正在跑的程式碼」。
#
# ------------------------------------------------------------
# 使用方式
# ------------------------------------------------------------
#   python regression_test.py                     # 跑全部案例
#   python regression_test.py --case pizza         # 只跑名稱包含 "pizza" 的案例
#   python regression_test.py --repeat 3           # 覆蓋每個案例的重跑次數
#   python regression_test.py --verbose            # 印出完整 graph 執行過程（各節點 log）
#
# 執行環境要求跟你平常跑後端一樣：Ollama 已啟動且模型已下載、
# .env 裡的 BRAVE_API_KEY 有效、ChromaDB 向量庫在 xuya_vdb/ 底下。
# 這支腳本會真的呼叫 Ollama 推論、真的打 Brave API，
# 不是 mock 測試，所以每跑一次都會消耗真實的搜尋額度與運算時間。
# ============================================================

import argparse
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, AIMessage

# 直接吃正式的 app_graph，確保測到的是「現在真正在跑」的版本
from graph_core import app_graph, _answer_numbers_traceable


# ============================================================
# 【1】建構初始狀態 —— 完全比照 graph_core.py 的 __main__ 測試區塊
# ============================================================
def _blank_state(question: str, history=None) -> dict:
    return {
        "chat_history": history or [],
        "messages": [HumanMessage(content=question)],
        "retry_count": 0,
        "search_calls": 0,
        "math_calls": 0,
        "plan": [],
        "facts": {},
        "searched_queries": [],
        "current_step": -1,
        "rag_context": "",
        "rag_hit_type": "none",
        "all_steps_done": True,
        "plan_decision": "undetermined",
        "search_notes": [],
    }


# ============================================================
# 【2】資料結構定義
# ============================================================
@dataclass
class TestCase:
    name: str                                       # 案例代號，方便用 --case 篩選
    question: str                                   # 使用者輸入
    check: Callable[[str, dict], "tuple[bool, str]"]  # (最終回覆, 最終state) -> (是否通過, 說明)
    note: str = ""                                  # 對應踩坑筆記第幾條，方便追查根因
    repeat: int = 1                                 # 這個案例要重跑幾次（抓取樣變異用）


@dataclass
class RunResult:
    case_name: str
    attempt: int
    passed: bool
    detail: str
    final_answer: str = ""
    elapsed_sec: float = 0.0
    error: Optional[str] = None


# ============================================================
# 【3】可重複使用的「不變條件」檢查函式
# ============================================================
def contains_all(*numbers_or_texts):
    """回覆裡必須同時出現這些內容（數字用字串比對，避免千分位逗號誤判）。"""
    def _check(answer: str, state: dict):
        missing = [str(n) for n in numbers_or_texts if str(n) not in answer]
        if missing:
            return False, f"回覆裡缺少必須出現的內容：{missing}"
        return True, "所有必要內容都有出現"
    return _check


def not_contains(*banned_texts):
    """回覆裡不該出現這些內容（用來防止已知的歷史錯誤重演）。"""
    def _check(answer: str, state: dict):
        hit = [t for t in banned_texts if t in answer]
        if hit:
            return False, f"回覆裡出現了不該出現的內容（歷史錯誤重演）：{hit}"
        return True, "沒有出現任何已知的錯誤內容"
    return _check


def used_math_not_search(answer: str, state: dict):
    """自足數學題的核心不變條件：不該打任何一次 Brave API。"""
    search_calls = state.get("search_calls", 0)
    if search_calls > 0:
        return False, (
            f"這題應該是純數學題，卻呼叫了 {search_calls} 次 Search_Agent"
            "（可能又被 RAG 短路或規劃官誤判成需要上網）"
        )
    return True, "沒有動用任何搜尋，正確走純數學路徑"


def used_search_at_least_once(answer: str, state: dict):
    """開放式查詢的核心不變條件：真的有觸發搜尋，不是被 RAG 或規劃官吃案。"""
    search_calls = state.get("search_calls", 0)
    if search_calls < 1:
        return False, "這題應該要上網查詢，但 search_calls 是 0，代表根本沒有真的搜尋"
    return True, f"有觸發搜尋，search_calls={search_calls}"


def has_search_notes(answer: str, state: dict):
    """開放式資訊查詢應該要有 search_notes，而不是被硬塞進數值萃取管線。"""
    notes = state.get("search_notes", []) or []
    if not notes:
        return False, "search_notes 是空的，這題可能又被當成單一數值查詢去萃取了"
    return True, f"search_notes 有 {len(notes)} 筆"


def no_garbled_fact_leak(answer: str, state: dict):
    """
    防止「12.5個基」那種被截斷的搜尋碎片直接洩漏給使用者。
    這裡只做關鍵字層級的粗略防呆，抓不到全部情況，
    但至少能攔住「12.5個基」這個已知案例重演。
    """
    if re.search(r'\d+個基(?!點)', answer):
        return False, "疑似出現被截斷的搜尋碎片（例如「12.5個基」缺了「點」字），數值萃取器可能又抓錯行了"
    return True, "沒有偵測到已知的碎片洩漏模式"


def numbers_are_traceable(answer: str, state: dict):
    """
    輸出層數字溯源：回覆裡的每個數字都必須能對應到 facts 或 search_notes，
    不能是模型自己心算/編造出來的。直接複用 graph_core 自己的溯源函式，
    確保測試標準跟正式程式碼裡的把關標準完全一致。
    """
    facts = state.get("facts", {}) or {}
    search_notes = state.get("search_notes", []) or []
    question = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            question = msg.content
    ok, unknown = _answer_numbers_traceable(answer, facts, question, search_notes)
    if not ok:
        return False, f"回覆裡有無法溯源的數字：{sorted(unknown)}（可能是模型自己編的）"
    return True, "回覆裡的數字全部可以溯源"


def not_opens_with_banned_phrase(answer: str, state: dict):
    """鐵則 8：不該用『根據查詢結果／根據計算結果』這種罐頭開場白。"""
    banned = ("根據查詢結果", "根據計算結果", "你好，我是一位專業的 AI 助理")
    stripped = answer.strip()
    for b in banned:
        if stripped.startswith(b):
            return False, f"回覆以禁用的罐頭開場白開頭：「{b}」（鐵則 8 沒被遵守，目前是已知的 P2 問題）"
    return True, "開場白沒有踩到鐵則 8 的紅線"


def no_merged_quotient_in_facts(answer: str, state: dict):
    """
    踩坑筆記第 17 條：直接檢查事實帳本裡有沒有『兩個商相加』的痕跡
    （例如「(70 除以 7 取整數) + (30 除以 7 取整數)」），
    不只看最終回覆的文字，因為就算回覆這次剛好講對了，
    帳本裡如果還是把兩種東西的商合併計算，下一次措辭改一下還是可能翻車。
    """
    pattern = re.compile(r'除以\s*\d+\s*取整數\)?\s*\+\s*\(?\d+\s*除以')
    for v in (state.get("facts", {}) or {}).values():
        if pattern.search(str(v)):
            return False, f"事實帳本裡偵測到『兩個商相加』的痕跡：{v}"
    return True, "事實帳本沒有偵測到商相加的可疑痕跡"


def combine(*checkers):
    """把多個檢查函式串成一個，全部通過才算通過；印出每一項各自的結果方便除錯。"""
    def _combined(answer: str, state: dict):
        lines = []
        all_ok = True
        for fn in checkers:
            ok, detail = fn(answer, state)
            mark = "✅" if ok else "❌"
            lines.append(f"        {mark} {fn.__name__}：{detail}")
            if not ok:
                all_ok = False
        return all_ok, "\n" + "\n".join(lines)
    return _combined


# ============================================================
# 【4】測試案例集 —— 全部來自踩坑筆記裡真實踩過的坑
# ============================================================
CASES = [
    # ------------------------------------------------------------
    # 案例 1：披薩分配（踩坑筆記第 6 條：白名單量詞永遠追不完）
    # 歷史錯誤：系統曾經漏了「片」這個量詞，把數字拿去搜尋引擎查。
    # ------------------------------------------------------------
    TestCase(
        name="pizza_split_self_contained_math",
        question="10片披薩分給4個人，每個人可以分到多少片？會剩下多少片？",
        check=combine(
            used_math_not_search,
            contains_all("2"),  # 10 // 4 = 2 片，10 % 4 = 2 片剩
        ),
        note="踩坑筆記第 6 條：白名單量詞追不完，改用『有無外部事實訊號』判斷。",
        repeat=3,  # 純數學、不打外部 API，多跑幾次不花錢，用來抓取樣變異
    ),

    # ------------------------------------------------------------
    # 案例 2：大魚小魚分配（同一條筆記的實測慘案：曾經搜出「一午二紅沙」）
    # ------------------------------------------------------------
    TestCase(
        name="fish_split_self_contained_math",
        question="大魚70條、小魚30條，平分給7個人，每個人可以分到幾條大魚、幾條小魚？各會剩下多少？",
        check=combine(
            used_math_not_search,
            contains_all("10", "4"),  # 70 // 7 = 10 條大魚，30 // 7 = 4 條小魚
            not_contains("一午二紅沙"),  # 防止那次已知的搜尋垃圾內容重演
            no_merged_quotient_in_facts,  # 防止「兩種東西的商加在一起」的變種重演
        ),
        note=(
            "踩坑筆記第 6 條：實測慘案，系統曾經把『大魚總數』拿去搜尋，搜出一句台灣俗諺。"
            "第 17 條：這支腳本第一次真正跑起來後，又抓到新變種——模型把兩種東西的商加在一起"
            "（(70//7)+(30//7)），已經用 AST 不變條件擋下，這裡繼續留著防止復發。"
        ),
        repeat=3,
    ),

    # ------------------------------------------------------------
    # 案例 3：組合數學（8選3 × 6選2，本輪對話修過的 C(n,k) 記法變異 bug）
    # ------------------------------------------------------------
    TestCase(
        name="combinatorics_group_selection",
        question=(
            "某公司技術部門有8位工程師，行銷部門有6位企劃，"
            "現在要成立專案小組需要從技術部門選3人，並且從行銷部門選2人，"
            "總共有多少不同種選法？"
        ),
        check=combine(
            used_math_not_search,  # 這題也曾經被 RAG manual 誤判短路，交給模型心算
            contains_all("840"),
        ),
        note="踩坑筆記第 7、16 條：小模型有時輸出 comb(n,k)、有時輸出 C(n,k)，且曾被 RAG 誤判短路。",
        repeat=3,
    ),

    # ------------------------------------------------------------
    # 案例 4：台北 101 vs 東京晴空塔（需要「搜尋 + 計算」串接）
    # 這題答案本身依賴即時搜尋結果，不鎖定具體數字，
    # 改用「數字溯源」做為不變條件：只要答案裡的數字都能對應到
    # 查證過的事實帳本，就不算模型亂編。
    # ------------------------------------------------------------
    TestCase(
        name="taipei101_vs_skytree_height_diff",
        question="請幫我分別查詢台北101與日本東京晴空塔的建築總高度（公尺），並計算晴空塔和台北101誰比誰高多少公尺？",
        check=combine(
            used_search_at_least_once,
            numbers_are_traceable,
        ),
        note="踩坑筆記第 13 條：搜尋摘要撈數字不可靠；這裡不驗真實高度，只驗『答案數字有沒有來源』。",
        repeat=1,  # 會真的打 Brave API，預設只跑一次省額度
    ),

    # ------------------------------------------------------------
    # 案例 5：昨天台灣的新聞（開放式資訊查詢，本輪對話修的 search_notes 分流）
    # ------------------------------------------------------------
    TestCase(
        name="taiwan_news_open_query",
        question="可以幫我上網搜尋昨天台灣的新聞嗎？",
        check=combine(
            used_search_at_least_once,
            has_search_notes,
            no_garbled_fact_leak,
            not_opens_with_banned_phrase,  # 目前已知會失敗（P2 未修），見下方說明
        ),
        note=(
            "踩坑筆記第 13 條 + 本輪修的 search_notes 分流。"
            "注意：not_opens_with_banned_phrase 這項目前預期會不通過（鐵則 8 尚未修），"
            "不是這支腳本壞了，是那個 P2 項目還沒處理，先留著當提醒。"
        ),
        repeat=1,
    ),
]


# ============================================================
# 【5】執行引擎
# ============================================================
def run_case(case: TestCase, attempt: int, verbose: bool) -> RunResult:
    state = _blank_state(case.question)
    final_answer = ""
    final_state = {}
    start = time.time()
    try:
        for output in app_graph.stream(state, {"recursion_limit": 30}):
            for key, value in output.items():
                if verbose:
                    print(f"      --- 經過房間: {key} ---")
                final_state.update(value)
                if key == "Final_Answer":
                    final_answer = value["messages"][-1].content
    except Exception as e:
        elapsed = time.time() - start
        return RunResult(
            case.name, attempt, False,
            f"執行過程拋出例外，還沒跑到 Final_Answer",
            error=f"{e}\n{traceback.format_exc()}",
            elapsed_sec=elapsed,
        )

    elapsed = time.time() - start

    if not final_answer:
        return RunResult(
            case.name, attempt, False,
            "沒有取得任何 Final_Answer 回覆（graph 可能卡住或提前結束）",
            elapsed_sec=elapsed,
        )

    ok, detail = case.check(final_answer, final_state)
    return RunResult(case.name, attempt, ok, detail, final_answer, elapsed)


def main():
    parser = argparse.ArgumentParser(description="多智能體回歸測試腳本")
    parser.add_argument("--case", type=str, default=None, help="只跑名稱包含這個字串的案例")
    parser.add_argument("--repeat", type=int, default=None, help="覆蓋每個案例的重跑次數")
    parser.add_argument("--verbose", action="store_true", help="印出完整 graph 執行過程")
    args = parser.parse_args()

    cases = CASES
    if args.case:
        cases = [c for c in cases if args.case.lower() in c.name.lower()]
        if not cases:
            print(f"⚠️ 找不到名稱包含「{args.case}」的案例，可用案例：")
            for c in CASES:
                print(f"   - {c.name}")
            sys.exit(1)

    print("=" * 70)
    print(f"🧪 開始執行回歸測試，共 {len(cases)} 個案例")
    print("=" * 70)

    all_results: "list[RunResult]" = []

    for case in cases:
        repeat = args.repeat if args.repeat is not None else case.repeat
        print(f"\n▶ 案例：{case.name}（重跑 {repeat} 次）")
        print(f"   問題：{case.question}")
        if case.note:
            print(f"   對應筆記：{case.note}")

        for attempt in range(1, repeat + 1):
            result = run_case(case, attempt, args.verbose)
            all_results.append(result)

            mark = "✅ 通過" if result.passed else "❌ 失敗"
            print(f"   [第 {attempt}/{repeat} 次] {mark}（耗時 {result.elapsed_sec:.1f} 秒）")
            print(f"      {result.detail}")
            if result.error:
                print(f"      錯誤堆疊：{result.error}")
            if not result.passed and result.final_answer:
                print(f"      實際回覆：{result.final_answer[:300]}")

    # --------------------------------------------------------
    # 總結報告
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 總結報告")
    print("=" * 70)

    by_case = {}
    for r in all_results:
        by_case.setdefault(r.case_name, []).append(r)

    total_pass = sum(1 for r in all_results if r.passed)
    total_run = len(all_results)

    for name, results in by_case.items():
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        mark = "✅" if passed == total else ("⚠️" if passed > 0 else "❌")
        print(f"  {mark} {name}：{passed}/{total} 次通過")

    print(f"\n  總計：{total_pass}/{total_run} 次執行通過")

    if total_pass < total_run:
        print("\n⚠️ 有案例沒有全部通過，往上找對應案例的詳細說明與『對應筆記』欄位定位根因。")
        sys.exit(1)
    else:
        print("\n🎉 全部通過！")
        sys.exit(0)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# graph_core.py
# ==========================================
# 航空母艦戰鬥群 (LangGraph) 核心定義檔  ── v2.1「任務清單 + 事實帳本」版
# ==========================================
# 【SA v2 改版總說明】(沿用 v2)：
#
# v1 的三個結構性病灶(101 那題連查三次的真正原因)：
#   病灶 A：Supervisor 每一步都「從零重新判斷」，reasoning 裡雖然寫出了 508，
#           但 reasoning 只被 print 就丟掉，沒存進背包 → 下一步完全不知道查過了。
#   病灶 B：沒有任何 Python 層級的「同一句關鍵字不准查第二次」硬防線。
#   病灶 C：鑑定士用「關鍵字掃描」判斷成敗，網頁摘要裡出現「錯誤」就誤殺；
#           而「508」這種語法合法、語意無意義的算式反而被判合格。
#
# v2 的三根新骨架：Planner 任務清單 / facts 事實帳本 / Supervisor 降級為進度管理員。
#
# ==========================================
# 【SA v2.1 本次追加的三件事】：
#
#   追加 1：RAG 艙室獨立 (rag_context)
#           v2 把 RAG 檢索結果塞在 messages 的 SystemMessage 裡，
#           那 Search_Agent 的關鍵字抽取器就會連同 6 篇不相關的履歷文件一起讀進去。
#           更嚴重的是：Planner 看不到知識庫，所以就算知識庫裡已經有標準答案，
#           它還是會排一個上網查詢的步驟 —— 白白浪費 Brave API，而且答案可能更差。
#           v2.1 把 RAG 拆成獨立艙室：Planner 與 Final_Answer 看得到，
#           Search_Agent / Math_Agent 完全看不到。
#
#   追加 2：三層拆解保底 (Planner Fallback Chain)
#           Planner 要輸出巢狀 JSON (list of object)，這對小模型是有難度的。
#           萬一 XUYA 底層模型撐不住，v2 會直接退回 v1 的鬼打牆模式 —— 等於白改。
#           v2.1 改成三層：
#             第一層 巢狀結構化輸出 (最完整)
#               ↓ 失敗
#             第二層 兩段式拆解：先要一個「純字串清單」的搜尋目標，再問一次要不要計算
#                    (純字串清單對小模型容易非常多，這是主要的保險絲)
#               ↓ 失敗
#             第三層 純 Python 正則啟發式：專門處理最常見的「A 與 B 的某屬性」比較題
#                    完全不依賴模型能力，實測可正確拆出台北101/晴空塔那題
#               ↓ 失敗
#             第四層 空清單 → 退回 LLM 自由判斷 (等同 v1 行為)
#
#   追加 3：所有可調參數搬到 config.py
#           模型名稱、次數上限、逾時秒數不再寫死在核心邏輯檔裡。
# ==========================================
import re
import ast
import copy
from typing import Annotated, Sequence, TypedDict, Literal, List, Dict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_API_BASE_URL

# 【SA v2.1】：參數改由 config.py 統一管理。
# 用 try/except 包起來的理由：萬一你還沒替換 config.py 就先跑了這支檔案，
# 也不會直接 ImportError 掛掉，而是自動使用下面的預設值。
try:
    from config import (
        MAIN_MODEL_NAME, VERIFY_MODEL_NAME,
        MAX_SEARCH_CALLS_PER_TURN, MAX_MATH_CALLS_PER_TURN,
        MAX_RETRY, MAX_STEP_ATTEMPTS, LLM_TIMEOUT_SECONDS,
    )
except ImportError:
    print("[系統] ⚠️ config.py 尚未加入 v2 參數區塊，暫時使用 graph_core 內建預設值。")
    MAIN_MODEL_NAME = "XUYA:latest"
    VERIFY_MODEL_NAME = "gemma3:4b"
    MAX_SEARCH_CALLS_PER_TURN = 6
    MAX_MATH_CALLS_PER_TURN = 4
    MAX_RETRY = 2
    MAX_STEP_ATTEMPTS = 3
    LLM_TIMEOUT_SECONDS = 90

# ==========================================
# 🚩 第零區：狀態旗標 (取代舊版的關鍵字掃描)
# ==========================================
# 【SA v2】：每個工具節點回報時，一律在訊息最前面掛上狀態旗標。
# 鑑定士只看這個旗標，不再去猜網頁內文裡的「錯誤」兩個字是不是代表失敗。
STATUS_OK = "【STATUS:OK】"
STATUS_FAIL = "【STATUS:FAIL】"


def _is_failed_message(msg) -> bool:
    """判斷某一則工作紀錄是不是失敗回報(只看開頭旗標，不掃描內文)。"""
    content = str(getattr(msg, "content", "") or "")
    return content.lstrip().startswith(STATUS_FAIL)


# ==========================================
# 🎒 第一區：定義「共用背包」 (AgentState)
# ==========================================
# 【SA 資料隔離升級 - Context Isolation】：
#   chat_history     -> 跨輪次持久保存的「乾淨」歷史對話 (只有 User 問題 + 最終回答)
#   messages         -> 「單次任務」的工作記憶區，每次新問題進來都是全新的一頁
#   next_node        -> 主管的派工決定
#   retry_count      -> Local Grader 專用的重試計數器
#   search_calls     -> 本輪搜尋總次數
#   math_calls       -> 本輪計算總次數
#
# 【SA v2 新增】：
#   plan             -> 任務清單，Supervisor 靠它知道進度
#   facts            -> 事實帳本 {"台北101 建築總高度": "508 公尺"}
#   searched_queries -> 本輪已真正打過 API 的關鍵字(Python 硬去重)
#   current_step     -> 主管指定的當前任務 id
#
# 【SA v2.1 新增】：
#   rag_context      -> RAG 檢索結果的獨立艙室。
#                       只有 Planner(判斷需不需要上網) 與 Final_Answer(客服題直接引用) 看得到，
#                       Search_Agent / Math_Agent 完全看不到，避免說明書干擾關鍵字抽取。
#   rag_hit_type     -> "manual"(高精準命中) / "auto"(勉強撈到) / "none"(什麼都沒有)
#   all_steps_done   -> 本輪任務是否全部成功，供 state_manager 決定要不要提供真人轉接
#
# 【SA v2.3 新增 ── 這是這一版最重要的欄位】：
#   plan_decision    -> 規劃官的「決策結論」，這是三態而不是兩態：
#                       "has_plan"         = 排出了任務清單，照著跑
#                       "no_tools_needed"  = 明確判定這題不需要任何工具，直接結案
#                       "undetermined"     = 三層拆解全部拋例外，真的不知道該怎麼辦
#
#                       為什麼非要獨立一個欄位？
#                       v2.2 把「判定不需要工具」和「拆解壞掉」兩種結論，
#                       全都壓縮成 plan = [] 這一個值往下傳。
#                       Supervisor 收到空清單時無從分辨，只好一律走 LLM 自由判斷 ——
#                       於是「請自我介紹一下」這種完全不需要上網的問題，
#                       規劃官明明答對了(空清單)，最後還是被 Supervisor 派去搜尋，
#                       白燒一次 Brave API，還撈回一堆「30秒自我介紹範本」的垃圾。
#
#                       結論本身沒問題，問題出在傳遞結論的管道只有兩格。
#                       這裡把管道加寬成三格，Supervisor 才能正確地什麼都不做。
class AgentState(TypedDict):
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str
    retry_count: int
    search_calls: int
    math_calls: int
    plan: List[Dict]
    facts: Dict[str, str]
    searched_queries: List[str]
    current_step: int
    rag_context: str
    rag_hit_type: str
    all_steps_done: bool
    plan_decision: str
    # 【SA v4.3 新增】search_notes -> 開放式資訊查詢(新聞、時事、一般話題)的原始搜尋摘要清單。
    # 跟 facts 分開存放的原因：facts 是「單一可驗證數值」帳本(508 公尺、56…)，
    # 下游有嚴格的數字溯源檢查；但「昨天台灣的新聞」這種問題根本沒有單一數值可萃取，
    # 硬塞進 facts 只會逼萃取器捏造一個假數字出來(見 search_node 的說明)。
    # search_notes 專門放這種「一段文字摘要」，Final_Answer 會直接引用去統整答案，
    # 不會被數字溯源檢查誤判成編造。
    search_notes: List[str]


# ==========================================
# 📋 第二區：定義各節點的「強制下拉式選單」(結構化輸出)
# ==========================================
class RouteDecision(BaseModel):
    """只有在 Planner 生不出清單時，Supervisor 才會退回用這個自由判斷。"""
    reasoning: str = Field(description="請用一句話（不超過 50 字）簡短說明你的判斷邏輯，絕對不要在這裡進行完整的計算、列點或推導過程！")
    next_node: Literal["Search_Agent", "Math_Agent", "FINISH"] = Field(
        description=(
            "1. Search_Agent：需要上網查詢未知資訊、最新數據時選擇。\n"
            "2. Math_Agent：有明確計算需求時選擇。\n"
            "3. FINISH：資料已備齊，可直接回答時選擇。"
        )
    )


# 【SA v2】：規劃官的「第一層」輸出格式(巢狀結構，最完整但對小模型最難)
class PlanStep(BaseModel):
    step_type: Literal["search", "math"] = Field(
        description="search=需要上網查一個具體事實；math=需要用計算機做一次運算"
    )
    target: str = Field(
        description=(
            "如果是 search：寫『單一對象 + 要查的屬性』的搜尋關鍵字，一次只能有一個對象。\n"
            "如果是 math：用中文寫清楚要算什麼(例如『東京晴空塔高度 減去 台北101高度』)。"
        )
    )


class TaskPlan(BaseModel):
    steps: List[PlanStep] = Field(description="完成這個問題所需要的步驟清單，最多 6 步")


# 【SA v2.1 新增】：規劃官的「第二層」保險絲 —— 兩段式拆解用的兩個超簡單結構。
# 純字串陣列 / 單一布林 + 字串，對 4B~8B 等級的模型來說幾乎不會失敗，
# 這是整條保底鏈裡最重要的一環。
class SearchTargets(BaseModel):
    targets: List[str] = Field(
        description="需要上網查詢的關鍵字清單，一個字串只能包含一個查詢對象。如果完全不需要上網，回傳空陣列。"
    )


class MathNeed(BaseModel):
    need_math: bool = Field(description="這個問題在查到資料之後，是否還需要做數值運算？")
    description: str = Field(description="如果需要，用中文描述要算什麼(例如『東京晴空塔高度 減去 台北101高度』)。不需要就填空字串。")


class SearchQuery(BaseModel):
    query: str = Field(description="要丟給搜尋引擎的精準關鍵字")


# 【SA v2】：搜尋結果的「數值萃取器」輸出格式
class ExtractedFact(BaseModel):
    found: bool = Field(description="搜尋結果中是否明確出現了要找的數值。沒有就填 false，絕對不要用你自己的記憶硬湊。")
    value: str = Field(description="找到的數值，含單位，例如 '508 公尺'。找不到就填空字串。")


class MathExpression(BaseModel):
    # 【SA v4.0】：欄位改成可容納「多行推導腳本」。
    # 單行算式（634 - 508）仍然完全支援，行為不變；
    # 但遇到後面步驟依賴前面結果的題目時，模型可以改寫成
    #     r1 = 75 % 8
    #     total = r1 + r2 + r3
    #     stickers = total // 3
    # 讓依賴關係由變數表達，而不是逼模型在一行裡湊出巢狀算式。
    expression: str = Field(
        description=(
            "要執行的數學內容。簡單題直接寫一行算式，例如 '634 - 508'；"
            "如果後面的步驟需要用到前面算出來的結果，就寫成多行、每行一個變數指派，"
            "例如 'r1 = 75 % 8\\nr2 = 52 % 8\\ntotal = r1 + r2\\nstickers = total // 3'。"
            "只能包含數字、變數名與 + - * / // % ** 運算，不要有任何中文或說明文字。"
        )
    )


# ==========================================
# 🧠 第三區：初始化大腦模型
# ==========================================
main_llm = ChatOllama(
    base_url=OLLAMA_API_BASE_URL,
    model=MAIN_MODEL_NAME,
    temperature=0.2,
    repeat_penalty=1.15,
    num_predict=800,
    stop=[],
    mirostat=0,
    top_p=0.9,
    top_k=40
)

verify_llm = ChatOllama(
    base_url=OLLAMA_API_BASE_URL,
    model=VERIFY_MODEL_NAME,
    temperature=0,
    num_predict=200
)

router_llm = ChatOllama(
    base_url=OLLAMA_API_BASE_URL,
    model=MAIN_MODEL_NAME,
    temperature=0,
    num_predict=300,
    stop=[],
    mirostat=0,
    top_p=0.9,
    top_k=40
)

planner_llm = ChatOllama(
    base_url=OLLAMA_API_BASE_URL,
    model=MAIN_MODEL_NAME,
    temperature=0,
    num_predict=500,
    stop=[],
    mirostat=0,
    top_p=0.9,
    top_k=40
)

supervisor_llm = router_llm.with_structured_output(RouteDecision)
planner_structured_llm = planner_llm.with_structured_output(TaskPlan)

import concurrent.futures


def invoke_with_timeout(llm, messages, timeout_sec: int = LLM_TIMEOUT_SECONDS,
                        fallback_text: str = "抱歉，這個問題目前超出我的處理能力，請換個方式再問一次，或將問題拆得更簡單一點。"):
    """
    帶超時保護的 LLM 呼叫。
    注意：Ollama 端的運算不會因為這個 timeout 就立刻停止，
    但至少能保護 Flask/WebSocket 主流程不會被單一難題永遠卡住。
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm.invoke, messages)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print(f"\n[系統守衛] 🛑 LLM 呼叫超過 {timeout_sec} 秒，強制中斷等待，回傳保底訊息！")
            class _Fallback:
                content = fallback_text
            return _Fallback()


# ==========================================
# 🔧 第三.六區：共用小工具
# ==========================================
# 【SA 中文數字單位正規化】：「億」「萬」這種換算是機械式規則，不該讓小模型心算
_CN_UNIT_MULTIPLIERS = [
    ("兆", 1_000_000_000_000),
    ("億", 100_000_000),
    ("千萬", 10_000_000),
    ("百萬", 1_000_000),
    ("萬", 10_000),
    ("仟", 1_000),
    ("千", 1_000),
]


def _round_number(value):
    value_round = round(value, 2)
    return int(value_round) if value_round == int(value_round) else value_round


def normalize_chinese_number_units(text: str) -> str:
    # 財報表格式寫法：「(百萬) 56000」「(仟元) 12000」
    table_pattern = r'[（(]\s*(千萬|百萬|億|兆|仟|千|萬)\s*元?\s*[)）]\s*(\d+(?:\.\d+)?)'

    def _replace_table(match):
        unit, num_str = match.group(1), match.group(2)
        multiplier = dict(_CN_UNIT_MULTIPLIERS)[unit]
        try:
            value_out = _round_number(float(num_str) * multiplier)
            return f"{match.group(0)}(={value_out})"
        except ValueError:
            return match.group(0)

    text = re.sub(table_pattern, _replace_table, text)

    # 「數字緊接著單位」寫法：「562.82億」
    inline_pattern = r'(\d+(?:\.\d+)?)(千萬|百萬|億|兆|萬|仟|千)'

    def _replace_inline(match):
        num_str, unit = match.group(1), match.group(2)
        multiplier = dict(_CN_UNIT_MULTIPLIERS)[unit]
        try:
            value_out = _round_number(float(num_str) * multiplier)
            return f"{num_str}{unit}(={value_out})"
        except ValueError:
            return match.group(0)

    text = re.sub(inline_pattern, _replace_inline, text)
    return text


def _render_facts(facts: dict) -> str:
    """把事實帳本印成人類/模型都好讀的乾淨清單。"""
    if not facts:
        return "（帳本目前是空的，尚未確認任何事實）"
    return "\n".join(f"- {k} ＝ {v}" for k, v in facts.items())


def _render_plan(plan: list) -> str:
    """把任務清單印成進度表，log 一眼就能看出卡在哪一步。"""
    if not plan:
        return "（無任務清單，走 LLM 自由判斷模式）"
    icon = {"pending": "⬜", "done": "✅", "failed": "❌"}
    return "\n".join(
        f"  {icon.get(s['status'], '⬜')} [{s['id']}] {s['type']}: {s['target']}"
        for s in plan
    )


def _extract_calc_value(calc_result: str) -> str:
    """
    【SA v2.3 新增】從計算機回傳的完整訊息中，只取出結果數值。

    計算機回的是「計算成功！算式 'comb(10, 3)' 的結果為：120」，
    這是給 log 看的、給人除錯用的格式，不該原封不動進事實帳本 ——
    因為 Final_Answer 會把帳本內容當成可引用素材直接抄給客人。
    這裡只留 120。
    """
    m = re.search(r'結果為：\s*(.+)$', (calc_result or "").strip())
    return m.group(1).strip() if m else (calc_result or "").strip()


def _numbers_in(text: str) -> set:
    """抽出字串裡所有的數字(去掉千分位逗號)，用來做「算式數字是否來自帳本」的溯源檢查。"""
    return set(re.findall(r'\d+(?:\.\d+)?', (text or "").replace(",", "")))


# 【SA v2 硬去重】：只要正規化後的關鍵字已經打過 API，就先嘗試加限定詞變形；
# 還是重複就直接回傳 None，讓上層判定為失敗，絕不再浪費一次 Brave 額度。
_QUERY_VARIANTS = ["維基百科", "官方 高度 公尺", "資料 規格"]


def _normalize_query(q: str) -> str:
    return re.sub(r'\s+', '', (q or "")).lower()


def _dedup_query(query: str, searched: list):
    norm_searched = {_normalize_query(s) for s in searched}
    if _normalize_query(query) not in norm_searched:
        return query
    for suffix in _QUERY_VARIANTS:
        candidate = f"{query} {suffix}"
        if _normalize_query(candidate) not in norm_searched:
            print(f"[網路戰士 Search_Agent] 🔁 關鍵字與先前重複，自動改寫為變形查詢：{candidate}")
            return candidate
    return None


# 【SA 保留 v1】：Python 強制拆分合併查詢(只在「無清單保底模式」下才會用到)
def _pick_unsearched_segment(query: str, prior_queries: list) -> str:
    segments = re.split(r'[,，、]|\s*(?:和|與|及|以及|跟|vs|VS)\s*', query)
    segments = [s.strip() for s in segments if s.strip()]
    if len(segments) <= 1:
        return query
    for seg in segments:
        if not any(seg in prior for prior in prior_queries):
            return seg
    return segments[0]


def _find_step(plan: list, step_id: int):
    return next((s for s in plan if s.get("id") == step_id), None)


def _build_plan(items: list) -> list:
    """
    把 [(type, target), ...] 轉成標準的任務清單結構，
    並強制 search 全部排在 math 前面(還沒查到數字就叫計算機動手，正是 v1 翻車的畫面)。
    """
    searches = [(t, g) for t, g in items if t == "search" and g]
    maths = [(t, g) for t, g in items if t == "math" and g]
    plan = []
    for i, (step_type, target) in enumerate((searches + maths)[:6]):
        plan.append({
            "id": i,
            "type": step_type,
            "target": target.strip(),
            "status": "pending",
            "attempts": 0,
            "result": ""
        })
    return plan


# 【SA v2.2 新增】prompt 回音特徵詞。
# 小模型在結構化輸出時，常常不是「回答問題」而是「把你寫的規則抄一遍」。
# 這些詞幾乎只會出現在指令文字裡，不會出現在真正的任務描述中，
# 拿來當偵測特徵非常可靠。
_PROMPT_ECHO_MARKERS = [
    "字眼", "需要進行數值運算", "如果需要", "請判斷", "不要輸出",
    "之類的", "就算需要", "規則：", "例如：", "絕對不可以", "請回傳",
]


def _sanitize_plan(plan: list) -> list:
    """
    【SA v2.2 新增】任務清單雜訊過濾器。

    為什麼需要這個？
    實測 log 裡，gemma3:4b 在第二層拆解時吐出了這樣一個 math 任務：
        「問題包含『計算』字眼，需要進行數值運算。」
    這句話是它把我寫在 prompt 裡的判斷規則【原文複述】回來當成答案 ——
    典型的 prompt 回音(prompt echo)。這種假任務會一路傳到 Math_Agent，
    而 Math_Agent 拿著一句沒有任何數字的中文去「翻譯成算式」，
    當然只能憑空捏造，最後就出現了 comb(10,3)*comb(7,2)... 這種完全虛構的算式。

    這裡在任務進入執行之前就先攔掉，比讓下游三道檢查去補救更乾淨。
    """
    if not plan:
        return plan

    cleaned = []
    for step in plan:
        target = step.get("target", "")

        # 規則 1：命中 prompt 回音特徵詞的任務一律丟棄
        if any(k in target for k in _PROMPT_ECHO_MARKERS):
            print(f"[規劃官聖騎士 Planner] 🧹 過濾掉疑似複述指令的假任務：{target!r}")
            continue

        # 【SA v3.1 新增】規則 1.5：search 任務的目標必須是「可以拿去搜尋的東西」
        #
        # 實測翻車現場：使用者問「我有10片披薩要怎麼分給4個人，一人可以吃幾片」，
        # 規劃官吐出的任務清單竟然是：
        #     [0] search: 4
        #     [1] search: 10
        #     [2] math: 4
        # 網路戰士就真的拿「4」去 Google，還登錄了「4 ＝ 4 公尺」這種荒謬的事實。
        #
        # 原因是舊版只檢查 math 任務有沒有指明運算內容，
        # 完全沒檢查 search 目標合不合理 ——「4」不是空字串就放行了。
        # 一個合理的搜尋關鍵字至少要有「查什麼東西」，不可能只是一個裸數字。
        if step.get("type") == "search":
            t = target.strip()
            # 純數字（含小數、逗號）或太短的片段，都不是有意義的搜尋目標
            if re.fullmatch(r'[\d\s,.，、]+', t) or len(t) < 3:
                print(f"[規劃官聖騎士 Planner] 🧹 過濾掉無意義的搜尋目標：{t!r}（裸數字或過短，不可能查到東西）")
                continue

        # 規則 2：math 任務如果整句話裡連一個數字、一個運算符號、一個運算動詞都沒有，
        #        代表它根本沒說要算什麼，留著只會逼下游硬編算式
        #
        # 【SA v2.3 迴歸修正】：v2.2 只檢查中文運算動詞(加減乘除相差倍…)，
        # 結果把「日本本州島面積 / 台灣本島面積」這個【完全合法】的任務誤殺了 ——
        # 因為它用的是 `/` 符號，不是「除」這個字。
        # 任務被清掉之後 Math_Agent 根本沒被叫，Final_Answer 只好自己心算，
        # 吐出「大約是6.36倍」(而且沒照題目要求四捨五入至整數)。
        # 這裡補上運算符號的判斷，寧可放行也不要再誤殺。
        if step.get("type") == "math":
            t = target.strip()
            # 【SA v3.1 新增】：math 目標如果只是一個裸數字，同樣是垃圾
            # （披薩題的 [2] math: 4 就是這樣混過去的，害算盤法師空轉三次）
            if re.fullmatch(r'[\d\s,.，、]+', t):
                print(f"[規劃官聖騎士 Planner] 🧹 過濾掉無意義的計算任務：{t!r}（只是一個裸數字，沒說要算什麼）")
                continue
            has_digit = bool(re.search(r'\d', target))
            has_symbol = bool(re.search(r'[\+\-\*/×÷%]', target))
            has_verb = bool(re.search(
                r'(加|減|乘|除|相差|差值|差距|倍|總和|合計|平均|百分比|比例|次方|階乘|排列|組合|扣掉|加起來|總計|換算|分給|平分|每人|一人)',
                target))
            if not has_digit and not has_symbol and not has_verb:
                print(f"[規劃官聖騎士 Planner] 🧹 過濾掉沒有指明運算內容的 math 任務：{target!r}")
                continue

        cleaned.append(step)

    # 重新編號，確保 id 連續(Supervisor 是靠 id 找任務的)
    for i, step in enumerate(cleaned):
        step["id"] = i
    return cleaned


# ==========================================
# 🧩 第三.七區：【SA v2.1 新增】純 Python 啟發式拆解器 (保底鏈第三層)
# ==========================================
# 這一段完全不依賴任何模型能力，專門處理最常見的「A 與 B 的某屬性」雙實體比較題。
# 它不聰明，但它 100% 可預測 —— 在小模型撐不住結構化輸出時，這就是最後的安全網。
_LEAD_PHRASES = [
    "請幫我分別查詢", "請幫我查詢", "幫我分別查詢", "請幫我查", "分別查詢",
    "幫我查詢", "請問一下", "查詢一下", "看一下", "查一下", "請幫我",
    "麻煩你", "我想問", "我想查", "告訴我", "請問", "幫我", "麻煩",
    "查詢", "分別", "我要", "給我", "查", "請",
]
_LEAD_PHRASES.sort(key=len, reverse=True)

# 常見的「可查詢屬性」關鍵字，長詞排前面確保優先匹配到最完整的詞
_ATTR_WORDS = [
    "建築總高度", "實收資本額", "海拔高度", "員工人數", "成立時間", "總高度",
    "資本額", "營業額", "市值", "股價", "高度", "人口", "面積", "長度",
    "重量", "票價", "房價", "溫度", "營收", "深度", "時速",
]
# 【SA v3.2 設計轉向】：純數學題的判斷改用「反向規則」
#
# 舊作法是白名單：列舉「加、減、乘、除、相差、平分、幾片…」。
# 這條路走不通 —— 中文的數學講法是無窮的，使用者每問一題新的量詞
# （披薩「片」、糖果「顆」、魚「條」）就要補一次清單，永遠追不完，
# 而且漏掉的那次就會像實測一樣，拿「大魚總數」去 Google 搜到「一午二紅沙」。
#
# 反過來問就簡單多了：「這題需不需要外部事實？」
# 需要外部事實的訊號種類是【有限且穩定】的 —— 上網、查詢、最新、現任、
# 今天、股價、匯率、天氣、誰是…… 這份清單不會膨脹。
#
# 於是規則變成：
#     題目裡有數字  +  沒有任何外部事實訊號  →  自足的數學題，直接交給計算機
#
# 實測驗證：披薩、糖果、大魚三題全部正確判定為純數學；
# 而「台北101 和晴空塔差多少」因為沒有數字、且要查高度，不會被誤判。
_NEEDS_EXTERNAL_FACT = (
    # 【SA v3.3 修正】：這份清單原本含有裸的「今天／昨天／明天」，結果誤傷慘重 ——
    # 「如果蘋果60顆、檸檬40顆【今天】要分給9個人」這種純分配題，
    # 只因為句子裡有「今天」兩個字就被判定成需要上網，前置判定直接放棄，
    # 題目落回 LLM 手上，算式失控成一長串 comb(10,2)*comb(5,1)/... 整題報廢。
    #
    # 時間詞本身不代表需要查外部資料，它只有在「綁著一個會變動的事實」時才算。
    # 所以改成：時間詞必須與天氣／股價／匯率這類動態資料同時出現才成立。
    r'(上網|網路搜尋|搜尋一下|查詢|查一下|幫我查|估狗|google|'
    r'最新|現任|目前的|現在的|即時|'
    r'股價|股票|匯率|利率|油價|房價|天氣|氣溫|下雨|颱風|新聞|'
    r'誰是|是誰|哪一年|哪一天|市值|營收|人口|面積|高度|海拔|排名|冠軍|'
    r'總統|首相|執行長|CEO|董事長)'
)

# 保留給第三層啟發式拆解使用（雙實體比較題），與純數學判斷無關
_MATH_HINT = r'(計算|算出|相差|差多少|高多少|多多少|少多少|誰比誰|總共|合計|幾倍|百分比|平均|加起來|總和|平分|分給|每人|一人可以|一個人可以)'

# 【SA v4.1 新增】明確的「要求上網」訊號 —— 修正 RAG 誤判蓋過使用者明確指令的 bug。
#
# 問題起因：planner_node 一開始就檢查 rag_hit_type == "manual"，
# 只要向量檢索覺得夠像知識庫裡的面試問答，就直接空清單結案，
# 完全沒有機會看到使用者這句話裡其實白紙黑字寫著「幫我上網搜尋」。
# 「昨天台灣的新聞」在向量空間上跟「柬埔寨專案」意外地近（0.528，低於救援門檻），
# 於是被邊界救援規則撈走，使用者明確的上網指令就這樣被蓋掉了。
#
# 這份清單刻意比 _NEEDS_EXTERNAL_FACT 窄很多、只挑「動詞是叫你去查」的強訊號
# （上網、搜尋、查詢、google…），不含「新聞、股價、天氣」這類名詞，
# 因為名詞可能只是問題內容的一部分（例如面試題「你對最新的 AI 趨勢有什麼看法」），
# 不代表使用者一定要工具去查即時資料；但「幫我上網查」這種動詞片語幾乎不會誤判。
_EXPLICIT_SEARCH_INTENT = r'(上網|網路搜尋|搜尋一下|幫我搜尋|查詢|查一下|幫我查|估狗|google)'

# 【SA v4.2 新增】把「幫我上網搜尋」這類指令性贅字從問題裡剝掉，只留下真正要查的主題，
# 這樣送進 search_web 的關鍵字才乾淨（不然 Brave 會拿「幫我上網搜尋昨天台灣的新聞」
# 整句去查，準確度比純關鍵字「昨天台灣的新聞」差）。
_SEARCH_TRIGGER_STRIP = re.compile(
    r'(可以幫我|請幫我|幫我|請你|請|可以|麻煩你|麻煩|上網幫我|上網搜尋|網路搜尋|搜尋一下|'
    r'幫我搜尋|搜尋|查詢一下|查一下|幫我查|查詢|估狗|google|一下|嗎|呢|喔|唷|？|\?|。)',
    re.IGNORECASE,
)


def _strip_search_trigger_words(question: str) -> str:
    core = _SEARCH_TRIGGER_STRIP.sub('', question or '').strip()
    return core if core else (question or '').strip()


# 【SA v4.3 新增】判斷一個搜尋目標是不是「單一可驗證數值」查詢(高度、股價、人口…)，
# 還是「開放式資訊」查詢(新聞、時事、某人對某議題的看法…)。
# 只有前者才適合走 ExtractedFact 數值萃取器；後者硬萃取只會逼小模型捏造假數字
# (實測案例：「昨天台灣的新聞」被萃取成「12.5個基」，是搜尋結果裡一句無關的
# 「央行升息半碼(12.5個基點)」被錯誤當成答案抓出來)。
_SINGLE_FACT_ATTRS = re.compile(
    r'(高度|海拔|面積|人口|身高|體重|年齡|歲數|股價|股票|市值|營收|資本額|'
    r'匯率|利率|油價|房價|多少人|多少錢|幾層|幾公里|幾公尺|排名|第幾名|冠軍|'
    r'是誰|誰是|哪一年|哪一天|成立於|創立於|創立時間|成立時間)'
)


def _is_single_fact_target(target: str) -> bool:
    return bool(_SINGLE_FACT_ATTRS.search(target or ''))


def _strip_lead(s: str) -> str:
    s = s.strip()
    changed = True
    while changed:
        changed = False
        for p in _LEAD_PHRASES:
            if s.startswith(p):
                s = s[len(p):].strip()
                changed = True
                break
    return re.sub(r'(的|之)$', '', s).strip()


def _heuristic_plan(question: str) -> list:
    """
    純正則拆解。只在能高度確定的情況下才回傳清單，否則回空陣列(寧可不做，也不要亂做)。
    實測：「請幫我分別查詢台北 101 與日本東京晴空塔的建築總高度（公尺），並計算…」
          → ['台北 101 建築總高度', '日本東京晴空塔 建築總高度'] + 一個 math 步驟
    """
    if not question:
        return []
    # 1. 找出問題裡的「屬性關鍵字」，找不到就放棄(代表這不是典型的數據查詢題)
    attr = next((w for w in _ATTR_WORDS if w in question), None)
    if not attr:
        return []
    # 2. 只看屬性關鍵字「之前」的那段文字，在裡面找「A 與 B」的比較結構
    head = question[:question.index(attr)]
    m = re.search(r'(.+?)\s*(?:與|和|跟|、|以及|及)\s*(.+)$', head)
    if not m:
        return []
    a, b = _strip_lead(m.group(1)), _strip_lead(m.group(2))
    # 3. 防呆：太長或太短的片段幾乎都是切錯了，寧可放棄
    if not a or not b or len(a) > 20 or len(b) > 20:
        return []

    items = [("search", f"{a} {attr}"), ("search", f"{b} {attr}")]
    if re.search(_MATH_HINT, question):
        items.append(("math", f"{b}的{attr} 減去 {a}的{attr}（求兩者差值）"))
    return _build_plan(items)


def _is_self_contained_math(question: str) -> bool:
    """
    【SA v3.2】判斷這題是不是「自足的數學題」——所有需要的數字都寫在題目裡，
    不需要上網查任何東西。

    判準只有兩條，刻意保持極簡：
      1. 題目裡至少有兩個數字（一個數字通常是「你幾歲」這類事實題，不是運算）
      2. 題目裡沒有任何「需要外部事實」的訊號

    為什麼不看有沒有數學動詞？因為那是無窮清單，追不完（見上方說明）。
    """
    if not question:
        return False
    nums = re.findall(r'\d+(?:\.\d+)?', question)
    if len(nums) < 2:
        return False
    if re.search(_NEEDS_EXTERNAL_FACT, question):
        return False
    return True


def _heuristic_math_only(question: str) -> list:
    """
    【SA v3.2 改寫】自足數學題的保底拆解器。

    只要 _is_self_contained_math 成立，就直接排一個 math 步驟，
    任務描述【用問題原文】——因為原文裡本來就有全部數字，
    交給算盤法師時既能通過數字溯源檢查，也保留了完整語境
    （「分給幾個人、還剩幾個」這種資訊，改寫成算式反而會遺失）。
    """
    if not _is_self_contained_math(question):
        return []
    return _build_plan([("math", question.strip())])


# ==========================================
# 🏢 第四區：定義專家房間 (Nodes)
# ==========================================

# 【0. 規劃官房間】── v2 新增節點，v2.1 加上三層保底鏈
def planner_node(state: AgentState):
    """
    整輪對話開場只跑一次，把使用者問題拆成明確的任務清單。

    【SA v2.1 重點】：
    1. 會先看 RAG 知識庫。高精準區命中(hit_type == manual)代表這是公司內部客服題，
       標準答案就在知識庫裡，直接回傳空清單走 FINISH —— 不浪費 API、答案也更準。
    2. 拆解採「三層保底鏈」，就算主模型結構化輸出能力不足，也不會退化成 v1 的鬼打牆。
    """
    msgs = list(state["messages"])
    question = ""
    for m in msgs:
        if isinstance(m, HumanMessage):
            question = m.content

    rag_context = state.get("rag_context", "") or ""
    rag_hit_type = state.get("rag_hit_type", "none")

    base_return = {
        "facts": {},
        "searched_queries": [],
        "current_step": -1,
        "retry_count": 0,
        "search_calls": 0,
        "math_calls": 0,
        "all_steps_done": True,
        "plan_decision": "undetermined",
        "search_notes": [],
    }

    # ---- 【SA v4.2 新增】最優先前置攔截：使用者明確要求上網 → 不靠模型判斷，直接排 search ----
    #
    # v4.1 原本只是「manual 命中時不要短路，讓題目繼續走三層拆解」，但實測發現
    # 就算不短路，交給模型的第一層（結構化拆解）判斷後仍然吐出空清單：
    #
    #   [規劃官聖騎士 Planner] ⚠️ ...含明確上網訊號，不短路，繼續交給規劃官判斷。
    #   [規劃官聖騎士 Planner] 📋 (第一層) 判定本題不需要任何工具(空清單)
    #
    # 原因：第一層系統提示的判斷準則第 1 條只講「你不確定的具體事實數字」
    # （高度、人口、股價…），範例也只示範「查兩棟建築高度」這種數字題。
    # 「幫我上網搜尋昨天台灣的新聞」不含這類數字，8B 小模型套不進這條規則，
    # 又符合第 5 條「問公司介紹/服務內容可回傳空清單」的模糊邊界，於是誤判成不需要工具。
    #
    # 跟純數學題的處理邏輯一樣（見下方 _is_self_contained_math）：與其繼續
    # 加強提示詞去說服一顆小模型，不如把「使用者已經明講要上網」這件事
    # 直接寫死成規則，程式碼凌駕於 LLM 判斷 —— 這是本專案一路以來的設計原則。
    # 一旦句子裡出現明確的上網／搜尋／查詢動詞，一律不經模型、直接排一個 search 步驟，
    # 查詢關鍵字用 _strip_search_trigger_words() 把「幫我」「上網」這些贅字剝掉，
    # 只留下真正的查詢主題（例如「昨天台灣的新聞」），零 API、零模型判斷。
    if re.search(_EXPLICIT_SEARCH_INTENT, question):
        search_query = _strip_search_trigger_words(question)
        search_plan = _build_plan([("search", search_query)])
        print(f"\n[規劃官聖騎士 Planner] 🌐 前置判定：使用者明確要求上網查詢 → 直接排 search 步驟（零 API、零模型判斷），查詢詞：「{search_query}」")
        print(_render_plan(search_plan))
        return {**base_return, "plan": search_plan, "plan_decision": "has_plan"}

    # ---- 【SA v4.3 修正】前置攔截：自足的數學題根本不必問模型（順序提前到 RAG 短路之前）----
    #
    # 這一步刻意放在【呼叫 LLM 之前】，也刻意放在【RAG manual 短路判斷之前】。
    #
    # 血淋淋的實測翻車：「某公司技術部門有8位工程師...從技術部選出3人、
    # 從行銷部選出2人，總共有多少種選法？」這種純數學題，向量檢索意外跟知識庫裡
    # 「工作經歷」主題的面試問答很接近(距離 0.403，還低於精準門檻，是真命中不是邊界救援)，
    # 於是被舊版的 RAG manual 短路判斷攔截，整題交給小模型憑印象亂答，
    # 完全沒進算盤法師，答案對不對純粹看運氣。
    #
    # 這跟原本只保護「明確上網意圖」的 v4.2 是同一種病：向量相似度是模糊分數，
    # 不該凌駕在「題目本身就是可以百分之百確定的自足數學題」這種決定性判斷之上。
    # 所以把這一段搬到 RAG manual 短路的前面 —— 只要題目自帶所有數字、
    # 又沒有任何需要外部資料的訊號，就不計較 RAG 弓箭手撈到了什麼，直接交給算盤法師。
    #
    # 其他實測慘案（同樣是讓模型自己判斷才會發生）：
    #   「10片披薩分4個人」  → 規劃官吐出 search:4 / search:10，網路戰士真的去 Google「4」
    #   「大魚70條小魚30條」 → 拆出 search:大魚總數，搜到「一午二紅沙」這句台灣俗諺
    # 這些題目的數字全部寫在題幹裡，一次 API 都不該打。
    if _is_self_contained_math(question):
        math_plan = _build_plan([("math", question.strip())])
        print("\n[規劃官聖騎士 Planner] 🧮 前置判定：題目自帶所有數字、且無需外部資料 → 純數學題，直接交給算盤法師（零 API、零模型判斷）")
        print(_render_plan(math_plan))
        return {**base_return, "plan": math_plan, "plan_decision": "has_plan"}

    # ---- 前置判斷：知識庫已有標準答案，就不要上網 ----
    # 【SA v4.1】：manual 命中（含邊界救援誤判）且使用者沒有明確上網指令、也不是自足數學題時，才短路結案。
    # 上面兩段 v4.2／v4.3 攔截已經先處理掉「有明確上網指令」與「自足數學題」這兩種決定性情況，
    # 所以這裡的 RAG 短路只會作用在真正的知識庫問答題上，能安全短路。
    if rag_hit_type == "manual":
        print("\n[規劃官聖騎士 Planner] 🎯 RAG 高精準區已命中標準答案，本輪不需要上網查詢，直接結案。")
        return {**base_return, "plan": [], "plan_decision": "no_tools_needed"}

    # ---- 保底鏈第一層：巢狀結構化輸出 ----
    # 【SA v2.2】：layer1_ok / layer2_ok 記錄的是「這一層有沒有成功回傳」，
    # 而不是「有沒有產出步驟」。空清單也是一種成功的答案。
    plan = []
    layer1_ok = False
    layer2_ok = False

    # 【SA v2.4 移除 rag_hint ── 這是「台積電股價」那題翻車的直接原因】
    #
    # v2.3 曾在這裡把 RAG 檢索結果塞進規劃官的提示，並附帶一句
    # 「如果下面的內容已經足以回答問題，請回傳空的 steps 清單」。
    # 當時的想法是省下不必要的 API 呼叫，但這個設計有個致命的邏輯漏洞：
    #
    #   高精準命中(manual)在本函式【更前面】就已經直接 return 結案了，
    #   所以這段程式碼實際上【只會在 auto 軌的情況下執行】。
    #   而 auto 軌是無條件撈 Top-K 的 —— 它從來不看相關性，
    #   你問「台積電股價多少」，它一樣塞兩大段張序亞的履歷給你。
    #
    # 結果就是：拿一份「不保證相關」的資料，去說服規劃官「這些夠回答了，別上網」。
    # 規劃官照做，回傳空清單，主管直接結案，網路戰士從頭到尾沒被叫過，
    # 使用者只拿到一句「很抱歉，我們無法找到您關於台積電股價的查詢結果」。
    #
    # 正解：規劃官只負責「這題需要哪些工具」，不該替 auto 軌背書。
    # 判斷 auto 軌內容夠不夠用，是盜賊客服在寫回覆時的工作 ——
    # 它拿到的提示已經明確標註了「相關性不保證」。
    rag_hint = ""

    sys_msg = SystemMessage(content=(
        "你是任務拆解專家。你『絕對沒有』任何常識與計算能力，你的工作【只有】把問題拆成步驟清單，"
        "【嚴禁】在這裡回答問題或算出任何答案。\n\n"
        "拆解規則：\n"
        "1. 問題裡每一個『你不確定的具體事實數字』(高度、人口、股價、資本額、現任人物…)，"
        "都要獨立成一個 search 步驟，一個步驟只能查一個對象。\n"
        "2. 如果問題比較 A 和 B 兩個對象，就要產生兩個 search 步驟(一個查 A、一個查 B)，"
        "【絕對不可以】把 A 和 B 寫在同一個 target 裡。\n"
        "3. 所有需要運算的部分，都要獨立成 math 步驟，並且一定要排在相關的 search 步驟後面。\n"
        "4. 純邏輯/排列組合題(題目文字裡就有全部數字)，不需要 search，只要 math 步驟。\n"
        "5. 如果問題不需要查也不需要算(例如問公司介紹、問履歷、問服務內容)，請回傳空的 steps 清單。\n"
        "6. 步驟總數不要超過 6 個。\n\n"
        "範例：\n"
        "問題：「請查台北101與東京晴空塔的總高度，並算出誰高多少公尺？」\n"
        "正確拆解：\n"
        "  [1] search / 台北101 建築總高度 公尺\n"
        "  [2] search / 東京晴空塔 建築總高度 公尺\n"
        "  [3] math   / 東京晴空塔高度 減去 台北101高度\n"
        "錯誤拆解(禁止)：\n"
        "  [1] search / 台北101 與 東京晴空塔 建築總高度   ← 兩個對象混在一起，會查到模糊的比較文章\n"
        + rag_hint
    ))

    try:
        result = planner_structured_llm.invoke([sys_msg, HumanMessage(content=f"請拆解這個問題：{question}")])
        items = [(s.step_type, (s.target or "").strip()) for s in (result.steps or [])]
        plan = _sanitize_plan(_build_plan(items))
        # 【SA v2.2 關鍵修正】：只要第一層「成功回傳」就算數，不管它給的是幾個步驟。
        #
        # v2.1 的致命邏輯錯誤：用 `if not plan` 判斷要不要往下一層，
        # 這等於把「模型正確判斷『這題不需要任何工具』所以回傳空清單」
        # 誤認成「模型壞掉了」，然後把工作交給能力更弱的第二層去亂編。
        # 「請自我介紹一下」那次翻車就是這樣來的：
        # 第一層明明答對了(空清單)，卻被當成失敗，第二層接手後編出
        # 「search: AI 語言模型 / search: 自我介紹」，白燒兩次 Brave API。
        #
        # 正解：只有「拋出例外」才算失敗。空清單是一個合法且有意義的答案。
        layer1_ok = True
        if plan:
            print("\n[規劃官聖騎士 Planner] 📋 (第一層) 結構化拆解成功")
        else:
            print("\n[規劃官聖騎士 Planner] 📋 (第一層) 判定本題不需要任何工具(空清單)，直接交給盜賊客服回答")
    except Exception as e:
        print(f"[規劃官聖騎士 Planner] ⚠️ (第一層) 巢狀結構化拆解失敗：{e}")

    # ---- 保底鏈第二層：兩段式拆解(純字串清單，小模型友善) ----
    # 【SA v2.2】：判斷條件從 `if not plan` 改成 `if not layer1_ok`。
    # 保底鏈只在「上一層真的壞掉」時才啟動，不會在上一層答對時越俎代庖。
    if not layer1_ok:
        print("[規劃官聖騎士 Planner] 🔄 (第二層) 改用兩段式拆解...")
        try:
            targets_llm = verify_llm.with_structured_output(SearchTargets)
            t_sys = SystemMessage(content=(
                "你是搜尋關鍵字拆解員。請把下面的問題，拆成『需要上網查詢的關鍵字清單』。\n"
                "規則：\n"
                "1. 一個字串只能包含【一個】查詢對象，絕對不可以把兩個對象寫在同一個字串裡。\n"
                "2. 每個字串請寫成『對象 + 要查的屬性』，例如 '台北101 建築總高度'。\n"
                "3. 如果問題完全不需要上網(例如純數學題、問你自己是誰、請你自我介紹、"
                "問公司服務內容、問某人的履歷經歷)，請回傳空陣列。\n"
                "4. 不要輸出任何解釋文字。"
            ))
            t_res = targets_llm.invoke([t_sys, HumanMessage(content=question)])
            items = [("search", t.strip()) for t in (t_res.targets or []) if t and t.strip()][:5]

            m_llm = verify_llm.with_structured_output(MathNeed)
            m_sys = SystemMessage(content=(
                "請判斷下面這個問題，在查到資料之後是否還需要做數值運算(加減乘除、求差值、算倍數等)。\n"
                "如果需要，description 請直接寫出『要算什麼』的具體內容"
                "(例如『東京晴空塔高度 減去 台北101高度』)，"
                "【絕對不要】把這段判斷規則本身複述回來當成答案。\n"
                "不要輸出任何解釋文字。"
            ))
            m_res = m_llm.invoke([m_sys, HumanMessage(content=question)])
            if m_res.need_math and (m_res.description or "").strip():
                items.append(("math", m_res.description.strip()))

            plan = _sanitize_plan(_build_plan(items))
            layer2_ok = True
            if plan:
                print("[規劃官聖騎士 Planner] 📋 (第二層) 兩段式拆解成功")
            else:
                print("[規劃官聖騎士 Planner] 📋 (第二層) 判定本題不需要任何工具(空清單)")
        except Exception as e:
            print(f"[規劃官聖騎士 Planner] ⚠️ (第二層) 兩段式拆解失敗：{e}")

    # ---- 保底鏈第三層：純 Python 正則啟發式 ----
    # 【SA v2.2】：同樣改成只在前兩層都真的壞掉時才啟動
    if not layer1_ok and not layer2_ok:
        print("[規劃官聖騎士 Planner] 🔄 (第三層) 改用純 Python 正則啟發式拆解...")
        plan = _sanitize_plan(_heuristic_plan(question))
        if plan:
            print("[規劃官聖騎士 Planner] 📋 (第三層) 正則啟發式拆解成功")

    # ---- 保底鏈第四層：放棄拆解，退回 LLM 自由判斷(等同 v1 行為) ----
    # 【SA v2.3】：這裡把「結論」明確化成三態，不再只丟一個空清單給 Supervisor 去猜。
    if plan:
        plan_decision = "has_plan"
        print("[規劃官聖騎士 Planner] 📋 本輪任務清單：")
        print(_render_plan(plan))
    elif layer1_ok or layer2_ok:
        # 有任何一層「成功回傳」但結果是空的 → 這是一個明確的判斷：本題不需要工具。
        #
        # 【SA v3.1 注意】：這裡的「空」有兩種來源，兩種都該走同樣的補救：
        #   (1) 模型本來就回傳空清單（判斷不需要工具）
        #   (2) 模型回了清單，但整份被 _sanitize_plan 判定為垃圾而清空
        #       —— 披薩題就是這種：拆出 search:4 / search:10 / math:4，全部被過濾掉。
        # 兩種情況都先用純 Python 確認一次是不是純數學題，是的話自己補上計算步驟，
        # 而不是放它去走 LLM 自由判斷（那才是真的會亂搜尋）。
        math_plan = _heuristic_math_only(question)
        if math_plan:
            plan = math_plan
            plan_decision = "has_plan"
            print("[規劃官聖騎士 Planner] 🧮 補救：偵測到這是純數學題，自動補上計算步驟")
            print(_render_plan(plan))
        else:
            plan_decision = "no_tools_needed"
            print("[規劃官聖騎士 Planner] ✅ 結論：本題不需要任何工具，直接交給盜賊客服回答(不呼叫任何 API)")
    else:
        plan_decision = "undetermined"
        print("[規劃官聖騎士 Planner] ⚠️ 三層拆解全部失敗，交由主管召喚師自由判斷")

    return {**base_return, "plan": plan, "plan_decision": plan_decision}


# 【1. 總機主管房間】
def supervisor_node(state: AgentState):
    """
    有清單時 → 純 Python 找出第一個 pending 的項目派工，完全不呼叫 LLM。
                好處：0 token、0 幻覺、0 鬼打牆，log 會直接印出進度表。
    沒清單時 → 才退回 v1 的 LLM 自由判斷(保底路徑)。
    """
    plan = copy.deepcopy(state.get("plan", []) or [])
    facts = dict(state.get("facts", {}) or {})
    searched = list(state.get("searched_queries", []) or [])
    cur_search_calls = state.get("search_calls", 0)
    cur_math_calls = state.get("math_calls", 0)

    carry = {
        "plan": plan,
        "facts": facts,
        "searched_queries": searched,
        "search_calls": cur_search_calls,
        "math_calls": cur_math_calls,
        "retry_count": 0,
    }

    # 🛑 【Python 物理絕對防禦】：v2 改看 STATUS 旗標，不再掃描內文關鍵字
    msgs = list(state["messages"])
    if msgs and _is_failed_message(msgs[-1]):
        last_name = getattr(msgs[-1], "name", "")
        if last_name in ["Math_Agent", "Search_Agent"]:
            print(f"\n[系統守衛] 🛑 偵測到 {last_name} 回報失敗旗標，記錄後繼續往下一項任務。")

    # ------------------------------------------------
    # (A) 有任務清單 → 純 Python 勾選模式
    # ------------------------------------------------
    if plan:
        for step in plan:
            if step["status"] == "pending" and step.get("attempts", 0) >= MAX_STEP_ATTEMPTS:
                step["status"] = "failed"
                print(f"[主管召喚師 Supervisor] ⚠️ 任務 [{step['id']}] {step['target']} 已嘗試 {step['attempts']} 次仍未完成，標記失敗並跳過。")

        print("\n[主管召喚師 Supervisor] 📋 目前進度：")
        print(_render_plan(plan))

        for step in plan:
            if step["status"] == "pending":
                node = "Search_Agent" if step["type"] == "search" else "Math_Agent"
                print(f"[主管召喚師 Supervisor] ➡️ 派工給 {node}：任務 [{step['id']}] {step['target']}")
                return {**carry, "plan": plan, "next_node": node, "current_step": step["id"]}

        print("[主管召喚師 Supervisor] 🎉 任務清單全數處理完畢，交給盜賊客服結案。")
        return {**carry, "plan": plan, "next_node": "FINISH", "current_step": -1}

    # ------------------------------------------------
    # (B) 【SA v2.3 新增】規劃官明確判定「不需要任何工具」→ 直接結案
    # ------------------------------------------------
    # 這一段就是「請自我介紹一下卻跑去 Google」的解藥。
    # v2.2 的 Supervisor 只看得到 plan == []，分不出這是「答對」還是「壞掉」，
    # 一律走下面的 LLM 自由判斷，於是 XUYA 又把它派去搜尋。
    # 現在改看規劃官傳下來的明確結論，該什麼都不做的時候就什麼都不做。
    if state.get("plan_decision") == "no_tools_needed":
        print("\n[主管召喚師 Supervisor] 🈳 規劃官已判定本題不需要任何工具，直接交給盜賊客服結案(零 API 呼叫)。")
        return {**carry, "next_node": "FINISH", "current_step": -1}

    # ------------------------------------------------
    # (C) 沒有任務清單、也沒有明確結論 → 退回 v1 的 LLM 自由判斷(保底)
    # ------------------------------------------------
    chat_history = list(state.get("chat_history", []))

    facts_note = (
        "\n\n【本輪已確認的事實帳本】(這些已經查到了，絕對不要再重複查)：\n"
        + _render_facts(facts)
    )
    searched_note = ""
    if searched:
        searched_note = "\n\n【本輪已經查詢過的關鍵字】(不要再查同樣的東西)：\n" + "\n".join(f"- {q}" for q in searched)

    # 【SA v2.4】：這裡原本也會把 RAG 內容餵給主管、並暗示「夠了就選 FINISH」，
    # 與規劃官那邊是同一個漏洞（精準命中早就短路了，所以只會餵到不保證相關的 auto 軌）。
    # 一併移除，主管只負責派工，不替 auto 軌的相關性背書。
    rag_note = ""

    sys_msg = SystemMessage(content=(
        "你是路由主管。你『絕對沒有』任何常識、歷史知識或數學能力！\n"
        "請嚴格遵守以下派工順序：\n"
        "1. 若問題是詢問張序亞的履歷，或【參考知識庫】裡已有答案，直接選 'FINISH'。\n"
        "2. 只要問題詢問『客觀事實、數據』，且【事實帳本裡還沒有這筆資料】，"
        "你【絕對不准】憑記憶回答，【強制】派給 'Search_Agent' 查詢！\n"
        "3. 【重要】如果某個數字【已經出現在下方的事實帳本裡】，就代表它查到了，"
        "【絕對不准】再派 Search_Agent 去查同一個東西，請直接進入下一步(計算或結案)。\n"
        "4. 拿到數字後若需計算，【絕對不准】在理由中心算，【強制】派給 'Math_Agent'！\n"
        "5. 比較兩個以上實體時，可以針對『不同實體』連續派工，但每次查的對象必須不同。\n"
        "6. 事實帳本已足夠回答問題時，選 'FINISH'。"
        + facts_note + searched_note + rag_note
    ))

    decision = supervisor_llm.invoke([sys_msg] + chat_history + msgs)
    print(f"\n[主管召喚師 Supervisor] 決定派工給: {decision.next_node} (理由: {decision.reasoning})")

    # 【SA 保留 v1】：攔截「假裝搜尋過」的幻覺
    has_search_record = any(getattr(m, "name", "") == "Search_Agent" for m in msgs)
    if decision.next_node == "Math_Agent" and not has_search_record and not facts:
        if any(kw in decision.reasoning for kw in ["搜尋結果", "查詢結果", "根據網路", "根據維基"]):
            print("[系統守衛] 🛑 偵測到主管聲稱『已搜尋』但這一輪其實從未執行搜尋，強制導正為 Search_Agent！")
            return {**carry, "next_node": "Search_Agent", "current_step": -1}

    return {**carry, "next_node": decision.next_node, "current_step": -1}


# 【2. 網路戰士房間】
def search_node(state: AgentState):
    """
    1. 關鍵字不再每次都靠小模型「重新抽取」—— 清單模式下直接用 Planner 定好的 target。
    2. Python 硬去重：已經打過 API 的關鍵字絕不再打第二次(省 Brave 額度)。
    3. 拿到結果後立刻做「數值萃取」，把乾淨的數字登錄進事實帳本。
    【SA v2.1】：本節點完全看不到 rag_context，避免說明書段落干擾關鍵字抽取。
    """
    msgs = list(state["messages"])
    plan = copy.deepcopy(state.get("plan", []) or [])
    facts = dict(state.get("facts", {}) or {})
    searched = list(state.get("searched_queries", []) or [])
    step_id = state.get("current_step", -1)
    step = _find_step(plan, step_id)

    search_calls = state.get("search_calls", 0) + 1
    retry_count = state.get("retry_count", 0)
    if msgs and getattr(msgs[-1], "name", "") == "Search_Agent":
        retry_count += 1

    print(f"[網路戰士 Search_Agent] 收到任務(本輪第 {search_calls} 次搜尋)，準備出擊...")

    # ---- 決定這次要查什麼 ----
    if step is not None:
        # 清單模式：關鍵字由 Planner 事先決定，重試時做機械式變形(不呼叫 LLM，省算力)
        base_query = step["target"]
        if retry_count == 1:
            base_query = f"{base_query} 維基百科"
        elif retry_count >= 2:
            base_query = re.sub(r'(建築|總|大約|請問|查詢)', '', base_query).strip()
        step["attempts"] = step.get("attempts", 0) + 1
    else:
        # 保底模式：沿用 v1 的小模型抽取 + 合併查詢拆分
        extractor = verify_llm.with_structured_output(SearchQuery)
        already_searched_hint = ""
        if searched:
            already_searched_hint = (
                "\n\n【注意】：這一輪已經用過以下查詢字串：\n"
                + "\n".join(f"- {q}" for q in searched)
                + "\n請針對『還沒查過的』對象提取關鍵字，不要重複查同一個東西！"
            )
        sys_msg = SystemMessage(content=(
            "你是一個關鍵字提取專家。請只根據下方【本輪工作紀錄】提取出最適合上網搜尋的精準關鍵字，"
            "絕對不要參考任何你自己記得的舊資訊或歷史對話。\n"
            "如果問題同時比較兩個以上的實體，請『一次只針對一個實體』提取關鍵字。\n"
            "如果詢問『現任』人物、職位，請加上『2026年 最新』等字眼。\n"
            "如果詢問台灣股票，請加上『台灣股市』或『台幣』等字眼，避免查到美股 ADR。"
            + already_searched_hint
        ))
        search_req = extractor.invoke([sys_msg] + msgs)
        base_query = _pick_unsearched_segment(search_req.query, searched)
        if base_query != search_req.query:
            print(f"[網路戰士 Search_Agent] ⚠️ 偵測到合併查詢，強制拆分鎖定單一對象: {base_query}")

    # ---- Python 硬去重(「不再浪費 API」的實作) ----
    final_query = _dedup_query(base_query, searched)
    if final_query is None:
        print(f"[網路戰士 Search_Agent] 🛑 關鍵字「{base_query}」及其所有變形都已查過，直接放棄本次搜尋以節省 API 額度。")
        return {
            "messages": [AIMessage(
                name="Search_Agent",
                content=f"{STATUS_FAIL} 關鍵字「{base_query}」本輪已重複查詢過，未再次呼叫搜尋 API。"
            )],
            "plan": plan, "facts": facts, "searched_queries": searched,
            "retry_count": retry_count, "search_calls": search_calls
        }

    print(f"[網路戰士 Search_Agent] 正在網路上揮劍尋找: {final_query}")
    from tools.web_search import search_web_ex
    payload = search_web_ex(final_query)
    searched.append(final_query)

    if not payload["ok"]:
        print(f"[網路戰士 Search_Agent] ❌ 搜尋未取得結果：{payload['message']}")
        return {
            "messages": [AIMessage(
                name="Search_Agent",
                content=f"{STATUS_FAIL} 搜尋「{final_query}」未取得有效結果：{payload['message']}"
            )],
            "plan": plan, "facts": facts, "searched_queries": searched,
            "retry_count": retry_count, "search_calls": search_calls
        }

    raw = normalize_chinese_number_units(payload["text"])

    # 【SA 保留 v1】：明確數字速查表
    annotated_numbers = re.findall(r'\(=([\d.]+)\)', raw)
    if annotated_numbers:
        unique_numbers = list(dict.fromkeys(annotated_numbers))
        raw = f"【已換算好的明確數字，請優先使用】：{('、'.join(unique_numbers))}\n\n" + raw

    fact_target = step["target"] if step is not None else final_query
    excerpt = raw[:800]

    # ---- 【SA v4.3 新增】分流：這是「單一可驗證數值」查詢，還是「開放式資訊」查詢？ ----
    #
    # 血淋淋的實測翻車（新聞題）：問「昨天台灣的新聞」，下面的 ExtractedFact 萃取器
    # 被逼著一定要從搜尋結果裡「找出一個數值」，結果從一段完全無關的
    # 「央行升息半碼(12.5個基點)」新聞片段裡硬抓了「12.5」出來，
    # 登錄成「昨天台灣的新聞 ＝ 12.5個基」這種語意不通的假事實，
    # 最後還被 Final_Answer 的確定性模板原封不動印給使用者看。
    #
    # 根因是這整條「數值萃取 → 事實帳本 → 數字溯源檢查」的管線，
    # 從設計上就只適合「有單一正確答案的事實題」(建築高度、股價、人口…)，
    # 對「新聞、話題、看法」這種開放式資訊完全文不對題。
    #
    # 解法：先用 _is_single_fact_target() 判斷這次查的到底是不是「量化屬性」，
    # 是，才進數值萃取這條管線；不是，就直接把原始搜尋摘要存進 search_notes，
    # 交給 Final_Answer 自己統整成一段文字回覆，不勉強蒸餾成一個數字。
    if _is_single_fact_target(fact_target):
        # ---- 【SA v2 核心】：數值萃取 → 登錄事實帳本 ----
        extracted_value = ""
        try:
            fact_extractor = verify_llm.with_structured_output(ExtractedFact)
            fact_sys = SystemMessage(content=(
                "你是數值萃取員。請【只】從下方搜尋結果的文字中，找出使用者要的那一個數值。\n"
                f"要找的目標是：{fact_target}\n\n"
                "規則：\n"
                "1. 只能使用搜尋結果裡真正出現的數字，【絕對禁止】使用你自己記憶中的數字。\n"
                "2. 找到請填 found=true，並在 value 填上『數字 + 單位』(例如 '508 公尺')。\n"
                "3. 搜尋結果裡如果沒有明確數字，請誠實填 found=false，value 留空。\n"
                "4. 不要輸出任何解釋文字。"
            ))
            fact_res = fact_extractor.invoke([fact_sys, HumanMessage(content=raw[:2500])])
            if fact_res.found and (fact_res.value or "").strip():
                extracted_value = fact_res.value.strip()
        except Exception as e:
            print(f"[網路戰士 Search_Agent] ⚠️ 數值萃取器異常({e})，改為保留原始摘要交給下游判讀。")

        if extracted_value:
            facts[fact_target] = extracted_value
            if step is not None:
                step["status"] = "done"
                step["result"] = extracted_value
            print(f"[網路戰士 Search_Agent] 📒 已登錄事實帳本：{fact_target} ＝ {extracted_value}")
            content = (
                f"{STATUS_OK}\n"
                f"【搜尋關鍵字：{final_query}】\n"
                f"【已確認事實】{fact_target} ＝ {extracted_value}\n"
                f"【原始摘要(節錄)】\n{excerpt}"
            )
        else:
            print(f"[網路戰士 Search_Agent] ⚠️ 搜尋有結果，但未能萃取出「{fact_target}」的明確數值。")
            content = (
                f"{STATUS_FAIL}\n"
                f"【搜尋關鍵字：{final_query}】\n"
                f"【問題】搜尋有回應，但結果中找不到「{fact_target}」的明確數值。\n"
                f"【原始摘要(節錄)】\n{excerpt}"
            )

        return {
            "messages": [AIMessage(name="Search_Agent", content=content)],
            "plan": plan,
            "facts": facts,
            "searched_queries": searched,
            "retry_count": retry_count,
            "search_calls": search_calls
        }

    # ---- 開放式資訊查詢：不萃取數值，直接把原始搜尋摘要存進 search_notes ----
    search_notes = list(state.get("search_notes", []) or [])
    search_notes.append(f"【查詢主題：{fact_target}】\n{excerpt}")
    if step is not None:
        step["status"] = "done"
        step["result"] = f"（已取得搜尋摘要，共 {len(excerpt)} 字，詳見 search_notes）"
    print(f"[網路戰士 Search_Agent] 📝 這是開放式資訊查詢，不勉強萃取單一數值，直接保留搜尋摘要（{len(excerpt)} 字）供 Final_Answer 統整。")
    content = (
        f"{STATUS_OK}\n"
        f"【搜尋關鍵字：{final_query}】\n"
        f"【這是開放式資訊查詢，摘要已保留供最終回覆整理引用】\n"
        f"【原始摘要(節錄)】\n{excerpt}"
    )

    return {
        "messages": [AIMessage(name="Search_Agent", content=content)],
        "plan": plan,
        "facts": facts,
        "searched_queries": searched,
        "retry_count": retry_count,
        "search_calls": search_calls,
        "search_notes": search_notes,
    }


# 【SA v2】：Search_Agent 出口的 Local Grader (f-b)，只看 STATUS 旗標
def search_grader(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    is_bad = _is_failed_message(last_msg)
    retry = state.get("retry_count", 0)

    if is_bad and retry < MAX_RETRY:
        print(f"[網路鑑定士 f-Search] ❌ 搜尋未取得有效事實，退回重搜 (第 {retry} 次重試)")
        return "retry"
    if is_bad:
        print("[網路鑑定士 f-Search] 🛑 已達重試上限，本項任務放棄，交回主管召喚師繼續下一項")
        return "pass"   # 【SA v2】：不再 give_up 直接結案，否則第二個對象永遠沒機會查
    print("[網路鑑定士 f-Search] ✅ 已取得明確事實，放行給主管召喚師")
    return "pass"


# 【3. 算盤法師房間】
def math_node(state: AgentState):
    """
    1. 上下文換血 —— 只餵「事實帳本 + 這一步要算什麼」，不再丟整坨搜尋雜訊。
    2. 合法性檢查升級 —— 額外要求算式必須含有運算子(擋掉 '508' 這種假合格)。
    3. 數字溯源檢查 —— 算式裡的數字必須來自事實帳本。
    """
    msgs = list(state["messages"])
    plan = copy.deepcopy(state.get("plan", []) or [])
    facts = dict(state.get("facts", {}) or {})
    step_id = state.get("current_step", -1)
    step = _find_step(plan, step_id)

    math_calls = state.get("math_calls", 0) + 1
    retry_count = state.get("retry_count", 0)
    if msgs and getattr(msgs[-1], "name", "") == "Math_Agent":
        retry_count += 1

    if step is not None:
        step["attempts"] = step.get("attempts", 0) + 1
        task_desc = step["target"]
    else:
        task_desc = ""
        for m in msgs:
            if isinstance(m, HumanMessage):
                task_desc = m.content

    print(f"[算盤法師 Math_Agent] 收到任務(本輪第 {math_calls} 次計算)，正在推導公式...")
    print(f"[算盤法師 Math_Agent] 📒 目前可用的事實帳本：\n{_render_facts(facts)}")

    # 【SA v3.2 新增】：任務描述如果【本身就是一條純運算式】，直接拿去算，不要問模型。
    #
    # 實測慘案：任務是「10 / 6」——這已經是一條可以直接執行的算式了，
    # 但舊版還是丟給 gemma3:4b「翻譯」，結果它吐出 comb(6, 1) / comb(10, 1)，
    # 算出 0.6；下一個任務「10 % 6」它又吐成 10 / 6，算出 1.667。
    # 兩個答案都是錯的，卻都通過了語法與溯源檢查（因為數字確實來自題目）。
    #
    # 這是典型的「讓模型做它不必做的事」。算式已經在手上，翻譯步驟只會引入錯誤。
    expr_direct = task_desc.strip()
    if re.fullmatch(r'[\d\.\+\-\*/%\(\)\s]+', expr_direct) and re.search(r'[\+\-\*/%]', expr_direct):
        print(f"[算盤法師 Math_Agent] ⚡ 任務本身就是純算式，跳過模型翻譯，直接計算：{expr_direct}")
        from tools.calculator import calculate_math
        result = calculate_math(expr_direct)
        if not result.startswith("計算失敗"):
            if step is not None:
                step["status"] = "done"
                step["result"] = result
                facts[step["target"]] = _extract_calc_value(result)
            else:
                facts[f"計算：{expr_direct}"] = _extract_calc_value(result)
            print(f"[算盤法師 Math_Agent] 📒 已登錄：{task_desc} ＝ {_extract_calc_value(result)}")
            return {
                "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_OK}\n【計算機結果】\n{result}")],
                "plan": plan, "facts": facts,
                "retry_count": retry_count, "math_calls": math_calls
            }
        # 直接算失敗就往下走，讓模型試著重新翻譯
        print(f"[算盤法師 Math_Agent] ⚠️ 直接計算失敗（{result}），改請模型重新翻譯。")

    # 【SA v3.5 新增】：重試時把上一次失敗的原因帶進提示。
    #
    # 舊版重試是「原封不動再問一次」—— 模型沒有任何新資訊，
    # 實測結果就是連續三次吐出一模一樣的錯誤算式，白白燒掉三次呼叫。
    # 既然檢查器已經明確知道錯在哪（漏了哪個數字、用錯什麼函式），
    # 就把它告訴模型，讓重試真的有機會改對。
    retry_hint = ""
    if msgs:
        last = msgs[-1]
        if getattr(last, "name", "") == "Math_Agent":
            last_content = str(getattr(last, "content", ""))
            if last_content.lstrip().startswith(STATUS_FAIL):
                problem = last_content.split("無法產生有效算式：")[-1].strip()
                if problem:
                    retry_hint = (
                        f"\n\n⚠️【上一次你答錯了，原因如下，請務必修正】：\n{problem[:200]}\n"
                        "請針對這個問題重新產生算式，不要再交出同樣的答案。"
                    )

    extractor = verify_llm.with_structured_output(MathExpression)
    sys_msg = SystemMessage(content=(
        "你是數學算式翻譯機。請把下方的計算任務，翻譯成一行純 Python 數學算式。\n\n"
        f"【可以使用的事實帳本】(只能用這裡面的數字，禁止使用你自己記憶中的任何數字)：\n{_render_facts(facts)}\n\n"
        f"【這一步要計算的任務】：{task_desc}\n\n"
        "嚴格規則：\n"
        "1. 只能輸出算式本身，例如 '634 - 508'，【絕對不能包含任何中文字、等號、單位或說明】。\n"
        "2. 算式裡【必須】至少有一個運算子(+ - * /)或函式，"
        "【禁止】只回一個光禿禿的數字(例如只回 '508' 是錯的)。\n"
        "3. 算式裡的每一個數字，都必須是上面事實帳本或任務描述裡出現過的數字。\n"
        # 【SA v3.3 新增】：教它分配題怎麼寫。
        # 實測「糖果25顆分給7個人，每人幾顆又剩多少」被翻成 comb(25,7)-(7*7)=480651，
        # 因為模型根本不知道「整除」和「取餘數」該用什麼符號，只好亂抓函式。
        "4. 【分配題】如果任務是「N 個東西分給 M 個人，每人幾個」，"
        "請用整數除法 N // M；如果問「還剩幾個」，請用取餘數 N % M。"
        "例如「25顆分給7個人每人幾顆」→ 25 // 7；「還剩幾顆」→ 25 % 7。\n"
        # 【SA v3.3 新增】：封鎖 comb 濫用。
        # 這是所有計算錯誤的共同元凶 —— 小模型學到了「數學任務就用 comb」，於是：
        #   糖果分配 → comb(25, 7) - (7 * 7)
        #   10 / 6   → comb(6, 1) / comb(10, 1)
        #   蘋果檸檬 → comb(10,2)*comb(5,1)/comb(3,1) - comb(7,1) - ...（長到爆行）
        # 全部通過語法與溯源檢查，因為數字確實來自題目。
        "5. 【嚴禁濫用組合函式】：comb / perm / factorial 只能用在"
        "題目明確詢問「組合」「排列」「有幾種選法」「挑選方式」的時候。"
        "分配、平分、相差、加總、倍數這些題目【一律不准】使用 comb，"
        "請老實用 + - * / // % 就好。\n"
        # 【SA v3.4 新增】：允許一次回多條算式。
        #
        # 實測情境：「有60顆蘋果、40顆檸檬，要分給9個人，每個人可以各拿多少？又會剩下多少？」
        # 這一題其實問了四件事（蘋果每人幾顆／蘋果剩幾顆／檸檬每人幾顆／檸檬剩幾顆），
        # 但舊版只准回一條算式，模型只好硬把它們湊成 (60//9)+(40//9) 這種
        # 語法正確、語意卻毫無意義的東西。
        #
        # 解法很單純：准它用分號隔開多條算式，每一條都會被獨立計算並個別回報。
        "6. 【題目問了好幾件事時】請用分號 ; 隔開多條算式，每條算式對應一個問題。"
        "例如「60顆蘋果分給9個人，每人幾顆又剩幾顆」→ 60 // 9 ; 60 % 9。"
        "如果題目同時問了兩種東西（蘋果和檸檬），就四條都寫出來："
        "60 // 9 ; 60 % 9 ; 40 // 9 ; 40 % 9。"
        "【絕對禁止】把兩種不同東西的商加在一起，例如「大魚70條、小魚30條分給7人，"
        "每人幾條大魚、幾條小魚」【絕對不可以】寫成 (70 // 7) + (30 // 7)，"
        "正確寫法一樣是四條分開的算式：70 // 7 ; 70 % 7 ; 30 // 7 ; 30 % 7。"
        "只要題目問的是「幾個 A、幾個 B」這種並列的兩種東西，答案就一定要分開，不能合併相加。\n"
        "7. 除了分號之外，【不要】輸出任何其他符號或文字，"
        "特別是不要在算式前面加冒號、等號或「答案」兩個字。\n"
        "8. 【題目給的每個數字都要用到】：如果題目寫了 60、40、9 三個數字，"
        "你的算式就必須把這三個數字都處理到，不可以只算其中一部分。\n"
        # 【SA v4.0 新增】：教它處理「後面的步驟需要前面的結果」這種題型。
        # 這是文具福袋題翻車的直接原因 —— 模型無法在一行裡表達步驟依賴，
        # 只好用猜的，把「每袋幾本」當成「剩下幾本」拿去換貼紙。
        "9. 【多步驟推導請用變數】：如果後面的計算需要用到前面算出來的結果，"
        "請改寫成多行，每行一個變數指派，用變數名表達依賴關係。例如：\n"
        "題目「75本、52支、38瓶分成8袋各剩多少？把剩下的每3件換1張貼紙可換幾張？」\n"
        "正確寫法（注意：換貼紙用的是【餘數】r1 r2 r3，不是每袋數量）：\n"
        "r1 = 75 % 8\n"
        "r2 = 52 % 8\n"
        "r3 = 38 % 8\n"
        "q1 = 75 // 8\n"
        "q2 = 52 // 8\n"
        "q3 = 38 // 8\n"
        "total_rem = r1 + r2 + r3\n"
        "stickers = total_rem // 3\n"
        "leftover = total_rem % 3\n"
        "變數名請取得有意義（r=餘數、q=每份數量），這樣你自己也比較不會混淆。"
        + retry_hint
    ))

    # 【SA v2 隔離重點】：刻意「不」傳入 msgs，只給乾淨的帳本與任務描述
    math_req = extractor.invoke([sys_msg, HumanMessage(content=f"請翻譯這個計算任務：{task_desc}")])

    # 【SA v4.0】：保留模型的原始輸出，清理失敗時才有東西可以看。
    # 之前「抽取結果：''」那個 log 完全看不出模型到底吐了什麼，無從診斷。
    raw_expr = (math_req.expression or "")
    expr = raw_expr.strip()

    # 【SA v3.6 修正】：清理算式前後的雜訊字元。
    #
    # 這裡我犯了跟「數學動詞白名單」一模一樣的錯誤兩次：
    #   v3.4：發現模型吐出 ':(60 // 9) + (40 // 9)'，於是我列舉了 : ： = ＝ 「」 引號 …
    #   v3.6：模型改吐 '60 // 9 ; 60 % 9 ; 40 // 9 ; 40 % 9}' —— 尾巴一個 }
    #         不在我的清單裡，同一條算式又被整條丟掉，重試三次三次都一樣。
    #
    # 教訓：不要列舉「有哪些垃圾」（無窮無盡），要定義「什麼是合法的」（很少而且固定）。
    # 一條算式的開頭只可能是數字、左括號，或函式名的第一個字母；
    # 結尾只可能是數字或右括號。其餘一律從邊緣剝掉。
    def _trim_to_math(e: str) -> str:
        # 【SA v3.6】開頭只認：數字、左括號、或 ASCII 字母（comb/perm/factorial 的開頭）。
        # 特別注意要用 isascii()：中文字的 isalpha() 也是 True，
        # 少了這道檢查，「答案：634 - 508」的「答」會被當成函式名開頭而留下來。
        e = e.strip()
        while e and not (e[0].isdigit() or e[0] == "(" or (e[0].isascii() and e[0].isalpha())):
            e = e[1:].lstrip()
        while e and not (e[-1].isdigit() or e[-1] == ")"):
            e = e[:-1].rstrip()
        return e

    expr = _trim_to_math(expr)

    # 【SA v4.4 新增】記法正規化 —— 把模型自己習慣的數學課本記法改寫成白名單認得的函式名。
    #
    # 實測翻車（8選3、6選2 選法題）：提示詞從頭到尾只教過 comb(n,k)，
    # 模型卻自己選擇更熟悉的課本寫法 C(8,3) * C(6,2)。這不是模型「看不懂題目」，
    # 是它輸出習慣跟指令不一致；語法檢查的白名單只認 comb/perm/factorial，
    # "C" 不在清單裡，於是連續三次判定為非法字元，整題失敗轉真人。
    #
    # 與其繼續在提示詞裡加字去說服模型「請用 comb」(已經加過一次了，這次還是沒用)，
    # 不如把「模型很可能會用的同義記法」直接正規化掉 —— 這跟數值萃取那次的教訓一樣：
    # 程式碼負責把輸入攤平成白名單認得的形式，不要期待小模型每次都乖乖照抄指令格式。
    # 只處理最常見的兩種課本記法：C(n,k) 組合數、P(n,k) 排列數。
    # 用 \b 避免誤傷 comb/perm 內部字母(例如 perm 開頭沒有裸 P 後面接括號的情況)。
    expr = re.sub(r'\bC\s*\(', 'comb(', expr)
    expr = re.sub(r'\bP\s*\(', 'perm(', expr)

    # ---- 【SA v4.0】鏈式推導：多行腳本優先走 calculate_script ----
    #
    # 為什麼要有這條路？實測「文具福袋題」暴露了單行算式的極限：
    #   「75本、52支、38瓶分成8袋各剩多少？再把剩下的每3件換1張貼紙」
    # 第二小題必須先知道三個餘數（3、4、6）加起來是 13，才能算 13 // 3。
    # 但規劃節點開場時不可能知道餘數是多少，模型也無法在一行裡表達這種依賴，
    # 結果它把「每袋幾本」(9+6+4) 當成「剩下幾本」除以 3，答出 6 張貼紙（正解 4 張）。
    #
    # 讓模型用變數寫幾行推導，依賴關係就由變數本身表達，
    # 而且執行過程完全確定 —— 不需要跑多輪迴圈、不需要模型記得中間結果。
    if "\n" in expr or re.search(r'^[A-Za-z_]\w*\s*=', expr.strip(), re.M):
        from tools.calculator import calculate_script
        script_res = calculate_script(expr)
        if script_res["ok"]:
            steps_txt = "、".join(
                f"{name or '結果'} = {e} → {v}" for name, e, v in script_res["steps"]
            )
            final_vars = script_res["vars"]
            print(f"[算盤法師 Math_Agent] 🔗 鏈式推導成功（{len(script_res['steps'])} 步）")
            for name, e, v in script_res["steps"]:
                print(f"           {str(name or '(算式)'):12} = {e:28} → {v}")
            if step is not None:
                step["status"] = "done"
                step["result"] = steps_txt
                facts[step["target"]] = steps_txt
            else:
                facts["推導結果"] = steps_txt
            return {
                "messages": [AIMessage(name="Math_Agent",
                                       content=f"{STATUS_OK}\n【逐步推導結果】\n{steps_txt}")],
                "plan": plan, "facts": facts,
                "retry_count": retry_count, "math_calls": math_calls
            }
        print(f"[算盤法師 Math_Agent] ⚠️ 鏈式推導失敗（{script_res['message']}），改以單行算式流程處理。")

    # 【SA v4.0】：拆成多條算式分別驗證與計算（沒有分號時就是單條，行為不變）
    sub_exprs = [_trim_to_math(e) for e in expr.split(";")]
    sub_exprs = [e for e in sub_exprs if e]
    if not sub_exprs:
        sub_exprs = [expr] if expr else []

    # 【SA v4.0 修正】：空算式要有自己的錯誤訊息。
    # 實測「45支鉛筆28塊橡皮擦分5位同學」那題，抽取結果是空字串，
    # 但錯誤訊息卻是「算式含有非數學字元(抽取結果：'')」——
    # 這句話對模型完全沒有指導性，重試時它不知道要改什麼，於是連續三次都吐空字串。
    if not sub_exprs:
        print(f"[算盤法師] ⚠️ 模型沒有產出任何算式（原始輸出：{raw_expr[:60]!r}）")
        return {
            "messages": [AIMessage(name="Math_Agent", content=(
                f"{STATUS_FAIL} 無法產生有效算式：你這次沒有輸出任何算式（結果是空的）。"
                f"請務必寫出實際的數學算式，例如 45 // 5 ; 45 % 5 ; 28 // 5 ; 28 % 5，"
                f"不要只回覆文字說明或空白。"
            ))],
            "plan": plan, "facts": facts,
            "retry_count": retry_count, "math_calls": math_calls
        }

    # ---- 檢查 1：語法合法性(允許 comb/perm/factorial) ----
    # 【SA v3.4 修正】：白名單補上 % 與 //。
    # 上一版才剛在提示裡教模型「求餘數請用 N % M」，
    # 但這裡的白名單 [\d\.\+\-\*/\(\),\s] 根本沒有 % ——
    # 等於一邊叫它用，一邊把用了的擋掉。
    def _syntax_of(e):
        s = re.sub(r'\b(comb|perm|factorial)\b', '', e)
        return bool(e) and bool(re.fullmatch(r'[\d\.\+\-\*/%\(\),\s]+', s))

    syntax_ok = all(_syntax_of(e) for e in sub_exprs)

    # ---- 【SA v4.5 新增】檢查 1.5：禁止把兩種不同東西的商/餘數直接加在一起 ----
    #
    # 實測翻車（大魚小魚分配題）：
    #   「大魚70條、小魚30條，平分給7個人，每個人可以分到幾條大魚、幾條小魚？」
    #   模型吐出 ((70 // 7) + (30 // 7))，把「大魚每人幾條」和「小魚每人幾條」
    #   這兩個題目明確要求【分開回答】的量加總成一個數字 14。
    #   語法檢查、運算檢查、數字溯源、數字完整性，四道既有檢查全部通過
    #   （因為 70、30、7 都確實來自題目，算式也真的有在運算）——
    #   問題不在「數字對不對」，是在「該分開的量被錯誤合併」。
    #
    # 這題本來就有規則 6 的提示詞範例（蘋果／檸檬分開算），
    # 但換一種措辭（大魚／小魚、用「幾條 X、幾條 Y」的句型）小模型就沒有類推過去。
    # 這正是這個專案一路以來的教訓：教規則不保證每次都被遵守，
    # 只要有辦法寫成程式碼可以驗證的「不變條件」，就不要只依賴提示詞。
    #
    # 這裡的不變條件很具體：一條算式的【最外層】如果是「兩個取整除或取餘數的結果相加」，
    # 幾乎不可能是正確答案 —— 分配題要的是「各自」的商或餘數，不是把商數加總。
    # 用 AST 直接檢查算式的語法樹結構，比在提示詞裡多舉一個例子更可靠。
    def _is_merged_quotients(e: str) -> bool:
        try:
            node = ast.parse(e, mode="eval").body
        except Exception:
            return False

        def _is_quotient_or_remainder(n):
            return isinstance(n, ast.BinOp) and isinstance(n.op, (ast.FloorDiv, ast.Mod))

        return (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and _is_quotient_or_remainder(node.left)
            and _is_quotient_or_remainder(node.right)
        )

    merged_quantities_ok = not any(_is_merged_quotients(e) for e in sub_exprs)

    # ---- 檢查 2：語意有效性 —— 必須真的在「算」東西 ----
    has_operation = all(
        bool(re.search(r'[\+\-\*/%]', e)) or bool(re.search(r'\b(comb|perm|factorial)\s*\(', e))
        for e in sub_exprs
    )

    # ---- 【SA v3.3 新增】檢查 2.5：封鎖組合函式濫用 ----
    #
    # 光在提示裡寫「不准濫用 comb」是不夠的 —— 小模型不會照做。
    # 這裡用程式碼硬擋：只有題目真的在問組合／排列，才准用 comb / perm / factorial。
    #
    # 為什麼這條這麼重要？因為 comb 濫用是【所有計算錯誤的共同元凶】：
    #   「糖果25顆分給7個人」  → comb(25, 7) - (7 * 7)      = 480651
    #   「10 / 6」            → comb(6, 1) / comb(10, 1)    = 0.6
    #   「大魚70條分7人」      → comb(140, 1) / comb(1, 1)   = 140
    #   「蘋果60檸檬40分9人」  → comb(10,2)*comb(5,1)/... 長到把 log 洗爆
    # 這些算式全部通過了語法檢查與數字溯源（數字確實來自題目），
    # 所以前面兩道檢查完全攔不住，只能在這裡單獨處理。
    uses_combinatorics = bool(re.search(r'\b(comb|perm|factorial)\s*\(', expr))
    question_wants_combinatorics = bool(re.search(
        r'(組合|排列|幾種|選法|挑選|抽出|抽取|階乘|不重複|順序)', task_desc))
    combinatorics_ok = (not uses_combinatorics) or question_wants_combinatorics

    # ---- 檢查 3：數字溯源 —— 算式裡的數字必須有來源 ----
    # 【SA v2.2 重要修正】：v2.1 在這裡開了一個大洞 ——
    # 只要算式用到 comb/perm/factorial，就整個跳過溯源檢查。
    # 實測後果：小模型吐出 comb(10,3)*comb(7,2)-factorial(5)/comb(4,1)，
    # 裡面 10/3/7/2/5/4 全部是憑空捏造的，帳本和題目裡根本沒有這些數字，
    # 卻因為「有 comb(」而暢行無阻，還被鑑定士蓋章合格。
    #
    # 正解：不要放行整個算式，而是把「合法數字的來源」擴大。
    # 真正的排列組合題，數字本來就寫在題目文字裡(例如「從10個人選3個」)，
    # 所以把「任務描述裡的數字」也納入合法來源，就能既不誤殺、又堵住捏造。
    allowed_numbers = set()
    for v in facts.values():
        allowed_numbers |= _numbers_in(str(v))
    allowed_numbers |= _numbers_in(task_desc)

    provenance_ok = True
    if syntax_ok and has_operation:
        expr_numbers = _numbers_in(expr)
        # 【SA v2.2 補洞】：這裡刻意【不】加上 `if allowed_numbers` 的守衛。
        # 第一版寫成 `if allowed_numbers and ...`，結果在「帳本和題目都沒有任何數字」時，
        # 整道檢查會靜靜地被跳過 —— 而那正是最該擋下的情況：
        # 算式裡有一堆數字、可用來源卻是零，代表這些數字 100% 是模型自己生出來的。
        # 現在改成：只要算式有數字、卻沒有任何一個對得上來源，一律判定為捏造。
        if expr_numbers and not (expr_numbers & allowed_numbers):
            provenance_ok = False

    # ---- 【SA v3.5 新增】檢查 4：完整性 —— 題目提到的數字都該被用到 ----
    #
    # 這條是「不變條件」而不是「公式」，這個區別很重要。
    #
    # 走錯的路：一題一題教它「分配題怎麼算、比例題怎麼算、追及問題怎麼算」——
    #          中文應用題的型態無窮無盡，這條路跟先前的「數學動詞白名單」一樣追不完。
    # 走對的路：不管什麼題型，正確的答案都必須滿足某些性質。
    #          其中最好用的一條就是：題目給你的數字，你不該無視它。
    #
    # 實測抓到的漏算：
    #   題目「有60顆蘋果、40顆檸檬，要分給9個人，每人各拿多少？又剩多少？」
    #   模型只吐出 60 // 9 ; 60 % 9 —— 蘋果算完了，檸檬的 40 從頭到尾沒被碰過，
    #   結果回覆變成「每個人可以各拿 6 個蘋果和檸檬」，把兩種水果混為一談。
    #
    # 只看題目本身的數字（不看帳本），因為帳本裡的數字可能是前面步驟的中間結果，
    # 不一定每個都要在這一步用到；但題目寫出來的數字，就是使用者要你處理的東西。
    completeness_ok = True
    unused_numbers = set()
    if syntax_ok and has_operation and step is not None:
        # 【SA v3.5 防誤判】：要區分「這個數字是待計算的數量」還是「專有名詞的一部分」。
        #
        # 誤判案例一：任務「東京晴空塔高度 減去 台北101高度」——
        #   101 是建築物名稱，不是數量，卻被判定成「漏算了 101」而擋下正確的 634 - 508。
        # 誤判案例二（第一版修法過頭）：把「中文字+數字」整段刪掉，
        #   結果「有大魚70條 小魚30條」的 70 和 30 也被一起吃掉，反而漏掉真正的漏算。
        #
        # 關鍵差異在【數字後面】：真正的數量後面會跟著量詞或單位（70條、60顆、9個、25片），
        # 而專有名詞裡的數字後面通常直接接別的字或結束（台北101高度、iPhone15）。
        # 所以改成正面表列：只認「數字 + 量詞」這種樣式。
        question_numbers = set(re.findall(r'(\d+(?:\.\d+)?)\s*(?=[顆個條片張份人位隻本台把杯瓶包盒袋箱元塊克公斤公克公尺公里秒分時天週月年%％]|$|[\s,，、。？?!！])', task_desc))
        expr_numbers = _numbers_in(expr)
        # 排除 0 和 1 這種常見的中性常數，避免誤判
        question_numbers -= {"0", "1"}
        unused_numbers = question_numbers - expr_numbers
        if question_numbers and unused_numbers:
            completeness_ok = False

    if not syntax_ok or not has_operation or not provenance_ok or not combinatorics_ok or not completeness_ok or not merged_quantities_ok:
        if not syntax_ok:
            reason = f"算式含有非數學字元(抽取結果：{expr!r})"
        elif not has_operation:
            reason = f"算式沒有任何運算，只是一個孤立的數字(抽取結果：{expr!r})，這代表它沒有真的在計算"
        elif not merged_quantities_ok:
            reason = (
                f"算式把兩種不同東西的商或餘數直接加在一起了(抽取結果：{expr[:80]!r})。"
                f"題目問的是「幾條/幾個 A、幾條/幾個 B」這種要【分開回答】的量，"
                f"不可以用 (a // n) + (b // n) 這種方式合併成一個數字。"
                f"請把每一種東西的商與餘數都拆成獨立的算式，用分號 ; 隔開，例如："
                f"70 // 7 ; 70 % 7 ; 30 // 7 ; 30 % 7。"
            )
        elif not combinatorics_ok:
            reason = (f"題目並不是在問組合或排列，卻使用了 comb/perm/factorial"
                      f"(抽取結果：{expr[:80]!r})。分配、平分、相差這類題目請用 + - * / // % 就好")
        elif not completeness_ok:
            # 【SA v3.5】：把漏掉的數字明確寫進失敗訊息，
            # 下一次重試時 math_node 會讀到這行，直接告訴模型「你漏了什麼」，
            # 比讓它盲目重猜有效得多。
            miss = "、".join(sorted(unused_numbers))
            reason = (f"算漏了：題目裡的數字 {miss} 在算式中完全沒有被使用"
                      f"(抽取結果：{expr[:80]!r})。題目給的每個數字都要處理到")
        else:
            reason = f"算式中的數字({expr!r})既不在事實帳本、也不在題目文字裡，疑似模型憑記憶編造"
        print(f"[算盤法師] ⚠️ {reason}")
        return {
            "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_FAIL} 無法產生有效算式：{reason}")],
            "plan": plan, "facts": facts,
            "retry_count": retry_count, "math_calls": math_calls
        }

    # 【SA v3.4】：逐條計算。單條時行為與舊版完全相同；
    # 多條時每條各自送進計算機，結果一起回報，供盜賊客服組成完整答案。
    from tools.calculator import calculate_math
    results = []
    for e in sub_exprs:
        print(f"[算盤法師 Math_Agent] 正在使用魔法計算機: {e}")
        r = calculate_math(e)
        if r.startswith("計算失敗"):
            return {
                "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_FAIL} {r}")],
                "plan": plan, "facts": facts,
                "retry_count": retry_count, "math_calls": math_calls
            }
        results.append((e, _extract_calc_value(r)))

    # 組成人類看得懂的結果字串
    # 【SA v3.6 修正】：不要把原始算式直接寫進帳本。
    # 實測畫面：使用者看到的回覆是「剩下的蘋果有 60 % 9 = 6 個」——
    # 對一般人來說 % 讀起來是「百分之」，整句話變得莫名其妙。
    # 帳本內容會被盜賊客服直接引用，所以在這裡就先翻成中文說法。
    def _humanize(e: str) -> str:
        h = e
        h = re.sub(r'(\d+(?:\.\d+)?)\s*//\s*(\d+(?:\.\d+)?)', r'\1 除以 \2 取整數', h)
        h = re.sub(r'(\d+(?:\.\d+)?)\s*%\s*(\d+(?:\.\d+)?)', r'\1 除以 \2 的餘數', h)
        return h

    if len(results) == 1:
        result_text = results[0][1]
    else:
        result_text = "、".join(f"{_humanize(e)} = {v}" for e, v in results)

    if step is not None:
        step["status"] = "done"
        step["result"] = result_text
        facts[step["target"]] = result_text
    else:
        facts[f"計算：{expr}"] = result_text

    print(f"[算盤法師 Math_Agent] 📒 已登錄：{task_desc} ＝ {result_text}")

    return {
        "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_OK}\n【計算機結果】\n{result_text}")],
        "plan": plan,
        "facts": facts,
        "retry_count": retry_count,
        "math_calls": math_calls
    }


# 【SA v2】：Math_Agent 出口的 Local Grader (f-c)
def math_grader(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    is_bad = _is_failed_message(last_msg)
    retry = state.get("retry_count", 0)

    if is_bad and retry < MAX_RETRY:
        print(f"[算盤鑑定士 f-Math] ❌ 計算結果不合格，退回重算 (第 {retry} 次重試)")
        return "retry"
    if is_bad:
        print("[算盤鑑定士 f-Math] 🛑 已達重試上限，本項任務放棄，交回主管召喚師繼續下一項")
        return "pass"
    print("[算盤鑑定士 f-Math] ✅ 計算結果合格，放行給主管召喚師")
    return "pass"



def _clean_internal_terms(raw_text: str) -> str:
    """
    【SA v3.0】把模型回覆裡洩漏的內部骨架清乾淨。

    抽成獨立函式的理由：極簡版與完整版兩條路徑都要用同一套清洗規則，
    寫在節點裡就得複製兩份，日後改一邊忘另一邊必定出事。

    這裡擋兩類東西：
      1. 出處框架：「根據公司知識庫的標準解答」「我們知道」「答案是」…
         模型會把提示裡的區塊標題當成可引用的出處名稱照抄。
      2. 區塊標題本身：「（本輪任務完成情況：…）」「（本輪調查過程紀錄…）」
         這是實測看到最誇張的一種 —— 模型直接把我的提示骨架印給面試官看。
    """
    clean_text = (raw_text or "").strip()
    clean_text = re.sub(r'^assistant[:\s\n]*', '', clean_text, flags=re.IGNORECASE).strip()

    patterns = [
        # --- 區塊標題整段外洩（含括號包起來的形式）---
        # 【SA v3.0 注意】：原本用 [^）)]* 會被內層括號提早截斷 ——
        # 「（本輪調查過程紀錄(佐證用)：無）」裡面的 (佐證用) 有自己的右括號，
        # 結果只清掉前半段、留下「：無）」這種殘骸。
        # 改成允許吃掉內層括號，直到遇到行尾或全形右括號為止。
        r'[（(]\s*本輪(任務完成情況|調查過程紀錄|調查紀錄)[\s\S]*?(?:）|\)\s*$|$)\s*',
        r'[（(]\s*(佐證用|僅供參考)[^）)]*[）)]\s*',
        r'^【?(本輪任務完成情況|本輪調查過程紀錄|上一輪聊了什麼|本次查證到的資料|這一題的參考答案)】?[：:].*$',
        # --- 出處框架 ---
        r'根據(公司)?知識庫的?(標準)?(解答|答案|內容|資料)[，,、：:]?\s*',
        r'根據(事實帳本|查證資料|調查紀錄|工作紀錄|參考答案)(的[\u4e00-\u9fff]{0,4})?[，,、]?\s*',
        r'(事實帳本|查證資料|本輪調查紀錄|內部紀錄)(中|裡|裡面)?[的]?',
        r'根據\s*(Math_Agent|Search_Agent|Final_Answer|Supervisor|Planner)\s*(的[\u4e00-\u9fff]{0,4})?[，,、]?\s*',
        r'我們知道[，,、：:]?\s*',
        r'^(因此[，,]?\s*)?答案(是|為)[：:]?\s*',
        # --- 罐頭開場白 ---
        r'^我可以回答你的問題了[。.]?\s*',
        r'^你想知道我?會?哪些[\u4e00-\u9fff]{0,6}嗎[？?]\s*',
        # --- 工具原始輸出 ---
        r'計算成功！?\s*',
        r'算式\s*[\'"][^\'"]*[\'"]\s*的結果為[：:]\s*',
    ]

    lines = clean_text.split("\n")
    cleaned = []
    for line in lines:
        for pat in patterns:
            line = re.sub(pat, '', line, flags=re.MULTILINE)
        cleaned.append(line)
    clean_text = "\n".join(cleaned)

    clean_text = re.sub(r'^[，,、。：:\s]+', '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    # 【SA v3.3 新增】：複讀偵測 —— 最後一道防線。
    #
    # 實測災情：「談談你的 DevOps 與 CI/CD 經驗」那題，模型把同一句
    # 「他熟練掌握 Docker 微服務容器化與 Image 打包…」連續輸出了幾十遍，
    # 整個聊天視窗被洗版。就算上游已經做了素材去重，
    # 小模型仍有機率自己陷進迴圈，所以這裡再擋一次。
    #
    # 作法：依句子切開，看到內容重複出現過的句子就停在那裡（保留前面正常的部分）。
    sentences = re.split(r'(?<=[。！？!?\n])', clean_text)
    kept, seen = [], set()
    for s in sentences:
        key = re.sub(r'\s+', '', s)
        if len(key) < 12:          # 太短的句子（例如「好的。」）不參與去重
            kept.append(s)
            continue
        if key in seen:
            print("[盜賊客服 Final_Answer] 🔁 偵測到複讀，已截斷重複內容。")
            break
        seen.add(key)
        kept.append(s)
    clean_text = "".join(kept)

    return clean_text.strip()


def _answer_numbers_traceable(answer: str, facts: dict, question: str, search_notes=None):
    """
    【SA v4.1 新增，v4.3 擴充】輸出層的數字溯源檢查 —— 第五道不變條件。

    為什麼需要這道？因為前四道不變條件全部作用在【算盤法師】身上，
    盜賊客服在寫最終回覆時，完全沒有程式碼在監督它有沒有亂編數字，
    只靠提示詞裡的鐵則 3 —— 而我們早就知道提示是請求、不是保證。

    實測災情（寶石題）：
      計算機算出 (32+25)//7 = 8、(32+25)%7 = 1，兩個數字都正確；
      但盜賊客服看到題目提到「紅寶石」「藍寶石」兩種，就自作主張分開算，
      回覆變成「每人 8 顆紅寶石和 3.57 顆藍寶石，剩下紅寶石 4 個、藍寶石 2 個」。
      3.57、4、2 這三個數字計算機從來沒算過 —— 工具全對，最後一棒把答案編掉。

    這是最危險的錯誤類型：log 全綠、鑑定士放行，但使用者拿到假答案。

    【SA v4.3】：search_notes 是開放式資訊查詢(新聞…)保留的原始搜尋摘要，
    這些文字裡本來就會出現大量日期、金額等數字，是合法引用來源，
    不該被當成「編造」，所以也要納入 allowed 集合，否則新聞摘要題
    會被這道檢查誤判、一路退到確定性模板，重演本次翻車。

    回傳 (是否全部可溯源, 無法溯源的數字集合)
    """
    allowed = set()
    for k, v in facts.items():
        allowed |= _numbers_in(str(k))
        allowed |= _numbers_in(str(v))
    for note in (search_notes or []):
        allowed |= _numbers_in(str(note))
    allowed |= _numbers_in(question)
    # 【SA v4.1】：把題目數字之間的「常見中間結果」也算成合法來源。
    # 例如寶石題「32 顆 + 25 顆混在一起」，回覆講「共 57 顆」是完全正確的推理，
    # 但 57 不在帳本裡（帳本存的是 (32+25)//7 = 8），少了這一步就會誤擋正確答案。
    # 只補加總與差值這兩種最常見的中間量，不做更複雜的推導。
    q_nums = [n for n in _numbers_in(question)]
    for i, a in enumerate(q_nums):
        for b in q_nums[i + 1:]:
            try:
                fa, fb = float(a), float(b)
                for mid in (fa + fb, abs(fa - fb)):
                    allowed.add(str(int(mid)) if mid == int(mid) else str(mid))
            except ValueError:
                continue
    # 常見的中性數字（序號、一半、百分比基準…）不列入追查，避免誤判
    allowed |= {"0", "1", "2", "3", "10", "100"}

    answer_numbers = _numbers_in(answer)
    unknown = answer_numbers - allowed
    return (not unknown), unknown


def _render_allowed_numbers(facts: dict, question: str, search_notes=None) -> str:
    """把「這一題可以使用的數字」整理成一行，供重寫時明確告知模型。"""
    allowed = set()
    for k, v in facts.items():
        allowed |= _numbers_in(str(v))
    for note in (search_notes or []):
        allowed |= _numbers_in(str(note))
    allowed |= _numbers_in(question)
    return "、".join(sorted(allowed, key=lambda x: (len(x), x))) or "（沒有任何可用數字）"


# 【4. 盜賊客服房間】
def final_answer_node(state: AgentState):
    print("\n[盜賊客服 Final_Answer] 資料收集完畢，正在撰寫最終回覆給客人...")

    msgs = list(state["messages"])
    plan = state.get("plan", []) or []
    facts = dict(state.get("facts", {}) or {})
    rag_context = state.get("rag_context", "") or ""
    rag_hit_type = state.get("rag_hit_type", "none")
    search_notes = list(state.get("search_notes", []) or [])

    current_question = ""
    for msg in msgs:
        if isinstance(msg, HumanMessage):
            current_question = msg.content

    # 【SA v2】：Python 先算好「有沒有東西沒查到」，直接把結論寫死在提示裡，
    # 不讓主模型自己判斷「資料夠不夠」—— 它每次都會覺得夠。
    failed_steps = [s for s in plan if s["status"] != "done"]
    all_done = len(failed_steps) == 0
    if failed_steps:
        gap_note = (
            "⚠️ 以下項目【沒有查到/算出結果】，你在回答中必須誠實告知使用者這些部分查詢失敗：\n"
            + "\n".join(f"- {s['target']}" for s in failed_steps)
        )
    else:
        gap_note = "所有預定的查詢與計算項目都已完成。"

    # 【SA v2.6 重大修正】：判斷「計算到底做完了沒」，改看任務清單的狀態旗標。
    #
    # v2.5 是用猜的：
    #     has_calc_result = any("計算" in k or re.search(r'[\+\-\*/]', k) for k in facts.keys())
    # 實測翻車現場：算盤法師明明算出 126 並登錄進帳本，鍵是
    #     「東京晴空塔高度 減去 台北101高度」
    # 中文「減去」不含 ASCII 運算符號，也沒有「計算」二字，於是被判定成沒算，
    # 提示最前面就被塞進「🚫 禁止自己算，請說計算步驟未能完成」——
    # 模型很聽話地照做，把一個已經算對的答案硬生生丟掉。
    #
    # 這是本專案第二次因為「掃描字串猜狀態」而出事（第一次是品管員掃「錯誤」二字）。
    # plan 裡本來就有 status 這個明確旗標，直接看它就好，不要再猜。
    done_math_steps = [s for s in plan if s.get("type") == "math" and s.get("status") == "done"]
    planned_math_steps = [s for s in plan if s.get("type") == "math"]

    if done_math_steps:
        has_calc_result = True          # 有 math 任務且已完成 → 一定算過了
    elif planned_math_steps:
        has_calc_result = False         # 排了 math 任務但沒完成 → 確實沒算成功
    else:
        # 沒有任務清單（保底模式）才退回看帳本內容，並且把中文運算詞也算進去
        has_calc_result = any(
            re.search(r'(計算|加|減|乘|除|相差|差值|倍|總和|合計|平均|[\+\-\*/])', k)
            for k in facts.keys()
        )

    question_wants_math = bool(re.search(_MATH_HINT, current_question))
    math_block_note = ""
    if question_wants_math and not has_calc_result:
        math_block_note = (
            "\n\n🚫【本輪最高優先警告】：使用者的問題要求做運算，"
            "但計算步驟【沒有完成】，可用資料裡沒有任何計算結果。\n"
            "你【絕對禁止】自己做任何加減乘除然後把答案寫出來。"
            "請如實列出已經查到的各項數值，並明確告訴使用者『計算步驟未能完成』。\n"
        )

    # 【SA v2.6 新增】：控制送進模型的總長度。
    # 你的 Modelfile 設定 PARAMETER num_ctx 4096，這是整個上下文的硬上限（含輸出）。
    # 而這段提示要塞：RAG 內容 + 查證資料 + 任務狀況 + 調查紀錄 + 歷史對話 + 十條鐵則，
    # 中文一個字大約就是一個 token，很容易在不知不覺中超過 4096，
    # 超過的部分會被【靜默截斷】—— 通常先被吃掉的就是最前面的身分設定與鐵則。
    # 所以這裡主動把佐證用的內容壓短，把預算留給真正要引用的知識庫與查證資料。
    scratch_str = ""
    for msg in msgs:
        if getattr(msg, "name", "") in ["Search_Agent", "Math_Agent"]:
            scratch_str += f"[{msg.name}]: {str(msg.content)[:280]}\n\n"
    if len(scratch_str) > 1200:
        scratch_str = scratch_str[:1200] + "…（後略）"
    if not scratch_str:
        scratch_str = "（本輪沒有動用搜尋或計算工具）"

    # 【SA v2.9 重大修正】：歷史對話污染是「每題開頭都在自我介紹」的元凶。
    #
    # 病灶：第一題問自我介紹，盜賊客服回了一段自我介紹，這段被存進 chat_history。
    # 之後每一題，這段自我介紹都會被當成「歷史」塞進提示裡。
    # XUYA 是 8B 小模型，分不清「歷史僅供語意連貫」和「要照著回」，
    # 就把整段自我介紹又抄一遍當開頭 —— 而且越積越多。
    # 更慘的是 101 那題：前面硬抄的自我介紹把 num_ctx 4096 塞爆，
    # 真正的搜尋結果與計算數字被擠出上下文，於是它說「資訊不足」。
    #
    # 修法有三層：
    #   1. 只保留【最近一輪】對話（不是全部），大幅減少被抄的素材
    #   2. 每則截到 60 字（連貫語意只需要知道剛剛聊到什麼，不需要全文）
    #   3. 提示裡把歷史的定位講得更死：只用來判斷「這題是不是延續前一題」，
    #      不是拿來抄的內容
    history_str = ""
    hist = list(state.get("chat_history", []))
    if hist:
        # 只取最後一輪（一問一答 = 最多 2 則），這是連貫語意需要的最小量
        recent = hist[-2:]
        for msg in recent:
            role = "使用者" if isinstance(msg, HumanMessage) else "AI"
            snippet = str(msg.content).replace("\n", " ")[:60]
            history_str += f"{role}剛剛說：{snippet}…\n"
    else:
        history_str = "（這是本次對話的第一個問題）"

    # 【SA v2.1】：客服題(RAG 命中)必須讓 Final_Answer 看得到知識庫，否則它會兩手空空。
    # 但同時要標明可信度，避免它把 Fallback 撈到的不相關段落當成標準答案在講。
    # 【SA v2.7】：這個區塊標題原本叫「公司知識庫標準解答」，
    # 模型會把「根據公司知識庫的標準解答，我們知道…」整句照抄給面試官。
    # 改成中性、不像出處名稱的措辭，從源頭減少複述的誘因（搭配下方 Python 清洗雙保險）。
    if rag_hit_type == "manual":
        rag_block = f"【這一題的參考答案，請用自己的話自然講出來，不要照抄格式】：\n{rag_context}"
    elif rag_hit_type == "auto" and rag_context.strip():
        rag_block = (
            "【幾段可能相關的參考資料 ── 相關性不保證，只有在確實對應到問題時才引用】：\n"
            + rag_context[:1500]
        )
    else:
        rag_block = "【參考資料】：本次未檢索到相關內容。"

    # ==========================================
    # 【SA v3.0 重大重構】：提示詞分成「極簡版」與「完整版」兩套
    # ==========================================
    # 為什麼要拆？因為提示詞肥大到把模型壓垮了。
    #
    # 這幾版我一路往上疊：鐵則 0~10、最高優先計算警告、五個資料區塊
    # （參考答案 / 查證資料 / 任務完成情況 / 調查紀錄 / 上一輪聊了什麼）。
    # 對一顆 8B、num_ctx 只有 4096 的模型，這已經遠遠超載。
    #
    # 實測崩潰現場（真實 log）：
    #   1. 模型把區塊標題原封不動印給面試官：
    #        「（本輪任務完成情況：所有預定的查詢與計算項目都已完成）」
    #        「（本輪調查過程紀錄(佐證用)：無）」
    #   2. 弓箭手明明撈對了（兵役 48 字、柬埔寨 76 字，主題票數 5/5、4/5），
    #      模型卻完全略過那份資料，直接把上一題的答案抄一遍。
    #
    # 原因不是模型笨，是它分不清「哪些是要照做的指令」「哪些是要引用的內容」
    # 「哪些是骨架不該講出來」—— 區塊一多就全糊在一起。
    #
    # 解法：依情境給不同複雜度的提示。
    #   90% 的題目是「RAG 命中 → 照著標準答案講」，這種只需要極簡提示，
    #   完全不需要計算警告、任務清單、調查紀錄那一整套。
    #   只有真的動用了搜尋／計算工具時，才需要完整版。
    used_tools = bool(plan) or bool(facts) or bool(search_notes)

    if rag_hit_type == "manual" and not used_tools:
        # ---------- 極簡版：知識庫命中，照著講就好 ----------
        # 【設計原則】：段落越少、指令越短，小模型越不會亂抄。
        # 這裡刻意【不】給上一題的答案，只給上一題的「問題」——
        # 模型就沒有可抄的答案文本，但仍能理解「裡面」「那個」指的是什麼。
        prev_q = ""
        hist = list(state.get("chat_history", []))
        for m in reversed(hist):
            if isinstance(m, HumanMessage):
                prev_q = str(m.content)[:50]
                break

        context_line = f"（面試官上一題問的是：{prev_q}）\n" if prev_q else ""

        sys_msg = SystemMessage(content=(
            "你是張序亞（Steven）的面試 AI 助理，正在回答面試官的提問。\n\n"
            f"{context_line}"
            f"面試官這一題問：{current_question}\n\n"
            f"這一題的參考答案：\n{rag_context}\n\n"
            "請用自然、專業的口吻，把上面的參考答案講給面試官聽。\n"
            "1. 【最重要】只能講參考答案裡有的內容。"
            "【嚴禁】補充任何參考答案裡沒有的技術名詞、程式語言、專案或經歷 ——"
            "你記憶中的東西一律不算數，寧可少講也不要編。"
            "如果參考答案沒有回答到面試官問的點，就誠實說這部分建議面試時直接跟序亞聊。\n"
            "2. 主語一律用「他」或「序亞」，不要用「我們」；"
            "只有介紹「你自己是誰」時才用「我」。\n"
            "3. 直接講內容，不要有「我可以回答你了」「你想知道…嗎」這種開場白，"
            "也不要加括號註記或說明你的思考過程。\n"
            "4. 只回答這一題，不要重複前面聊過的內容。"
        ))
        response = invoke_with_timeout(main_llm, [sys_msg])
        clean_text = _clean_internal_terms(response.content)
        print(f"[盜賊客服 Final_Answer] 📝 採用極簡提示（知識庫命中，未動用工具）")
        return {
            "messages": [AIMessage(name="Final_Answer", content=clean_text)],
            "all_steps_done": True
        }

    # ---------- 完整版：有動用搜尋／計算工具時才用 ----------
    # 【SA v4.3 新增】開放式搜尋摘要區塊 —— 跟 facts(單一數值帳本) 分開顯示，
    # 避免模型把「新聞摘要」誤認成「查證數字」硬套進鐵則 4 的計算邏輯。
    if search_notes:
        search_notes_block = (
            "【本次搜尋到的原始資料 ── 這是一般性資訊(例如新聞)，"
            "請你自己統整成一段自然的白話摘要回答使用者，"
            "只能講這份資料裡確實提到的內容，不要延伸、不要補充資料外的具體數字、"
            "人名、事件】：\n" + "\n\n".join(search_notes)[:1800]
        )
    else:
        search_notes_block = "【本次搜尋到的原始資料】：（本輪沒有這類開放式搜尋資料）"

    sys_msg = SystemMessage(content=(
        # 【SA v2.6 重大修正】：這裡原本寫「你是一位專業的 AI 助理。」
        #
        # 問題在於：透過 API 傳送 system role 訊息時，會【覆蓋掉 Modelfile 裡的 SYSTEM 設定】。
        # 使用者在 Modelfile 已經寫好完整的面試助理人設（代表張序亞、如何應對面試官、
        # 知識盲區怎麼回答…），但每一次呼叫都被上面那一行洗掉，
        # 模型只好照念「你好，我是一位專業的 AI 助理」——
        # 即使 RAG 已經正確撈到標準答案（實測距離 0.130），它也視而不見。
        #
        # 現在把身分設定寫回來，並且明確要求「知識庫有標準答案時就以它為準」。
        "你是一位專屬的面試 AI 助理，代表軟體工程師張序亞（Steven）。\n"
        "你的職責是專業、自信且友善地向面試官介紹序亞的技術能力、專案經驗與人格特質。\n"
        "回答時保持工程師的務實與客觀，不要浮誇，也不要自行捏造任何經歷。\n"
        + math_block_note + "\n"
        f"【目前使用者的問題】：\n{current_question}\n\n"
        f"{rag_block}\n\n"
        f"【本次查證到的資料 ── 這是你唯一可以引用的『外部查詢數字』來源】：\n{_render_facts(facts)}\n\n"
        f"{search_notes_block}\n\n"
        f"【本輪任務完成情況】：\n{gap_note}\n\n"
        f"【本輪調查過程紀錄(佐證用)】：\n{scratch_str}\n\n"
        f"【上一輪聊了什麼，只用來判斷本題是否延續前文，不是要你複述】：\n{history_str}\n\n"
        "請嚴格遵守：\n"
        "【鐵則 0 ── 參考答案優先】：如果上方出現【這一題的參考答案】，"
        "那就是這一題的正確內容，請用你自己的話自然地講出來，"
        "可以潤飾語氣但不要改變事實。"
        "【嚴禁】把「這一題的參考答案」「根據知識庫」「我們知道」「答案是」"
        "這類框架文字照抄進回覆——直接講內容就好，就像你本來就知道一樣。\n"
        "【鐵則 1 ── 只答當前問題】：只回答『目前使用者的問題』這一題。"
        "【嚴禁】把上一輪的自我介紹、上一題的答案，重複抄到這一題的開頭。"
        "每一題都是獨立回答，不要用「你好，我是張序亞…」當每一題的開場白，"
        "那段話只在使用者【第一次】問你是誰、或請你自我介紹時才需要講。\n"
        "【鐵則 2】：只有當本題明顯是延續上一題時（例如上一題問專案、這題問『那個專案多久』），"
        "才需要參考上一輪聊了什麼；否則完全忽略歷史，直接回答當前問題。\n"
        "【鐵則 3】：你只能使用【公司知識庫】【本次查證到的資料】或【本次搜尋到的原始資料】裡"
        "明確出現的數字與名稱作答，"
        "絕對不准使用你自己記憶中的人名、年份、數字！查無資料就誠實說查詢失敗，絕不編造。\n"
        "【鐵則 4 ── 最重要，絕無例外】：你【完全不會算數】。"
        "如果使用者問的是差值、總和、比例、倍數，而【查證資料裡沒有現成的計算結果】，"
        "你【絕對禁止】自己在心裡做任何加減乘除然後把答案寫出來，"
        "只能誠實說明『目前只查到 A 和 B 的數值，計算步驟未能完成』。"
        "但反過來說，只要資料裡【已經有】計算結果，就直接把那個結果講出來，不要再說沒算完。\n"
        "【鐵則 5】：直接、自然地把結論講出來就好。"
        "【嚴禁】出現任何內部字眼，包括但不限於「事實帳本」「查證資料」「調查紀錄」"
        "「根據 Math_Agent」「根據 Search_Agent」「計算成功」「算式」等等。"
        "使用者是來問問題的，不需要知道系統內部長什麼樣子。\n"
        "【鐵則 6】：如果問題比較兩個以上對象，必須分別給出每個對象的明確數字，再說明差異。\n"
        "【鐵則 7 ── 人稱一致】：談到張序亞本人的經歷時一律用「他」或「序亞」，"
        "不要一下說「我有相關經驗」一下又說「他以 Docker…」。"
        "只有介紹「你自己是誰」的時候才用「我」。\n"
        "【鐵則 8 ── 不要開場白】：直接回答問題，"
        "不要用「你好，我是一位專業的 AI 助理」「根據你的問題，我查到了相關資料」"
        "這類罐頭開場白浪費對方的時間。\n"
        "【鐵則 9 ── 時效性資料要標註】：如果答案是股價、匯率、天氣這類會隨時間變動的數字，"
        "請說明這是網路搜尋到的結果，並提醒使用者以官方即時資料為準。\n"
        "【鐵則 10 ── 知識盲區】：如果問題問到的細節不在上方任何資料裡，"
        "請誠實說明這部分建議直接在面試中與序亞深入討論，絕對不要自行編造經歷。"
    ))

    response = invoke_with_timeout(main_llm, [sys_msg])

    clean_text = _clean_internal_terms(response.content)

    # 【SA v4.1 新增，v4.3 擴充】：輸出層數字溯源 —— 發現編造就給一次重寫機會。
    #
    # 只在「有動用工具」的路徑做這道檢查，因為這時候才有明確的數字來源可以比對。
    # 純知識問答（履歷題）不做，那類回答本來就會出現知識庫裡的各種年份與數量。
    # 【v4.3】：search_notes(開放式搜尋摘要，如新聞) 也納入檢查範圍 ——
    # 一樣要防止模型在整理摘要時，把摘要裡沒有的數字自己編出來。
    if facts or search_notes:
        ok, unknown = _answer_numbers_traceable(clean_text, facts, current_question, search_notes)
        if not ok:
            bad = "、".join(sorted(unknown))
            print(f"[盜賊客服 Final_Answer] 🚨 偵測到回覆中有無法溯源的數字：{bad}，要求重寫一次。")
            retry_msg = SystemMessage(content=(
                "你是張序亞（Steven）的面試 AI 助理。\n\n"
                f"面試官的問題：{current_question}\n\n"
                f"【這一題唯一可以使用的數字】：{_render_allowed_numbers(facts, current_question, search_notes)}\n\n"
                f"【已經算出來的結果】：\n{_render_facts(facts)}\n\n"
                + (f"【搜尋摘要原文，只能引用裡面出現過的內容】：\n" + "\n\n".join(search_notes)[:1800] + "\n\n" if search_notes else "")
                + f"⚠️ 你上一次的回覆出現了 {bad} 這些數字，"
                "它們【不在】上面的結果裡，是你自己編的。\n\n"
                "請重寫一次回覆。硬性規定：\n"
                "1. 只能使用上面列出的數字，一個都不准多。\n"
                "2. 【不要】自己把題目拆成好幾份分開計算 —— "
                "上面的結果已經是完整答案了，照著講就好。\n"
                "3. 如果上面的結果沒有涵蓋到問題的某個部分，"
                "就誠實說那部分沒有算出來，不要用猜的補上。\n"
                "4. 直接講結論，不要開場白，不要加括號註記。"
            ))
            response2 = invoke_with_timeout(main_llm, [retry_msg])
            clean_text2 = _clean_internal_terms(response2.content)
            ok2, unknown2 = _answer_numbers_traceable(clean_text2, facts, current_question, search_notes)
            if ok2:
                print("[盜賊客服 Final_Answer] ✅ 重寫後所有數字都可溯源。")
                clean_text = clean_text2
            elif facts:
                # 【SA v4.1】重寫還是編 → 不再讓模型自由發揮，直接用確定性模板輸出。
                # 寧可講得像機器人，也不要給面試官一個看起來很順但數字是假的答案。
                # 這個模板只適用於「有 facts(單一數值)」的情況，因為它就是把 facts 逐條印出來。
                print(f"[盜賊客服 Final_Answer] 🛑 重寫後仍有編造數字（{'、'.join(sorted(unknown2))}），改用確定性模板輸出。")
                clean_text = (
                    f"{current_question}\n\n"
                    "計算結果如下：\n"
                    + "\n".join(f"・{k} ＝ {v}" for k, v in facts.items())
                )
            else:
                # 【SA v4.3 新增】：純開放式搜尋(沒有 facts、只有 search_notes)的確定性後備方案。
                # facts 是空的，上面那個「計算結果如下」模板完全不適用(印出來會是空清單)，
                # 這也是這次「12.5個基」事故的直接肇因 —— 舊版沒有這個分支，
                # 空 facts 也硬套進同一個模板，於是使用者收到的訊息看起來像沒填完的表格。
                # 這裡改成：老實告訴使用者摘要沒能完全整理乾淨，附上搜尋到的原始重點，
                # 讓使用者自己判斷，而不是用一個編造或空洞的句子交差。
                print(f"[盜賊客服 Final_Answer] 🛑 重寫後仍有編造數字（{'、'.join(sorted(unknown2))}），且本題無 facts、改用原始搜尋摘要後備輸出。")
                first_note = search_notes[0] if search_notes else "（沒有可用的搜尋摘要）"
                clean_text = (
                    f"{current_question}\n\n"
                    "我整理摘要時抓不準確切數字，直接附上搜尋到的原始重點給你參考：\n\n"
                    + first_note[:600]
                )

    return {
        "messages": [AIMessage(name="Final_Answer", content=clean_text)],
        "all_steps_done": all_done
    }


# ==========================================
# 🗺️ 第五區：畫地圖與建立動線 (Graph Edges)
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("Planner", planner_node)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Search_Agent", search_node)
workflow.add_node("Math_Agent", math_node)
workflow.add_node("Final_Answer", final_answer_node)

# 【SA v2 動線】：開場先經過規劃官，產生任務清單後才交給主管召喚師執行
workflow.add_edge(START, "Planner")
workflow.add_edge("Planner", "Supervisor")


def routing_logic(state: AgentState):
    """總次數上限的最後一道物理煞車。"""
    next_node = state["next_node"]

    if next_node == "Search_Agent" and state.get("search_calls", 0) >= MAX_SEARCH_CALLS_PER_TURN:
        print(f"\n[系統守衛] 🛑 這一輪 Search_Agent 已達呼叫上限({MAX_SEARCH_CALLS_PER_TURN}次)，強制結案避免卡死！")
        return "Final_Answer"

    if next_node == "Math_Agent" and state.get("math_calls", 0) >= MAX_MATH_CALLS_PER_TURN:
        print(f"\n[系統守衛] 🛑 這一輪 Math_Agent 已達呼叫上限({MAX_MATH_CALLS_PER_TURN}次)，強制結案避免卡死！")
        return "Final_Answer"

    if next_node == "FINISH":
        return "Final_Answer"
    return next_node


workflow.add_conditional_edges(
    "Supervisor",
    routing_logic,
    {
        "Search_Agent": "Search_Agent",
        "Math_Agent": "Math_Agent",
        "Final_Answer": "Final_Answer"
    }
)

# 【SA v2】：兩個 Grader 都不再有 "give_up → 直接結案" 這條路。
# 失敗達上限就 pass 回主管，由主管標記 failed 後繼續跑下一項，
# 最後由 Final_Answer 誠實告知哪些沒查到 —— 比整題暴斃好太多。
workflow.add_conditional_edges(
    "Search_Agent",
    search_grader,
    {"retry": "Search_Agent", "pass": "Supervisor"}
)
workflow.add_conditional_edges(
    "Math_Agent",
    math_grader,
    {"retry": "Math_Agent", "pass": "Supervisor"}
)

workflow.add_edge("Final_Answer", END)

app_graph = workflow.compile()
print(f"[系統] 🗺️ 航空母艦地圖建立完成 (v2.1)！主模型={MAIN_MODEL_NAME} / 驗證模型={VERIFY_MODEL_NAME}")

# ==========================================
# 🎮 第六區：本機獨立測試區塊
# ==========================================
if __name__ == "__main__":
    print("\n========================================================")
    print("🚀 航空母艦戰鬥群 v2.1：雙實體比較題回歸測試")
    print("========================================================")

    def _blank_state(question, history=None):
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

    q1 = "請幫我分別查詢台北 101 與日本東京晴空塔的建築總高度（公尺），並計算晴空塔和台北 101 誰比誰高多少公尺？"
    print(f"👤 [使用者 Q1]: {q1}\n")

    a1_text = ""
    for output in app_graph.stream(_blank_state(q1), {"recursion_limit": 30}):
        for key, value in output.items():
            print(f"--- 經過房間: {key} ---")
            if key == "Final_Answer":
                a1_text = value["messages"][-1].content

    print(f"\n🤖 [A1 最終回答]:\n{a1_text}\n")

    q2 = "請上網查詢台灣高鐵最新的實收資本額大約是多少新台幣？接著幫我計算：如果每股面額 10 元，總共有多少股？"
    print(f"👤 [使用者 Q2]: {q2}\n")

    a2_text = ""
    hist = [HumanMessage(content=q1), AIMessage(content=a1_text)]
    for output in app_graph.stream(_blank_state(q2, hist), {"recursion_limit": 30}):
        for key, value in output.items():
            print(f"--- 經過房間: {key} ---")
            if key == "Final_Answer":
                a2_text = value["messages"][-1].content

    print("\n========================================================")
    print(f"🤖 [A2 最終回答]:\n{a2_text}")
    print("========================================================")
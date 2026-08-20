# -*- coding: utf-8 -*-
# graph_core.py
# ==========================================
# 航空母艦戰鬥群 (LangGraph) 核心定義檔  ── v2「任務清單 + 事實帳本」版
# ==========================================
# 【SA v2 改版總說明】：
#
# v1 的三個結構性病灶(101 那題連查三次的真正原因)：
#   病灶 A：Supervisor 每一步都「從零重新判斷」，它的 reasoning 裡雖然已經寫出
#           「台北 101 的高度為 508 公尺」，但 reasoning 只被 print 出來就丟掉，
#           沒有存進背包 → 下一步的 Supervisor 完全不知道這件事已經查到了，
#           於是又照規則 2 派給 Search_Agent，形成「知道答案卻一直重查」的鬼打牆。
#   病灶 B：沒有任何 Python 層級的「同一句關鍵字不准查第二次」硬防線。
#           _pick_unsearched_segment 只在查詢字串含有逗號/「與」時才會拆分，
#           「台北 101 建築總高度 公尺」這種沒有分隔符的字串會原封不動放行。
#   病灶 C：品管員(Grader)用「關鍵字掃描」判斷成敗。
#           搜尋結果是整段網頁摘要，只要網頁裡剛好出現「錯誤」「失敗」兩個字，
#           一個完全正確的搜尋就會被判不合格；反過來，Math_Agent 抽出「508」
#           這種語意上毫無意義、但語法合法的算式，計算機回「計算成功」，
#           品管員就放行了 → 假合格。
#
# v2 的三根新骨架：
#   骨架 1：Planner_Agent(規劃官) ── 開場一次性把問題拆成明確的任務清單
#           例：[查 台北101 建築總高度] → [查 東京晴空塔 建築總高度] → [算 兩者差值]
#   骨架 2：facts 事實帳本 ── 查到的數字寫進帳本，Math_Agent 只看帳本(乾淨)，
#           不再從一大坨網頁雜訊裡自己挑數字
#   骨架 3：Supervisor 降級為「進度管理員」 ── 有清單時用純 Python 勾選下一項，
#           完全不呼叫 LLM(0 token、0 幻覺、0 迴圈)，只有清單生不出來時才退回 LLM 判斷
#
# 另外把品管員從「關鍵字掃描」改成「狀態旗標(STATUS 標記)」，
# 徹底根除病灶 C 的雙向誤判。
# ==========================================
import re
import copy
from typing import Annotated, Sequence, TypedDict, Literal, List, Dict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_API_BASE_URL

# ==========================================
# 🚩 第零區：狀態旗標 (取代舊版的關鍵字掃描)
# ==========================================
# 【SA v2 新增】：每個工具節點回報時，一律在訊息最前面掛上狀態旗標。
# 品管員只看這個旗標，不再去猜網頁內文裡的「錯誤」兩個字是不是代表失敗。
# 這是 v1 最容易誤判、也最難 debug 的地方，這裡用最笨但最可靠的方式解決。
STATUS_OK = "【STATUS:OK】"
STATUS_FAIL = "【STATUS:FAIL】"


def _is_failed_message(msg) -> bool:
    """判斷某一則工作紀錄是不是失敗回報(只看開頭旗標，不掃描內文)。"""
    content = str(getattr(msg, "content", "") or "")
    return content.lstrip().startswith(STATUS_FAIL)


# ==========================================
# 🎒 第一區：定義「共用背包」 (AgentState)
# ==========================================
# 【SA 資料隔離升級 - Context Isolation】(v1 保留)：
#   chat_history -> 跨輪次持久保存的「乾淨」歷史對話 (只有 User 問題 + 最終回答)
#   messages     -> 「單次任務」的工作記憶區，每次新問題進來都是全新的一頁
#   next_node    -> 主管的派工決定
#   retry_count  -> Local Grader 專用的重試計數器，避免無限迴圈
#   search_calls -> 本輪搜尋總次數
#   math_calls   -> 本輪計算總次數
#
# 【SA v2 新增的四個欄位，這是整個改版的核心】：
#   plan             -> 任務清單。每一項長這樣：
#                       {"id": 0, "type": "search", "target": "台北101 建築總高度",
#                        "status": "pending|done|failed", "attempts": 0, "result": ""}
#                       Supervisor 靠這個知道「哪些做完了、下一步該做什麼」，
#                       這正是你自己點出的「Supervisor 沒有清楚進度清單」的解法。
#   facts            -> 事實帳本 {"台北101 建築總高度": "508 公尺"}。
#                       查到的數字立刻登錄，Math_Agent / Final_Answer 只吃這裡的乾淨資料。
#   searched_queries -> 這一輪已經真正打過 API 的關鍵字。Python 層硬去重，
#                       同一句話絕不會打第二次 Brave API(解決你點出的「重複查詢浪費 API」)。
#   current_step     -> 主管指定「現在要做清單裡的哪一項」的 id。
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


# ==========================================
# 📋 第二區：定義各節點的「強制下拉式選單」(結構化輸出)
# ==========================================
class RouteDecision(BaseModel):
    """v1 保留：只有在 Planner 生不出清單時，Supervisor 才會退回用這個自由判斷。"""
    reasoning: str = Field(description="請用一句話（不超過 50 字）簡短說明你的判斷邏輯，絕對不要在這裡進行完整的計算、列點或推導過程！")
    next_node: Literal["Search_Agent", "Math_Agent", "FINISH"] = Field(
        description=(
            "1. Search_Agent：需要上網查詢未知資訊、最新數據時選擇。\n"
            "2. Math_Agent：有明確計算需求時選擇。\n"
            "3. FINISH：資料已備齊，可直接回答時選擇。"
        )
    )


# 【SA v2 新增】：規劃官的輸出格式。
# 刻意設計得非常窄 —— 只有「做什麼類型」跟「對象是誰」兩個欄位，
# 目的是逼小模型不要在這裡寫作文、不要順手把題目解掉，只做「拆解」這一件事。
class PlanStep(BaseModel):
    step_type: Literal["search", "math"] = Field(
        description="search=需要上網查一個具體事實；math=需要用計算機做一次運算"
    )
    target: str = Field(
        description=(
            "如果是 search：寫『單一對象 + 要查的屬性』的搜尋關鍵字，"
            "一次只能有一個對象，絕對不可以把兩個對象寫在同一句(例如只能寫『台北101 建築高度』)。\n"
            "如果是 math：用中文寫清楚要算什麼(例如『東京晴空塔高度 減去 台北101高度』)。"
        )
    )


class TaskPlan(BaseModel):
    steps: List[PlanStep] = Field(description="完成這個問題所需要的步驟清單，最多 6 步")


class SearchQuery(BaseModel):
    query: str = Field(description="要丟給搜尋引擎的精準關鍵字")


# 【SA v2 新增】：搜尋結果的「數值萃取器」輸出格式。
# 這是 v2 提升正確率最大的一根槓桿：
# v1 是把整坨網頁摘要原封不動塞進背包，讓下游 Math_Agent 自己在雜訊裡撈數字(它撈不到)；
# v2 在搜尋當下就先撈一次，只把撈到的乾淨數值登錄進帳本。
class ExtractedFact(BaseModel):
    found: bool = Field(description="搜尋結果中是否明確出現了要找的數值。沒有就填 false，絕對不要用你自己的記憶硬湊。")
    value: str = Field(description="找到的數值，含單位，例如 '508 公尺'。找不到就填空字串。")


class MathExpression(BaseModel):
    expression: str = Field(description="要執行的純數學算式，例如 '634 - 508'")


# ==========================================
# 🧠 第三區：初始化大腦模型
# ==========================================
# 【SA 大小模型分工】(v1 保留)：
#   main_llm   -> Supervisor 保底判斷 / Planner / Final_Answer
#   verify_llm -> 關鍵字抽取、數值萃取、算式抽取這類窄任務
MAIN_MODEL_NAME = "XUYA:latest"
VERIFY_MODEL_NAME = "gemma3:4b"  # 8GB VRAM：主模型 4.9GB + 驗證模型 3.3GB

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

# 路由/規劃專用：temperature=0，要的是可預期而不是創意
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

# 【SA v2 新增】：規劃官需要輸出一個 list of object，比單純選擇題吃力一點，
# 所以 num_predict 給 500，比 router 寬鬆但仍遠低於 main_llm，避免它寫作文。
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

# ==========================================
# ⚙️ 第三.五區：各種上限常數
# ==========================================
MAX_RETRY = 2                      # 單一節點連續失敗重試上限
MAX_SEARCH_CALLS_PER_TURN = 6      # 【SA v2 調整】：v1 設 4，多實體題(2 個對象 + 重試)很容易撞牆；
                                   # 因為 v2 已經有硬去重，重複查詢不會再吃額度，所以可以放寬到 6
MAX_MATH_CALLS_PER_TURN = 4
MAX_STEP_ATTEMPTS = 3              # 【SA v2 新增】：同一個清單項目最多嘗試 3 次，超過就標記 failed 跳過，
                                   # 避免某一項卡死拖垮整條流程(v1 沒有這層，只能靠總次數上限硬撞)

import concurrent.futures

LLM_TIMEOUT_SECONDS = 90


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
# 【SA 中文數字單位正規化】(v1 保留)：
# 「億」「萬」這種單位換算是機械式規則，不該讓小模型自己心算。
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
    """【SA v2 新增】：把事實帳本印成人類/模型都好讀的乾淨清單。"""
    if not facts:
        return "（帳本目前是空的，尚未確認任何事實）"
    return "\n".join(f"- {k} ＝ {v}" for k, v in facts.items())


def _render_plan(plan: list) -> str:
    """【SA v2 新增】：把任務清單印成進度表，log 一眼就能看出卡在哪一步。"""
    if not plan:
        return "（無任務清單，走 LLM 自由判斷模式）"
    icon = {"pending": "⬜", "done": "✅", "failed": "❌"}
    return "\n".join(
        f"  {icon.get(s['status'], '⬜')} [{s['id']}] {s['type']}: {s['target']}"
        for s in plan
    )


def _numbers_in(text: str) -> set:
    """抽出字串裡所有的數字(去掉千分位逗號)，用來做「算式數字是否來自帳本」的溯源檢查。"""
    return set(re.findall(r'\d+(?:\.\d+)?', (text or "").replace(",", "")))


# 【SA v2 新增 - 硬去重】：v1 只在字串含分隔符時才拆，完全沒有「同一句不准查第二次」的防線。
# 這裡改成：只要正規化後的關鍵字已經打過 API，就先嘗試加限定詞變形；
# 還是重複就直接回傳 None，讓上層節點判定為失敗，絕不再浪費一次 Brave 額度。
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


# 【SA 保留 v1】：Python 強制拆分合併查詢(只在「無清單保底模式」下才會用到)。
# 【SA v2 加強】：分隔符補上「跟」「以及」，並允許沒有空白的寫法。
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


# ==========================================
# 🏢 第四區：定義專家房間 (Nodes)
# ==========================================

# 【0. 規劃官房間】── v2 全新節點
def planner_node(state: AgentState):
    """
    【SA v2 新增】：整輪對話開場只跑一次，把使用者問題拆成明確的任務清單。

    為什麼一定要有這個節點？
    v1 的 Supervisor 是「每一步重新讀題、重新判斷」，等於每一步都在賭小模型這次
    會不會想起來「101 已經查過了」。實測結果就是連續三次都沒想起來。
    把「拆解」從「派工」裡切出來、而且只做一次，之後的派工就退化成
    「照著清單勾選」的純機械動作 —— 機械動作不會有幻覺，也不會鬼打牆。
    """
    msgs = list(state["messages"])
    question = ""
    for m in msgs:
        if isinstance(m, HumanMessage):
            question = m.content

    sys_msg = SystemMessage(content=(
        "你是任務拆解專家。你『絕對沒有』任何常識與計算能力，你的工作【只有】把問題拆成步驟清單，"
        "【嚴禁】在這裡回答問題或算出任何答案。\n\n"
        "拆解規則：\n"
        "1. 問題裡每一個『你不確定的具體事實數字』(高度、人口、股價、資本額、現任人物…)，"
        "都要獨立成一個 search 步驟，一個步驟只能查一個對象。\n"
        "2. 如果問題比較 A 和 B 兩個對象，就要產生兩個 search 步驟(一個查 A、一個查 B)，"
        "【絕對不可以】把 A 和 B 寫在同一個 target 裡。\n"
        "3. 所有需要運算的部分(加減乘除、比較差值、換算)，都要獨立成 math 步驟，"
        "並且一定要排在相關的 search 步驟後面。\n"
        "4. 如果問題純粹是邏輯/排列組合題，題目文字裡就有全部數字，那就不需要 search，只要 math 步驟。\n"
        "5. 如果問題不需要查也不需要算(例如問公司介紹、問履歷)，請回傳空的 steps 清單。\n"
        "6. 步驟總數不要超過 6 個。\n\n"
        "範例：\n"
        "問題：「請查台北101與東京晴空塔的總高度，並算出誰高多少公尺？」\n"
        "正確拆解：\n"
        "  [1] search / 台北101 建築總高度 公尺\n"
        "  [2] search / 東京晴空塔 建築總高度 公尺\n"
        "  [3] math   / 東京晴空塔高度 減去 台北101高度\n"
        "錯誤拆解(禁止)：\n"
        "  [1] search / 台北101 與 東京晴空塔 建築總高度   ← 兩個對象混在一起，會查到模糊的比較文章\n"
    ))

    plan = []
    try:
        result = planner_structured_llm.invoke([sys_msg, HumanMessage(content=f"請拆解這個問題：{question}")])
        raw_steps = (result.steps or [])[:6]
        # 【SA v2 防呆】：強制把所有 search 排在 math 前面。
        # 小模型偶爾會把 math 排在前面，那會導致還沒查到數字就叫計算機動手 —— 正是這次翻車的畫面。
        search_steps = [s for s in raw_steps if s.step_type == "search"]
        math_steps = [s for s in raw_steps if s.step_type == "math"]
        ordered = search_steps + math_steps
        for i, s in enumerate(ordered):
            target = (s.target or "").strip()
            if not target:
                continue
            plan.append({
                "id": i,
                "type": s.step_type,
                "target": target,
                "status": "pending",
                "attempts": 0,
                "result": ""
            })
    except Exception as e:
        # 【SA v2 保底】：規劃失敗不讓整條流程死掉，退回 v1 的「LLM 自由判斷」模式。
        # log 會明確印出來，方便你判斷是不是主模型的結構化輸出能力不足。
        print(f"[規劃官 Planner] ⚠️ 任務拆解失敗({e})，本輪退回 LLM 自由判斷模式")
        plan = []

    if plan:
        print("\n[規劃官 Planner] 📋 本輪任務清單已產生：")
        print(_render_plan(plan))
    else:
        print("\n[規劃官 Planner] 📋 未產生任務清單(可能是純知識題)，交由 Supervisor 自由判斷")

    return {
        "plan": plan,
        "facts": {},
        "searched_queries": [],
        "current_step": -1,
        "retry_count": 0,
        "search_calls": 0,
        "math_calls": 0
    }


# 【1. 總機主管房間】
def supervisor_node(state: AgentState):
    """
    【SA v2 改版】：主管從「每步重新讀題的決策者」降級成「照清單勾選的進度管理員」。

    有清單時 → 純 Python 找出第一個 pending 的項目派工，完全不呼叫 LLM。
                好處：0 token、0 幻覺、0 鬼打牆，而且 log 會直接印出進度表。
    沒清單時 → 才退回 v1 的 LLM 自由判斷(保底路徑，行為與舊版一致)。
    """
    plan = copy.deepcopy(state.get("plan", []) or [])
    facts = dict(state.get("facts", {}) or {})
    searched = list(state.get("searched_queries", []) or [])
    cur_search_calls = state.get("search_calls", 0)
    cur_math_calls = state.get("math_calls", 0)

    # 每一次派工都原封不動帶著這些欄位，不依賴「沒回傳的 key 會自動延續」這個假設
    carry = {
        "plan": plan,
        "facts": facts,
        "searched_queries": searched,
        "search_calls": cur_search_calls,
        "math_calls": cur_math_calls,
        "retry_count": 0,
    }

    # 🛑 【Python 物理絕對防禦】(v2 改版)：
    # v1 是掃描訊息內文有沒有「錯誤」兩個字 —— 但搜尋結果裡本來就常常出現這兩個字，
    # 等於埋了一顆隨機引爆的地雷。v2 改看開頭的 STATUS 旗標，只在真的失敗時才煞車。
    msgs = list(state["messages"])
    if msgs and _is_failed_message(msgs[-1]):
        last_name = getattr(msgs[-1], "name", "")
        if last_name in ["Math_Agent", "Search_Agent"]:
            print(f"\n[系統守衛] 🛑 偵測到 {last_name} 回報失敗旗標，記錄後繼續往下一項任務。")

    # ------------------------------------------------
    # (A) 有任務清單 → 純 Python 勾選模式
    # ------------------------------------------------
    if plan:
        # 先把「已經試太多次」的項目標記為 failed，避免單一項目卡死整條流程
        for step in plan:
            if step["status"] == "pending" and step.get("attempts", 0) >= MAX_STEP_ATTEMPTS:
                step["status"] = "failed"
                print(f"[主管 Supervisor] ⚠️ 任務 [{step['id']}] {step['target']} 已嘗試 {step['attempts']} 次仍未完成，標記失敗並跳過。")

        print("\n[主管 Supervisor] 📋 目前進度：")
        print(_render_plan(plan))

        for step in plan:
            if step["status"] == "pending":
                node = "Search_Agent" if step["type"] == "search" else "Math_Agent"
                print(f"[主管 Supervisor] ➡️ 派工給 {node}：任務 [{step['id']}] {step['target']}")
                return {**carry, "plan": plan, "next_node": node, "current_step": step["id"]}

        print("[主管 Supervisor] 🎉 任務清單全數處理完畢，交給客服公關結案。")
        return {**carry, "plan": plan, "next_node": "FINISH", "current_step": -1}

    # ------------------------------------------------
    # (B) 沒有任務清單 → 退回 v1 的 LLM 自由判斷(保底)
    # ------------------------------------------------
    chat_history = list(state.get("chat_history", []))

    # 【SA v2 加強】：把「已確認的事實帳本」明確餵給主管。
    # v1 主管的 reasoning 寫出「508 公尺」卻沒地方存，下一步就忘光 —— 這裡把它補起來。
    facts_note = (
        "\n\n【本輪已確認的事實帳本】(這些已經查到了，絕對不要再重複查)：\n"
        + _render_facts(facts)
    )
    searched_note = ""
    if searched:
        searched_note = "\n\n【本輪已經查詢過的關鍵字】(不要再查同樣的東西)：\n" + "\n".join(f"- {q}" for q in searched)

    sys_msg = SystemMessage(content=(
        "你是路由主管。你『絕對沒有』任何常識、歷史知識或數學能力！\n"
        "下方會提供兩種資料：\n"
        "【過去對話歷史】：僅供你判斷『這一題是否延續上一題』，若無關請完全忽略。\n"
        "【本輪工作紀錄】：這一題目前已經查到/算到的資料。\n\n"
        "請嚴格遵守以下派工順序：\n"
        "1. 若問題是詢問張序亞的履歷，直接選 'FINISH'。\n"
        "2. 只要問題詢問『客觀事實、數據』，且【事實帳本裡還沒有這筆資料】，"
        "你【絕對不准】憑記憶回答，【強制】派給 'Search_Agent' 查詢！\n"
        "3. 【重要】如果某個數字【已經出現在下方的事實帳本裡】，就代表它查到了，"
        "【絕對不准】再派 Search_Agent 去查同一個東西，請直接進入下一步(計算或結案)。\n"
        "4. 拿到數字後若需計算，【絕對不准】在理由中心算，【強制】派給 'Math_Agent'！\n"
        "5. 比較兩個以上實體時，可以針對『不同實體』連續派工，但每次查的對象必須不同。\n"
        "6. 事實帳本已足夠回答問題時，選 'FINISH'。"
        + facts_note + searched_note
    ))

    decision = supervisor_llm.invoke([sys_msg] + chat_history + msgs)
    print(f"\n[主管 Supervisor] 決定派工給: {decision.next_node} (理由: {decision.reasoning})")

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
    【SA v2 改版重點】：
    1. 關鍵字不再每次都靠小模型「重新抽取」—— 清單模式下直接用 Planner 定好的 target，
       這從源頭消滅了「每次都抽出同一句話」的可能。
    2. Python 硬去重：已經打過 API 的關鍵字絕不再打第二次(省 Brave 額度)。
    3. 拿到結果後立刻做「數值萃取」，把乾淨的數字登錄進事實帳本，
       塞進 messages 的只有摘要，不再是整坨網頁雜訊。
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
        # 清單模式：關鍵字由 Planner 事先決定，重試時才做機械式變形(不呼叫 LLM，省算力)
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

    # ---- Python 硬去重(這一段就是「不再浪費 API」的實作) ----
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

    # ---- 【SA v2 核心新增】：數值萃取 → 登錄事實帳本 ----
    fact_target = step["target"] if step is not None else final_query
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

    excerpt = raw[:800]

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


# 【SA v2 改版】：Search_Agent 出口的 Local Grader (f-b)
# v1 掃描內文關鍵字 → 網頁摘要裡出現「錯誤」就誤殺；v2 只看開頭的 STATUS 旗標。
def search_grader(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    is_bad = _is_failed_message(last_msg)
    retry = state.get("retry_count", 0)

    if is_bad and retry < MAX_RETRY:
        print(f"[品管員 f-Search] ❌ 搜尋未取得有效事實，退回重搜 (第 {retry} 次重試)")
        return "retry"
    if is_bad:
        print("[品管員 f-Search] 🛑 已達重試上限，本項任務放棄，交回主管繼續下一項")
        return "pass"   # 【SA v2 調整】：v1 是 give_up 直接跳結案，會讓「第二個對象」永遠沒機會查；
                        # v2 改為交回主管，由主管把這一項標記 failed 後繼續跑下一項，整題才不會半路死掉
    print("[品管員 f-Search] ✅ 已取得明確事實，放行給主管")
    return "pass"


# 【3. 算盤法師房間】
def math_node(state: AgentState):
    """
    【SA v2 改版重點】：
    1. 上下文換血 —— 不再把整坨搜尋雜訊丟給小模型，只餵「事實帳本 + 這一步要算什麼」。
       v1 就是因為餵了一大坨網頁文字，gemma3:4b 才會抽出
       「台北 101 建築總高度 公尺 = 508」這種中文段落當算式。
    2. 合法性檢查升級 —— v1 只檢查「有沒有中文」，所以「508」這種光禿禿的數字
       會被判定合法、計算機回「計算成功」、品管員放行 → 假合格(這次翻車的第三個原因)。
       v2 額外要求算式必須含有運算子或白名單函式。
    3. 數字溯源檢查 —— 算式裡的數字必須來自事實帳本，防止小模型憑記憶塞數字進來。
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

    extractor = verify_llm.with_structured_output(MathExpression)
    sys_msg = SystemMessage(content=(
        "你是數學算式翻譯機。請把下方的計算任務，翻譯成一行純 Python 數學算式。\n\n"
        f"【可以使用的事實帳本】(只能用這裡面的數字，禁止使用你自己記憶中的任何數字)：\n{_render_facts(facts)}\n\n"
        f"【這一步要計算的任務】：{task_desc}\n\n"
        "嚴格規則：\n"
        "1. 只能輸出算式本身，例如 '634 - 508'，【絕對不能包含任何中文字、等號、單位或說明】。\n"
        "2. 算式裡【必須】至少有一個運算子(+ - * /)或函式，"
        "【禁止】只回一個光禿禿的數字(例如只回 '508' 是錯的)。\n"
        "3. 算式裡的每一個數字，都必須是上面事實帳本裡出現過的數字。\n"
        "4. 排列組合請用 comb(n, k) / perm(n, k) / factorial(n)，不要用 C(n,k) 這種課本記號。"
    ))

    # 【SA v2 隔離重點】：這裡刻意「不」傳入 msgs，只給乾淨的帳本與任務描述。
    # 這是把 Math_Agent 從雜訊裡救出來最有效的一刀。
    math_req = extractor.invoke([sys_msg, HumanMessage(content=f"請翻譯這個計算任務：{task_desc}")])

    expr = (math_req.expression or "").strip()

    # ---- 檢查 1：語法合法性(允許 comb/perm/factorial) ----
    stripped_for_check = re.sub(r'\b(comb|perm|factorial)\b', '', expr)
    syntax_ok = bool(expr) and bool(re.fullmatch(r'[\d\.\+\-\*/\(\),\s]+', stripped_for_check))

    # ---- 檢查 2：【SA v2 新增】語意有效性 —— 必須真的在「算」東西 ----
    has_operation = bool(re.search(r'[\+\-\*/]', expr)) or bool(re.search(r'\b(comb|perm|factorial)\s*\(', expr))

    # ---- 檢查 3：【SA v2 新增】數字溯源 —— 算式裡的數字要來自帳本 ----
    # 【SA v2 注意】：排列組合/階乘題的數字通常直接寫在題目裡、不會經過搜尋，
    # 帳本當然對不上，所以只要算式用到白名單函式就跳過這道檢查，避免誤殺。
    uses_whitelisted_func = bool(re.search(r'\b(comb|perm|factorial)\s*\(', expr))
    provenance_ok = True
    if facts and syntax_ok and has_operation and not uses_whitelisted_func:
        fact_numbers = set()
        for v in facts.values():
            fact_numbers |= _numbers_in(str(v))
        expr_numbers = _numbers_in(expr)
        # 完全沒有任何一個數字對得上帳本 → 幾乎可以確定是小模型憑記憶亂編的
        if expr_numbers and not (expr_numbers & fact_numbers):
            provenance_ok = False

    if not syntax_ok or not has_operation or not provenance_ok:
        if not syntax_ok:
            reason = f"算式含有非數學字元(抽取結果：{expr!r})"
        elif not has_operation:
            reason = f"算式沒有任何運算，只是一個孤立的數字(抽取結果：{expr!r})，這代表它沒有真的在計算"
        else:
            reason = f"算式中的數字({expr!r})沒有任何一個來自事實帳本，疑似模型憑記憶編造"
        print(f"[算盤法師] ⚠️ {reason}")
        return {
            "messages": [AIMessage(
                name="Math_Agent",
                content=f"{STATUS_FAIL} 無法產生有效算式：{reason}"
            )],
            "plan": plan, "facts": facts,
            "retry_count": retry_count, "math_calls": math_calls
        }

    print(f"[算盤法師 Math_Agent] 正在使用魔法計算機: {expr}")
    from tools.calculator import calculate_math
    result = calculate_math(expr)

    if result.startswith("計算失敗"):
        return {
            "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_FAIL} {result}")],
            "plan": plan, "facts": facts,
            "retry_count": retry_count, "math_calls": math_calls
        }

    # 計算成功 → 一樣登錄進事實帳本，Final_Answer 就能直接引用
    if step is not None:
        step["status"] = "done"
        step["result"] = result
        facts[step["target"]] = result
    else:
        facts[f"計算：{expr}"] = result

    print(f"[算盤法師 Math_Agent] 📒 已登錄事實帳本：{task_desc} ＝ {result}")

    return {
        "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_OK}\n【計算機結果】\n{result}")],
        "plan": plan,
        "facts": facts,
        "retry_count": retry_count,
        "math_calls": math_calls
    }


# 【SA v2 改版】：Math_Agent 出口的 Local Grader (f-c)，一樣只看 STATUS 旗標
def math_grader(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    is_bad = _is_failed_message(last_msg)
    retry = state.get("retry_count", 0)

    if is_bad and retry < MAX_RETRY:
        print(f"[品管員 f-Math] ❌ 計算結果不合格，退回重算 (第 {retry} 次重試)")
        return "retry"
    if is_bad:
        print("[品管員 f-Math] 🛑 已達重試上限，本項任務放棄，交回主管繼續下一項")
        return "pass"
    print("[品管員 f-Math] ✅ 計算結果合格，放行給主管")
    return "pass"


# 【4. 客服公關房間】
def final_answer_node(state: AgentState):
    print("\n[客服公關 Final_Answer] 資料收集完畢，正在撰寫最終回覆給客人...")

    msgs = list(state["messages"])
    plan = state.get("plan", []) or []
    facts = dict(state.get("facts", {}) or {})

    current_question = ""
    for msg in msgs:
        if isinstance(msg, HumanMessage):
            current_question = msg.content

    # 【SA v2 新增】：Python 先算好「有沒有東西沒查到」，直接把結論寫死在提示裡，
    # 不讓主模型自己判斷「資料夠不夠」—— 它每次都會覺得夠。
    failed_steps = [s for s in plan if s["status"] != "done"]
    if failed_steps:
        gap_note = (
            "⚠️ 以下項目【沒有查到/算出結果】，你在回答中必須誠實告知使用者這些部分查詢失敗：\n"
            + "\n".join(f"- {s['target']}" for s in failed_steps)
        )
    else:
        gap_note = "所有預定的查詢與計算項目都已完成。"

    # 本輪的原始工作紀錄(給模型當佐證，但主要依據仍是事實帳本)
    scratch_str = ""
    for msg in msgs:
        if getattr(msg, "name", "") in ["Search_Agent", "Math_Agent"]:
            scratch_str += f"[{msg.name}]: {msg.content[:600]}\n\n"
    if not scratch_str:
        scratch_str = "（本輪沒有動用搜尋或計算工具）"

    history_str = ""
    hist = list(state.get("chat_history", []))
    if hist:
        for msg in hist:
            role = "使用者" if isinstance(msg, HumanMessage) else "AI"
            history_str += f"{role}: {msg.content}\n"
    else:
        history_str = "（無歷史對話）"

    sys_msg = SystemMessage(content=(
        "你是一位專業的 AI 助理。\n\n"
        f"【目前使用者的問題】：\n{current_question}\n\n"
        f"【已確認的事實帳本 ── 這是你唯一可以引用的數字來源】：\n{_render_facts(facts)}\n\n"
        f"【本輪任務完成情況】：\n{gap_note}\n\n"
        f"【本輪調查過程紀錄(佐證用)】：\n{scratch_str}\n\n"
        f"【過去對話歷史，僅供語意連貫參考】：\n{history_str}\n\n"
        "請嚴格遵守：\n"
        "【鐵則 1】：只回答『目前使用者的問題』，不要主動重複過去對話的內容。\n"
        "【鐵則 2】：只有當問題明確延續過去對話時，才可以引用【過去對話歷史】。\n"
        "【鐵則 3】：你只能使用【事實帳本】裡明確出現的數字與名稱作答，"
        "絕對不准使用你自己記憶中的人名、年份、數字！查無資料就誠實說查詢失敗，絕不編造。\n"
        "【鐵則 4 ── 最重要，絕無例外】：你【完全不會算數】。"
        "如果使用者問的是差值、總和、比例、倍數，而【事實帳本裡沒有現成的計算結果】，"
        "你【絕對禁止】自己在心裡做任何加減乘除然後把答案寫出來，"
        "只能誠實說明『目前只查到 A 和 B 的數值，計算步驟未能完成』。"
        "就算你覺得那個減法很簡單、你一定算得對，也一樣禁止 —— 這是系統設計的紅線。\n"
        "【鐵則 5】：請自然地陳述結果，絕對不要說出「根據 Math_Agent」「事實帳本」等內部字眼。\n"
        "【鐵則 6】：如果問題比較兩個以上對象，必須分別給出每個對象的明確數字，再說明差異。"
    ))

    response = invoke_with_timeout(main_llm, [sys_msg])

    # 🛡️ 切除模型溢出的 "assistant" 標籤
    clean_text = response.content.strip()
    clean_text = re.sub(r'^assistant[:\s\n]*', '', clean_text, flags=re.IGNORECASE).strip()

    return {"messages": [AIMessage(name="Final_Answer", content=clean_text)]}


# ==========================================
# 🗺️ 第五區：畫地圖與建立動線 (Graph Edges)
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("Planner", planner_node)          # 【SA v2 新增節點】
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Search_Agent", search_node)
workflow.add_node("Math_Agent", math_node)
workflow.add_node("Final_Answer", final_answer_node)

# 【SA v2 動線調整】：開場先經過規劃官，產生任務清單後才交給主管執行
workflow.add_edge(START, "Planner")
workflow.add_edge("Planner", "Supervisor")


def routing_logic(state: AgentState):
    """總次數上限的最後一道物理煞車(v1 保留，數值放寬)。"""
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

# 【SA v2 說明】：兩個 Grader 現在都不再有 "give_up → 直接結案" 這條路。
# 原因：v1 只要第一個對象查失敗就整題陣亡，第二個對象根本沒機會查。
# v2 改成失敗達上限就 pass 回主管，由主管把該項標記 failed 後繼續跑下一項，
# 最後由 Final_Answer 誠實告知哪些沒查到 —— 這比整題暴斃好太多。
workflow.add_conditional_edges(
    "Search_Agent",
    search_grader,
    {
        "retry": "Search_Agent",
        "pass": "Supervisor"
    }
)
workflow.add_conditional_edges(
    "Math_Agent",
    math_grader,
    {
        "retry": "Math_Agent",
        "pass": "Supervisor"
    }
)

workflow.add_edge("Final_Answer", END)

app_graph = workflow.compile()
print("[系統] 🗺️ 航空母艦地圖與動線建立完成！準備出航！")

# ==========================================
# 🎮 第六區：本機獨立測試區塊
# ==========================================
if __name__ == "__main__":
    print("\n========================================================")
    print("🚀 航空母艦戰鬥群：跨輪次資料隔離測試")
    print("========================================================")

    q1 = "請幫我分別查詢台北 101 與日本東京晴空塔的建築總高度（公尺），並計算晴空塔和台北 101 誰比誰高多少公尺？"
    print(f"👤 [使用者 Q1]: {q1}\n")

    state1 = {
        "chat_history": [],
        "messages": [HumanMessage(content=q1)],
        "retry_count": 0,
        "search_calls": 0,
        "math_calls": 0,
        "plan": [],
        "facts": {},
        "searched_queries": [],
        "current_step": -1
    }

    a1_text = ""
    for output in app_graph.stream(state1, {"recursion_limit": 30}):
        for key, value in output.items():
            print(f"--- 經過房間: {key} ---")
            if key == "Final_Answer":
                a1_text = value["messages"][-1].content

    print(f"\n🤖 [A1 最終回答]:\n{a1_text}\n")

    # 第二輪：確認跨輪次資料隔離仍然有效
    q2 = "請上網查詢台灣高鐵最新的實收資本額大約是多少新台幣？接著幫我計算：如果每股面額 10 元，總共有多少股？"
    print(f"👤 [使用者 Q2]: {q2}\n")

    state2 = {
        "chat_history": [HumanMessage(content=q1), AIMessage(content=a1_text)],
        "messages": [HumanMessage(content=q2)],
        "retry_count": 0,
        "search_calls": 0,
        "math_calls": 0,
        "plan": [],
        "facts": {},
        "searched_queries": [],
        "current_step": -1
    }

    a2_text = ""
    for output in app_graph.stream(state2, {"recursion_limit": 30}):
        for key, value in output.items():
            print(f"--- 經過房間: {key} ---")
            if key == "Final_Answer":
                a2_text = value["messages"][-1].content

    print("\n========================================================")
    print(f"🤖 [A2 最終回答]:\n{a2_text}")
    print("========================================================")
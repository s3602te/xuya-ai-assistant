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
    expression: str = Field(description="要執行的純數學算式，例如 '634 - 508'")


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
            has_digit = bool(re.search(r'\d', target))
            has_symbol = bool(re.search(r'[\+\-\*/×÷%]', target))
            has_verb = bool(re.search(
                r'(加|減|乘|除|相差|差值|差距|倍|總和|合計|平均|百分比|比例|次方|階乘|排列|組合|扣掉|加起來|總計|換算)',
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
_MATH_HINT = r'(計算|算出|相差|差多少|高多少|多多少|少多少|誰比誰|總共|合計|幾倍|百分比|平均|加起來|總和)'


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


def _heuristic_math_only(question: str) -> list:
    """
    【SA v2.3 新增】純數學題的保底拆解器。

    實測情境：「從10個人中選3個有幾種組合?」
    這題完全不需要上網，規劃官照理應該排一個 math 步驟(規則 4)，
    但 XUYA 回了空清單，害得整題只能靠 Supervisor 的 LLM 保底路徑救回來。
    能救回來是運氣好，不該當成常態。

    這裡用純 Python 補上：問題裡同時出現「數字」和「運算意圖」，
    而且沒有任何需要外部查證的跡象時，就直接排一個 math 步驟，
    任務描述直接用問題原文(Math_Agent 本來就吃得下原文，
    而且原文裡的數字正好能通過溯源檢查)。
    """
    if not question:
        return []
    if not re.search(r'\d', question):
        return []
    if not re.search(_MATH_HINT + r'|(幾種|幾個|幾倍|多少|排列|組合|階乘|次方)', question):
        return []
    # 出現這些字眼代表需要外部事實，就不是純數學題，交給其他層處理
    if re.search(r'(上網|查詢|查一下|最新|現任|目前|今天|股價|匯率|天氣)', question):
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
    }

    # ---- 前置判斷：知識庫已有標準答案，就不要上網 ----
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
        # 先用純 Python 再確認一次是不是純數學題(規劃官偶爾會漏掉這種)。
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

    # ---- 【SA v2 核心】：數值萃取 → 登錄事實帳本 ----
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

    # 【SA v2 隔離重點】：刻意「不」傳入 msgs，只給乾淨的帳本與任務描述
    math_req = extractor.invoke([sys_msg, HumanMessage(content=f"請翻譯這個計算任務：{task_desc}")])

    expr = (math_req.expression or "").strip()

    # ---- 檢查 1：語法合法性(允許 comb/perm/factorial) ----
    stripped_for_check = re.sub(r'\b(comb|perm|factorial)\b', '', expr)
    syntax_ok = bool(expr) and bool(re.fullmatch(r'[\d\.\+\-\*/\(\),\s]+', stripped_for_check))

    # ---- 檢查 2：語意有效性 —— 必須真的在「算」東西 ----
    has_operation = bool(re.search(r'[\+\-\*/]', expr)) or bool(re.search(r'\b(comb|perm|factorial)\s*\(', expr))

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

    if not syntax_ok or not has_operation or not provenance_ok:
        if not syntax_ok:
            reason = f"算式含有非數學字元(抽取結果：{expr!r})"
        elif not has_operation:
            reason = f"算式沒有任何運算，只是一個孤立的數字(抽取結果：{expr!r})，這代表它沒有真的在計算"
        else:
            reason = f"算式中的數字({expr!r})既不在事實帳本、也不在題目文字裡，疑似模型憑記憶編造"
        print(f"[算盤法師] ⚠️ {reason}")
        return {
            "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_FAIL} 無法產生有效算式：{reason}")],
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

    if step is not None:
        step["status"] = "done"
        step["result"] = result
        # 【SA v2.3 修正】：帳本只存「乾淨的數值」，不要存整串工具輸出。
        # 實測 log 顯示，v2.2 把 "計算成功！算式 'comb(10, 3)' 的結果為：120" 整串塞進帳本，
        # 而 Final_Answer 拿到帳本後直接照抄，面試官看到的回覆變成
        # 「根據事實帳本，我查詢到了答案。計算成功！算式 'comb(10, 3)' 的結果為：120」——
        # 內部術語和工具原始字串全部外洩，鐵則 5 形同虛設。
        # 在源頭就把數值切乾淨，比事後叫模型「不要說出內部字眼」可靠得多。
        facts[step["target"]] = _extract_calc_value(result)
    else:
        facts[f"計算：{expr}"] = _extract_calc_value(result)

    # 【SA v2.4】：這行原本印的是計算機的完整輸出，讓人誤以為帳本沒清乾淨。
    # 實際存進帳本的一直都是清洗後的數值，這裡改印同一個值，log 與實際狀態才一致。
    print(f"[算盤法師 Math_Agent] 📒 已登錄：{task_desc} ＝ {_extract_calc_value(result)}")

    return {
        "messages": [AIMessage(name="Math_Agent", content=f"{STATUS_OK}\n【計算機結果】\n{result}")],
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


# 【4. 盜賊客服房間】
def final_answer_node(state: AgentState):
    print("\n[盜賊客服 Final_Answer] 資料收集完畢，正在撰寫最終回覆給客人...")

    msgs = list(state["messages"])
    plan = state.get("plan", []) or []
    facts = dict(state.get("facts", {}) or {})
    rag_context = state.get("rag_context", "") or ""
    rag_hit_type = state.get("rag_hit_type", "none")

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

    history_str = ""
    hist = list(state.get("chat_history", []))
    if hist:
        for msg in hist:
            role = "使用者" if isinstance(msg, HumanMessage) else "AI"
            history_str += f"{role}: {str(msg.content)[:200]}\n"
        # 【SA v2.6】：只保留最近的部分，同樣是為了控制 num_ctx 預算
        if len(history_str) > 900:
            history_str = "…（較早的對話已略過）\n" + history_str[-900:]
    else:
        history_str = "（無歷史對話）"

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
        f"【本輪任務完成情況】：\n{gap_note}\n\n"
        f"【本輪調查過程紀錄(佐證用)】：\n{scratch_str}\n\n"
        f"【過去對話歷史，僅供語意連貫參考】：\n{history_str}\n\n"
        "請嚴格遵守：\n"
        "【鐵則 0 ── 參考答案優先】：如果上方出現【這一題的參考答案】，"
        "那就是這一題的正確內容，請用你自己的話自然地講出來，"
        "可以潤飾語氣但不要改變事實。"
        "【嚴禁】把「這一題的參考答案」「根據知識庫」「我們知道」「答案是」"
        "這類框架文字照抄進回覆——直接講內容就好，就像你本來就知道一樣。\n"
        "【鐵則 1】：只回答『目前使用者的問題』，不要主動重複過去對話的內容。\n"
        "【鐵則 2】：只有當問題明確延續過去對話時，才可以引用【過去對話歷史】。\n"
        "【鐵則 3】：你只能使用【公司知識庫】或【本次查證到的資料】裡明確出現的數字與名稱作答，"
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

    # 🛡️ 切除模型溢出的 "assistant" 標籤
    clean_text = response.content.strip()
    clean_text = re.sub(r'^assistant[:\s\n]*', '', clean_text, flags=re.IGNORECASE).strip()

    # 【SA v2.4 新增】：Python 強制清洗內部術語。
    #
    # 光靠鐵則叫模型「不要說出內部字眼」是無效的 —— 實測它照樣回出
    #   「根據事實帳本，我們知道這樣子的組合總共有 120 種。」
    # 因為提示裡的區塊標題就寫著「事實帳本」，模型自然把它當成可引用的出處名稱。
    #
    # 上一版已經把標題改成中性用語，這裡再補一道 Python 後處理當作保險：
    # 提示詞是「請求」，正則替換是「保證」，兩層一起做才不會漏。
    _INTERNAL_TERMS = [
        # 【SA v2.7 擴充】：補上這次實測外洩的框架用語。
        # 這些是提示裡的區塊標題與句式，模型會把它們當成可引用的出處名稱照抄，
        # 例如實測出現的「根據公司知識庫的標準解答，我們知道：…答案是：…」。
        r'根據(公司)?知識庫的?(標準)?(解答|答案|內容|資料)[，,、：:]?\s*',
        r'根據(事實帳本|查證資料|調查紀錄|工作紀錄)(的[\u4e00-\u9fff]{0,4})?[，,、]?\s*',
        r'(事實帳本|查證資料|本輪調查紀錄|內部紀錄)(中|裡|裡面)?[的]?',
        r'根據\s*(Math_Agent|Search_Agent|Final_Answer|Supervisor|Planner)\s*(的[\u4e00-\u9fff]{0,4})?[，,、]?\s*',
        r'我們知道[，,、：:]?\s*',
        r'^(因此[，,]?\s*)?答案(是|為)[：:]?\s*',
        r'計算成功！?\s*',
        r'算式\s*[\'"][^\'"]*[\'"]\s*的結果為[：:]\s*',
    ]
    # 【SA v2.7】：逐行套用，因為「答案是」那條綁了行首錨點 ^，需要對每一行分別比對
    lines = clean_text.split("\n")
    cleaned_lines = []
    for line in lines:
        for pat in _INTERNAL_TERMS:
            line = re.sub(pat, '', line, flags=re.MULTILINE)
        cleaned_lines.append(line)
    clean_text = "\n".join(cleaned_lines)
    # 清掉替換後可能留下的多餘標點與空白
    clean_text = re.sub(r'^[，,、。\s]+', '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

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
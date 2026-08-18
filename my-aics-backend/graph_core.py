# -*- coding: utf-8 -*-
# graph_core.py
# ==========================================
# 航空母艦戰鬥群 (LangGraph) 核心定義檔
# ==========================================
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_API_BASE_URL

# ==========================================
# 🎒 第一區：定義「共用背包」 (AgentState)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str

# ==========================================
# 📋 第二區：定義主管的「強制下拉式選單」
# ==========================================
class RouteDecision(BaseModel):
    reasoning: str = Field(description="請詳細說明判斷邏輯：使用者的最新問題是什麼？我們現在缺什麼資訊？")
    next_node: Literal["Search_Agent", "Math_Agent", "FINISH"] = Field(
        description=(
            "1. Search_Agent：僅限查詢『外部即時資訊』(如今天股價、天氣、新聞)。履歷資料已經在知識庫，絕對禁止對履歷使用網路搜尋！\n"
            "2. Math_Agent：僅限『使用者的問題』明確要求加減乘除時。不要因為知識庫裡有年份或數字就亂算！\n"
            "3. FINISH：如果知識庫已有答案(如自我介紹、經歷)，或者其他 Agent 已經把答案找回來/算出來了，請立刻選擇 FINISH。"
        )
    )

# ==========================================
# 🧠 第三區：初始化大腦模型
# ==========================================
llm = ChatOllama(
    base_url=OLLAMA_API_BASE_URL,
    model="XUYA:latest",
    temperature=0  
)
supervisor_llm = llm.with_structured_output(RouteDecision)

# ==========================================
# 🏢 第四區：定義專家房間 (Nodes)
# ==========================================
# 【1. 總機主管房間】
def supervisor_node(state: AgentState):
    messages = state["messages"]
    
    # 【SA 動態連擊升級 v2】：賦予主管分辨 RAG 與外部資訊的能力
    sys_msg = SystemMessage(content=(
        "你是冷靜聰明的路由主管。請『只針對使用者最後一句話』進行派工判斷。\n"
        "【鐵則 1 - 知識庫優先】：張序亞的履歷已經在你最上面的『動態系統變數』裡了！只要是問序亞的經歷、自我介紹，直接選 'FINISH'，絕對不准派給 Search_Agent 或 Math_Agent！\n"
        "【鐵則 2 - 數學觸發時機】：只有當使用者明確要求『計算、算錢、算時間』時才派給 Math_Agent。如果只是單純的閒聊，即使裡面有數字，也絕對不准算數學！\n"
        "【鐵則 3 - 動態連擊邏輯】：如果任務需要多步驟（例如：查股價再算股數），第一步『必須』先派給 'Search_Agent'。等下一輪你看到 Search_Agent 把股價找回來後，你才可以派給 'Math_Agent' 去計算。"
    ))
    decision = supervisor_llm.invoke([sys_msg] + messages)
    print(f"\n[主管 Supervisor] 決定派工給: {decision.next_node} (理由: {decision.reasoning})")
    return {"next_node": decision.next_node}

# 【2. 網路戰士房間】
class SearchQuery(BaseModel):
    query: str = Field(description="要丟給搜尋引擎的精準關鍵字")

def search_node(state: AgentState):
    print("[網路戰士 Search_Agent] 收到任務，準備出擊...")
    extractor = llm.with_structured_output(SearchQuery)
    # 【SA 強化】：要求戰士專注於台幣計價
    sys_msg = SystemMessage(content=(
        "你是一個關鍵字提取專家。請從對話中提取出最適合上網搜尋的精準關鍵字。\n"
        "如果使用者詢問台灣股票，請務必在關鍵字中加上『台灣股市』或『台幣』等字眼，以確保查到的是台股代號的價格，而非美股 ADR。"
    ))
    search_req = extractor.invoke([sys_msg] + state["messages"])
    
    print(f"[網路戰士 Search_Agent] 正在網路上揮劍尋找: {search_req.query}")
    from tools.web_search import search_web
    result = search_web(search_req.query)
    
    return {"messages": [AIMessage(name="Search_Agent", content=f"【網路搜尋結果】\n{result}")]}

# 【3. 算盤法師房間】
class MathExpression(BaseModel):
    expression: str = Field(description="要執行的純數學算式，例如 '1000000 / 2380'")

def math_node(state: AgentState):
    print("[算盤法師 Math_Agent] 收到任務，正在推導公式...")
    extractor = llm.with_structured_output(MathExpression)
    # 【SA 極限強化】：教導法師如何過濾網頁雜訊
    sys_msg = SystemMessage(content=(
        "你是一個極度嚴謹的物理與數學翻譯專家。\n"
        "【規則 1】：如果搜尋結果中出現多個股價數字，請優先採用『Yahoo股市』或標示為『收盤價/成交價』的最新數字，排除新聞標題預測的數字。\n"
        "【規則 2】：請仔細比對使用者的問題。如果使用者有 100 萬，算式裡就必須是 1000000。\n"
        "【規則 3】：注意單位一致性。如果你取得的股價是美金 (USD)，必須換算；優先使用台幣 (TWD) 報價。\n"
        "【規則 4】：將問題轉換為單純的 Python 數學算式，絕對不能包含任何中文字或特殊符號。"
    ))
    math_req = extractor.invoke([sys_msg] + state["messages"])
    
    print(f"[算盤法師 Math_Agent] 正在使用魔法計算機: {math_req.expression}")
    from tools.calculator import calculate_math
    result = calculate_math(math_req.expression)
    
    return {"messages": [AIMessage(name="Math_Agent", content=f"【計算機結果】\n{result}")]}

# 【4. 客服公關房間】
def final_answer_node(state: AgentState):
    print("[客服公關 Final_Answer] 資料收集完畢，正在撰寫最終回覆給客人...")
    
    # 【SA 防漏氣修復】：將背包裡的所有對話與調查結果「平面化」成純文字報告
    context_str = ""
    for msg in state["messages"]:
        # 抓出發言者的角色或名字
        role_name = msg.name if msg.name else msg.type
        context_str += f"[{role_name}]: {msg.content}\n\n"
        
    # 將整份報告塞進單一個 SystemMessage 中，確保 Llama 3 絕對不會格式錯亂
    sys_msg = SystemMessage(content=(
        "你是一位專業的 AI 助理。請閱讀以下的【調查過程紀錄】，並針對最後一個使用者的問題，用流暢、專業的繁體中文給出最終回答。\n"
        "【嚴格規定】：請自然地回答結果，絕對不要說出「根據 Math_Agent」、「從上述對話可見」或透露調查過程的生硬字眼。\n"
        "--------------------------\n"
        f"【調查過程紀錄】：\n{context_str}"
    ))
    
    # 乾淨俐落：只丟一個訊息給模型，保證 100% 穩定輸出
    response = llm.invoke([sys_msg])

# ==========================================
    # 🛡️ 【SA 終極防護】：切除 Llama 3 溢出的 "assistant" 標籤
    # ==========================================
    import re
    clean_text = response.content.strip()
    
    # 1. 使用 re.IGNORECASE 來確保不分大小寫的匹配 (相容性最高)
    # 2. 匹配 "assistant" 加上任何接續的空白、冒號或換行
    clean_text = re.sub(r'^assistant[:\s\n]*', '', clean_text, flags=re.IGNORECASE).strip()
    
    # 【SA 注意】：這裡原本寫錯了，不能回傳 response.content，必須回傳處理過的 clean_text！
    # ==========================================

    return {"messages": [AIMessage(name="Final_Answer", content=clean_text)]}


# ==========================================
# 🗺️ 第五區：畫地圖與建立動線 (Graph Edges)
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Search_Agent", search_node)
workflow.add_node("Math_Agent", math_node)
workflow.add_node("Final_Answer", final_answer_node)

workflow.add_edge(START, "Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next_node"],
    {
        "Search_Agent": "Search_Agent",
        "Math_Agent": "Math_Agent",
        "FINISH": "Final_Answer"
    }
)

workflow.add_edge("Search_Agent", "Supervisor")
workflow.add_edge("Math_Agent", "Supervisor")
workflow.add_edge("Final_Answer", END)

app_graph = workflow.compile()
print("[系統] 🗺️ 航空母艦地圖與動線建立完成！準備出航！")

# ==========================================
# 🎮 第六區：本機獨立測試區塊
# ==========================================
if __name__ == "__main__":
    print("\n========================================================")
    print("🚀 航空母艦戰鬥群：終極光速挑戰測試")
    print("========================================================")
    
    test_question = "如果光速是每秒 299792458 公尺，而地球到月球的距離大約是 384400 公里，請問光從地球走到月球需要多少秒？"
    print(f"👤 [使用者]: {test_question}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=test_question)]
    }
    
    final_answer_text = ""
    
    # 啟動航空母艦！
    for output in app_graph.stream(initial_state, {"recursion_limit": 10}):
        for key, value in output.items():
            print(f"--- 經過房間: {key} ---")
            # 【SA 修復】：直接從迴圈中抓取 Final_Answer 的結果，不需要 get_state
            if key == "Final_Answer":
                final_answer_text = value['messages'][-1].content
            
    print("\n========================================================")
    print(f"🤖 [最終回答]:\n{final_answer_text}")
    print("========================================================")
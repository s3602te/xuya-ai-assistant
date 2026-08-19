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
    reasoning: str = Field(description="請說明你的判斷邏輯。為什麼選擇這個節點？")
    next_node: Literal["Search_Agent", "Math_Agent", "FINISH"] = Field(
        description=(
            "1. Search_Agent：需要上網查詢未知資訊、最新數據時選擇。\n"
            "2. Math_Agent：有明確計算需求時選擇。\n"
            "3. FINISH：資料已備齊，可直接回答時選擇。"
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
    
    # 🛑 【Python 物理絕對防禦】：保護 GPU 免受無限迴圈之苦
    if len(messages) > 0:
        last_msg = messages[-1]
        if getattr(last_msg, "name", "") in ["Math_Agent", "Search_Agent"]:
            if any(error_kw in last_msg.content for error_kw in ["錯誤", "失敗", "異常", "Error"]):
                print("\n[系統守衛] 🛑 偵測到專家執行異常！Python 物理煞車已啟動，強制結案！")
                return {"next_node": "FINISH"}

    # 【SA 強制剝奪認知】：徹底封鎖大模型的自信心，強制它依賴工具
    sys_msg = SystemMessage(content=(
        "你是路由主管。你『絕對沒有』任何常識、歷史知識或數學能力！\n"
        "請嚴格遵守以下派工順序：\n"
        "1. 若問題是詢問張序亞的履歷，直接選 'FINISH'。\n"
        "2. 只要問題詢問『客觀事實、數據、現任人物、資本額』，你【絕對不准】憑記憶回答，【強制】第一步先派給 'Search_Agent' 查詢！\n"
        "3. 拿到 Search_Agent 回報的數字後，若需計算，【絕對不准】在理由中心算，【強制】派給 'Math_Agent'！\n"
        "4. 只有背包裡已經有【計算機結果】或【搜尋結果】時，才准選 'FINISH'。"
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
        "你是物理與數學翻譯專家。\n"
        "請仔細閱讀 Search_Agent 找回來的數字（注意單位，1億 = 100000000），並將問題轉換為正確的 Python 數學算式。\n"
        "絕對不能包含中文字。"
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
        role_name = getattr(msg, "name", msg.type)
        context_str += f"[{role_name}]: {msg.content}\n\n"
        
    # 【SA 記憶隔離】：強制客服公關只針對「最後一個問題」作答
    sys_msg = SystemMessage(content=(
        "你是一位專業的 AI 助理。請閱讀以下的【調查過程紀錄】。\n"
        "【最高鐵則 1】：請『只針對最後一個問題』給出解答，絕對不准重複或提及之前歷史對話的答案（例如不要把前一題的總統跟這題的高鐵混在一起）！\n"
        "【最高鐵則 2】：請自然地陳述計算或搜尋結果，絕對不要說出「根據 Math_Agent」等內部字眼。\n"
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

# 【SA 精準防呆邏輯】：只攔截死迴圈，不干涉正常連擊
def routing_logic(state: AgentState):
    next_node = state["next_node"]
    
    if len(state["messages"]) > 0:
        last_msg = state["messages"][-1]
        
        # 真正的物理煞車：嚴格禁止主管「連續兩步」派給同一個專家 (防止卡死在同一個房間)
        # 但允許 Search -> Math -> Search 這種跨房間的連擊！
        if getattr(last_msg, "name", "") == next_node and next_node != "FINISH":
            print(f"\n[系統守衛] 🛑 偵測到主管試圖連續重複呼叫 {next_node}！物理煞車啟動，強制結案！")
            return "Final_Answer"
            
    # 如果主管判斷任務完成，就正常走向客服公關
    if next_node == "FINISH":
        return "Final_Answer"
    return next_node

# 【正確】：將判斷條件綁定為 routing_logic，確保 Python 物理煞車生效！
workflow.add_conditional_edges(
    "Supervisor",
    routing_logic,  # <--- 注意這裡！替換成我們寫好的函式名稱
    {
        "Search_Agent": "Search_Agent",
        "Math_Agent": "Math_Agent",
        "Final_Answer": "Final_Answer" # <--- 因為 routing_logic 回傳的是 "Final_Answer"
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
// ============================
// 元件與靜態資源引入開始
// ============================
// 1. 引入 React 核心模組，用於狀態管理與生命週期控制
import { useState, useEffect, useRef } from 'react'
// 2. 引入 mermaid 套件，用於渲染動態架構圖
import mermaid from 'mermaid'
// 3. 匯入 V1.0 靜態圖片資源
import archImageV1 from './assets/ivtc-line-architecture.png'
// 4. 匯入 V2.0 靜態圖片資源：改用 PNG 格式，解決 SVG 透明背景在深色模式下變成全黑的問題
import archImageV2 from './assets/my-line-architecture.png'
// 5. 匯入 V3.0 多智能體靜態圖片資源 (根據你提供的檔名)
import archImageV3 from './assets/multi-agent+mcp-architecture.png'
// ============================
// 元件與靜態資源引入結束
// ============================

export default function Architecture() {
  // ============================
  // 狀態管理與 DOM 參考開始
  // ============================
  // 1. 宣告容器的 DOM 參考，用於後續控制內部滾動條位置
  const containerRef = useRef(null);

  // 2. 追蹤使用者在輸入框中鍵入的密碼
  const [password, setPassword] = useState('')
  // 3. 紀錄當前頁面是否已成功解鎖
  const [isUnlocked, setIsUnlocked] = useState(false)
  // 4. 紀錄密碼驗證失敗時要顯示的錯誤提示訊息
  const [errorMsg, setErrorMsg] = useState('')

  // 5. 追蹤目前輪播展示的架構版本 (0 = V1.0, 1 = V2.0)
  const [activeVersion, setActiveVersion] = useState(0)
  // 6. 儲存要全螢幕放大的靜態圖片來源
  const [lightboxImage, setLightboxImage] = useState(null)
  // 7. 控制 Mermaid 終端機是否開啟全螢幕
  const [isMermaidFullscreen, setIsMermaidFullscreen] = useState(false)
  // ============================
  // 狀態管理與 DOM 參考結束
  // ============================


  // ============================
  // Mermaid 防彈版語法變數定義開始
  // ============================
  // 1. 定義 V1.0 LINE 企業客服架構
  const mermaidCodeV1 = `
graph TD
    User((使用者)) -->|"發送訊息/圖片"| LINE[LINE Platform]
    LINE -->|"Webhook"| IIS[IIS 伺服器]
    IIS -->|"轉發 API"| Core[Python App.py]

    subgraph sg1 ["前置處理與防呆緩衝"]
        Core --> Buffer[多執行緒訊息水桶]
        Buffer -->|"純文字 5 秒 / 圖片 10 秒"| Guard[Regex 財務防護網 & 狀態機]
    end

    subgraph sg2 ["核心雙軌檢索 - RAG 路由"]
        Guard --> HasImage{是否包含圖片?}
        HasImage -->|"純文字"| Router{Stage 1: 檢索路由}
        
        Router -->|"優先查找"| DB_A[("ChromaDB 軌道 A<br/>CSV 手動高精準")]
        DB_A -->|"L2 距離 < 2.0 (命中)"| Ans_A[直接回傳標準解答]
        
        DB_A -->|"未命中 / 分數過大"| DB_B[("ChromaDB 軌道 B<br/>PDF 自動擴展")]
        DB_B -->|"提取 Top-K 參考段落"| Ans_B[彙整參考知識]
    end

    subgraph sg3 ["多模態 AI 生成 - Ollama"]
        Ans_B --> LLM[IVTC 語言模型]
        HasImage -->|"圖片+文字"| Vision[IVTC_Vision 多模態模型]
    end

    Ans_A --> Output[整合最終回覆]
    LLM --> Output
    Vision --> Output

    Output -->|"Reply API"| LINE
    LINE -->|"傳送解答"| User
  `;

  // 2. 定義 V2.0 Web 全端 AI 助理架構
  const mermaidCodeV2 = `
graph TD
    classDef frontend fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff;
    classDef backend fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff;
    classDef database fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff;
    classDef ai fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff;

    subgraph ClientLayer ["用戶端層 (Client Layer)"]
        UI["React + Vite 前端<br/>(Chatroom.jsx)"]:::frontend
    end

    subgraph ServerLayer ["後端邏輯層 (Backend - app.py)"]
        API["Flask API 路由<br/>(/api/web_chat)"]:::backend
        RAG{"雙軌 RAG 路由模組<br/>(search_knowledge)"}:::backend
        Prompt["Prompt 組合器<br/>(上下文與防護網)"]:::backend
    end

    subgraph DataLayer ["資料持久層 (Data Layer)"]
        SQLite[("SQLite 關聯資料庫<br/>(歷史對話記憶庫)")]:::database
        DB_A[("ChromaDB 軌道 A<br/>(xuya_qa_manual 高精準)")]:::database
        DB_B[("ChromaDB 軌道 B<br/>(xuya_qa_auto 自動擴展)")]:::database
    end

    subgraph AILayer ["AI 引擎層 (Local)"]
        Ollama["Ollama 引擎<br/>(XUYA:latest)"]:::ai
    end

    UI -->|"1. POST 發送訊息 (含 session_id)"| API
    API -->|"2. 寫入用戶提問 & 讀取歷史對話"| SQLite
    API -->|"3. 語意向量化與搜尋"| RAG
    
    RAG -->|"4a. 優先查找 (L2 < 2.0 命中)"| DB_A
    RAG -.->|"4b. 若未命中則 Fallback 查找"| DB_B
    
    DB_A -->|"回傳 100% 標準解答"| Prompt
    DB_B -.->|"提取 Top-K 參考段落"| Prompt
    SQLite -->|"傳入歷史上下文 (Context History)"| Prompt

    Prompt -->|"5. 組合提示詞並呼叫 API"| Ollama
    Ollama -->|"6. 回傳生成的回答"| API
    
    API -->|"7. 寫入 AI 回答紀錄"| SQLite
    API -->|"8. 回傳 JSON Response"| UI
  `;

  // 3. 定義 V3.0 多智能體架構
  const mermaidCodeV3 = `
flowchart TD
    %% ==========================================
    %% 全域樣式與色塊定義
    %% ==========================================
    classDef cicd fill:#EDE7F6,stroke:#5E35B1,stroke-width:2px,color:#311B92;
    classDef entry fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef backend fill:#E0F2F1,stroke:#00695C,stroke-width:2px,color:#004D40;
    classDef agent fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C;
    classDef infra fill:#FBE9E7,stroke:#D84315,stroke-width:2px,color:#4E342E;

    %% ==========================================
    %% 階段 0：自動化建置與容器交付
    %% ==========================================
    subgraph S0 ["【Stage 0】CI/CD 自動化建置與極速交付管線 (與執行期解耦)"]
        A1["0-1. Git Push 觸發 GitHub Actions"]:::cicd --> A2["0-2. 自動編譯 React 前端並打包入 Flask 後端"]:::cicd
        A2 --> A3["0-3. 封裝 Docker 映像檔並推至 Docker Hub"]:::cicd
        A3 --> A4["0-4. 目標機器一鍵 Docker Compose 啟動 (掛載 Volume 持久化)"]:::cicd
    end

    %% ==========================================
    %% 階段 1：使用者統一接入點
    %% ==========================================
    subgraph S1 ["【Stage 1】前端統一入口 (Web 網址 / LINE LIFF 內嵌)"]
        U1(("👤 使用者 / 面試官")):::entry
        U1 -->|點擊 LINE 選單或直接開網址| E1["React SPA 前端介面 (支援 LIFF 容器與獨立瀏覽器)"]:::entry
        E1 -->|Cloudflare Tunnel / ngrok 穿透| E2["Step 1. WebSocket 雙向連線 / REST API 請求"]:::entry
    end

    %% ==========================================
    %% 階段 2：後端穩定性與狀態緩衝
    %% ==========================================
    subgraph S2 ["【Stage 2】後端微服務與狀態緩衝層 (Flask + SQLite)"]
        E2 --> B1["Step 2. 多執行緒訊息水桶 (Threading 緩衝，防止碎語高頻請求)"]:::backend
        B1 --> B2["Step 3. 狀態機與對話管理 (Session 綁定 & SQLite 紀錄)"]:::backend
        B2 --> B3["Step 4. RAG 智慧弓箭手檢索 (雙軌 ChromaDB：CSV 精準軌 / PDF 擴展軌)"]:::backend
    end

    %% ==========================================
    %% 階段 3：LangGraph 多智能體大腦
    %% ==========================================
    subgraph S3 ["【Stage 3】LangGraph 多智能體協同大腦 (航空母艦戰鬥群)"]
        B3 --> G1["Step 5-1. Planner 規劃官 (三層保底拆解 + 硬規則攔截，產出任務清單)"]:::agent
        G1 --> G2["Step 5-2. Supervisor 路由主管 (純 Python 狀態進度派工，零 Token 消耗)"]:::agent
        
        G2 -->|分派聯網任務| W1["Search_Agent 網路戰士 (呼叫 MCP: Brave Search + 記憶體快取)"]:::agent
        G2 -->|分派計算任務| W2["Math_Agent 算盤法師 (呼叫 MCP: AST 安全計算機 + 鏈式推導)"]:::agent
        
        W1 --> C1{"網路鑑定士 Grader (狀態旗標檢查)"}:::agent
        C1 -->|未達標重試| W1
        C1 -->|合格登錄事實帳本| G2

        W2 --> C2{"算盤鑑定士 Grader (數字溯源檢驗)"}:::agent
        C2 -->|未達標重試| W2
        C2 -->|合格登錄事實帳本| G2

        G2 -->|所有任務完成 / 無需工具| G3["Step 5-3. Final_Answer 盜賊客服 (整合帳本、去內部詞、輸出層數字溯源)"]:::agent
    end

    %% ==========================================
    %% 階段 4：底層算力與回傳
    %% ==========================================
    subgraph S4 ["【Stage 4】底層雙模型與即時推播"]
        G1 -.->|調用| M1["XUYA 主模型 (8B 自然語言統整)"]:::infra
        G3 -.->|調用| M1
        W1 -.->|數值萃取| M2["Gemma 驗證小模型 (4B 結構化抽取)"]:::infra
        W2 -.->|算式翻譯| M2
        
        G3 -->|Step 6. 格式化解答產生| R1["WebSocket 即時推播 (廣播至前端 UI)"]:::backend
        R1 -->|Step 7. 無延遲渲染答案| U1
    end

    %% 連接交付到服務啟動
    A4 -.->|服務運行於 Port 5000| B1
  `;
  // ============================
  // Mermaid 防彈版語法變數定義結束
  // ============================


  // ============================
  // Mermaid 初始化與動態重繪邏輯開始
  // ============================
  useEffect(() => {
    // 1. 判斷畫面是否解鎖，確保 DOM 存在才啟動引擎
    if (isUnlocked) {
      // 2. 初始化 Mermaid 配置：深色主題、允許較寬鬆的安全層級
      mermaid.initialize({
        startOnLoad: true,
        theme: 'dark',
        securityLevel: 'loose',
        // 3. 取消強制縮放限制，確保 Mermaid 以 100% 原始解析度清晰繪製
        flowchart: { useMaxWidth: false }
      });
      // 4. 給予極短延遲確保 DOM 已更新，強制解析畫面上的 Mermaid 標籤
      setTimeout(() => {
        mermaid.contentLoaded();
      }, 50);
    }
    // 5. 監聽版本切換與全螢幕狀態，變更時觸發重繪
  }, [isUnlocked, activeVersion, isMermaidFullscreen]);
  // ============================
  // Mermaid 初始化與動態重繪邏輯結束
  // ============================


// ============================
  // 視窗與容器滾動控制開始
  // ============================
  useEffect(() => {
    // 1. 強制隱藏瀏覽器外層捲軸，避免產生雙捲軸
    document.body.style.overflow = 'hidden';

    // 2. 離開頁面時自動復原，確保不影響其他頁面
    return () => {
      document.body.style.overflow = '';
    };
  }, []);

  useEffect(() => {
    // 3. 處理行動裝置或切換頁面時的捲軸位置殘留問題
    window.scrollTo(0, 0);

    // 4. 若頁面已解鎖且內部容器成功掛載，將內部容器重置回頂部
    if (isUnlocked && containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [isUnlocked]);
  // ============================
  // 視窗與容器滾動控制結束
  // ============================


  // ============================
  // 密碼驗證與解鎖邏輯開始
  // ============================
  const handleUnlock = () => {
    // 1. 定義高強度密碼的正規表達式 (Regex)
    // 規則條件：
    // - (?=.*[a-z]) : 至少包含一個小寫英文字母
    // - (?=.*[A-Z]) : 至少包含一個大寫英文字母
    // - (?=.*\d)    : 至少包含一個數字
    // - (?=.*[!@#$%^&*]) : 至少包含一個特殊符號
    // - .{8,12}     : 總長度限制為 8 到 12 個字元
    const regex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*]).{8,12}$/

    // 2. 執行驗證流程
    if (password === 'Abcd0304!') {
      // 2-1. 驗證通過指定授權密碼：畫面歸零、設定解鎖狀態、清空錯誤訊息
      window.scrollTo(0, 0);
      setIsUnlocked(true)
      setErrorMsg('')
    } else if (!regex.test(password)) {
      // 2-2. 驗證格式失敗：提示密碼強度與格式要求
      setErrorMsg('密碼格式錯誤：需 8-12 位，含大小寫字母、數字與特殊符號。')
    } else {
      // 2-3. 格式正確但密碼錯誤：提示向擁有者索取
      setErrorMsg('密碼驗證失敗，請向求職者索取正確密碼！')
    }
  }
  // ============================
  // 密碼驗證與解鎖邏輯結束
  // ============================


  // ============================
  // 畫面 A：權限驗證 (上鎖狀態) 渲染開始
  // ============================
  if (!isUnlocked) {
    return (
      // 1. 渲染滿版的登入背景與容器
      <div className="w-full h-[100dvh] flex flex-col items-center justify-center bg-gray-900 p-4">
        <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl max-w-md w-full border border-gray-700 text-center">
          <div className="text-5xl mb-4">🔒</div>
          <h2 className="text-2xl font-bold text-white mb-2">機密架構文件</h2>
          <p className="text-gray-400 text-sm mb-6">此區域僅限面試環節展示，請輸入授權密碼解鎖。</p>

          {/* 2. 密碼輸入框區塊 */}
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleUnlock()}
            placeholder="請輸入密碼..."
            className="w-full bg-gray-900 text-white border border-gray-600 rounded-lg px-4 py-3 mb-4 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
          />

          {/* 3. 錯誤訊息提示區：根據 errorMsg 狀態判斷，有錯誤內容時才會渲染出提示文字 */}
          {errorMsg && <p className="text-red-400 text-sm mb-4 text-left">{errorMsg}</p>}

          {/* 4. 觸發驗證與解鎖按鈕 */}
          <button
            onClick={handleUnlock}
            className="w-full bg-blue-600 text-white font-bold py-3 rounded-lg hover:bg-blue-700 transition-colors shadow-lg shadow-blue-900/50"
          >
            驗證並解鎖
          </button>
        </div>
      </div>
    )
  }
  // ============================
  // 畫面 A：權限驗證 (上鎖狀態) 渲染結束
  // ============================


  // ============================
  // 畫面 B：系統架構圖 (解鎖狀態) 渲染開始
  // ============================
return (
    // 1. 最外層主視窗容器：設定 h-screen 與 overflow-y-auto，並加上 pb-24 保證能滑過第 10 張卡片
    <div ref={containerRef} className="w-full h-screen p-6 md:p-8 pt-14 md:pt-8 pb-24 bg-gray-50 overflow-y-auto animate-fade-in transition-all duration-300">

      {/* ============================ */}
      {/* 靜態圖片點擊放大全螢幕 (Lightbox) 彈窗區塊開始 */}
      {/* ============================ */}
      {lightboxImage && (
        // 1. 全螢幕黑色半透明遮罩背景
        <div
          className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 cursor-zoom-out animate-fade-in"
          onClick={() => setLightboxImage(null)}
        >
          {/* 2. 置中的放大圖片 */}
          <img
            src={lightboxImage}
            alt="放大架構圖"
            className="max-w-full max-h-full object-contain shadow-2xl rounded-lg"
          />
          {/* 3. 右上角關閉圖示 */}
          <div className="absolute top-6 right-6 text-white text-3xl font-bold">✕</div>
        </div>
      )}
      {/* ============================ */}
      {/* 靜態圖片點擊放大全螢幕 (Lightbox) 彈窗區塊結束 */}
      {/* ============================ */}

      {/* 2. 內容最大寬度與置中容器 */}
      <div className="max-w-7xl 2xl:max-w-[1600px] mx-auto transition-all duration-300">

{/* ============================ */}
        {/* 標題與版本切換按鈕區塊開始        */}
        {/* ============================ */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 pb-4 gap-4">
          {/* 1. 頁面主標題 */}
          <h1 className="text-3xl font-extrabold text-gray-900 border-b-4 border-blue-500 pb-2 inline-block self-start">
            系統架構藍圖 (System Architecture)
          </h1>
          
          {/* 2. 版本切換選單 (Dropdown) */}
          <div className="relative w-full md:w-auto self-start md:self-auto">
            <select
              value={activeVersion}
              onChange={(e) => setActiveVersion(Number(e.target.value))}
              className="appearance-none w-full md:w-72 bg-white border-2 border-gray-300 text-gray-800 py-2.5 px-4 pr-10 rounded-lg font-bold shadow-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-colors cursor-pointer text-base md:text-sm"
            >
              <option value={0}>V1.0 LINE 企業客服</option>
              <option value={1}>V2.0 Web 全端 AI 助理</option>
              <option value={2}>V3.0 多智能體協作 (Agentic)</option>
            </select>
            {/* 自訂下拉箭頭圖示 */}
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
              <svg className="fill-current h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
              </svg>
            </div>
          </div>
        </div>
        {/* ============================ */}
        {/* 標題與版本切換按鈕區塊結束        */}
        {/* ============================ */}

{/* ============================================================ */}
        {/* 圖表展示雙欄網格（桌機版 lg: 左右並排 / 手機版單欄堆疊）開始 */}
        {/* ============================================================ */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8 items-stretch">          
          {/* 左側：PNG 靜態架構總覽 */}
          <div className="w-full h-[520px] bg-white rounded-xl shadow-xl overflow-hidden border border-gray-200 flex flex-col transition-all duration-300">
            {/* 1. 圖片頂部標題列 */}
            <div className="bg-gray-100 px-4 py-2 border-b border-gray-200 text-sm font-bold text-gray-600 flex justify-between items-center shrink-0">
              <span>
                {activeVersion === 0 && "V1.0 LINE 企業客服 (靜態總覽)"}
                {activeVersion === 1 && "V2.0 Web 全端 AI 助理 (靜態總覽)"}
                {activeVersion === 2 && "V3.0 多智能體協作架構 (靜態總覽)"}
              </span>
              <span
                className="text-xs text-blue-500 cursor-pointer hover:underline font-semibold"
                onClick={() => setLightboxImage(activeVersion === 0 ? archImageV1 : (activeVersion === 1 ? archImageV2 : archImageV3))}
              >
                🔍 點擊放大
              </span>
            </div>
            {/* 2. 圖片展示區塊 */}
            <div className="bg-gray-50 flex-1 flex items-center justify-center p-2 overflow-hidden">
              <img
                src={activeVersion === 0 ? archImageV1 : (activeVersion === 1 ? archImageV2 : archImageV3)}
                alt="專案系統架構圖"
                className="w-full h-full object-contain block animate-fade-in cursor-zoom-in hover:opacity-95 transition-opacity"
                onClick={() => setLightboxImage(activeVersion === 0 ? archImageV1 : (activeVersion === 1 ? archImageV2 : archImageV3))}
              />
            </div>
          </div>

          {/* 右側：Mermaid 終端機即時預覽 */}
          <div className={isMermaidFullscreen
            ? "fixed inset-0 z-[80] bg-black p-4 flex flex-col animate-fade-in"
            : "w-full h-[520px] bg-black rounded-xl border border-gray-700 shadow-xl overflow-hidden flex flex-col transition-all duration-300"
          }>
            {/* 1. 終端機頂部控制列 */}
            <div className="bg-gray-800 px-4 py-2 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500 cursor-pointer" onClick={() => setIsMermaidFullscreen(false)}></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500 cursor-pointer" onClick={() => setIsMermaidFullscreen(!isMermaidFullscreen)}></div>
                <span className="ml-4 text-gray-400 text-xs font-mono">
                  {activeVersion === 0 && "architecture_v1_line.md"}
                  {activeVersion === 1 && "architecture_v2_web.md"}
                  {activeVersion === 2 && "architecture_v3_agent.md"} - Mermaid Live Preview
                </span>
              </div>
              {/* 2. 全螢幕切換按鈕 */}
              <button
                onClick={() => setIsMermaidFullscreen(!isMermaidFullscreen)}
                className="text-gray-400 hover:text-white text-lg transition-colors font-bold"
                title={isMermaidFullscreen ? "還原視窗" : "全螢幕放大"}
              >
                {isMermaidFullscreen ? "✖" : "⛶"}
              </button>
            </div>

            {/* 3. 滾動區塊 */}
            <div className="flex-1 p-4 overflow-auto bg-gray-900 cursor-move">
              <div className="w-fit mx-auto min-w-full flex justify-center">
                <pre key={`${activeVersion}-${isMermaidFullscreen}`} className="mermaid text-xs md:text-sm animate-fade-in">
                  {activeVersion === 0 && mermaidCodeV1}
                  {activeVersion === 1 && mermaidCodeV2}
                  {activeVersion === 2 && mermaidCodeV3}
                </pre>
              </div>
            </div>
          </div>         
        </div>
        {/* ============================================================ */}
        {/* 圖表展示雙欄網格（桌機版 lg: 左右並排 / 手機版單欄堆疊）結束 */}
        {/* ============================================================ */}

        {/* ============================ */}
        {/* 核心技術說明網格區塊 (原始 HTML 結構) 開始 */}
        {/* ============================ */}

        {/* 1. 當 activeVersion 為 0 時，渲染 V1.0 的技術說明網格 */}
        {activeVersion === 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-fade-in">
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-blue-500">
              <h3 className="text-xl font-bold mb-2">1. LINE Bot API</h3>
              <p className="text-gray-600 text-sm leading-relaxed">處理使用者的圖文訊息，並透過 Webhook 將事件安全地轉發至內部網路。</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-green-500">
              <h3 className="text-xl font-bold mb-2">2. IIS 伺服器 (C#)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">作為企業防火牆內的前線接收端，進行基礎的流量過濾與格式轉換。</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-purple-500">
              <h3 className="text-xl font-bold mb-2">3. 多執行緒緩衝 (Threading)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">針對 LINE 使用者常有的「碎語」習慣，實作 5~10 秒的延遲收容機制，避免頻繁觸發 AI。</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-orange-500">
              <h3 className="text-xl font-bold mb-2">4. 狀態機管理</h3>
              <p className="text-gray-600 text-sm leading-relaxed">結合正則表達式 (Regex)，實作「AI 自動服務」、「等待統編」與「真人接手」等多重狀態切換。</p>
            </div>
          </div>
        )}

        {/* 2. 當 activeVersion 為 1 時，渲染 V2.0 的技術說明網格 */}
        {activeVersion === 1 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-fade-in">
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-blue-500">
              <h3 className="text-xl font-bold mb-2">1. 前端介面 (React + Vite)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">負責處理使用者輸入、狀態管理 (State) 與條件渲染 (Conditional Rendering)，並使用 Tailwind 實現完美 RWD。</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-green-500">
              <h3 className="text-xl font-bold mb-2">2. 網路穿透 (Ngrok)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">作為安全的 API Gateway，將外部的 HTTPS 請求精準路由至本地端的 Python 服務伺服器。</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-purple-500">
              <h3 className="text-xl font-bold mb-2">3. 後端邏輯 (Python)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">處理 API 路由與跨域請求 (CORS)，並負責將資料整理後對接底層的 AI 模型。</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-orange-500">
              <h3 className="text-xl font-bold mb-2">4. 大型語言模型 (Ollama)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">本地端運行的 AI 引擎，提供低延遲、高隱私的自然語言生成服務。</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-yellow-500">
              <h3 className="text-xl font-bold mb-2">5. 向量檢索 (ChromaDB 雙軌 RAG)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">實作高精準 (A軌) 與自動擴展 (B軌) 的雙軌檢索機制，透過 Cosine Similarity 嚴謹比對，大幅降低大型語言模型的幻覺 (Hallucination)。</p>
            </div>
          </div>
        )}

        {/* 3. 當 activeVersion 為 2 時，渲染 V3.0 的技術說明網格 */}
        {activeVersion === 2 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-fade-in">
            {/* 模組 1：CI/CD 與極速交付 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-indigo-500">
              <h3 className="text-xl font-bold mb-2">1. CI/CD 自動化建置與 Docker 交付</h3>
              <p className="text-gray-600 text-sm leading-relaxed">透過 GitHub Actions 建置自動化流水線，編譯 React 前端並注入 Flask 後端打包成映像檔推至 Docker Hub，實現目標主機一鍵 Docker Compose 掛載持久化 Volume 啟動。</p>
            </div>

            {/* 模組 2：多執行緒緩衝與雙向通訊 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-teal-500">
              <h3 className="text-xl font-bold mb-2">2. WebSocket 即時通訊與多執行緒訊息水桶</h3>
              <p className="text-gray-600 text-sm leading-relaxed">針對用戶碎語實作 5~10 秒的滑動視窗緩衝機制，防止高頻併發請求擊垮 AI；透過 WebSocket 實現雙向非同步推播，並內建「真人客服接管」狀態機與計時歸零重置機制。</p>
            </div>

            {/* 模組 3：RAG 智慧弓箭手 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-purple-500">
              <h3 className="text-xl font-bold mb-2">3. RAG 智慧弓箭手（雙軌檢索與主題投票）</h3>
              <p className="text-gray-600 text-sm leading-relaxed">建置手動精準軌（CSV）與自動擴展軌（PDF）雙軌 ChromaDB。實作「多候選主題投票」與 0.85 距離動態過濾，並採「上下文獨立艙室」設計，避免知識庫雜訊污染其他 Agent 的關鍵字抽取。</p>
            </div>

            {/* 模組 4：Planner 規劃官 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-amber-500">
              <h3 className="text-xl font-bold mb-2">4. Planner 規劃官（三層保底與硬規則攔截）</h3>
              <p className="text-gray-600 text-sm leading-relaxed">負責任務開場解析。實作「巢狀 JSON ➔ 兩段式目標提取 ➔ 純 Python 正則啟發式」三層降級保險絲，內建 Prompt 回音雜訊清洗，並在前置硬編碼純數學規則攔截，杜絕無效外部 API 調用。</p>
            </div>

            {/* 模組 5：Supervisor 路由主管 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-red-500">
              <h3 className="text-xl font-bold mb-2">5. Supervisor 路由主管（零 Token 派工）</h3>
              <p className="text-gray-600 text-sm leading-relaxed">降級傳統 LLM 路由為純 Python 狀態機。嚴格依據「任務清單 (Plan)」與「事實帳本 (Facts)」管理執行進度，達成 0 運算 Token 消耗、0 幻覺與硬性去重派工控制。</p>
            </div>

            {/* 模組 6：Search_Agent 網路戰士 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-blue-500">
              <h3 className="text-xl font-bold mb-2">6. Search_Agent 網路戰士（快取與數值萃取）</h3>
              <p className="text-gray-600 text-sm leading-relaxed">調用 Brave Search API 獲取外部即時數據。內建進程內記憶體快取（TTL 600s）與自動改寫重試機制，並調用小模型精準萃取結構化數值，直接登錄進事實帳本供下游使用。</p>
            </div>

            {/* 模組 7：Math_Agent 算盤法師 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-green-500">
              <h3 className="text-xl font-bold mb-2">7. Math_Agent 算盤法師（AST 鏈式推導）</h3>
              <p className="text-gray-600 text-sm leading-relaxed">將運算任務轉譯為多行 Python 腳本。採用 AST 抽象語法樹沙盒評估與嚴格白名單（開放 comb/perm），支援多變數鏈式依賴推導，徹底封鎖 LLM 心算幻覺。</p>
            </div>

            {/* 模組 8：雙軌 Grader 鑑定士 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-orange-500">
              <h3 className="text-xl font-bold mb-2">8. 雙軌 Grader 鑑定士（狀態旗標與物理防禦）</h3>
              <p className="text-gray-600 text-sm leading-relaxed">在專家節點出口實作物理檢驗閘門。捨棄脆弱的內文關鍵字掃描，改以【STATUS:OK/FAIL】旗標判定成敗；算盤鑑定士額外執行「數字溯源」與「數字完整性」驗證，未達標則觸發帶因重試。</p>
            </div>

            {/* 模組 9：Final_Answer 盜賊客服 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-rose-500">
              <h3 className="text-xl font-bold mb-2">9. Final_Answer 盜賊客服（雙套提示與輸出溯源）</h3>
              <p className="text-gray-600 text-sm leading-relaxed">依檢索狀況動態切換極簡/完整提示詞，並以正則清理內部骨架詞彙；輸出端實作「最後一棒數字溯源防線」，偵測到未授權數據即觸發重寫或降級套用確定性模板。</p>
            </div>

            {/* 模組 10：健康檢查與可觀測性守衛 */}
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-cyan-500">
              <h3 className="text-xl font-bold mb-2">10. 系統健康檢查與超時守衛</h3>
              <p className="text-gray-600 text-sm leading-relaxed">提供 `/api/health` 端點，一鍵掃描 Ollama 模型就緒狀態、Brave 金鑰、ChromaDB 向量庫與 LangGraph 圖編譯；搭配 ThreadPool 執行緒超時熔斷保護，避免單一複雜請求阻塞服務。</p>
            </div>
          </div>
        )}
        {/* ============================ */}
        {/* 核心技術說明網格區塊 (原始 HTML 結構) 結束 */}
        {/* ============================ */}

      </div>
    </div>
  )
}
// ============================
// 畫面 B：系統架構圖 (解鎖狀態) 渲染結束
// ============================
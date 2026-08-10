// ============================
// 元件與靜態資源引入開始
// ============================
import { useState, useEffect, useRef } from 'react'
import mermaid from 'mermaid'                                                   // 1. 引入 mermaid 套件，用於渲染動態架構圖
import archImageV1 from './assets/創群LINE架構.png' // 2. 匯入 V1.0 靜態圖片資源
import archImageV2 from './assets/我的LINE架構.png'                               // 3. 【修復】改用 PNG 格式，解決 SVG 透明背景在深色模式下變成全黑的問題
// ============================
// 元件與靜態資源引入結束
// ============================

export default function Architecture() {
  // ============================
  // 狀態管理與 DOM 參考開始
  // ============================
  const containerRef = useRef(null); // 1. 宣告容器的 DOM 參考，用於後續控制內部滾動條位置

  const [password, setPassword] = useState('')        // 2. 追蹤使用者在輸入框中鍵入的密碼
  const [isUnlocked, setIsUnlocked] = useState(false) // 3. 紀錄當前頁面是否已成功解鎖
  const [errorMsg, setErrorMsg] = useState('')        // 4. 紀錄密碼驗證失敗時要顯示的錯誤提示訊息

  const [activeVersion, setActiveVersion] = useState(0)                 // 5. 追蹤目前輪播展示的架構版本 (0 = V1.0, 1 = V2.0)
  const [lightboxImage, setLightboxImage] = useState(null)              // 6. 儲存要全螢幕放大的靜態圖片來源
  const [isMermaidFullscreen, setIsMermaidFullscreen] = useState(false) // 7. 控制 Mermaid 終端機是否開啟全螢幕
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
  // ============================
  // Mermaid 防彈版語法變數定義結束
  // ============================


  // ============================
  // Mermaid 初始化與動態重繪邏輯開始
  // ============================
  useEffect(() => {
    // 1. 判斷畫面是否解鎖，確保 DOM 存在才啟動引擎
    if (isUnlocked) {
      mermaid.initialize({
        startOnLoad: true,
        theme: 'dark', // 2. 設定深色主題
        securityLevel: 'loose',
        // 3. 取消強制縮放限制，確保 Mermaid 以 100% 原始解析度清晰繪製
        flowchart: { useMaxWidth: false } 
      });
      // 4. 給予極短延遲確保 DOM 已更新，強制解析畫面上的 Mermaid 標籤
      setTimeout(() => {
        mermaid.contentLoaded();
      }, 50);
    }
  }, [isUnlocked, activeVersion, isMermaidFullscreen]); // 5. 監聽版本與全螢幕狀態，變更時觸發重繪
  // ============================
  // Mermaid 初始化與動態重繪邏輯結束
  // ============================


  // ============================
  // 視窗與容器滾動控制開始
  // ============================
  // 處理行動裝置或切換頁面時的捲軸位置殘留問題
  useEffect(() => {
    // 1. 強制將全域視窗滾動至最頂部
    window.scrollTo(0, 0);
    
    // 2. 若頁面已解鎖且內部容器成功掛載，將內部容器也重置回頂部
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
    // 定義高強度密碼的正規表達式 (Regex)
    // 規則條件：
    // 1. (?=.*[a-z]) : 至少包含一個小寫英文字母
    // 2. (?=.*[A-Z]) : 至少包含一個大寫英文字母
    // 3. (?=.*\d)    : 至少包含一個數字
    // 4. (?=.*[!@#$%^&*]) : 至少包含一個特殊符號
    // 5. .{8,12}     : 總長度限制為 8 到 12 個字元
    const regex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*]).{8,12}$/

    // 執行驗證流程
    if (password === 'Abcd0304!') {
        // 1. 驗證通過指定授權密碼：畫面歸零、設定解鎖狀態、清空錯誤訊息
        window.scrollTo(0, 0); 
        setIsUnlocked(true)
        setErrorMsg('')
    } else if (!regex.test(password)) {
        // 2. 驗證格式失敗：提示密碼強度與格式要求
        setErrorMsg('密碼格式錯誤：需 8-12 位，含大小寫字母、數字與特殊符號。')
    } else {
        // 3. 格式正確但密碼錯誤：提示向擁有者索取
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
      <div className="w-full h-[100dvh] flex flex-col items-center justify-center bg-gray-900 p-4">
        <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl max-w-md w-full border border-gray-700 text-center">
          <div className="text-5xl mb-4">🔒</div>
          <h2 className="text-2xl font-bold text-white mb-2">機密架構文件</h2>
          <p className="text-gray-400 text-sm mb-6">此區域僅限面試環節展示，請輸入授權密碼解鎖。</p>
          
          <input 
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleUnlock()}
            placeholder="請輸入密碼..."
            className="w-full bg-gray-900 text-white border border-gray-600 rounded-lg px-4 py-3 mb-4 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
          />
          
          {/* 根據 errorMsg 狀態判斷：有錯誤內容時才會渲染出提示文字 */}
          {errorMsg && <p className="text-red-400 text-sm mb-4 text-left">{errorMsg}</p>}
          
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
  // 將 containerRef 綁定於最外層容器，確保滾動條重置邏輯生效
  return (
    <div ref={containerRef} className="w-full h-[100dvh] p-8 pt-14 md:pt-8 bg-gray-50 overflow-y-auto animate-fade-in transition-all duration-300">
      
      {/* ============================ */}
      {/* 靜態圖片點擊放大全螢幕 (Lightbox) 彈窗區塊開始 */}
      {/* ============================ */}
      {lightboxImage && (
        <div 
          className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 cursor-zoom-out animate-fade-in"
          onClick={() => setLightboxImage(null)} 
        >
          <img 
            src={lightboxImage} 
            alt="放大架構圖" 
            className="max-w-full max-h-full object-contain shadow-2xl rounded-lg"
          />
          <div className="absolute top-6 right-6 text-white text-3xl font-bold">✕</div>
        </div>
      )}
      {/* ============================ */}
      {/* 靜態圖片點擊放大全螢幕 (Lightbox) 彈窗區塊結束 */}
      {/* ============================ */}

      <div className="max-w-4xl mx-auto">
        
        {/* ============================ */}
        {/* 標題與版本切換按鈕區塊開始        */}
        {/* ============================ */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 pb-4">
          <h1 className="text-3xl font-extrabold text-gray-900 mb-4 md:mb-0 border-b-4 border-blue-500 pb-2 inline-block">
            系統架構藍圖 (System Architecture)
          </h1>
          
          {/* 版本切換按鈕群 */}
          <div className="flex bg-gray-200 p-1 rounded-lg">
            <button 
              onClick={() => setActiveVersion(0)}
              className={`px-4 py-2 text-sm font-bold rounded-md transition-all duration-200 ${activeVersion === 0 ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              V1.0 LINE 企業客服
            </button>
            <button 
              onClick={() => setActiveVersion(1)}
              className={`px-4 py-2 text-sm font-bold rounded-md transition-all duration-200 ${activeVersion === 1 ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              V2.0 Web 全端 AI
            </button>
          </div>
        </div>
        {/* ============================ */}
        {/* 標題與版本切換按鈕區塊結束        */}
        {/* ============================ */}
        
        {/* ============================ */}
        {/* 架構圖展示區 (靜態圖片) 開始     */}
        {/* ============================ */}
        <div className="w-full bg-white rounded-xl mb-8 shadow-lg overflow-hidden border border-gray-200 transition-opacity duration-300">
          <div className="bg-gray-100 px-4 py-2 border-b border-gray-200 text-sm font-bold text-gray-600 flex justify-between items-center">
            <span>{activeVersion === 0 ? "V1.0 LINE 企業客服 (靜態總覽)" : "V2.0 Web 全端 AI 助理 (靜態總覽)"}</span>
            <span 
              className="text-xs text-blue-500 cursor-pointer hover:underline font-semibold" 
              onClick={() => setLightboxImage(activeVersion === 0 ? archImageV1 : archImageV2)}
            >
              🔍 點擊放大
            </span>
          </div>
          {/* 【確保清晰】使用 object-contain 與淺色背景，配合 PNG 圖片，保證不裁切且完美顯示 */}
          <div className="bg-gray-50 flex justify-center w-full">
            <img 
              src={activeVersion === 0 ? archImageV1 : archImageV2} 
              alt="專案系統架構圖" 
              className="w-full max-h-[500px] object-contain block animate-fade-in cursor-zoom-in hover:opacity-90 transition-opacity"
              onClick={() => setLightboxImage(activeVersion === 0 ? archImageV1 : archImageV2)} 
            />
          </div>
        </div>
        {/* ============================ */}
        {/* 架構圖展示區 (靜態圖片) 結束     */}
        {/* ============================ */}

        {/* ============================ */}
        {/* 架構圖展示區 (Mermaid 終端機) 開始 */}
        {/* ============================ */}
        <div className={isMermaidFullscreen 
            ? "fixed inset-0 z-[80] bg-black p-4 flex flex-col animate-fade-in" 
            : "w-full bg-black rounded-xl border border-gray-700 shadow-2xl overflow-hidden mb-8 transition-all duration-300"
        }>
          {/* 終端機頂部控制列 */}
          <div className="bg-gray-800 px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500 cursor-pointer" onClick={() => setIsMermaidFullscreen(false)}></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500 cursor-pointer" onClick={() => setIsMermaidFullscreen(!isMermaidFullscreen)}></div>
              <span className="ml-4 text-gray-400 text-xs font-mono">
                {activeVersion === 0 ? "architecture_v1_line.md" : "architecture_v2_web.md"} - Mermaid Live Preview
              </span>
            </div>
            {/* 全螢幕切換按鈕 */}
            <button 
              onClick={() => setIsMermaidFullscreen(!isMermaidFullscreen)}
              className="text-gray-400 hover:text-white text-lg transition-colors font-bold"
              title={isMermaidFullscreen ? "還原視窗" : "全螢幕放大"}
            >
              {isMermaidFullscreen ? "✖" : "⛶"}
            </button>
          </div>
          
          {/* 3. 【核心修復：防裁切的滾動區塊】 */}
          {/* 使用 overflow-auto 允許自由滾動，並利用 w-fit mx-auto 保證過大的圖表不會被 flex 置中裁斷左半邊 */}
          <div className="flex-1 p-6 overflow-auto bg-gray-900 cursor-move">
            <div className="w-fit mx-auto">
              {/* 利用 key 強制 React 銷毀並重建 DOM 節點，解決 Mermaid 無法動態重繪的問題 */}
              <pre key={`${activeVersion}-${isMermaidFullscreen}`} className="mermaid text-sm animate-fade-in">
                {activeVersion === 0 ? mermaidCodeV1 : mermaidCodeV2}
              </pre>
            </div>
          </div>
        </div>
        {/* ============================ */}
        {/* 架構圖展示區 (Mermaid 終端機) 結束 */}
        {/* ============================ */}


        {/* ============================ */}
        {/* 核心技術說明網格區塊 (原始 HTML 結構) 開始 */}
        {/* ============================ */}
        
        {/* 當 activeVersion 為 0 時，渲染 V1.0 的 HTML 區塊 */}
        {activeVersion === 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-blue-500">
              <h3 className="text-xl font-bold mb-2">LINE Bot API</h3>
              <p className="text-gray-600 text-sm leading-relaxed">處理使用者的圖文訊息，並透過 Webhook 將事件安全地轉發至內部網路。</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-green-500">
              <h3 className="text-xl font-bold mb-2">IIS 伺服器 (C#)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">作為企業防火牆內的前線接收端，進行基礎的流量過濾與格式轉換。</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-purple-500">
              <h3 className="text-xl font-bold mb-2">多執行緒緩衝 (Threading)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">針對 LINE 使用者常有的「碎語」習慣，實作 5~10 秒的延遲收容機制，避免頻繁觸發 AI。</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-orange-500">
              <h3 className="text-xl font-bold mb-2">狀態機管理</h3>
              <p className="text-gray-600 text-sm leading-relaxed">結合正則表達式 (Regex)，實作「AI 自動服務」、「等待統編」與「真人接手」等多重狀態切換。</p>
            </div>
          </div>
        )}

        {/* 當 activeVersion 為 1 時，渲染 V2.0 的 HTML 區塊 */}
        {activeVersion === 1 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-blue-500">
              <h3 className="text-xl font-bold mb-2">前端介面 (React + Vite)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">負責處理使用者輸入、狀態管理 (State) 與條件渲染 (Conditional Rendering)，並使用 Tailwind 實現完美 RWD。</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-green-500">
              <h3 className="text-xl font-bold mb-2">網路穿透 (Ngrok)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">作為安全的 API Gateway，將外部的 HTTPS 請求精準路由至本地端的 Python 服務伺服器。</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-purple-500">
              <h3 className="text-xl font-bold mb-2">後端邏輯 (Python)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">處理 API 路由與跨域請求 (CORS)，並負責將資料整理後對接底層的 AI 模型。</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-orange-500">
              <h3 className="text-xl font-bold mb-2">大型語言模型 (Ollama)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">本地端運行的 AI 引擎，提供低延遲、高隱私的自然語言生成服務。</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-yellow-500">
              <h3 className="text-xl font-bold mb-2">向量檢索 (ChromaDB 雙軌 RAG)</h3>
              <p className="text-gray-600 text-sm leading-relaxed">實作高精準 (A軌) 與自動擴展 (B軌) 的雙軌檢索機制，透過 Cosine Similarity 嚴謹比對，大幅降低大型語言模型的幻覺 (Hallucination)。</p>
            </div>
          </div>
        )}
        {/* ============================ */}
        {/* 核心技術說明網格區塊 (原始 HTML 結構) 結束 */}
        {/* ============================ */}

      </div>
    </div>
  )
  // ============================
  // 畫面 B：系統架構圖 (解鎖狀態) 渲染結束
  // ============================
}
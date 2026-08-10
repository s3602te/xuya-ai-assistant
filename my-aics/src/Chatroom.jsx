// ============================
// 元件與模組引入開始
// ============================
import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
// ============================
// 元件與模組引入結束
// ============================

function App() {
  // ============================
  // 狀態管理 (State & Ref) 開始
  // ============================
  // 1. 基礎對話與 UI 狀態
  const [inputText, setInputText] = useState('')        // 追蹤輸入框內容
  const [isLoading, setIsLoading] = useState(false)     // 控制思考中動畫與按鈕鎖定
  const [isSidebarOpen, setIsSidebarOpen] = useState(false) // 控制側邊欄開關狀態
  
  // 2. 語音合成 (TTS) 狀態追蹤
  const [speakingIndex, setSpeakingIndex] = useState(null) // 紀錄目前正在朗讀的訊息索引
  const [isPaused, setIsPaused] = useState(false)          // 紀錄語音是否處於暫停狀態
  
  // 3. 會話 (Session) 與訊息狀態管理
  const [currentSessionId, setCurrentSessionId] = useState(null) // 追蹤當前對話的唯一識別碼
  const [chatHistory, setChatHistory] = useState([])             // 儲存歷史對話清單
  const [messages, setMessages] = useState([                     // 儲存當前畫面的對話內容，並設定預設歡迎詞
    { role: 'ai', text: '你好！我是這位張序亞的專屬 AI 助理。您可以問我任何關於他專案、技術或開發過程的問題！' }
  ])

  // 4. 建立 DOM 節點參考，用於對話區域自動滾動
const chatContainerRef = useRef(null)
  // ============================
  // 狀態管理 (State & Ref) 結束
  // ============================


  // ============================
  // 訪客身分驗證邏輯開始
  // ============================
  // 利用 useState 的惰性初始化 (Lazy Initialization) 產生並記錄無痕身分證
  const [userId] = useState(() => {
    // 1. 嘗試從瀏覽器本機儲存 (localStorage) 讀取歷史身分證
    let id = localStorage.getItem('interview_guest_id')
    if (!id) {
      // 2. 若無紀錄，則動態生成一組隨機 UUID (例如: guest_x8f9a...)
      id = 'guest_' + Math.random().toString(36).substring(2, 15)
      // 3. 寫入 localStorage 以便下次存取
      localStorage.setItem('interview_guest_id', id)
    }
    return id
  })
  // ============================
  // 訪客身分驗證邏輯結束
  // ============================


  // ============================
  // 佈景主題 (Theme) 切換邏輯開始
  // ============================
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'system') // 讀取預設主題
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false)                       // 控制主題下拉選單

  useEffect(() => {
    // 1. 取得根節點與監聽系統深淺色偏好
    const root = window.document.documentElement
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    
    // 2. 定義套用主題的執行函式
    const applyTheme = () => {
      if (theme === 'dark' || (theme === 'system' && mediaQuery.matches)) {
        root.classList.add('dark')
      } else {
        root.classList.remove('dark')
      }
    }

    // 3. 執行主題套用，並同步儲存至 localStorage
    applyTheme()
    localStorage.setItem('theme', theme)

    // 4. 設定系統主題變更的監聽器 (當選擇 system 模式時即時響應)
    const listener = () => { if (theme === 'system') applyTheme() }
    mediaQuery.addEventListener('change', listener)
    
    // 5. 元件卸載時清除監聽器
    return () => mediaQuery.removeEventListener('change', listener)
  }, [theme])
  // ============================
  // 佈景主題 (Theme) 切換邏輯結束
  // ============================


  // ============================
  // 生命週期與自動化操作開始
  // ============================
  // 1. 畫面初次載入時，自動向後端請求歷史對話清單
  useEffect(() => {
    fetchSessions()
  }, [])

  // 2. 監聽訊息列表或讀取狀態變化，自動將對話框滾動至最底層
  useEffect(() => {
    // 只針對聊天區塊進行內部滑動，絕對不會干擾到外層標題列
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      })
    }
  }, [messages, isLoading])
  // ============================
  // 生命週期與自動化操作結束
  // ============================


  // 快捷提問選項 (Quick Chips)
  const quickChips = [
    "請自我介紹一下",
    "談談你的 DevOps 與 CI/CD 經驗",
    "RAG 雙軌架構怎麼運作？",   
    "C# 銀行家捨入法是什麼？" 
  ]


  // ============================
  // 後端 API 與會話管理開始
  // ============================
  // 取得側邊欄清單 API
  const fetchSessions = async () => {
    try {
      // 1. 夾帶 userId 向後端請求該訪客專屬的歷史紀錄
      const res = await fetch(`/api/chat_sessions?user_id=${userId}`)
      if (res.ok) {
        const data = await res.json()
        setChatHistory(data) // 2. 更新狀態以渲染側邊欄
      }
    } catch (e) {
      console.error("無法取得歷史對話", e)
    }
  }

  // 點擊側邊欄項目，載入特定歷史對話 API
  const loadSession = async (sessionId) => {
    // 1. 切換對話前，先關閉側邊欄並中斷正在進行的語音
    setIsSidebarOpen(false)
    stopSpeaking()
    setCurrentSessionId(sessionId)
    
    try {
      // 2. 請求特定會話的詳細對話紀錄
      const res = await fetch(`/api/chat_sessions/${sessionId}`)
      if (res.ok) {
        const data = await res.json()
        setMessages(data) // 3. 覆蓋當前畫面訊息
      }
    } catch (e) {
      console.error("無法載入對話內容", e)
    }
  }
  // ============================
  // 後端 API 與會話管理結束
  // ============================


  // ============================
  // 語音合成 (TTS) 與互動輔助開始
  // ============================
  // 停止語音朗讀與重置狀態
  const stopSpeaking = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
    }
    setSpeakingIndex(null)
    setIsPaused(false)
  }

  // 進階版語音朗讀處理 (支援播放/暫停/繼續)
  const handleSpeak = (text, index) => {
    // 1. 檢查瀏覽器相容性
    if (!('speechSynthesis' in window)) {
      return alert('您的瀏覽器不支援語音朗讀功能。')
    }

    // 2. 判斷是否點擊同一個訊息：執行暫停或繼續
    if (speakingIndex === index) {
      if (isPaused) {
        window.speechSynthesis.resume()
        setIsPaused(false)
      } else {
        window.speechSynthesis.pause()
        setIsPaused(true)
      }
      return
    }

    // 3. 若點擊新訊息：先強制停止舊語音，再建立新語音實例
    stopSpeaking()
    setSpeakingIndex(index)
    setIsPaused(false)

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = 'zh-TW' 
      utterance.rate = 1.0 
      
    // 4. 篩選優先使用的優質中文語音 (如 Yating 或 Google 語音)
      const voices = window.speechSynthesis.getVoices()
      const preferredVoice = voices.find(voice => 
        voice.lang.includes('zh-TW') && (voice.name.includes('Yating') || voice.name.includes('Google'))
      ) || voices.find(voice => voice.lang.includes('zh-TW'))
      
      if (preferredVoice) {
        utterance.voice = preferredVoice
      }

    // 5. 綁定事件監聽器，確保朗讀結束或報錯時能重置 UI 狀態
    utterance.onend = () => {
      setSpeakingIndex(null)
      setIsPaused(false)
    }
    utterance.onerror = () => {
      setSpeakingIndex(null)
      setIsPaused(false)
    }

      window.speechSynthesis.speak(utterance)
  }

  // 一鍵複製功能
  const handleCopy = (text) => {
    navigator.clipboard.writeText(text)
    alert('已成功複製回覆內容！')
  }
  // ============================
  // 語音合成 (TTS) 與互動輔助結束
  // ============================


  // ============================
  // 對話紀錄編輯與刪除邏輯開始
  // ============================
  const [activeMenuId, setActiveMenuId] = useState(null) // 追蹤側邊欄被展開的下拉選單項目

  // 定義對話框狀態
  const [renameModal, setRenameModal] = useState({ isOpen: false, sessionId: null, oldTitle: '', newTitle: '' })
  const [deleteModal, setDeleteModal] = useState({ isOpen: false, sessionId: null })

  // 打開重新命名彈窗
  const handleRenameSession = (e, sessionId, oldTitle) => {
    e.stopPropagation()
    setActiveMenuId(null) // 關閉側邊欄原生下拉選單
    setRenameModal({ isOpen: true, sessionId, oldTitle, newTitle: oldTitle }) 
  }

  // 執行重新命名 API
  const confirmRename = async () => {
    const { sessionId, oldTitle, newTitle } = renameModal
    const trimmedTitle = newTitle.trim()
    
    // 1. 驗證標題是否為空或未修改
    if (!trimmedTitle || trimmedTitle === oldTitle) {
      setRenameModal({ isOpen: false, sessionId: null, oldTitle: '', newTitle: '' })
      return
    }

    try {
      // 2. 傳送 PUT 請求更新標題
      const res = await fetch(`/api/chat_sessions/${sessionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: trimmedTitle })
      })
      if (res.ok) fetchSessions() // 3. 成功後刷新側邊欄
    } catch (error) {
      console.error("重命名失敗", error)
    }
    setRenameModal({ isOpen: false, sessionId: null, oldTitle: '', newTitle: '' })
  }

  // 打開刪除確認彈窗
  const handleDeleteSession = (e, sessionId) => {
    e.stopPropagation()
    setActiveMenuId(null)
    setDeleteModal({ isOpen: true, sessionId })
  }

  // 執行刪除 API
  const confirmDelete = async () => {
    const { sessionId } = deleteModal
    try {
      // 1. 傳送刪除請求
      const res = await fetch(`/api/chat_sessions/${sessionId}`, { method: 'DELETE' })
      if (res.ok) {
        // 2. 檢查：若刪除的剛好是正在觀看的對話，則強制重置為主畫面
        if (currentSessionId === sessionId) handleNewChat() 
        fetchSessions()
      }
    } catch (error) {
      console.error("刪除失敗", error)
    }
    setDeleteModal({ isOpen: false, sessionId: null })
  }
  // ============================
  // 對話紀錄編輯與刪除邏輯結束
  // ============================


  // ============================
  // 訊息傳送核心邏輯開始
  // ============================
  const handleSendMessage = async (textToSend) => {
    // 1. 驗證輸入內容是否為空或系統正在處理中
    const messageContent = textToSend || inputText
    if (!messageContent.trim() || isLoading) return

    // 2. 傳送新訊息時，立刻中斷正在進行的 TTS 語音播放
    stopSpeaking()

    // 3. 更新畫面：將使用者訊息加入列表，並清空輸入框，啟動 loading 狀態
    const userMsg = messageContent
    setMessages(prev => [...prev, { role: 'user', text: userMsg }])
    setInputText('')
    setIsLoading(true)// 開啟「思考中」動畫

    /*  原本要用暴力鎖定 但是會無法重新輸入內容
    // ★ 終極修復 1：當按下送出、輸入框變成 disabled 導致鍵盤瞬間收起時，iOS 會忘記把畫面拉下來。
    // 這裡給予 0.1 秒的延遲，強制把卡在半空中的 iOS 視窗「扯」回原位！
    setTimeout(() => {
      window.scrollTo(0, 0);
      document.body.scrollTop = 0;
    }, 100);*/

    try {
      // 4. 發送 POST 請求至 Python 後端
      const response = await fetch('/api/web_chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: messageContent,
          session_id: currentSessionId, // 若為 null，後端將視為全新對話
          user_id: userId               // 夾帶訪客身分證進行權限辨識
        }) 
      })
      
      if (!response.ok) throw new Error('伺服器連線異常')
      
      const data = await response.json()

      // 5. 關閉 loading 狀態，並將 AI 回覆渲染至畫面
      setIsLoading(false)
      setMessages(prev => [...prev, { role: 'ai', text: data.reply }])

      // 6. 首次對話處理：若後端回傳了新的 session_id，則更新當前狀態並重整側邊欄
      if (!currentSessionId && data.session_id) {
        setCurrentSessionId(data.session_id)
        fetchSessions() // 重新整理側邊欄清單
      }

    } catch (error) {
      console.error('連線失敗', error)
      setIsLoading(false)
      setMessages(prev => [...prev, { role: 'ai', text: '【系統提示】伺服器連線失敗，請確認後端 API 是否已啟動。' }])
    }
  }

  // 觸發新增對話 (重置當前狀態)
  const handleNewChat = () => {
    stopSpeaking()
    setCurrentSessionId(null) // 清除 Session ID，讓後端知道要開新對話
    setMessages([{ role: 'ai', text: '你好！我是這位求職者的專屬 AI 助理。您可以重新開始提問！' }])
    setIsSidebarOpen(false)
  }
  // ============================
  // 訊息傳送核心邏輯結束
  // ============================


  return (
  // ============================
  // 核心外框佈局開始
  // ============================

  //採用 absolute inset-0 完美貼合父元件 (PageController) 的佈局，解決滾動條溢出問題
    
    /*  原本要用暴力鎖定 但是會無法重新輸入內容
    // 暴力修：加入 fixed inset-0 雙重鎖定，確保它像釘子一樣釘在螢幕上，並加上 flex-col 正確排版
    <div className="fixed inset-0 w-full h-[100dvh] bg-gray-100 dark:bg-gray-900 font-sans flex flex-col overflow-hidden transition-colors duration-300">*/

    // 改用 h-full 貼合 PageController 的高度，解決初始畫面需要往下滑的溢出問題
    // <div className="w-full h-full bg-gray-100 dark:bg-gray-900 font-sans flex flex-col overflow-hidden transition-colors duration-300">
    <div className="absolute inset-0 bg-gray-100 dark:bg-gray-900 font-sans flex flex-col overflow-hidden transition-colors duration-300">
      
      {/* ============================ */}
      {/* 側邊欄 (Sidebar) 區域開始      */}
      {/* ============================ */}
      <div 
        className={`absolute inset-y-0 left-0 z-40 w-64 bg-gray-900 dark:bg-black text-white transform transition-transform duration-300 ease-in-out flex flex-col shadow-2xl ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="p-4 border-b border-gray-700 flex justify-between items-center">
          <span className="font-bold text-lg tracking-wide">對話紀錄</span>
          <button onClick={() => setIsSidebarOpen(false)} className="p-1 hover:bg-gray-800 rounded">✖</button>
        </div>
        
        {/* 新增對話按鈕 */}
        <div className="p-3 border-b border-gray-700">
          <button onClick={handleNewChat} className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-sm font-semibold py-2 rounded-lg transition">
            <span>＋</span> 新增面試提問
          </button>
        </div>

        {/* 歷史對話清單渲染區 */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin">
          {chatHistory.length === 0 && <div className="text-gray-500 text-sm text-center mt-4">目前尚無歷史紀錄</div>}
          
          {chatHistory.map((history) => (
            <div key={history.id} className={`relative flex items-center justify-between w-full px-2 py-1.5 text-sm rounded-lg transition group ${
                currentSessionId === history.id ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`}
            >
              {/* 左側：對話標題按鈕 */}
              <button onClick={() => loadSession(history.id)} className="flex-1 text-left truncate pl-1 pr-2 py-1">
                💬 {history.title}
              </button>

              {/* 右側：動作選單按鈕 (電腦版 Hover 顯示，手機版可直點) */}
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveMenuId(activeMenuId === history.id ? null : history.id);
                }}
                className="p-1 px-2 rounded opacity-100 md:opacity-0 group-hover:opacity-100 hover:bg-black/30 transition flex-shrink-0"
              >
                ⋮
              </button>

              {/* 下拉選單實體 (重新命名 / 刪除) */}
              {activeMenuId === history.id && (
                <>
                  {/* 透明遮罩：點擊選單外圍即可自動關閉選單 */}
                  <div className="fixed inset-0 z-40" onClick={(e) => { e.stopPropagation(); setActiveMenuId(null); }}></div>
                  <div className="absolute right-2 top-8 w-32 bg-gray-800 border border-gray-600 rounded-lg shadow-xl z-50 overflow-hidden text-gray-200">
                    <button onClick={(e) => handleRenameSession(e, history.id, history.title)} className="w-full text-left px-4 py-2 hover:bg-gray-700 flex items-center gap-2">
                      ✏️ 重新命名
                    </button>
                    <button onClick={(e) => handleDeleteSession(e, history.id)} className="w-full text-left px-4 py-2 hover:bg-gray-700 text-red-400 border-t border-gray-600 flex items-center gap-2">
                      🗑️ 刪除
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      </div>
      {/* ============================ */}
      {/* 側邊欄 (Sidebar) 區域結束      */}
      {/* ============================ */}

      {/* ============================ */}
      {/* 側邊欄全螢幕透明遮罩開始        */}
      {/* ============================ */}
      {isSidebarOpen && (
        <div className="absolute inset-0 bg-black/50 z-30 transition-opacity" onClick={() => setIsSidebarOpen(false)} />
      )}
      {/* ============================ */}
      {/* 側邊欄全螢幕透明遮罩結束        */}
      {/* ============================ */}

      {/* ============================ */}
      {/* 主聊天畫面區塊開始             */}
      {/* ============================ */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
              
        {/* 頂部導覽與標題列 */}
        <header className="bg-blue-600 dark:bg-gray-800 text-white p-4 shadow-md flex items-center shrink-0 transition-colors duration-300 relative z-40">
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="p-1.5 hover:bg-blue-700 dark:hover:bg-gray-700 rounded-lg transition mr-2"
          >
            ☰
          </button>

          {/* 深淺色主題切換下拉選單 (佈局避開右上角的全局控制按鈕) */}
          <div className="relative mr-3">
            <button 
              onClick={() => setIsThemeMenuOpen(!isThemeMenuOpen)}
              className="p-1.5 hover:bg-blue-700 dark:hover:bg-gray-700 rounded-lg transition text-xl flex-shrink-0"
              title="主題設定"
            >
              {theme === 'light' ? '☀️' : theme === 'dark' ? '🌙' : '💻'}
            </button>

            {/* 下拉選單實體 */}
            {isThemeMenuOpen && (
              <>
                {/* 點擊選單外自動關閉的隱形遮罩 */}
                <div className="fixed inset-0 z-40" onClick={() => setIsThemeMenuOpen(false)} />
                <div className="absolute left-0 mt-2 w-32 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 z-50 overflow-hidden text-sm text-gray-800 dark:text-gray-200">
                  <button onClick={() => { setTheme('system'); setIsThemeMenuOpen(false) }} className="w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 flex justify-between items-center">
                    <span>系統</span>{theme === 'system' && <span>✓</span>}
                  </button>
                  <button onClick={() => { setTheme('light'); setIsThemeMenuOpen(false) }} className="w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 flex justify-between items-center border-t border-gray-100 dark:border-gray-700">
                    <span>淺色</span>{theme === 'light' && <span>✓</span>}
                  </button>
                  <button onClick={() => { setTheme('dark'); setIsThemeMenuOpen(false) }} className="w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 flex justify-between items-center border-t border-gray-100 dark:border-gray-700">
                    <span>深色</span>{theme === 'dark' && <span>✓</span>}
                  </button>
                </div>
              </>
            )}
          </div>

          {/* 標題靠左對齊 */}
          <h1 className="text-lg font-bold flex-1 text-left truncate">專屬 AI 助理</h1>
          
          {/* 隱形佔位區塊，防止標題被最外層的選單按鈕遮擋 */}
          <div className="w-12"></div>
        </header>

        {/* 核心對話顯示區域 */}
        <main ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 md:max-w-3xl md:mx-auto md:w-full">
        {messages.map((msg, index) => (
          <div key={index} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
            <div className={`max-w-[85%] rounded-2xl p-3 shadow-sm leading-relaxed ${
              msg.role === 'user' 
                ? 'bg-blue-500 dark:bg-blue-600 text-white rounded-br-none' 
                : 'bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 rounded-bl-none border border-gray-200 dark:border-gray-700'
            }`}>
                {/* 根據角色判斷：AI 使用 Markdown 渲染器，使用者維持純文字以防跑版 */}
              {msg.role === 'ai' ? (
                  <div className="markdown-body space-y-2 text-[15px]">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              ) : (
                  <span className="whitespace-pre-wrap text-[15px]">{msg.text}</span>
              )}
            </div>

              {/* AI 回覆專屬操作列：朗讀與複製功能 */}
            {msg.role === 'ai' && (
                <div className="flex gap-3 mt-1 ml-2 text-xs text-gray-400 dark:text-gray-500">
                  <button onClick={() => handleSpeak(msg.text, index)} className="hover:text-blue-600 dark:hover:text-blue-400 transition flex items-center gap-1">
                    {speakingIndex === index ? (isPaused ? '▶️ 繼續' : '⏸️ 暫停') : '🔊 朗讀'}
                  </button>
                  
                  {/* 只有在朗讀當前這句話時，才顯示停止按鈕 */}
                  {speakingIndex === index && (
                    <button onClick={stopSpeaking} className="hover:text-red-500 transition flex items-center gap-1">⏹️ 停止</button>
                  )}
                  
                  <button onClick={() => handleCopy(msg.text)} className="hover:text-blue-600 dark:hover:text-blue-400 transition flex items-center gap-1">
                  📋 複製
                </button>
              </div>
            )}
          </div>
        ))}

        {/* 思考中的跳動點點動畫 */}
        {isLoading && (
          <div className="flex justify-start">
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl rounded-bl-none p-4 shadow-sm flex items-center space-x-2">
                <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
          </div>
        )}
      </main>
        
        {/* ========================================== */}
        {/* 底部輸入與快捷按鈕區域開始      */}
        {/* ========================================== */}
        <footer className="bg-gray-100 dark:bg-gray-900 shrink-0 transition-colors duration-300">
          
          {/* 快捷提問選項列 */}
          <div className="px-4 py-2">
            <div className="flex gap-2 overflow-x-auto scrollbar-none md:max-w-3xl md:mx-auto">
          {quickChips.map((chip, idx) => (
                <button key={idx} onClick={() => handleSendMessage(chip)} disabled={isLoading}
                  className="whitespace-nowrap bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-gray-700 text-xs font-medium px-3 py-1.5 rounded-full border border-blue-200 dark:border-gray-600 shadow-sm transition disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            >
              {chip}
            </button>
          ))}
          </div>
        </div>

          {/* 訊息輸入框 */}
          <div className="px-4 pb-4">
            <div className="flex gap-2 md:max-w-3xl md:mx-auto relative bg-white dark:bg-gray-800 p-2 rounded-full shadow-[0_2px_15px_rgba(0,0,0,0.08)] border border-gray-200 dark:border-gray-700 transition-colors duration-300">
          <input 
            type="text" 
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder={isLoading ? "AI 正在思考中..." : "請問關於序亞的問題..."}

            // 原本要用暴力鎖定 但是會無法重新輸入內容
            // /* ★ 終極鎖定：當輸入框被點擊時，如果 iOS 試圖把網頁往上推，0.1秒內強制把它扯回最頂端！ */
            // onFocus={() => setTimeout(() => window.scrollTo(0, 0), 100)}
            // /* ★ 終極修復 2：當使用者點擊螢幕空白處手動收起鍵盤時，也強制把視窗拉回原位 */
            // onBlur={() => setTimeout(() => window.scrollTo(0, 0), 100)}

            /* ★ 核心修復：拔除 disabled={isLoading}！
            handleSendMessage 裡面本來就有 if (isLoading) return 擋住連點，
            不設成 disabled 就不會觸發 iOS Safari 的輸入框當機 Bug！ */

            disabled={isLoading} // AI 思考時鎖定輸入框，防止狂按 Enter
            className="flex-1 bg-transparent px-4 py-2 focus:outline-none text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 transition-colors text-base"
          />
          <button 
            onClick={() => handleSendMessage()}
                disabled={isLoading} // 按鈕也一起鎖定
                className="bg-blue-600 dark:bg-blue-700 text-white px-6 py-2 rounded-full font-semibold hover:bg-blue-700 dark:hover:bg-blue-600 transition disabled:bg-blue-300 disabled:cursor-not-allowed text-sm"
          >
            送出
          </button>
          </div>
        </div>
      </footer>
        {/* ========================================== */}
        {/* 底部輸入與快捷按鈕區域結束      */}
        {/* ========================================== */}

      </div>
      {/* ============================ */}
      {/* 主聊天畫面區塊結束             */}
      {/* ============================ */}
      
      {/* ============================ */}
      {/* 彈跳視窗 (Modals) 區塊開始      */}
      {/* ============================ */}
      {/* 1. 重新命名對話彈窗 */}
      {renameModal.isOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden border border-gray-200 dark:border-gray-700">
            <div className="p-5">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">重新命名對話</h3>
              <input 
                autoFocus
                type="text" 
                value={renameModal.newTitle}
                onChange={(e) => setRenameModal({ ...renameModal, newTitle: e.target.value })}
                onKeyPress={(e) => e.key === 'Enter' && confirmRename()}
                className="w-full bg-gray-100 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg px-4 py-2 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700">
              <button 
                onClick={() => setRenameModal({ isOpen: false, sessionId: null, oldTitle: '', newTitle: '' })}
                className="flex-1 py-3 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition font-medium"
              >
                取消
              </button>
              <div className="w-[1px] bg-gray-200 dark:bg-gray-700"></div>
              <button 
                onClick={confirmRename}
                className="flex-1 py-3 text-blue-600 dark:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition font-bold"
              >
                儲存
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 2. 刪除確認彈窗 */}
      {deleteModal.isOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden border border-gray-200 dark:border-gray-700">
            <div className="p-6 text-center">
              <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mx-auto mb-4">
                <span className="text-red-500 text-xl">🗑️</span>
              </div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">刪除對話紀錄</h3>
              <p className="text-gray-500 dark:text-gray-400 text-sm">此動作無法復原，確定要永久刪除這筆對話嗎？</p>
            </div>
            <div className="flex bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700">
              <button 
                onClick={() => setDeleteModal({ isOpen: false, sessionId: null })}
                className="flex-1 py-3 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition font-medium"
              >
                取消
              </button>
              <div className="w-[1px] bg-gray-200 dark:bg-gray-700"></div>
              <button 
                onClick={confirmDelete}
                className="flex-1 py-3 text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition font-bold"
              >
                確認刪除
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default App
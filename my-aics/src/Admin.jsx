// ============================
// 元件與模組引入開始
// ============================
// 1. 引入 React 核心 Hooks，用於狀態管理與生命週期控制
import { useState, useEffect, useRef } from 'react'
// ============================
// 元件與模組引入結束
// ============================

export default function Admin() {
  // ============================
  // 狀態管理開始
  // ============================
  // 1. 權限與密碼狀態：控制管理員是否登入與錯誤提示
  const [isUnlocked, setIsUnlocked] = useState(false)
  const [password, setPassword] = useState('')
  const [errorMsg, setErrorMsg] = useState('')

  // 2. 客服後台資料狀態：儲存所有對話、當前選中的客戶、對話 ID、訊息內容與輸入框文字
  const [sessions, setSessions] = useState([])
  const [selectedUserId, setSelectedUserId] = useState(null)
  const [selectedSessionId, setSelectedSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')

  // 3. UI 輔助狀態：控制側邊欄、左側客戶抽屜開關，以及防止按鈕重複點擊的處理狀態
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [expandedUsers, setExpandedUsers] = useState({}) // 控制左側「抽屜」開關狀態
  const [isProcessingAction, setIsProcessingAction] = useState(false) // 控制按鈕處理中狀態，防止重複點擊

  // 4. 真人模式狀態：記錄真人模式的閒置倒數秒數
  const [humanCountdown, setHumanCountdown] = useState(null)

  // 5. 主題切換狀態：與前台同步，預設讀取 localStorage
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'system')
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false)

  // 6. DOM 參考：用於新訊息抵達時，自動將畫面滾動到底部
  const messagesEndRef = useRef(null)
  // ============================
  // 狀態管理結束
  // ============================

  // ============================
  // 自動化操作與資料撈取開始
  // ============================
  // 1. 佈景主題切換邏輯：監聽主題設定或系統偏好以動態套用深淺色 (移植自前台)
  useEffect(() => {
    const root = window.document.documentElement
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const applyTheme = () => {
      if (theme === 'dark' || (theme === 'system' && mediaQuery.matches)) {
        root.classList.add('dark')
      } else {
        root.classList.remove('dark')
      }
    }
    applyTheme()
    localStorage.setItem('theme', theme)
    const listener = () => { if (theme === 'system') applyTheme() }
    mediaQuery.addEventListener('change', listener)
    return () => mediaQuery.removeEventListener('change', listener)
  }, [theme])

  // 2. 左側名單輪詢邏輯：每 3 秒自動刷新一次，確保能看見新進線的客戶
  useEffect(() => {
    if (!isUnlocked) return
    const fetchAllSessions = async () => {
      try {
        const res = await fetch(`/api/chat_sessions?user_id=admin`)
        if (res.ok) {
          const data = await res.json()
          setSessions(data)
          
          // 3. 解決閉包陷阱：利用 prev 取得當前真實狀態。若目前無展開的資料夾，預設幫客服打開第一筆
          setExpandedUsers(prev => {
            if (data.length > 0 && Object.keys(prev).length === 0) {
              return { [data[0].user_id]: true };
            }
            return prev;
          });
        }
      } catch (e) {
        console.error("無法取得全站歷史對話", e)
      }
    }
    fetchAllSessions()
    const interval = setInterval(fetchAllSessions, 3000) //將輪詢時間 每 3 秒 fetch 一次左側的客戶清單。
    return () => clearInterval(interval)
  }, [isUnlocked])

  // 4. 右側對話內容輪詢邏輯：選中特定對話時，每 2 秒撈取該對話歷史訊息
  useEffect(() => {
    if (!selectedSessionId) return
    setIsProcessingAction(false) // 切換對話時，解開按鈕鎖定

    const fetchMessages = async () => {
      try {
        const res = await fetch(`/api/chat_sessions/${selectedSessionId}`)
        if (res.ok) {
          const data = await res.json()
          setMessages(prev => {
            // 5. 畫面滾動控制：只有當對話長度增加(代表有新訊息)時，才將畫面平滑滾動到底部
            if (prev.length !== data.length) {
              setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
            }
            return data
          })
        }
      } catch (e) {
        console.error("無法載入對話內容", e)
      }
    }
    fetchMessages()
    // 為了簡單起見，後台採用短輪詢每 2 秒更新一次對話內容
    const interval = setInterval(fetchMessages, 2000) //將輪詢時間 每 2 秒 fetch 一次右側的對話內容。
    return () => clearInterval(interval)
  }, [selectedSessionId])
  // ============================
  // 自動化操作與資料撈取結束
  // ============================

  // ============================
  // 工具與防呆邏輯開始
  // ============================
  // 1. 登入密碼驗證邏輯
  const handleUnlock = () => {
    if (password === 'Admin0304!') {
      setIsUnlocked(true)
      setErrorMsg('')
    } else {
      setErrorMsg('密碼錯誤，拒絕存取。')
    }
  }

  // 2. 監聽側邊欄開關：如果側邊欄被收回，在此攔截並可關閉殘留的狀態
  useEffect(() => {
    if (!isSidebarOpen) {
       // 如果未來有 activeMenuId 等狀態，可以在這裡一併清空
    }
  }, [isSidebarOpen]);

  // 3. 資料整理：將平坦的 session 清單依照 user_id 進行分組，以便抽屜化渲染
  const groupedSessions = sessions.reduce((acc, session) => {
    if (!acc[session.user_id]) acc[session.user_id] = []
    acc[session.user_id].push(session)
    return acc
  }, {})

  // 4. 抽屜控制：切換特定客戶資料夾的展開/收合狀態
  const toggleUserFolder = (userId) => {
    setExpandedUsers(prev => ({ ...prev, [userId]: !prev[userId] }))
  }

  // 5. 核心防呆：精準判斷目前是否處於「正在轉接」或「真人服務中」
  const checkIsHumanMode = () => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const text = messages[i].text;
      // 若包含轉接字眼，允許客服按「結束真人」
      if (text.includes('正在為您轉接真人客服')) return true;
      
      // 若發現系統或前台喊停的字眼，同步解除按鈕鎖定
      if (text.includes('真人服務結束') || text.includes('已自動切回 AI') || text.includes('AI 繼續為您服務') || text.includes('AI 客服將重新為您服務')) return false;
    }
    return false;
  }
  const isHumanModeActive = checkIsHumanMode()

  // 6. 提取對話時間：取得最後一筆訊息的資料庫時間，做為絕對時間起算點
  const lastMessageTimeStr = messages.length > 0 ? messages[messages.length - 1].created_at : null;
  // ============================
  // 工具與防呆邏輯結束
  // ============================

  // ============================
  // 真人模式閒置倒數計時器引擎開始
  // ============================
  // 1. 定義時間參考變數，避免頻繁渲染
  const targetTimeRef = useRef(null);

  // 2. 當對話或狀態更新時，計算出「未來的過期絕對時間」
  useEffect(() => {
    if (isHumanModeActive) {
      if (lastMessageTimeStr) {
        // 將 SQLite 的 "YYYY-MM-DD HH:MM:SS" 轉為 JS 可讀的 ISO 格式
        const safeTimeStr = lastMessageTimeStr.replace(' ', 'T');
        const messageTime = new Date(safeTimeStr).getTime();
        
        // 鎖定未來時間：最後一筆訊息的發送時間 + 30 秒
        targetTimeRef.current = messageTime + 30000;
      } else {
        // 備用方案：如果剛好沒有時間戳，預設給 30 秒
        targetTimeRef.current = Date.now() + 30000;
      }
    } else {
      targetTimeRef.current = null;
      setHumanCountdown(null);
    }
  // 監聽 lastMessageTimeStr 確保客服或客人講話時能刷新 30 秒
  }, [messages.length, isHumanModeActive, lastMessageTimeStr]);

  // 3. 獨立高頻計時器：精準比對絕對時間，解決鬼打牆與切換誤差
  useEffect(() => {
    const timer = setInterval(() => {
      if (!targetTimeRef.current) return;
      
      const remain = Math.ceil((targetTimeRef.current - Date.now()) / 1000);
      if (remain > 0) {
        setHumanCountdown(remain);
      } else {
        setHumanCountdown(0);
        // 核心修復：清空目標時間讓計算暫停，不使用 clearInterval 破壞引擎
        targetTimeRef.current = null; 
      }
    }, 200); // 提高結算頻率讓畫面秒數不卡頓

    return () => clearInterval(timer); // 只有當客服登出離開 Admin 頁面時，才真正銷毀引擎
  }, []);
  // ============================
  // 真人模式閒置倒數計時器引擎結束
  // ============================

  // ============================
  // 客服訊息傳送邏輯開始
  // ============================
  const handleAdminAction = async (actionType) => {
    // 1. 空白防呆：若是回覆且無文字則拒絕執行
    if (actionType === 'reply' && !inputText.trim()) return

    // 2. 按鈕鎖定：按下「結束真人」立刻鎖定並顯示「處理中...」
    if (actionType === 'end_human') {
      setIsProcessingAction(true)
    }

    const payload = {
      user_id: selectedUserId,
      message: inputText,
      action: actionType // 'reply' 或 'end_human'
    }

    try {
      // 3. 呼叫 API 送出指令
      await fetch('/api/admin_reply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (actionType === 'reply') setInputText('')
    } catch (e) {
      console.error("客服指令發送失敗", e)
    } finally {
      // 4. 強制解除：無論成功或失敗，利用 finally 確保解除「結束真人」的鎖定狀態
      if (actionType === 'end_human') setIsProcessingAction(false) 
    }
  }
  // ============================
  // 客服訊息傳送邏輯結束
  // ============================

  // ============================
  // 畫面渲染：上鎖狀態開始
  // ============================
  if (!isUnlocked) {
    return (
      // 1. 渲染滿版的登入背景與容器
      <div className="w-full h-[100dvh] flex flex-col items-center justify-center bg-gray-900 p-4 font-sans">
        <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl max-w-md w-full border border-gray-700 text-center">
          <div className="text-5xl mb-4">🎧</div>
          <h2 className="text-2xl font-bold text-white mb-2">客服中央控制台</h2>
          <p className="text-gray-400 text-sm mb-6">請輸入管理員密碼以進入後台。</p>
          
          {/* 2. 密碼輸入框與錯誤訊息顯示區 */}
          <input 
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleUnlock()}
            placeholder="請輸入密碼..."
            className="w-full bg-gray-900 text-white border border-gray-600 rounded-lg px-4 py-3 mb-4 focus:outline-none focus:ring-2 focus:ring-orange-500 transition"
          />
          {errorMsg && <p className="text-red-400 text-sm mb-4">{errorMsg}</p>}
          
          {/* 3. 觸發登入按鈕 */}
          <button onClick={handleUnlock} className="w-full bg-orange-600 text-white font-bold py-3 rounded-lg hover:bg-orange-700 transition shadow-lg">
            登入後台
          </button>
        </div>
      </div>
    )
  }
  // ============================
  // 畫面渲染：上鎖狀態結束
  // ============================

  {/* ============================ */}
  {/* 畫面渲染：客服後台主畫面開始 */}
  {/* ============================ */}
  return (
    // 1. 最外層滿版主視窗容器
    <div className="w-full h-[100dvh] bg-gray-100 dark:bg-gray-900 flex font-sans overflow-hidden">
      
      {/* RWD 側邊欄遮罩 */}
      {isSidebarOpen && (
        <div className="absolute inset-0 bg-black/50 z-[55] md:hidden transition-opacity" onClick={() => setIsSidebarOpen(false)} />
      )}

      {/* 3. 左側客戶清單容器 (抽屜式設計) */}
      <div 
        className={`absolute md:relative inset-y-0 left-0 z-[60] md:z-40 w-64 bg-gray-900 dark:bg-black text-white transform transition-all duration-300 ease-in-out flex flex-col shadow-2xl shrink-0 ${
          isSidebarOpen ? 'translate-x-0 ml-0' : '-translate-x-full md:-ml-64'
        }`}
      >
        <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-orange-600 dark:bg-orange-700 text-white">
          <span className="font-bold text-lg tracking-wide">👥 線上客戶清單</span>
          <button onClick={() => setIsSidebarOpen(false)} className="md:hidden p-1 hover:bg-orange-800 rounded">✖</button>
        </div>

        {/* 4. 迴圈渲染各客戶資料夾與對話紀錄 */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin">
          {Object.entries(groupedSessions).map(([uid, userSessions]) => {
            // 判斷該客戶是否有待處理的紅點對話
            const hasUnresolved = userSessions.some(s => s.needs_attention)

            return (
            <div key={uid} className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
              
              {/* 4-1. 抽屜標題 (客戶 ID 及未處理紅點) */}
              <button 
                onClick={() => toggleUserFolder(uid)}
                className="w-full text-left p-3 bg-gray-800 hover:bg-gray-700 text-white font-bold flex justify-between items-center transition"
              >
                <span className="truncate pr-2 text-sm flex items-center gap-2">
                  <span className="text-xl">{expandedUsers[uid] ? '📂' : '📁'}</span>
                  ID: {uid.split('_')[1] || uid}
                </span>
                {/* 抽屜關閉時，在外層顯示閃爍紅點 */}
                {hasUnresolved && !expandedUsers[uid] && (
                  <span className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)] shrink-0"></span>
                )}
              </button>
              
              {/* 4-2. 抽屜內容 (歷史對話列表) */}
              {expandedUsers[uid] && (
                <div className="bg-gray-900 p-2 space-y-1 border-t border-gray-700">
                  {userSessions.map(session => (
                    <button 
                      key={session.id} 
                      onClick={() => { 
                        setSelectedSessionId(session.id); 
                        setSelectedUserId(session.user_id); 
                        // 核心修復：只有手機版(寬度<768)才自動收起側邊欄，電腦 Web 版保持常駐！
                        if (window.innerWidth < 768) setIsSidebarOpen(false); 
                      }}
                      className={`w-full text-left p-2 rounded-md transition text-sm flex justify-between items-center ${selectedSessionId === session.id ? 'bg-orange-500 text-white' : 'text-gray-400 hover:bg-gray-700 hover:text-white'}`}
                    >
                      <span className="truncate">💬 {session.title || '未命名對話'}</span>
                      {/* 若對話有待處理狀態，於對話後方顯示紅點 */}
                      {session.needs_attention && (
                        <span className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)] shrink-0 ml-2" title="待處理"></span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )})}
          {/* 4-3. 空清單防呆提示 */}
          {Object.keys(groupedSessions).length === 0 && <div className="text-gray-500 text-center mt-6 text-sm">目前無對話紀錄</div>}
        </div>
      </div>

      {/* 5. 右側對話與控制主區塊 */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden bg-gray-50 dark:bg-gray-900">
        
        {/* 6. 右側上方 Header 區塊 */}
        <header className="p-4 pr-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm flex items-center justify-between shrink-0 z-10 transition-colors duration-300 relative gap-2">
          
          <div className="flex items-center min-w-0">
            {/* 側邊欄切換開關 */}
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition mr-2 text-gray-800 dark:text-gray-200 shrink-0"
            >
              ☰
            </button>

            {/* 深淺色主題切換下拉選單 */}
            <div className="relative mr-3 shrink-0">
              <button 
                onClick={() => setIsThemeMenuOpen(!isThemeMenuOpen)}
                className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition text-xl flex-shrink-0 text-gray-800 dark:text-gray-200"
                title="主題設定"
              >
                {theme === 'light' ? '☀️' : theme === 'dark' ? '🌙' : '💻'}
              </button>

              {/* 下拉選單實體區塊 */}
              {isThemeMenuOpen && (
                <>
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

            {/* 當前選擇狀態文字標題 */}
            <div className="min-w-0 flex items-center gap-2">
              <h1 className="text-base md:text-lg font-bold text-gray-800 dark:text-white truncate">
                {selectedSessionId ? "正在檢視對話" : "客服中央控制台"}
              </h1>
              {selectedUserId && <span className="text-xs text-gray-500 dark:text-gray-400 truncate hidden md:inline">Target ID: {selectedUserId}</span>}
            </div>
          </div>

          {/* 7. 結束真人控制按鈕區 */}
          <div className="flex items-center">
            {selectedSessionId && (
              <button 
                onClick={() => handleAdminAction('end_human')} 
                disabled={!isHumanModeActive || isProcessingAction}
                className={`px-3 md:px-4 py-2 rounded-lg text-xs md:text-sm font-bold shadow-sm transition shrink-0 whitespace-nowrap ${
                  (!isHumanModeActive || isProcessingAction)
                    ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed' 
                    : 'bg-red-500 hover:bg-red-600 text-white'
                }`}
              >
                {isHumanModeActive ? (isProcessingAction ? '處理中...' : '⏹️ 結束真人') : '🔒 非真人模式'}
              </button>
            )}
          </div>
        </header>

        {/* 8. 對話內容與輸入框渲染邏輯 */}
        {selectedSessionId ? (
          <>
            {/* 8-1. 動態進度條倒數計時橫幅 (僅在真人模式下呈現) */}
            {isHumanModeActive && humanCountdown !== null && humanCountdown > 0 && (
              <div className="px-4 pt-4 shrink-0 bg-gray-50 dark:bg-gray-900 animate-fade-in">
                {/* 外層容器：將遮罩動畫關在圓角邊框內 */}
                <div className={`relative overflow-hidden flex items-center justify-center gap-2 py-2 px-4 rounded-full border shadow-sm text-sm font-bold tracking-wide transition-colors z-0 bg-white dark:bg-gray-800 ${
                  humanCountdown <= 10 
                    ? 'border-red-300 dark:border-red-700 text-red-600 dark:text-red-400' 
                    : 'border-orange-300 dark:border-orange-700 text-orange-600 dark:text-orange-400'
                }`}>
                  {/* 內層遮罩：利用 width 百分比產生水準退去動畫 */}
                  <div 
                    className={`absolute inset-y-0 left-0 -z-10 transition-all duration-1000 ease-linear ${
                      humanCountdown <= 10 ? 'bg-red-100 dark:bg-red-900/30' : 'bg-orange-100 dark:bg-orange-900/30'
                    }`}
                    style={{ width: `${(humanCountdown / 30) * 100}%` }}
                  />
                  <span className={humanCountdown <= 10 ? 'animate-pulse' : ''}>⏳</span>
                  <span>
                    {humanCountdown <= 10 
                      ? `注意！閒置過久，將於 ${humanCountdown} 秒後自動切回 AI` 
                      : `真人客服閒置倒數：${humanCountdown} 秒`}
                  </span>
                </div>
              </div>
            )}

            {/* 8-2. 對話歷史紀錄列表 */}
            <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-start' : 'items-end'}`}>
                  <div className={`max-w-[85%] rounded-2xl p-3 shadow-sm text-[15px] whitespace-pre-wrap ${
                    msg.role === 'user' 
                      ? 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-none' 
                      : 'bg-orange-100 dark:bg-orange-900/30 text-orange-900 dark:text-orange-100 rounded-br-none border border-orange-200 dark:border-orange-800'
                  }`}>
                    <div className="text-[11px] font-bold mb-1 opacity-50">{msg.role === 'user' ? '客戶' : 'AI / 客服'}</div>
                    {msg.text}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* 8-3. 真人客服輸入與送出控制區 */}
            <footer className="p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 shrink-0">
              <div className="flex items-end gap-2 max-w-4xl mx-auto relative bg-gray-50 dark:bg-gray-900 p-2 rounded-3xl border border-gray-300 dark:border-gray-600">
                <textarea 
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    // 攔截 Enter 送出 (Shift+Enter 允許換行)
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault(); 
                      handleAdminAction('reply');
                    }
                  }}
                  placeholder={!isHumanModeActive ? "請等候客戶轉接真人..." : "回覆客戶訊息..."}
                  disabled={!isHumanModeActive}
                  rows={1}
                  className="flex-1 bg-transparent px-4 py-2 focus:outline-none text-gray-900 dark:text-white placeholder-gray-500 text-base disabled:opacity-50 resize-none max-h-32 overflow-y-auto min-h-[40px] scrollbar-thin"
                />
                <button 
                  onClick={() => handleAdminAction('reply')}
                  disabled={!isHumanModeActive || !inputText.trim()}
                  className="bg-orange-600 text-white px-6 py-2 rounded-full font-bold hover:bg-orange-700 disabled:bg-orange-300 disabled:cursor-not-allowed transition text-sm shadow-sm h-10 shrink-0"
                >
                  送出
                </button>
              </div>
            </footer>
          </>
        ) : (
          /* 8-4. 未選擇任何對話時的佔位畫面 */
          <div className="flex-1 flex items-center justify-center text-gray-400 bg-gray-100 dark:bg-gray-900">
            <div className="text-center">
              <div className="text-4xl mb-4">👋</div>
              <p>請從左側抽屜選擇一位客戶開始服務</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
{/* ============================ */}
{/* 畫面渲染：客服後台主畫面結束 */}
{/* ============================ */}
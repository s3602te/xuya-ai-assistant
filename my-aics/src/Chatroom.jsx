// ============================
// 元件與模組引入開始
// ============================
// 1. 引入 React 核心模組，用於狀態管理與生命週期控制
import { useState, useRef, useEffect } from 'react'
// 2. 引入 ReactMarkdown 套件，用於解析 AI 回覆的 Markdown 語法
import ReactMarkdown from 'react-markdown'
// 3. 引入 WebSocket 客戶端，用於與後端建立即時通訊隧道
import { io } from 'socket.io-client' 
// 4. 引入 LINE LIFF SDK，用於獲取 LINE 用戶的專屬身分
import liff from '@line/liff'         
// ============================
// 元件與模組引入結束
// ============================

function App() {
  // ============================
  // 狀態管理 (State & Ref) 開始
  // ============================
  // 1. 基礎對話與 UI 狀態管理
  const [inputText, setInputText] = useState('')            // 追蹤輸入框內容
  const [isLoading, setIsLoading] = useState(false)         // 控制思考中動畫與按鈕鎖定狀態
  const [isSidebarOpen, setIsSidebarOpen] = useState(false) // 控制側邊欄的開關狀態

  // 2. 語音合成 (TTS) 狀態追蹤
  const [isHumanMode, setIsHumanMode] = useState(false)    // 追蹤目前是否為真人客服接手模式
  const [speakingIndex, setSpeakingIndex] = useState(null) // 紀錄目前正在朗讀的訊息索引值
  const [isPaused, setIsPaused] = useState(false)          // 紀錄語音是否處於暫停狀態

  // 3. 會話 (Session) 與訊息狀態管理
  const [currentSessionId, setCurrentSessionId] = useState(null) // 追蹤當前對話的唯一識別碼
  const [chatHistory, setChatHistory] = useState([])             // 儲存側邊欄歷史對話清單
  const [messages, setMessages] = useState([                     // 儲存當前畫面的對話內容，並設定預設歡迎詞
    { role: 'ai', text: '你好！我是這位張序亞的專屬 AI 助理。您可以問我任何關於他專案、技術或開發過程的問題！' }
  ])

  // 4. DOM 節點參考 (Refs)
  const chatContainerRef = useRef(null) // 用於對話區域自動滾動至最底部
  const inputRef = useRef(null)         // 用來強制對焦輸入框，提升使用者體驗

  // 5. 彈跳視窗狀態管理
  const [customConfirm, setCustomConfirm] = useState({ isOpen: false, pendingAction: null }) // 自訂中斷轉接警告視窗

  // 6. 倒數計時狀態追蹤
  const [countdown, setCountdown] = useState(null) // null 代表未啟動，數字代表剩餘倒數秒數
  
  // 7. 畫面掛載狀態管理 (防止側邊欄初始載入時產生殘影閃爍)
  const [isMounted, setIsMounted] = useState(false)
  useEffect(() => setIsMounted(true), [])
  // ============================
  // 狀態管理 (State & Ref) 結束
  // ============================


  // ============================
  // 訪客身分驗證邏輯開始
  // ============================
  // 1. 定義身分驗證相關狀態 (將同步改為非同步管理)
  const [userId, setUserId] = useState(null)
  const [isInitializingId, setIsInitializingId] = useState(true)

  useEffect(() => {
    const initializeApp = async () => {
      try {
        // 2. 初始化 LIFF SDK 
        await liff.init({ liffId: '2009807796-K9a09Udj' });

        if (liff.isInClient()) {
          // 3-A. 情境 A：使用者從 LINE 官方帳號內點擊開啟網頁
          const profile = await liff.getProfile();
          
          // 取得使用者的 LINE 顯示名稱與 ID 前 4 碼，確保同名者不重複
          const displayName = profile.displayName;
          const shortId = profile.userId.substring(0, 4);
          
          // 組合出供後台識別的完美 ID 格式 (例如: LINE_使用者名稱-A1B2)
          const customUserId = `LINE_${displayName}-${shortId}`;
          
          setUserId(customUserId); 
          console.log('✅ 已取得 LINE 身分認證：', customUserId);
        } else {
          // 3-B. 情境 B：面試官使用電腦或一般瀏覽器開啟 (Fallback 機制)
          let id = localStorage.getItem('interview_guest_id');
          if (!id) {
            id = 'guest_' + Math.random().toString(36).substring(2, 15);
            localStorage.setItem('interview_guest_id', id);
          }
          setUserId(id);
          console.log('✅ 已配發訪客無痕身分：', id);
        }
      } catch (error) {
        console.error('LIFF 初始化失敗:', error);
        // 4. 防呆保護：若 LIFF 伺服器異常，強制發配一組亂數身分，確保系統仍可運作
        setUserId('guest_fallback_' + Math.random().toString(36).substring(2, 8));
      } finally {
        // 5. 驗證完畢，解除 Loading 封印，允許畫面渲染
        setIsInitializingId(false);
      }
    };

    initializeApp();
  }, []);
  // ============================
  // 訪客身分驗證邏輯結束
  // ============================


  // ============================
  // 佈景主題 (Theme) 切換邏輯開始
  // ============================
  // 1. 定義主題切換狀態與下拉選單狀態
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'system') 
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false)                       

  useEffect(() => {
    // 2. 取得根節點並監聽作業系統的深淺色偏好
    const root = window.document.documentElement
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

    // 3. 定義套用主題的執行邏輯
    const applyTheme = () => {
      if (theme === 'dark' || (theme === 'system' && mediaQuery.matches)) {
        root.classList.add('dark')
      } else {
        root.classList.remove('dark')
      }
    }

    // 4. 執行套用並同步儲存至 localStorage
    applyTheme()
    localStorage.setItem('theme', theme)

    // 5. 設定系統主題變更的監聽器，確保 'system' 模式能即時響應
    const listener = () => { if (theme === 'system') applyTheme() }
    mediaQuery.addEventListener('change', listener)

    // 6. 元件卸載時清除監聽器，釋放記憶體
    return () => mediaQuery.removeEventListener('change', listener)
  }, [theme])
  // ============================
  // 佈景主題 (Theme) 切換邏輯結束
  // ============================


  // ============================
  // WebSocket 即時通訊掛載開始
  // ============================
  useEffect(() => {
    // 1. 生命週期防護：等待身分驗證完成後，才允許建立 WebSocket 連線
    if (isInitializingId || !userId) return;

    // 2. 建立與後端伺服器的常時連線隧道
    const socket = io('/')

    socket.on('connect', () => {
      console.log('✅ WebSocket 隧道已接通！身分：', userId)
    })

    // 3. 監聽並接收 AI 或真人的回覆訊息
    socket.on('chat_reply', (data) => {
      console.log('📩 收到 WebSocket 訊息：', data)
      if (data.session_id === userId) {
        // 解除按鈕鎖定與思考中動畫
        setIsLoading(false)

        // 4. 關鍵字攔截引擎：根據後端指令動態啟動或關閉倒數計時器
        const replyText = data.reply || '';
        if (replyText.includes('是否轉接真人客服') || 
            replyText.includes('請在 30 秒內輸入貴公司統編') || 
            replyText.includes('已為您重新計時 30 秒')) {
          setCountdown(30); // 觸發 30 秒倒數
        } else if (replyText.includes('選擇超時') || 
                   replyText.includes('統編輸入超時') || 
                   replyText.includes('已收到統編') || 
                   replyText.includes('AI 繼續為您服務')) {
          setCountdown(null); // 任務完成或超時，安全關閉計時器
        }

        // 5. 更新對話狀態並清除舊的選項按鈕
        setMessages(prev => {
          const newMessages = prev.map(m => ({ ...m }));
          if (newMessages.length > 0) delete newMessages[newMessages.length - 1].options;
          return [...newMessages, { role: data.role || 'ai', text: data.reply, options: data.options }]
        })
      }
    })

    // 監聽來自大聲公 (websocket_manager) 的廣播
    socket.on('state_update', (data) => {
      console.log('📡 收到狀態切換廣播：', data)
      if (data.session_id === userId) {
        setIsHumanMode(data.state === 'human')
      }
    })

    // 7. 元件卸載時自動斷開連線
    return () => socket.disconnect() 
  }, [userId, isInitializingId]) 
  // ============================
  // WebSocket 即時通訊掛載結束
  // ============================


  // ============================
  // 倒數計時器 (Timer) 驅動引擎開始
  // ============================
  useEffect(() => {
    // 1. 狀態防護：若狀態為 null (未啟動) 則不執行任何動作
    if (countdown === null) return;

    // 2. 建立計時器：每秒觸發一次扣減邏輯
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer); // 歸零時銷毀實體
          return null;          // 重置狀態以隱藏 UI
        }
        return prev - 1;
      });
    }, 1000);

    // 3. 清理機制：元件卸載或狀態突變時強制清除計時器，防止 Memory Leak
    return () => clearInterval(timer);
  }, [countdown]);
  // ============================
  // 倒數計時器 (Timer) 驅動引擎結束
  // ============================


  // ============================
  // 生命週期與自動化操作開始
  // ============================
  // 1. 初次載入自動獲取對話：系統初始掛載時向後端請求歷史清單
  useEffect(() => {
    if (userId) {
      fetchSessions().then(data => {
        if (!currentSessionId && data && data.length > 0) {
          loadSession(data[0].id, true); // initial load 設為 true 避免觸發離開警告
        }
      });
    }
  }, [userId])

  // 2. 視圖自動跟隨：監聽對話變化，自動將畫面平滑滾動至最底部
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      })
    }
  }, [messages, isLoading])

  // 3. 輸入框焦點管理：AI 回覆完畢後，強制重新將焦點拉回輸入框
  useEffect(() => {
    if (!isLoading && inputRef.current) {
      setTimeout(() => inputRef.current.focus(), 10)
    }
  }, [isLoading])
  // ============================
  // 生命週期與自動化操作結束
  // ============================


  // ============================
  // 靜態常數宣告開始
  // ============================
  // 1. 定義快捷提問選項 (Quick Chips)，用於 AI 模式下快速發問
  const quickChips = [
    "請自我介紹一下",
    "談談你的 DevOps 與 CI/CD 經驗",
    "我要找真人客服",
    "RAG 雙軌架構怎麼運作？",
    "C# 銀行家捨入法是什麼？"
  ]
  // ============================
  // 靜態常數宣告結束
  // ============================


  // ============================
  // 後端 API 與會話管理開始
  // ============================
  // 1. 自訂防呆攔截邏輯：檢查是否處於轉接或需特殊處理狀態
  const triggerActionWithCheck = (actionFn) => {
    const lastMsg = messages.length > 0 ? messages[messages.length - 1].text : '';
    // 把「統編格式錯誤」也加入攔截字典中，填補漏網之魚！
    const isTransferring = isHumanMode || lastMsg.includes('正在為您轉接真人客服') || lastMsg.includes('是否轉接真人客服') || lastMsg.includes('輸入貴公司統編') || lastMsg.includes('統編格式錯誤');

    if (isTransferring) {
      // 觸發自訂 UI 警告，並將目標動作儲存進狀態中
      setCustomConfirm({ isOpen: true, pendingAction: actionFn })
    } else {
      actionFn()
    }
  }

  // 2. 執行確認離開邏輯：發送中斷指令並執行原本儲存的動作
  const executePendingAction = () => {
    // 偷偷發送隱藏指令，強制後端大腦解除轉接鎖定
    fetch('/api/web_chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: '取消轉接重啟AI', session_id: currentSessionId, user_id: userId })
    }).catch(e => console.log(e));

    if (customConfirm.pendingAction) customConfirm.pendingAction()
    setCustomConfirm({ isOpen: false, pendingAction: null })
  }

  // 3. 獲取會話清單 API：夾帶 userId 向後端取得歷史對話
  const fetchSessions = async () => {
    try {
      const res = await fetch(`/api/chat_sessions?user_id=${userId}`)
      if (res.ok) {
        const data = await res.json()
        setChatHistory(data)
        return data; 
      }
    } catch (e) {
      console.error("無法取得歷史對話", e)
    }
    return [];
  }

  // 4. 載入特定對話 API：獲取指定對話的歷史內容並重建前端狀態
  const loadSession = async (sessionId, isInitialLoad = false) => {
    const doLoad = async () => {
      // 4-1. 介面初始化與狀態清理
      if (window.innerWidth < 768) setIsSidebarOpen(false)
      stopSpeaking()
      setCountdown(null) // 切換歷史對話時，強制關閉倒數計時器
      setCurrentSessionId(sessionId)
      setIsHumanMode(false) // 切換對話時，預設解除前端真人狀態
      setIsLoading(true) // 載入歷史紀錄時，先開啟讀取動畫，避免畫面空轉

      try {
        // 4-2. 向後端請求該對話的完整內容
        const res = await fetch(`/api/chat_sessions/${sessionId}`)
        if (res.ok) {
          const data = await res.json()

          // 4-3. 狀態重建 (1)：若最後一句是問轉接，將選項按鈕補回畫面
          const processedMessages = data.map((msg, index) => {
            const newMsg = { ...msg };
            // 如果對話的最後一句話是問是否轉接，強制把按鈕補回去！
            if (index === data.length - 1 && newMsg.role === 'ai') {
              if (newMsg.text.includes('是否轉接真人客服')) {
                newMsg.options = ['是', '否'];
              }
            }
            return newMsg;
          });

          // 4-4. 狀態重建 (2)：掃描對話歷史推斷並恢復真人模式狀態
          let restoredHumanMode = false;
          for (let i = processedMessages.length - 1; i >= 0; i--) {
            const text = processedMessages[i].text;
            if (text.includes('正在為您轉接真人客服')) {
              restoredHumanMode = true;
              break;
            }
            if (text.includes('真人服務結束') || text.includes('已自動切回 AI') || text.includes('AI 繼續為您服務') || text.includes('AI 客服將重新為您服務')) {
              restoredHumanMode = false;
              break;
            }
          }

          // 將推斷出來的狀態與訊息套用回畫面
          setIsHumanMode(restoredHumanMode)
          setMessages(processedMessages)
        }
      } catch (e) {
        console.error("無法載入對話內容", e)
      } finally {
        setIsLoading(false) // 無論成功或失敗，最後都解除讀取動畫
      }
    };

    // 4-5. 防呆控制：根據呼叫來源決定是否需經過確認視窗攔截
    if (isInitialLoad) {
      doLoad();
    } else {
      triggerActionWithCheck(doLoad);
    }
  }
  // ============================
  // 後端 API 與會話管理結束
  // ============================


  // ============================
  // 語音合成 (TTS) 與互動輔助開始
  // ============================
  // 1. 停止語音朗讀邏輯
  const stopSpeaking = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
    }
    setSpeakingIndex(null)
    setIsPaused(false)
  }

  // 2. 進階語音朗讀控制 (支援播放、暫停、繼續)
  const handleSpeak = (text, index) => {
    // 2-1. 檢查瀏覽器 API 支援度
    if (!('speechSynthesis' in window)) {
      return alert('您的瀏覽器不支援語音朗讀功能。')
    }

    // 2-2. 暫停與繼續判斷：若點擊同一個正在朗讀的訊息
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

    // 2-3. 初始化新朗讀：停止舊語音，設定新訊息
    stopSpeaking()
    setSpeakingIndex(index)
    setIsPaused(false)

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'zh-TW'
    utterance.rate = 1.0

    // 2-4. 語音篩選：優先選用語音品質較佳的 Yating 或 Google 引擎
    const voices = window.speechSynthesis.getVoices()
    const preferredVoice = voices.find(voice =>
      voice.lang.includes('zh-TW') && (voice.name.includes('Yating') || voice.name.includes('Google'))
    ) || voices.find(voice => voice.lang.includes('zh-TW'))

    if (preferredVoice) {
      utterance.voice = preferredVoice
    }

    // 2-5. 綁定事件：結束或報錯時重置 UI
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

  // 3. 一鍵複製邏輯
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
  // 1. 定義彈窗與選單追蹤狀態
  const [activeMenuId, setActiveMenuId] = useState(null) 

  // 2. 監聽側邊欄關閉事件，同步關閉展開的選單
  useEffect(() => {
    if (!isSidebarOpen) {
      setActiveMenuId(null);
    }
  }, [isSidebarOpen]);

  // 定義對話框狀態
  const [renameModal, setRenameModal] = useState({ isOpen: false, sessionId: null, oldTitle: '', newTitle: '' })
  const [deleteModal, setDeleteModal] = useState({ isOpen: false, sessionId: null })

  // 3. 開啟重新命名彈窗
  const handleRenameSession = (e, sessionId, oldTitle) => {
    e.stopPropagation()
    setActiveMenuId(null) // 關閉側邊欄原生下拉選單
    setRenameModal({ isOpen: true, sessionId, oldTitle, newTitle: oldTitle })
  }

  // 4. 執行重新命名 API 要求
  const confirmRename = async () => {
    const { sessionId, oldTitle, newTitle } = renameModal
    const trimmedTitle = newTitle.trim()

    if (!trimmedTitle || trimmedTitle === oldTitle) {
      setRenameModal({ isOpen: false, sessionId: null, oldTitle: '', newTitle: '' })
      return
    }

    try {
      // 傳送 PUT 請求更新標題
      const res = await fetch(`/api/chat_sessions/${sessionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: trimmedTitle })
      })
      if (res.ok) fetchSessions() // 成功後刷新側邊欄
    } catch (error) {
      console.error("重命名失敗", error)
    }
    setRenameModal({ isOpen: false, sessionId: null, oldTitle: '', newTitle: '' })
  }

  // 5. 開啟刪除對話彈窗
  const handleDeleteSession = (e, sessionId) => {
    e.stopPropagation()
    setActiveMenuId(null)
    setDeleteModal({ isOpen: true, sessionId })
  }

  // 6. 執行刪除 API 要求
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
  // 1. 執行訊息傳送與畫面更新
  const handleSendMessage = async (textToSend) => {
    // 1-1. 空白防呆與重複點擊防護
    const messageContent = textToSend || inputText
    if (!messageContent.trim() || isLoading) return

    // 1-2. 傳送時立即中斷語音與清除倒數
    stopSpeaking()
    setCountdown(null)

    // 1-3. 更新畫面狀態，並移除畫面上殘留的舊按鈕選項
    setMessages(prev => {
      const newMessages = prev.map(m => ({ ...m }));
      if (newMessages.length > 0) delete newMessages[newMessages.length - 1].options;
      return [...newMessages, { role: 'user', text: messageContent }];
    })

    setInputText('')
    // 開啟「思考中」動畫
    setIsLoading(true)

    try {
      // 1-4. 封裝資料並向後端發起 POST 請求
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

      // 1-5. 動畫狀態解除：真人模式立刻解鎖；AI 模式則等待 WebSocket 回應
      if (isHumanMode) {
        setIsLoading(false)
      }

      // 1-6. 更新首次對話的 session_id，並重整側邊欄列表
      if (!currentSessionId && data.session_id) {
        setCurrentSessionId(data.session_id)
        fetchSessions() // 這裡只純粹刷新側邊欄，不會再干擾畫面上的按鈕了！
      }

    } catch (error) {
      console.error('連線失敗', error)
      setIsLoading(false)
      setMessages(prev => [...prev, { role: 'ai', text: '【系統提示】伺服器連線失敗，請確認後端 API 是否已啟動。' }])
    }
  }

  // 2. 觸發新增對話：透過防呆包裝，清除當前狀態以重新開始
  const handleNewChat = () => {
    // 改用自訂檢查函式包覆
    triggerActionWithCheck(() => {
      // 【修改】：如果是手機版，點擊新增對話後自動隱藏側邊欄
      if (window.innerWidth < 768) setIsSidebarOpen(false)
      stopSpeaking()
      setCountdown(null) // 新增對話時，強制關閉倒數計時器
      setCurrentSessionId(null)
      setMessages([{ role: 'ai', text: '你好！我是這位求職者的專屬 AI 助理。您可以重新開始提問！' }])
      setIsHumanMode(false)
    })
  }
  // ============================
  // 訊息傳送核心邏輯結束
  // ============================


  // ============================
  // 畫面渲染區塊開始
  // ============================
  // 1. 若身分尚在初始化，渲染滿版 Loading 防止畫面破圖
  if (isInitializingId) {
    return (
      <div className="absolute inset-0 bg-gray-100 dark:bg-gray-900 flex flex-col items-center justify-center transition-colors duration-300">
        <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
        <p className="text-gray-600 dark:text-gray-300 font-medium tracking-widest animate-pulse">系統環境初始化中...</p>
      </div>
    );
  }

  return (
    // 2. 主視窗外層容器：設定滿版屬性與響應式背景
    <div className="absolute inset-0 bg-gray-100 dark:bg-gray-900 font-sans flex overflow-hidden transition-colors duration-300">

      {/* ============================ */}
      {/* 側邊欄 (Sidebar) 區域開始      */}
      {/* ============================ */}
      <div
        // 3. 側邊欄容器：利用行內樣式 visibility 控制隱身，徹底解決初始掛載的閃爍黑影
        style={{ visibility: isMounted ? 'visible' : 'hidden' }}
        className={`fixed md:relative inset-y-0 left-0 z-[60] md:z-40 w-64 bg-gray-900 dark:bg-black text-white transform flex flex-col shadow-2xl shrink-0 
          ${isMounted ? 'transition-transform duration-300 ease-in-out' : ''} 
          ${isSidebarOpen ? 'translate-x-0 ml-0' : '-translate-x-full md:-ml-64'}
        `}
      >
        <div className="p-4 border-b border-gray-700 flex justify-between items-center">
          <span className="font-bold text-lg tracking-wide">對話紀錄</span>
          <button onClick={() => setIsSidebarOpen(false)} className="md:hidden p-1 hover:bg-gray-800 rounded">✖</button>
        </div>

        {/* 4. 新增對話控制按鈕 */}
        <div className="p-3 border-b border-gray-700">
          <button onClick={handleNewChat} className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-sm font-semibold py-2 rounded-lg transition">
            <span>＋</span> 新增面試提問
          </button>
        </div>

        {/* 5. 歷史對話清單渲染迴圈 */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin">
          {chatHistory.length === 0 && <div className="text-gray-500 text-sm text-center mt-4">目前尚無歷史紀錄</div>}

          {chatHistory.map((history) => (
            <div key={history.id} className={`relative flex items-center justify-between w-full px-2 py-1.5 text-sm rounded-lg transition group ${currentSessionId === history.id ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`}
            >
              {/* 5-1. 對話標題按鈕 */}
              <button onClick={() => loadSession(history.id)} className="flex-1 text-left truncate pl-1 pr-2 py-1">
                💬 {history.title}
              </button>

              {/* 5-2. 動作選單展開按鈕 */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveMenuId(activeMenuId === history.id ? null : history.id);
                }}
                className="p-1 px-2 rounded opacity-100 md:opacity-0 group-hover:opacity-100 hover:bg-black/30 transition flex-shrink-0"
              >
                ⋮
              </button>

              {/* 5-3. 動作下拉選單實體 (重命名與刪除) */}
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
      {/* 6. 手機版遮罩：層級提升確保蓋住其他元件，強迫使用者透過點擊外圍關閉側邊欄 */}
      {isSidebarOpen && (
        <div className="absolute inset-0 bg-black/50 z-[55] md:hidden transition-opacity" onClick={() => setIsSidebarOpen(false)} />
      )}
      {/* ============================ */}
      {/* 側邊欄全螢幕透明遮罩結束        */}
      {/* ============================ */}

      {/* ============================ */}
      {/* 主聊天畫面區塊開始             */}
      {/* ============================ */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">

        {/* 7. 動態標題列：依據真人或 AI 模式自動切換背景色系 */}
        <header className={`text-white p-4 shadow-md flex items-center shrink-0 transition-colors duration-500 relative z-40 ${isHumanMode ? 'bg-orange-500 dark:bg-orange-600' : 'bg-blue-600 dark:bg-gray-800'
          }`}>
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-1.5 hover:bg-white/20 rounded-lg transition mr-2"
          >
            ☰
          </button>

          {/* 8. 佈景主題切換選單區塊 */}
          <div className="relative mr-3">
            <button
              onClick={() => setIsThemeMenuOpen(!isThemeMenuOpen)}
              className="p-1.5 hover:bg-white/20 rounded-lg transition text-xl flex-shrink-0"
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

          {/* 9. 標題文字區 */}
          <h1 className="text-lg font-bold flex-1 text-left truncate flex items-center gap-2">
            {isHumanMode ? (
              <><span>👩‍💼</span> 真人客服服務中...</>
            ) : (
              <><span>🤖</span> 專屬 AI 助理</>
            )}
          </h1>

          <div className="w-12"></div>
        </header>

        {/* 10. 核心對話內容渲染區域 */}
        <main ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 md:max-w-3xl md:mx-auto md:w-full">
          {messages.map((msg, index) => (
            <div key={index} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              
              {/* 10-1. 訊息泡泡與渲染邏輯 */}
              <div className={`max-w-[85%] rounded-2xl p-3 shadow-sm leading-relaxed ${msg.role === 'user'
                ? (isHumanMode ? 'bg-orange-500 dark:bg-orange-600 text-white rounded-br-none' : 'bg-blue-500 dark:bg-blue-600 text-white rounded-br-none')
                : 'bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 rounded-bl-none border border-gray-200 dark:border-gray-700'
                }`}>
                {/* AI 訊息使用 Markdown 渲染；其餘(如真人、用戶)維持純文字換行渲染 */}
                {msg.role === 'ai' ? (
                  <div className="markdown-body space-y-2 text-[15px]">
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap text-[15px] break-words">{msg.text}</div>
                )}
              </div>

              {/* 10-2. 動態選項按鈕渲染 */}
              {msg.options && msg.options.length > 0 && (
                <div className="flex gap-2 mt-2 ml-1">
                  {msg.options.map((opt, i) => {
                    const isLastMessage = index === messages.length - 1;
                    return (
                      <button
                        key={i}
                        onClick={() => handleSendMessage(opt)}
                        disabled={isLoading}
                        className="bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 border border-blue-600 dark:border-blue-400 px-4 py-1.5 rounded-full text-sm font-medium hover:bg-blue-50 dark:hover:bg-gray-700 transition shadow-sm disabled:opacity-50"
                      >
                        {opt}
                      </button>
                    )
                  })}
                </div>
              )}

              {/* 10-3. TTS 朗讀與複製工具列 (僅針對非用戶訊息顯示) */}
              {msg.role !== 'user' && (
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

          {/* 10-4. 思考中的載入動畫 */}
          {isLoading && (
            <div className="flex flex-col items-start mt-4">
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

          {/* 11. 快捷鍵區塊 (根據模式動態切換) */}
          <div className="px-4 py-2 transition-all duration-300">
            <div className="flex gap-2 overflow-x-auto scrollbar-none md:max-w-3xl md:mx-auto">
              {!isHumanMode ? (
                // 11-1. AI 模式下顯示常規快捷問題
                quickChips.map((chip, idx) => (
                  <button key={idx} onClick={() => handleSendMessage(chip)} disabled={isLoading}
                    className="whitespace-nowrap bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-gray-700 text-xs font-medium px-3 py-1.5 rounded-full border border-blue-200 dark:border-gray-600 shadow-sm transition disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                  >
                    {chip}
                  </button>
                ))
              ) : (
                // 11-2. 真人模式下顯示專屬逃生門，觸發重啟 AI
                <button onClick={() => handleSendMessage("問題已解決，謝謝客服！")} disabled={isLoading}
                  className="whitespace-nowrap bg-orange-50 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 hover:bg-orange-100 dark:hover:bg-orange-800 text-xs font-bold px-4 py-1.5 rounded-full border border-orange-200 dark:border-orange-700 shadow-sm transition disabled:opacity-50 disabled:cursor-not-allowed shrink-0 flex items-center gap-1"
                >
                  ⏹️ 結束真人服務 (轉回AI)
                </button>
              )}
            </div>
          </div>

          {/* 12. 動態進度條倒數計時橫幅 */}
          {countdown !== null && countdown > 0 && (
            <div className="px-4 pb-3 md:max-w-3xl md:mx-auto animate-fade-in">
              <div className={`relative overflow-hidden flex items-center justify-center gap-2 py-1.5 px-4 rounded-full border shadow-sm text-sm font-bold tracking-wide transition-colors z-0 bg-white dark:bg-gray-800 ${
                  countdown <= 10 
                    ? 'border-red-300 dark:border-red-700 text-red-600 dark:text-red-400' 
                    : 'border-orange-300 dark:border-orange-700 text-orange-600 dark:text-orange-400'
                }`}>
                {/* 12-1. 內層遮罩：寬度隨時間縮減以實現水準退去動畫 */}
                <div 
                  className={`absolute inset-y-0 left-0 -z-10 transition-all duration-1000 ease-linear ${
                    countdown <= 10 ? 'bg-red-100 dark:bg-red-900/30' : 'bg-orange-100 dark:bg-orange-900/30'
                  }`}
                  style={{ width: `${(countdown / 30) * 100}%` }}
                />
                <span className={countdown <= 10 ? 'animate-pulse' : ''}>⏳</span>
                <span>系統等待回應中，將於 {countdown} 秒後失效</span>
              </div>
            </div>
          )}

          {/* 13. 訊息輸入框與送出按鈕區域 */}
          <div className="px-4 pb-4">
            <div className={`flex items-end gap-2 md:max-w-3xl md:mx-auto relative bg-white dark:bg-gray-800 p-2 rounded-3xl shadow-[0_2px_15px_rgba(0,0,0,0.08)] border transition-colors duration-300 ${isHumanMode ? 'border-orange-300 dark:border-orange-700' : 'border-gray-200 dark:border-gray-700'}`}>

              {/* 13-1. 多行輸入文字框區塊 */}
              <textarea
                ref={inputRef}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  // 判斷：按下 Enter 且沒有按住 Shift 時，執行送出
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault(); // 阻止原生 Enter 的換行行為
                    handleSendMessage();
                  }
                }}
                placeholder={isLoading ? "訊息傳送中..." : (isHumanMode ? "輸入訊息給客服... " : "請問關於序亞的問題... ")}
                disabled={isLoading}
                rows={1}
                className="flex-1 bg-transparent px-4 py-2 focus:outline-none text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 transition-colors text-base resize-none max-h-32 overflow-y-auto min-h-[40px] scrollbar-thin"
              />
              
              {/* 13-2. 送出觸發按鈕 */}
              <button
                onClick={() => handleSendMessage()}
                disabled={isLoading}  // 按鈕維持鎖定，防止使用者狂按送出
                className={`text-white px-6 py-2 rounded-full font-semibold transition disabled:cursor-not-allowed text-sm h-10 ${isHumanMode
                  ? 'bg-orange-500 hover:bg-orange-600 disabled:bg-orange-300'
                  : 'bg-blue-600 dark:bg-blue-700 hover:bg-blue-700 dark:hover:bg-blue-600 disabled:bg-blue-300'
                  }`}
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
      
      {/* 14. 重新命名對話彈窗 */}
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

      {/* 15. 刪除確認彈窗 */}
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
      
      {/* 16. 終止轉接自訂警告視窗 */}
      {customConfirm.isOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden border border-gray-200 dark:border-gray-700">
            <div className="p-6 text-center">
              <div className="w-12 h-12 rounded-full bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center mx-auto mb-4">
                <span className="text-orange-500 text-xl">⚠️</span>
              </div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">終止轉接確認</h3>
              <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed">系統正在處理轉接程序，如果現在離開或開啟新對話，將會終止真人轉接。確定要離開嗎？</p>
            </div>
            <div className="flex bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setCustomConfirm({ isOpen: false, pendingAction: null })}
                className="flex-1 py-3 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition font-medium"
              >
                取消
              </button>
              <div className="w-[1px] bg-gray-200 dark:bg-gray-700"></div>
              <button
                onClick={executePendingAction}
                className="flex-1 py-3 text-orange-600 dark:text-orange-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition font-bold"
              >
                確定離開
              </button>
            </div>
          </div>
        </div>
      )}
      {/* ============================ */}
      {/* 彈跳視窗 (Modals) 區塊結束      */}
      {/* ============================ */}
    </div>
  )
}

export default App
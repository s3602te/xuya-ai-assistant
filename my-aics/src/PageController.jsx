// ============================
// 元件與模組引入開始
// ============================
import { useState } from 'react'
import Chatroom from './Chatroom.jsx'        // 載入 AI 對話聊天室元件
import Architecture from './Architecture.jsx'  // 載入系統架構展示元件
import Deployment from './Deployment.jsx'    // 載入 CI/CD 部署流程展示元件
// ============================
// 元件與模組引入結束
// ============================

export default function PageController() {
  // ============================
  // 狀態管理 (State) 開始
  // ============================
  const [currentPage, setCurrentPage] = useState('chat') // 1. 初始化當前顯示頁面的狀態，預設為 'chat' (聊天室)
  const [isMenuOpen, setIsMenuOpen] = useState(false)    // 2. 初始化全螢幕選單的開關狀態，預設為 false (關閉)
  // ============================
  // 狀態管理 (State) 結束
  // ============================

  // ============================
  // 頁面切換邏輯處理開始
  // ============================
  const handleSwitchPage = (pageKey) => {
    setCurrentPage(pageKey) // 1. 接收選單傳入的 pageKey，更新當前頁面狀態
    setIsMenuOpen(false)    // 2. 切換頁面後，自動觸發選單關閉動作
  }
  // ============================
  // 頁面切換邏輯處理結束
  // ============================

  return (
    /* ============================ */
    /* 核心外框佈局開始               */
    /* ============================ */
    /* 採用 100dvh 鎖定全螢幕，精準解決行動裝置 (如 iOS Safari) 網址列造成的捲軸穿透與版面溢出問題 */
    <div className={`fixed inset-0 w-full h-[100dvh] overflow-hidden transition-colors duration-300 ${
      currentPage === 'chat' ? 'bg-gray-100 dark:bg-gray-900' : 'bg-gray-900 dark:bg-black'
    }`}>
      
      {/* ============================ */}
      {/* 1. 懸浮導覽按鈕開始             */}
      {/* ============================ */}
      {/* 1. 設定 z-50 確保按鈕永遠位於最頂層，不受任何子元件圖層干擾 */}
      <button 
        onClick={() => setIsMenuOpen(!isMenuOpen)} // 2. 點擊時反轉 isMenuOpen 狀態 (開啟/關閉)
        className="fixed top-4 right-4 z-50 w-12 h-12 bg-gray-900/80 backdrop-blur-md text-white rounded-full flex items-center justify-center shadow-lg border border-gray-700 hover:bg-gray-800 active:scale-95 transition-all"
        aria-label="Toggle Menu"
      >
        {/* 3. 根據選單狀態動態切換圖示：開啟時顯示 ✕，關閉時顯示 ☰ */}
        {isMenuOpen ? (
          <span className="text-xl font-bold">✕</span>
        ) : (
          <span className="text-xl">☰</span>
        )}
      </button>
      {/* ============================ */}
      {/* 1. 懸浮導覽按鈕結束             */}
      {/* ============================ */}

      {/* ============================ */}
      {/* 2. 全螢幕毛玻璃選單開始         */}
      {/* ============================ */}
      {/* 1. 當 isMenuOpen 為 true 時，才會掛載此選單區塊 */}
      {/* 2. 設定 z-40 確保圖層位於懸浮按鈕之下、主畫面之上 */}
      {isMenuOpen && (
        <div className="fixed inset-0 z-40 bg-black/60 backdrop-blur-md flex flex-col items-center justify-center space-y-6 animate-fade-in">
          <p className="text-gray-400 text-sm tracking-widest uppercase font-semibold mb-2">
            請選擇展示頁面
          </p>

          {/* 3. 各頁面按鈕：點擊時執行 handleSwitchPage 並傳入對應的 pageKey */}
          <button 
            onClick={() => handleSwitchPage('chat')} 
            className={`w-64 py-3.5 rounded-2xl font-bold text-lg shadow-md transition-all ${
              currentPage === 'chat' 
                ? 'bg-blue-600 text-white ring-2 ring-blue-400 shadow-blue-500/30' 
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            💬 專屬 AI 聊天室
          </button>

          <button 
            onClick={() => handleSwitchPage('app1')} 
            className={`w-64 py-3.5 rounded-2xl font-bold text-lg shadow-md transition-all ${
              currentPage === 'app1' 
                ? 'bg-green-600 text-white ring-2 ring-green-400 shadow-green-500/30' 
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            🚀 架構使用技術展示
          </button>

          <button 
            onClick={() => handleSwitchPage('app2')} 
            className={`w-64 py-3.5 rounded-2xl font-bold text-lg shadow-md transition-all ${
              currentPage === 'app2' 
                ? 'bg-pink-600 text-white ring-2 ring-pink-400 shadow-pink-500/30' 
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            🎨 Docker CI/CD展示
          </button>
        </div>
      )}
      {/* ============================ */}
      {/* 2. 全螢幕毛玻璃選單結束         */}
      {/* ============================ */}

      {/* ============================ */}
      {/* 3. 主畫面內容展示區開始         */}
      {/* ============================ */}
      {/* 1. 利用 w-full h-full 佔滿 100% 核心外框空間 */}
      {/* 2. 依據 currentPage 狀態進行動態條件渲染 (Conditional Rendering) */}
      <div className="w-full h-full">
        {currentPage === 'chat' && <Chatroom />}
        {currentPage === 'app1' && <Architecture />}
        {currentPage === 'app2' && <Deployment />}
      </div>
      {/* ============================ */}
      {/* 3. 主畫面內容展示區結束         */}
      {/* ============================ */}

    </div>
    /* ============================ */
    /* 核心外框佈局結束               */
    /* ============================ */
  )
}
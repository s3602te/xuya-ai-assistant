// ============================
// 核心模組與樣式引入開始
// ============================
import { StrictMode } from 'react'                // 1. 載入 React 嚴格模式 (用於開發環境的潛在錯誤檢查與警告)
import { createRoot } from 'react-dom/client'     // 2. 載入 React 18 的最新 DOM 渲染 API
import './index.css'                              // 3. 載入全域 CSS 樣式 (包含 Tailwind CSS 的基礎設定)

import PageController from './PageController.jsx' // 4. 載入應用程式的總路由器/控制器 (App Shell 核心)
// ============================
// 核心模組與樣式引入結束
// ============================


// ============================
// 應用程式初始化與渲染開始
// ============================
// 1. 尋找 DOM 節點：鎖定 index.html 檔案中 id 為 'root' 的空 <div>
// 2. 建立根節點：使用 createRoot 建立 React 的虛擬 DOM 根節點
createRoot(document.getElementById('root')).render(
  <StrictMode>
    {/* 3. 畫面掛載：將 PageController 作為唯一的頂層元件 (Entry Point) 注入到畫布中 */}
    <PageController /> 
  </StrictMode>,
)
// ============================
// 應用程式初始化與渲染結束
// ============================
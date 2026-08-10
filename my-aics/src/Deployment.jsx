// ============================
// 元件與模組引入開始
// ============================
import { useState, useEffect, useRef } from 'react'
// ============================
// 元件與模組引入結束
// ============================

export default function Deployment() {
  // ============================
  // DOM 參考設定開始
  // ============================
  // 1. 宣告容器的 DOM 參考，用於後續精準控制內部滾動條位置
  const containerRef = useRef(null);
  // ============================
  // DOM 參考設定結束
  // ============================

  // ============================
  // 視窗與容器滾動控制開始
  // ============================
  // 1. 核心修復：確保切換到此頁面時，強制讓「此容器內部」的捲軸歸零置頂，解決排版溢出的連帶問題
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, []);
  // ============================
  // 視窗與容器滾動控制結束
  // ============================
  
  // ============================
  // 狀態管理開始
  // ============================
  // 1. 模擬終端機指令的打字機特效狀態，追蹤目前互動展示的步驟 (預設為 1)
  const [activeStep, setActiveStep] = useState(1)
  // ============================
  // 狀態管理結束
  // ============================

  // ============================
  // 核心外框佈局開始
  // ============================
  return (
    // 1. 綁定 containerRef 參考，並使用 h-[100dvh] 鎖定高度與 overflow-y-auto 允許內部內容獨立滾動
    <div ref={containerRef} className="w-full h-[100dvh] p-8 bg-gray-900 overflow-y-auto animate-fade-in font-sans">
      <div className="max-w-5xl mx-auto">
        
        {/* 頁面標題與描述區塊 */}
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400 mb-4">
            持續整合與部署 (CI/CD Pipeline)
          </h1>
          <p className="text-gray-400">
            全自動化交付流程：從 Git Push 到 Docker 容器化上線，實現零停機更新。
          </p>
        </div>

        {/* ============================ */}
        {/* 核心流水線展示區 (互動卡片) 開始 */}
        {/* ============================ */}
        <div className="flex flex-col md:flex-row gap-6 mb-12">
          
          {/* 1. Step 1: GitHub 版本控制 */}
          {/* 滑鼠懸停時觸發更新 activeStep 狀態為 1 */}
          <div 
            onMouseEnter={() => setActiveStep(1)}
            className={`flex-1 p-6 rounded-2xl border-2 transition-all duration-300 cursor-pointer ${
              activeStep === 1 ? 'bg-gray-800 border-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.3)]' : 'bg-gray-800/50 border-gray-700'
            }`}
          >
            <div className="text-4xl mb-4">🐙</div>
            <h3 className="text-xl font-bold text-white mb-2">1. 版本控制</h3>
            <p className="text-gray-400 text-sm">
              開發完成後，透過 Git 將程式碼推送至 GitHub 儲存庫，作為自動化的觸發起點。
            </p>
          </div>

          {/* 2. Step 2: GitHub Actions 自動化腳本 */}
          {/* 滑鼠懸停時觸發更新 activeStep 狀態為 2 */}
          <div 
            onMouseEnter={() => setActiveStep(2)}
            className={`flex-1 p-6 rounded-2xl border-2 transition-all duration-300 cursor-pointer ${
              activeStep === 2 ? 'bg-gray-800 border-purple-500 shadow-[0_0_20px_rgba(168,85,247,0.3)]' : 'bg-gray-800/50 border-gray-700'
            }`}
          >
            <div className="text-4xl mb-4">⚙️</div>
            <h3 className="text-xl font-bold text-white mb-2">2. GitHub Actions</h3>
            <p className="text-gray-400 text-sm">
              偵測到 main 分支更新後，自動觸發 CI/CD 腳本，進行環境建置、依賴安裝與測試。
            </p>
          </div>

          {/* 3. Step 3: Docker 容器化部署 */}
          {/* 滑鼠懸停時觸發更新 activeStep 狀態為 3 */}
          <div 
            onMouseEnter={() => setActiveStep(3)}
            className={`flex-1 p-6 rounded-2xl border-2 transition-all duration-300 cursor-pointer ${
              activeStep === 3 ? 'bg-gray-800 border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.3)]' : 'bg-gray-800/50 border-gray-700'
            }`}
          >
            <div className="text-4xl mb-4">🐳</div>
            <h3 className="text-xl font-bold text-white mb-2">3. Docker 部署</h3>
            <p className="text-gray-400 text-sm">
              透過 Dockerfile 將前端靜態資源打包成獨立的容器 (Container)，確保執行環境高度一致並推送到主機。
            </p>
          </div>
        </div>
        {/* ============================ */}
        {/* 核心流水線展示區 (互動卡片) 結束 */}
        {/* ============================ */}


        {/* ============================ */}
        {/* 互動式終端機展示區開始           */}
        {/* ============================ */}
        <div className="bg-black rounded-xl border border-gray-700 shadow-2xl overflow-hidden">
          
          {/* 終端機頂部控制列 (模擬 macOS 視窗風格) */}
          <div className="bg-gray-800 px-4 py-2 flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="ml-4 text-gray-400 text-xs font-mono">bash - deployment_process</span>
          </div>
          
          {/* 終端機內容區塊 (根據 activeStep 進行動態條件渲染) */}
          <div className="p-6 font-mono text-sm">
            
            {/* 當前狀態為 Step 1 時渲染 */}
            {activeStep === 1 && (
              <div className="text-green-400 animate-pulse">
                $ git add .<br/>
                $ git commit -m "feat: upgrade to modern React + Vite"<br/>
                $ git push origin main<br/>
                <span className="text-gray-400 mt-2 block">➔ Push successful. Triggering workflow...</span>
              </div>
            )}
            
            {/* 當前狀態為 Step 2 時渲染 */}
            {activeStep === 2 && (
              <div className="text-purple-400 animate-pulse">
                $ actions/checkout@v4<br/>
                $ npm install<br/>
                $ npm run build<br/>
                <span className="text-gray-400 mt-2 block">➔ Build completed in 2.4s. Assets ready for containerization.</span>
              </div>
            )}
            
            {/* 當前狀態為 Step 3 時渲染 */}
            {activeStep === 3 && (
              <div className="text-emerald-400 animate-pulse">
                $ docker build -t my-aics-frontend:latest .<br/>
                $ docker stop my-aics-container || true<br/>
                $ docker run -d -p 80:80 --name my-aics-container my-aics-frontend:latest<br/>
                <span className="text-gray-400 mt-2 block">➔ Container running securely on port 80. Deployment LIVE.</span>
              </div>
            )}

          </div>
        </div>
        {/* ============================ */}
        {/* 互動式終端機展示區結束           */}
        {/* ============================ */}

      </div>
    </div>
  )
}
// ============================
// 核心外框佈局結束
// ============================
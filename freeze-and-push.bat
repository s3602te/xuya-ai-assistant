@echo off
chcp 65001 >nul
echo =========================================
echo 🚀 開始執行 Xuya-AI 自動化更新與推送程序...
echo =========================================

echo [1/4] 啟動虛擬環境並更新 requirements.txt...
call my-aics-backend\myai\Scripts\activate
cd my-aics-backend
pip freeze > requirements.txt
cd ..

echo [2/4] 將所有變更加入 Git 追蹤...
git add .

echo [3/4] 提交程式碼...
set /p commit_msg="請輸入這次更新的 Git 註解 (Commit Message): "
git commit -m "%commit_msg%"

echo [4/4] 推送至 GitHub 觸發 CI/CD...
git push

echo =========================================
echo 🎉 推送完成！請前往 GitHub 檢查 CI/CD 狀態。
echo =========================================
pause
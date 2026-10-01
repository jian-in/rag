@echo off
chcp 65001 >nul
REM ============================================================
REM  停止 RAG 知识库服务（结束占用端口 7860 的进程）
REM ============================================================

set "FOUND="
echo [停止] 正在查找占用端口 7860 的进程 ...

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7860" ^| findstr "LISTENING"') do (
    set "FOUND=1"
    echo        结束进程 PID=%%P
    taskkill /PID %%P /F >nul 2>&1
)

if not defined FOUND (
    echo        未发现监听 7860 的进程，服务可能未启动。
) else (
    echo [完成] RAG 服务已停止。
)

pause

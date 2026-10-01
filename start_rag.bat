@echo off
chcp 65001 >nul
REM ============================================================
REM  启动 RAG 知识库服务（端口 7860）
REM  双击运行；日志直接显示在本窗口，按 Ctrl+C 可停止
REM ============================================================

cd /d "%~dp0"

if not exist "RAG\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 RAG\Scripts\python.exe
    echo        请确认已在项目根目录，且虚拟环境存在。
    pause
    exit /b 1
)

echo [启动] 正在启动 RAG 服务，端口 7860 ...
echo        启动较慢（需加载 torch / embedding 模型），请耐心等待。
echo        就绪后访问: http://127.0.0.1:7860
echo        停止服务: 在本窗口按 Ctrl+C，或运行 stop_rag.bat
echo ------------------------------------------------------------

"RAG\Scripts\python.exe" main.py

echo ------------------------------------------------------------
echo [已退出] 服务已停止。
pause

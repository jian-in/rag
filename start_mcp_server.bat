@echo off
REM Start the RAG MCP HTTP server (streamable HTTP /mcp + SSE /sse on 8766).
REM Used by the logon scheduled task "RAG-MCP-Server"; can also be run by hand.
cd /d "%~dp0"
set "MCP_HTTP_PORT=8766"
"RAG\Scripts\python.exe" -u mcp_http_server.py

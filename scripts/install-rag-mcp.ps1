# Register the RAG knowledge-base MCP HTTP server (http://127.0.0.1:8766/mcp)
# to start silently at Windows sign-in, mirroring install-memory-hub.ps1.
#
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-rag-mcp.ps1 [-Start]
#   -Start  also launch the server now (skipped if it already answers /health)

param(
    [switch]$Start
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$pythonwPath = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'RAG\Scripts\pythonw.exe')).Path
$serverScript = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'mcp_http_server.py')).Path

$startupRoot = [Environment]::GetFolderPath('Startup')
if (-not (Test-Path -LiteralPath $startupRoot -PathType Container)) {
    throw "Windows Startup folder was not found: $startupRoot"
}

$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path $startupRoot 'RagKnowledgeMCP.lnk'
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonwPath
$shortcut.Arguments = '"' + $serverScript + '"'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.WindowStyle = 7
$shortcut.IconLocation = "$pythonwPath,0"
$shortcut.Description = 'Start the RAG knowledge-base MCP server (127.0.0.1:8766) at Windows sign-in.'
$shortcut.Save()

$started = $false
if ($Start) {
    $healthy = $false
    $request = @{ Uri = 'http://127.0.0.1:8766/health'; TimeoutSec = 2 }
    if ($PSVersionTable.PSVersion.Major -ge 7) { $request.NoProxy = $true }  # 5.1 has no -NoProxy
    try {
        $health = Invoke-RestMethod @request
        $healthy = $health.ok -eq $true
    } catch {
        $healthy = $false
    }
    if (-not $healthy) {
        Start-Process -FilePath $pythonwPath -ArgumentList ('"' + $serverScript + '"') -WorkingDirectory $projectRoot -WindowStyle Hidden
        $started = $true
    }
}

[pscustomobject]@{
    StartupShortcut = $shortcutPath
    ServiceExecutable = $pythonwPath
    Endpoint = 'http://127.0.0.1:8766/mcp'
    LogFile = Join-Path $env:LOCALAPPDATA 'RagKnowledgeMCP\server.log'
    StartedNow = $started
}

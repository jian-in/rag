# Stop the RAG knowledge-base MCP HTTP server started by install-rag-mcp.ps1 or
# start_mcp_server.bat. Only stops this project's python process listening on
# the MCP port, never an unrelated program that happens to use the port.

param(
    [int]$Port = 8766
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$expectedPythons = @(
    (Join-Path $projectRoot 'RAG\Scripts\python.exe')
    (Join-Path $projectRoot 'RAG\Scripts\pythonw.exe')
) | ForEach-Object { [System.IO.Path]::GetFullPath($_) }

$listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
if (-not $listeners) {
    Write-Output "No process is listening on port $Port."
    exit 0
}

foreach ($processId in $listeners) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if ($null -eq $process) { continue }
    $executable = if ($process.ExecutablePath) { [System.IO.Path]::GetFullPath($process.ExecutablePath) } else { '' }
    # The venv launcher re-executes the base interpreter, so also accept any
    # python whose command line runs this project's mcp_http_server.py.
    $isOurs = ($process.CommandLine -like '*mcp_http_server.py*') -and (
        ($expectedPythons -contains $executable) -or ($process.Name -like 'python*.exe')
    )
    if (-not $isOurs) {
        throw "PID $processId on port $Port is not this project's MCP server: $($process.CommandLine)"
    }
    Stop-Process -Id $processId
    Write-Output "Stopped RAG MCP server process $processId."
}

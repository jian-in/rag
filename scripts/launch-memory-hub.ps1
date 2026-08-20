param(
    [switch]$SyncOnly,
    [switch]$NoAutoSync
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot 'RAG\Scripts\python.exe'
$arguments = @('-m', 'memory_hub')

if ($SyncOnly) {
    $arguments += '--sync-only'
}
if ($NoAutoSync) {
    $arguments += '--no-auto-sync'
}

& $pythonPath @arguments
exit $LASTEXITCODE

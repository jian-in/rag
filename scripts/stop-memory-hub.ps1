$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$expectedPythons = @(
    (Resolve-Path -LiteralPath (Join-Path $projectRoot 'RAG\Scripts\python.exe')).Path
    (Resolve-Path -LiteralPath (Join-Path $projectRoot 'RAG\Scripts\pythonw.exe')).Path
)
$venvConfigPath = Join-Path $projectRoot 'RAG\pyvenv.cfg'
if (Test-Path -LiteralPath $venvConfigPath -PathType Leaf) {
    $homeLine = Get-Content -LiteralPath $venvConfigPath | Where-Object { $_ -match '^home\s*=\s*(.+)$' } | Select-Object -First 1
    if ($homeLine -match '^home\s*=\s*(.+)$') {
        $basePythonRoot = $Matches[1].Trim()
        foreach ($name in @('python.exe', 'pythonw.exe')) {
            $candidate = Join-Path $basePythonRoot $name
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $expectedPythons += [System.IO.Path]::GetFullPath($candidate)
            }
        }
    }
}
$expectedPythons = $expectedPythons | Select-Object -Unique
$stateRoot = Join-Path $env:LOCALAPPDATA 'FeiliMemoryHub'
$pidPath = Join-Path $stateRoot 'server.pid'

if (-not (Test-Path -LiteralPath $pidPath -PathType Leaf)) {
    Write-Output 'No managed Memory Hub process is recorded.'
    exit 0
}

$serverPid = 0
if (-not [int]::TryParse((Get-Content -LiteralPath $pidPath -Raw).Trim(), [ref]$serverPid)) {
    throw "Invalid PID file: $pidPath"
}

$process = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid"
if ($null -eq $process) {
    Remove-Item -LiteralPath $pidPath
    Write-Output 'The recorded Memory Hub process has already stopped.'
    exit 0
}

$resolvedExecutable = [System.IO.Path]::GetFullPath($process.ExecutablePath)
if ($expectedPythons -notcontains $resolvedExecutable -or $process.CommandLine -notlike '*-m memory_hub*') {
    throw "PID $serverPid does not belong to the managed Memory Hub process."
}

Stop-Process -Id $serverPid
Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
Write-Output "Stopped Feili Memory Hub process $serverPid."

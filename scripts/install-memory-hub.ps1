$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$pythonwPath = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'RAG\Scripts\pythonw.exe')).Path
$edgeCandidates = @(
    'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
)
$edgePath = $edgeCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1

if (-not $edgePath) {
    throw 'Microsoft Edge was not found in a standard installation path.'
}

$startupRoot = [Environment]::GetFolderPath('Startup')
$desktopRoot = [Environment]::GetFolderPath('Desktop')
if (-not (Test-Path -LiteralPath $startupRoot -PathType Container)) {
    throw "Windows Startup folder was not found: $startupRoot"
}
if (-not (Test-Path -LiteralPath $desktopRoot -PathType Container)) {
    throw "Desktop folder was not found: $desktopRoot"
}

$shell = New-Object -ComObject WScript.Shell

$startupShortcutPath = Join-Path $startupRoot 'FeiliMemoryHub.lnk'
$startupShortcut = $shell.CreateShortcut($startupShortcutPath)
$startupShortcut.TargetPath = $pythonwPath
$startupShortcut.Arguments = '-m memory_hub'
$startupShortcut.WorkingDirectory = $projectRoot
$startupShortcut.WindowStyle = 7
$startupShortcut.IconLocation = "$pythonwPath,0"
$startupShortcut.Description = 'Start Feili Personal Memory Hub at Windows sign-in.'
$startupShortcut.Save()

$desktopShortcutPath = Join-Path $desktopRoot '绯梨记忆观测台.lnk'
$desktopShortcut = $shell.CreateShortcut($desktopShortcutPath)
$desktopShortcut.TargetPath = $edgePath
$desktopShortcut.Arguments = '--app=http://127.0.0.1:8771/'
$desktopShortcut.WorkingDirectory = $projectRoot
$desktopShortcut.WindowStyle = 1
$desktopShortcut.IconLocation = "$edgePath,0"
$desktopShortcut.Description = 'Open Feili Personal Memory Observatory in an independent window.'
$desktopShortcut.Save()

[pscustomobject]@{
    StartupShortcut = $startupShortcutPath
    DesktopShortcut = $desktopShortcutPath
    ServiceExecutable = $pythonwPath
    WindowExecutable = $edgePath
}

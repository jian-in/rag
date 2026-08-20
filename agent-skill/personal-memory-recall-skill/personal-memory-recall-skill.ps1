$ErrorActionPreference = 'Stop'
$skillRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Get-Command python -ErrorAction SilentlyContinue

if ($null -eq $python) {
    [Console]::Error.WriteLine('{"error":"Python 3.10+ was not found.","error_type":"runtime","hint":"Install Python and retry."}')
    exit 1
}

& $python.Source (Join-Path $skillRoot 'scripts\memory_client.py') @args
exit $LASTEXITCODE

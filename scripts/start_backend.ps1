param(
    [int]$Port = 8010,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtualenv Python not found: $Python"
}

$env:PYTHONDONTWRITEBYTECODE = "1"
& $Python -m uvicorn app.main:app --reload --host $HostAddress --port $Port

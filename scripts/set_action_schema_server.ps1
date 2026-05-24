param(
    [Parameter(Mandatory = $true)]
    [string]$PublicUrl,
    [string]$InputPath = "action_schema.gpt-control.yaml",
    [string]$OutputPath = "action_schema.local-tunnel.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$PublicUrl = $PublicUrl.TrimEnd("/")
if ($PublicUrl -notmatch "^https://") {
    throw "PublicUrl must start with https://"
}

if (-not (Test-Path $InputPath)) {
    throw "Input schema not found: $InputPath"
}

$content = Get-Content $InputPath -Raw
$content = $content -replace "(?m)^servers:\r?\n\s+- url: .+$", "servers:`n  - url: $PublicUrl"
Set-Content -Path $OutputPath -Value $content -Encoding UTF8

Write-Host "Wrote $OutputPath with server URL $PublicUrl"

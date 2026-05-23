param(
    [string]$ConfigPath = "$env:USERPROFILE\.cloudflared\investment-agent-backend.yml"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared is not installed or not on PATH."
}

if (-not (Test-Path $ConfigPath)) {
    throw "Cloudflare tunnel config not found: $ConfigPath"
}

cloudflared tunnel --config $ConfigPath run investment-agent-backend

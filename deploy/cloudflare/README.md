# Cloudflare Tunnel Setup

This keeps the backend running on your PC while giving Custom GPT a public HTTPS URL.

## Target Architecture

```text
Custom GPT
  -> https://api.your-domain.com
  -> Cloudflare Tunnel
  -> http://127.0.0.1:8010 on this PC
  -> local workers and SQLite
```

## One-Time Setup

1. Install `cloudflared` from Cloudflare.
2. Log in:

```powershell
cloudflared tunnel login
```

3. Create a named tunnel:

```powershell
cloudflared tunnel create investment-agent-backend
```

4. Route your DNS hostname to the tunnel:

```powershell
cloudflared tunnel route dns investment-agent-backend api.your-domain.com
```

5. Copy `deploy/cloudflare/config.yml.example` to:

```text
%USERPROFILE%\.cloudflared\investment-agent-backend.yml
```

6. Edit the copied config:

```yaml
hostname: api.your-domain.com
credentials-file: C:/Users/user/.cloudflared/<actual-tunnel-id>.json
httpHostHeader: api.your-domain.com
```

## Start Locally

Start the API server:

```powershell
.\scripts\start_backend.ps1 -Port 8010
```

Start the tunnel:

```powershell
.\scripts\start_cloudflare_tunnel.ps1 -ConfigPath "$env:USERPROFILE\.cloudflared\investment-agent-backend.yml"
```

If PowerShell blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1 -Port 8010
powershell -ExecutionPolicy Bypass -File .\scripts\start_cloudflare_tunnel.ps1 -ConfigPath "$env:USERPROFILE\.cloudflared\investment-agent-backend.yml"
```

Check from a browser:

```text
https://api.your-domain.com/health
```

## Generate Custom GPT Schema

```powershell
.\scripts\set_action_schema_server.ps1 -PublicUrl "https://api.your-domain.com"
```

If PowerShell blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\set_action_schema_server.ps1 -PublicUrl "https://api.your-domain.com"
```

Paste `action_schema.local-tunnel.yaml` into the Custom GPT Actions schema window.

## Security Checklist

- Set a strong `BACKEND_API_KEY` in `.env`.
- Use the same value as `X-API-Key` in Custom GPT authentication.
- Do not commit tunnel credential JSON files or `cert.pem`.
- Keep the PC awake and connected to the internet while auto-trading is running.
- Restart the backend and workers after code changes.

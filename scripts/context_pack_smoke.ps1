param(
  [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
  [string]$Goal = "Scale Wohper from GLOBAL-L3ATTN smoke to controlled multi-agent runtime without context bloat",
  [string]$Out = "state/context_pack_smoke_2026-07-02.json"
)

$ErrorActionPreference = "Stop"

& $Python tools/context_pack.py `
  --goal $Goal `
  --mode plan `
  --config config/context.config.json `
  --out $Out `
  --format json
if ($LASTEXITCODE -ne 0) {
  throw "context_pack.py failed with exit code $LASTEXITCODE"
}

& $Python -m json.tool $Out | Out-Null
if ($LASTEXITCODE -ne 0) {
  throw "context pack output is not valid JSON"
}

Write-Host "CONTEXT_PACK_READY=$Out"
Get-Item $Out | Select-Object FullName,Length,LastWriteTime

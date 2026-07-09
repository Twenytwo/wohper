param(
    [string]$Root = "C:\WohperStorage",
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 9100,
    [string]$ServerScript = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found. Install Python 3 or add it to PATH."
}

New-Item -ItemType Directory -Force -Path $Root | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "experts") | Out-Null

if (-not $ServerScript) {
    $repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
    $ServerScript = Join-Path $repoRoot "tools\zc_expert_storage_server.py"
}

if (-not (Test-Path -LiteralPath $ServerScript)) {
    throw "Server script not found: $ServerScript"
}

Write-Host "Wohper expert storage"
Write-Host "  root: $Root"
Write-Host "  experts: $(Join-Path $Root "experts")"
Write-Host "  listen: http://$HostAddress`:$Port"
Write-Host ""
Write-Host "Keep this window open while the master is running."

py $ServerScript --root $Root --host $HostAddress --port $Port

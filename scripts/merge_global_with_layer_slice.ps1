param(
  [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
  [string]$GlobalCore = "models/wohper/GLM-5.2.INT4.GLOBAL-MICRO/dense_core.bin",
  [string]$LayerCore = "models/wohper/GLM-5.2.INT4.L3-L5-E0E1/dense_core.bin",
  [string]$OutDir = "models/wohper/GLM-5.2.INT4.GLOBAL-L3-L5-SMOKE",
  [string]$RemoteFetchEndpoint = ""
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$mergeArgs = @(
  "tools/merge_smoke_models.py",
  "--global-core", $GlobalCore,
  "--layer-core", $LayerCore,
  "--out", (Join-Path $OutDir "dense_core.bin"),
  "--index-out", (Join-Path $OutDir "dense_core.index.json"),
  "--experts-out-dir", (Join-Path $OutDir "experts")
)
if ($RemoteFetchEndpoint) {
  $mergeArgs += @("--remote-fetch-endpoint", $RemoteFetchEndpoint)
}

New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "experts") | Out-Null

& $Python @mergeArgs
if ($LASTEXITCODE -ne 0) {
  throw "merge_smoke_models.py failed with exit code $LASTEXITCODE"
}

Write-Host "MERGED_SMOKE_READY=$OutDir"
Get-ChildItem -Path $OutDir -Recurse -File | Select-Object FullName,Length,LastWriteTime

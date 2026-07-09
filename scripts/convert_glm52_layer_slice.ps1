param(
  [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
  [string]$MetadataDir = "models/zai-org/GLM-5.2",
  [string]$OutDir = "models/wohper/GLM-5.2.INT4.L3-L5-E0E1",
  [string]$LayerRange = "3,5",
  [int]$ExpertsPerLayer = 2,
  [int]$ActiveExperts = 1,
  [int]$ChunkMb = 8
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "experts") | Out-Null

& $Python tools/stream_convert_glm52.py `
  --metadata-dir $MetadataDir `
  --out (Join-Path $OutDir "dense_core.bin") `
  --experts-dir (Join-Path $OutDir "experts") `
  --index-out (Join-Path $OutDir "dense_core.index.json") `
  --resume-ledger "state/glm52_layer_slice_$($LayerRange.Replace(',', '_'))_ledger_2026-07-02.json" `
  --quant int4 `
  --layer-range $LayerRange `
  --experts-per-layer $ExpertsPerLayer `
  --active-experts $ActiveExperts `
  --chunk-mb $ChunkMb `
  --http-retries 8 `
  --retry-base-sleep 1.5 `
  --skip-existing
if ($LASTEXITCODE -ne 0) {
  throw "stream_convert_glm52.py failed with exit code $LASTEXITCODE"
}

Write-Host "LAYER_SLICE_READY=$OutDir"
Get-ChildItem -Path $OutDir -File | Select-Object Name,Length,LastWriteTime
Get-ChildItem -Path (Join-Path $OutDir "experts") -File | Select-Object Name,Length,LastWriteTime

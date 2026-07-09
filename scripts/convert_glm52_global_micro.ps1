param(
  [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
  [string]$MetadataDir = "models/zai-org/GLM-5.2",
  [string]$OutDir = "models/wohper/GLM-5.2.INT4.GLOBAL-MICRO",
  [int]$Rows = 4096,
  [int]$StartRow = 0
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "experts") | Out-Null
$safeName = (Split-Path -Leaf $OutDir) -replace '[^A-Za-z0-9_.-]', '_'
$ledgerPath = "state/glm52_global_${safeName}_rows_${StartRow}_${Rows}_ledger_2026-07-03.json"

& $Python tools/stream_convert_glm52.py `
  --metadata-dir $MetadataDir `
  --out (Join-Path $OutDir "dense_core.bin") `
  --experts-dir (Join-Path $OutDir "experts") `
  --index-out (Join-Path $OutDir "dense_core.index.json") `
  --resume-ledger $ledgerPath `
  --quant int4 `
  --layer-range "0,1" `
  --experts-per-layer 0 `
  --active-experts 0 `
  --pack-global-into-layer0 `
  --tensor-regex "^(model\.embed_tokens\.weight|model\.norm\.weight|lm_head\.weight)$" `
  --global-row-limit $Rows `
  --global-row-start $StartRow `
  --chunk-mb 8 `
  --http-retries 8 `
  --retry-base-sleep 1.5 `
  --allow-empty-dense
if ($LASTEXITCODE -ne 0) {
  throw "stream_convert_glm52.py failed with exit code $LASTEXITCODE"
}

Write-Host "GLOBAL_MICRO_READY=$OutDir"
Get-ChildItem -Path $OutDir -File | Select-Object Name,Length,LastWriteTime

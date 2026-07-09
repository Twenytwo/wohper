param(
  [string]$Python = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
  [string]$ModelDir = "models/deepseek-ai/DeepSeek-V4-Flash",
  [string]$Report = "state/deepseek_v4_flash_metadata_sync_local_2026-07-04.json"
)

$ErrorActionPreference = "Stop"
& $Python tools/download_hf_metadata_only.py `
  --repo-id deepseek-ai/DeepSeek-V4-Flash `
  --revision main `
  --out-dir $ModelDir `
  --report $Report `
  --max-file-bytes 33554432 `
  --max-total-bytes 268435456
if ($LASTEXITCODE -ne 0) {
  throw "metadata sync failed with exit code $LASTEXITCODE"
}

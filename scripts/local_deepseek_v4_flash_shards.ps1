param(
  [string]$Python = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
  [string]$ModelDir = "models/deepseek-ai/DeepSeek-V4-Flash",
  [string]$Report = "state/deepseek_v4_flash_shard_download_local_2026-07-04.json",
  [switch]$Execute,
  [double]$MinFreeAfterGb = 250,
  [int]$Limit = 0,
  [int]$ChunkMb = 1,
  [int]$TimeoutSec = 90,
  [int]$ProgressIntervalSec = 15
)

$ErrorActionPreference = "Stop"
$argsList = @(
  "tools/download_hf_safetensor_shards.py",
  "--repo-id", "deepseek-ai/DeepSeek-V4-Flash",
  "--revision", "main",
  "--model-dir", $ModelDir,
  "--out", $Report,
  "--min-free-after-gb", "$MinFreeAfterGb",
  "--chunk-mb", "$ChunkMb",
  "--timeout-sec", "$TimeoutSec",
  "--progress-interval-sec", "$ProgressIntervalSec",
  "--limit", "$Limit"
)
if ($Execute) {
  $argsList += "--execute"
}
& $Python @argsList
if ($LASTEXITCODE -ne 0) {
  throw "shard downloader failed with exit code $LASTEXITCODE"
}

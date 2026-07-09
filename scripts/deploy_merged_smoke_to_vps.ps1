param(
  [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519_navi_vps",
  [string]$Remote = "deploy@128.140.34.162",
  [string]$LocalDir = "models/wohper/GLM-5.2.INT4.GLOBAL-L3-L5-SMOKE",
  [string]$RemoteName = "GLM-5.2.INT4.GLOBAL-L3-L5-SMOKE"
)

$ErrorActionPreference = "Stop"

$remoteBase = "/home/deploy/hermes-agent/loop_workspace"
$remoteDir = "$remoteBase/models/wohper/$RemoteName"

if (!(Test-Path (Join-Path $LocalDir "dense_core.bin"))) {
  throw "Missing dense_core.bin in $LocalDir"
}

ssh -i $Key $Remote "mkdir -p '$remoteDir/experts' '$remoteBase/scripts' '$remoteBase/models/wohper/remote-worker-sim/experts'"
scp -i $Key (Join-Path $LocalDir "dense_core.bin") "${Remote}:$remoteDir/dense_core.bin"
scp -i $Key (Join-Path $LocalDir "dense_core.shards.json") "${Remote}:$remoteDir/dense_core.shards.json"
if (Test-Path (Join-Path $LocalDir "dense_core.index.json")) {
  scp -i $Key (Join-Path $LocalDir "dense_core.index.json") "${Remote}:$remoteDir/dense_core.index.json"
}
Get-ChildItem -Path (Join-Path $LocalDir "experts") -Filter "*.zcblk" -File | ForEach-Object {
  scp -i $Key $_.FullName "${Remote}:$remoteBase/models/wohper/remote-worker-sim/experts/$($_.Name)"
}
$smokeScripts = @(
  "scripts/vps_merged_smoke.sh",
  "scripts/vps_chat_task4_smoke.sh",
  "scripts/vps_worker_fetch_flow_smoke.sh"
)
foreach ($script in $smokeScripts) {
  if (Test-Path $script) {
    scp -i $Key $script "${Remote}:$remoteBase/$script"
  }
}
ssh -i $Key $Remote "chmod +x '$remoteBase'/scripts/*.sh && cd '$remoteDir' && python3 - <<'PY'
import json
from pathlib import Path
p = Path('dense_core.shards.json')
if not p.exists():
    raise SystemExit('missing dense_core.shards.json')
d = json.loads(p.read_text(encoding='utf-8'))
print('remote_shard_index experts=', len(d.get('experts', [])))
print('remote_shard_index remote_fetch=', d.get('remote_fetch'))
PY"

Write-Host "MERGED_SMOKE_DEPLOYED=$remoteDir"

param(
  [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519_navi_vps",
  [string]$Remote = "deploy@128.140.34.162",
  [string]$LocalDir = "models/wohper/GLM-5.2.INT4.GLOBAL-MICRO",
  [string]$RemoteDir = "/home/deploy/hermes-agent/loop_workspace/models/wohper/GLM-5.2.INT4.GLOBAL-MICRO"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path (Join-Path $LocalDir "dense_core.bin"))) {
  throw "Missing dense_core.bin in $LocalDir. Run scripts\convert_glm52_global_micro.ps1 first."
}

ssh -i $Key $Remote "mkdir -p '$RemoteDir/experts' /home/deploy/hermes-agent/loop_workspace/scripts"
scp -i $Key (Join-Path $LocalDir "dense_core.bin") "${Remote}:$RemoteDir/dense_core.bin"
if (Test-Path (Join-Path $LocalDir "dense_core.index.json")) {
  scp -i $Key (Join-Path $LocalDir "dense_core.index.json") "${Remote}:$RemoteDir/dense_core.index.json"
}
scp -i $Key "scripts/vps_global_micro_smoke.sh" "${Remote}:/home/deploy/hermes-agent/loop_workspace/scripts/vps_global_micro_smoke.sh"
ssh -i $Key $Remote "chmod +x /home/deploy/hermes-agent/loop_workspace/scripts/vps_global_micro_smoke.sh"

Write-Host "GLOBAL_MICRO_DEPLOYED=$RemoteDir"

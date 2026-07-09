param(
    [string]$VpsHost = "128.140.34.162",
    [string]$VpsUser = "deploy",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_navi_vps",
    [string]$RemoteRoot = "/home/deploy/hermes-agent/loop_workspace"
)

$ErrorActionPreference = "Stop"

$localCore = "models/wohper/GLM-5.2.INT4.WORKER-L3E0/dense_core.bin"
$localSidecar = "models/wohper/GLM-5.2.INT4.WORKER-L3E0/dense_core.shards.json"
$localConverter = "tools/stream_convert_glm52.py"
$remoteModelDir = "$RemoteRoot/models/wohper/GLM-5.2.INT4.WORKER-L3E0"
$remote = "$VpsUser@$VpsHost"

foreach ($path in @($localCore, $localSidecar, $localConverter)) {
    if (-not (Test-Path $path)) {
        throw "Missing local file: $path"
    }
}

ssh -i $KeyPath $remote "cd $RemoteRoot && mkdir -p $remoteModelDir tools cache/glm52-worker-l3e0"
scp -i $KeyPath $localCore "${remote}:$remoteModelDir/dense_core.bin"
scp -i $KeyPath $localSidecar "${remote}:$remoteModelDir/dense_core.shards.json"
scp -i $KeyPath $localConverter "${remote}:$RemoteRoot/tools/stream_convert_glm52.py"

$remoteCheck = @"
cd $RemoteRoot &&
PYTHONDONTWRITEBYTECODE=1 python3 tools/stream_convert_glm52.py \
  --metadata-dir models/zai-org/GLM-5.2 \
  --out models/wohper/GLM-5.2.INT4.WORKER-L3E0/dense_core.bin \
  --resume-ledger state/glm52_worker_l3e0_ledger_2026-07-02.json \
  --skip-existing \
  --layer-range 3,4 \
  --worker-endpoint http://127.0.0.1:9101 \
  --worker-policy all &&
./engine/zc_infer_core/target/release/zc_remote_fetch_smoke \
  --model models/wohper/GLM-5.2.INT4.WORKER-L3E0/dense_core.bin \
  --endpoint http://127.0.0.1:9101 \
  --cache-dir cache/glm52-worker-l3e0 \
  --layer-id 3 \
  --expert-id 0
"@

ssh -i $KeyPath $remote $remoteCheck

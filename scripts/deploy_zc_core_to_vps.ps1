param(
  [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_navi_vps",
  [string]$HostName = "deploy@128.140.34.162",
  [string]$RemoteRoot = "/home/deploy/hermes-agent/loop_workspace"
)

$ErrorActionPreference = "Stop"
$LocalRoot = Split-Path -Parent $PSScriptRoot
$Archive = Join-Path $LocalRoot "zc-core-deploy.zip"
$HashManifest = Join-Path $LocalRoot "zc-core-deploy.sha256"

function Invoke-Checked {
  param(
    [string]$Label,
    [scriptblock]$Command
  )
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit code $LASTEXITCODE"
  }
}

if (Test-Path $Archive) {
  Remove-Item -LiteralPath $Archive -Force
}
if (Test-Path $HashManifest) {
  Remove-Item -LiteralPath $HashManifest -Force
}

$items = @(
  "engine",
  "tools",
  "scripts",
  "docs/linux-environment-setup.md",
  "docs/local-inference-engine-spec.md",
  "docs/linux-benchmark-runbook.md",
  "PROJECT_NOTEBOOK.md"
) | ForEach-Object { Join-Path $LocalRoot $_ }

$criticalFiles = @(
  "engine/zc_infer_core/src/lib.rs",
  "engine/zc_infer_core/src/model_format.rs",
  "engine/zc_infer_core/src/compute.rs",
  "engine/zc_infer_core/src/deepseek_v4.rs",
  "engine/zc_infer_core/src/glm_dsa_indexer.rs",
  "engine/zc_infer_core/src/server/generation.rs",
  "engine/zc_infer_core/src/server/api.rs",
  "engine/zc_infer_core/src/bin/zc_infer_server.rs",
  "engine/zc_infer_core/src/bin/zc_remote_fetch_smoke.rs",
  "tools/convert_safetensors.py",
  "tools/check_glm52_reference_readiness.py",
  "tools/check_glm52_indexer_long_context.py",
  "tools/compare_glm52_trace_summaries.py",
  "tools/glm52_reference_trace.py",
  "tools/merge_smoke_models.py",
  "tools/plan_glm52_expert_coverage.py",
  "tools/plan_glm52_global_vocab_shards.py",
  "tools/retarget_expert_plan_model.py",
  "tools/stream_convert_glm52.py",
  "tools/summarize_zc_perf_log.py",
  "tools/audit_transformer_math_log.py",
  "tools/build_expert_catalog.py",
  "tools/top8_decision_gate.py",
  "tools/validate_quality_prompts.py",
  "tools/check_deepspec_integration_readiness.py",
  "tools/check_deepseek_v4_flash_readiness.py",
  "tools/plan_deepseek_v4_flash_inventory.py",
  "tools/download_hf_metadata_only.py",
  "tools/check_deepseek_v4_flash_tokenizer_contract.py",
  "tools/render_deepseek_v4_prompt.py",
  "tools/deepseek_v4_converter_dry_run.py",
  "tools/deepseek_v4_runtime_smoke_gate.py",
  "tools/tokenizer_chat_smoke.py",
  "tools/zc_quality_bench.py",
  "tools/zc_socket_smoke_client.py",
  "config/quality_prompts.small.json",
  "config/quality_prompts.v2.small.json",
  "config/deepseek_v4_flash.contract.json",
  "config/deepseek_v4_flash.tokenizer_contract.json",
  "config/deepseek_v4_flash.prompt_smoke.json",
  "config/deepseek_v4_quality_prompts.small.json",
  "README.md",
  "docs/release-readiness.md",
  "docs/expert-storage-workflow.md",
  "docs/test-matrix.md",
  "docs/model-quality-prompt-set.md",
  "docs/reference-parity-contract.md",
  "docs/artifact-hygiene-and-release-boundary.md",
  "docs/deepspec-integration-plan.md",
  "docs/deepseek-v4-flash-runtime-plan.md",
  "scripts/deepspec_integration_preflight.sh",
  "scripts/vps_deepseek_v4_flash_contract.sh",
  "scripts/vps_deepseek_v4_flash_inventory_gate.sh",
  "scripts/vps_deepseek_v4_flash_inventory_metadata.sh",
  "scripts/vps_deepseek_v4_flash_converter_dry_run.sh",
  "scripts/vps_deepseek_v4_flash_runtime_smoke_gate.sh",
  "scripts/vps_cleanup_glm52_artifacts.sh",
  "scripts/vps_deepseek_v4_flash_metadata_sync.sh",
  "scripts/vps_deepseek_v4_flash_tokenizer_contract.sh",
  "scripts/vps_merged_smoke.sh",
  "scripts/vps_chat_task4_smoke.sh",
  "scripts/vps_chat_stability_smoke.sh",
  "scripts/vps_quality_bench.sh",
  "scripts/vps_validate_quality_prompts.sh",
  "scripts/vps_perf_profile_top4_summary.sh",
  "scripts/vps_transformer_math_audit.sh",
  "scripts/vps_sampling_smoke.sh",
  "scripts/vps_sampling_top4_smoke.sh",
  "scripts/vps_reference_readiness.sh",
  "scripts/vps_indexer_long_context_guard.sh",
  "scripts/vps_glm52_reference_trace.sh",
  "scripts/vps_worker_fetch_flow_smoke.sh",
  "scripts/vps_expert_storage_preflight.sh",
  "scripts/vps_build_top4_expert_catalog.sh",
  "scripts/vps_top8_decision_gate.sh",
  "scripts/vps_open_repo_preflight.sh",
  "scripts/vps_public_test_matrix.sh",
  "scripts/merge_global_with_layer_slice.ps1",
  "PROJECT_NOTEBOOK.md"
)

$hashLines = foreach ($rel in $criticalFiles) {
  $localPath = Join-Path $LocalRoot $rel
  if (!(Test-Path $localPath)) {
    throw "Missing critical deploy file: $rel"
  }
  $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $localPath).Hash.ToLowerInvariant()
  "$hash  $($rel.Replace('\','/'))"
}
[System.IO.File]::WriteAllText(
  $HashManifest,
  (($hashLines -join "`n") + "`n"),
  [System.Text.Encoding]::ASCII
)

Compress-Archive -Path $items -DestinationPath $Archive -Force
Invoke-Checked "scp archive" { scp -i $KeyPath $Archive "${HostName}:${RemoteRoot}/zc-core-deploy.zip" }
Invoke-Checked "scp hash manifest" { scp -i $KeyPath $HashManifest "${HostName}:${RemoteRoot}/zc-core-deploy.sha256" }
Invoke-Checked "remote zip extract" {
  ssh -i $KeyPath $HostName "cd ${RemoteRoot} && python3 -m zipfile -e zc-core-deploy.zip . && chmod +x scripts/*.sh"
}

foreach ($rel in $criticalFiles) {
  $localPath = Join-Path $LocalRoot $rel
  $remoteRel = $rel.Replace('\','/')
  $remoteDir = Split-Path -Parent $remoteRel
  if ($remoteDir) {
    Invoke-Checked "remote mkdir $remoteDir" {
      ssh -i $KeyPath $HostName "mkdir -p '${RemoteRoot}/${remoteDir}'"
    }
  }
  Invoke-Checked "scp critical $remoteRel" {
    scp -i $KeyPath $localPath "${HostName}:${RemoteRoot}/${remoteRel}"
  }
}

Invoke-Checked "remote sha256 verification" {
  ssh -i $KeyPath $HostName "cd ${RemoteRoot} && sha256sum -c zc-core-deploy.sha256 && rm zc-core-deploy.zip zc-core-deploy.sha256"
}
Remove-Item -LiteralPath $Archive -Force
Remove-Item -LiteralPath $HashManifest -Force

Write-Host "Deployed zc core files to ${HostName}:${RemoteRoot}"
Write-Host "Next:"
Write-Host "  cd ${RemoteRoot}"
Write-Host "  scripts/setup_linux_zc_core_env.sh   # first time only"
Write-Host "  scripts/linux_bench_zc_core.sh"

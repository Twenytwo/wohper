# Linux Benchmark Runbook

Purpose: validate the Rust Core I/O Subsystem on Linux with a fake MoE `MODEL.bin`.

## Deploy From Windows

From this workspace:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy_zc_core_to_vps.ps1
```

Default target:

```text
deploy@128.140.34.162:/home/deploy/hermes-agent/loop_workspace
```

## Run On Linux

On the VPS:

```bash
cd /home/deploy/hermes-agent/loop_workspace
scripts/linux_bench_zc_core.sh
```

The script:

1. builds `engine/zc_infer_core` in release mode;
2. generates a fake `MODEL.bin` if missing;
3. runs `io_bench`;
4. writes a timestamped log under `logs/`.

## Default Fake Model

```text
layers=16
experts_per_layer=8
active_experts=2
dense_size=8mb
expert_size=14mb
```

Approximate size: around 2 GiB after 2MB alignment.

## Override Example

```bash
LAYERS=32 \
EXPERTS_PER_LAYER=8 \
ACTIVE_EXPERTS=2 \
DENSE_SIZE=8mb \
EXPERT_SIZE=16mb \
ROUNDS=4 \
COMPRESSED_BUFFER_MB=256 \
RUNTIME_BUFFER_MB=256 \
scripts/linux_bench_zc_core.sh
```

## Metrics To Watch

`io_bench` prints:

- submitted bytes;
- completed bytes;
- completed read count;
- elapsed seconds;
- completed GiB/s;
- process RSS from `/proc/self/status`.

System-side checks:

```bash
free -h
vmstat 1
iostat -xz 1
cat /proc/pressure/memory
```

The first pass is successful if:

- `ModelManifest::load()` accepts the fake model;
- all reads complete without alignment errors;
- RSS stays near fixed buffer size plus normal overhead;
- throughput is plausibly NVMe-bound.

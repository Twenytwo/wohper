//! GPU spike (step 1 of docs/gpu-offload-plan.md): FP8-E4M3 GEMV with
//! UE8M0 group scales on the RTX 2070 Super via cudarc + NVRTC, checked
//! against a scalar CPU reference with the engine's exact LUT semantics.
//!
//! Self-contained on synthetic data: validates kernel correctness and
//! measures VRAM-resident GEMV throughput before any model plumbing.
//! Run inside the dev container WITH the GPU:
//!   docker run --rm --gpus all -v $PWD:/workspace -w /workspace \
//!     -e CARGO_TARGET_DIR=/cargo-target zc-infer-dev \
//!     ./target-or-cargo run --release --features gpu --bin gpu_gemv_spike

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

/// FP8-E4M3 decode identical to `fp8_e4m3_base_lut` in the engine:
/// 1 sign, 4 exponent (bias 7), 3 mantissa; e=15&m=7 -> NaN treated as 0
/// (the engine LUT stores finite values only).
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let man = (byte & 0x07) as f32;
    if exp == 0x0F && man == 7.0 {
        return 0.0;
    }
    if exp == 0 {
        sign * man * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    }
}

/// UE8M0 scale: 2^(byte-127), same as `ue8m0_scale_lut`.
fn ue8m0_to_f32(byte: u8) -> f32 {
    2.0f32.powi(byte as i32 - 127)
}

const KERNEL: &str = r#"
extern "C" __global__ void fp8_gemv(
    const unsigned char* __restrict__ weight,   // rows x cols FP8 bytes
    const unsigned char* __restrict__ scales,   // (rows/rg) x scale_cols UE8M0
    const float* __restrict__ lut,              // 256-entry FP8 LUT
    const float* __restrict__ scale_lut,        // 256-entry UE8M0 LUT
    const float* __restrict__ input,            // cols
    float* __restrict__ output,                 // rows
    int rows, int cols, int col_group, int scale_cols, int row_group)
{
    // One warp per row; lanes stride the columns of each 128-wide group so
    // the per-group partial sums stay in registers, scale folded per group.
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const unsigned char* wrow = weight + (long long)row * cols;
    const unsigned char* srow = scales + (long long)(row / row_group) * scale_cols;
    float acc = 0.0f;
    for (int group = 0; group < scale_cols; ++group) {
        float partial = 0.0f;
        int start = group * col_group;
        for (int c = start + lane; c < start + col_group; c += 32) {
            partial += lut[wrow[c]] * input[c];
        }
        // warp reduce
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) {
            acc += scale_lut[srow[group]] * partial;
        }
    }
    if (lane == 0) {
        output[row] = acc;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // wo_b-like shape: the single largest attention projection.
    let rows: usize = 7168;
    let cols: usize = 8192;
    let col_group: usize = 128;
    let scale_cols = cols / col_group;
    let row_group: usize = 128;
    let scale_rows = rows / row_group;

    // Deterministic synthetic data (no external RNG dep).
    let mut seed = 0x243F6A8885A308D3u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let weight: Vec<u8> = (0..rows * cols).map(|_| (next() & 0xFF) as u8).collect();
    // Scales near 1.0 (bytes ~127) so sums stay finite.
    let scales: Vec<u8> = (0..scale_rows * scale_cols)
        .map(|_| 120 + (next() % 15) as u8)
        .collect();
    let input: Vec<f32> = (0..cols)
        .map(|_| ((next() % 2000) as f32 / 1000.0) - 1.0)
        .collect();

    let fp8_lut: Vec<f32> = (0..=255u8).map(fp8_e4m3_to_f32).collect();
    let scale_lut: Vec<f32> = (0..=255u8).map(ue8m0_to_f32).collect();

    // CPU reference (scalar, group-wise like the engine kernel).
    let cpu_started = Instant::now();
    let mut cpu_out = vec![0.0f32; rows];
    for row in 0..rows {
        let wrow = &weight[row * cols..(row + 1) * cols];
        let srow = &scales[(row / row_group) * scale_cols..(row / row_group + 1) * scale_cols];
        let mut acc = 0.0f32;
        for group in 0..scale_cols {
            let start = group * col_group;
            let mut partial = 0.0f32;
            for c in start..start + col_group {
                partial += fp8_lut[wrow[c] as usize] * input[c];
            }
            acc += scale_lut[srow[group] as usize] * partial;
        }
        cpu_out[row] = acc;
    }
    let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

    // GPU
    let device = CudaDevice::new(0)?;
    println!("gpu={}", device.name()?);
    let ptx = compile_ptx(KERNEL)?;
    device.load_ptx(ptx, "spike", &["fp8_gemv"])?;
    let kernel = device.get_func("spike", "fp8_gemv").unwrap();

    let d_weight = device.htod_sync_copy(&weight)?;
    let d_scales = device.htod_sync_copy(&scales)?;
    let d_lut = device.htod_sync_copy(&fp8_lut)?;
    let d_scale_lut = device.htod_sync_copy(&scale_lut)?;
    let d_input = device.htod_sync_copy(&input)?;
    let mut d_output = device.alloc_zeros::<f32>(rows)?;

    let warps_per_block = 8usize;
    let config = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(warps_per_block as u32), 1, 1),
        block_dim: ((warps_per_block * 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! launch {
        () => {
            unsafe {
                kernel.clone().launch(
                    config,
                    (
                        &d_weight,
                        &d_scales,
                        &d_lut,
                        &d_scale_lut,
                        &d_input,
                        &mut d_output,
                        rows as i32,
                        cols as i32,
                        col_group as i32,
                        scale_cols as i32,
                        row_group as i32,
                    ),
                )
            }
        };
    }
    // Warm + correctness.
    launch!()?;
    device.synchronize()?;
    let gpu_out = device.dtoh_sync_copy(&d_output)?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (cpu, gpu) in cpu_out.iter().zip(&gpu_out) {
        let abs = (cpu - gpu).abs();
        let rel = abs / cpu.abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }

    // Bench: 50 launches (weights resident, like the real decode loop).
    let iterations = 50;
    device.synchronize()?;
    let gpu_started = Instant::now();
    for _ in 0..iterations {
        launch!()?;
    }
    device.synchronize()?;
    let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let macs = (rows * cols) as f64;
    println!(
        "shape={}x{} cpu_scalar_ms={:.1} gpu_ms={:.3} gpu_gmacs={:.1} max_abs_err={:.3e} max_rel_err={:.3e}",
        rows,
        cols,
        cpu_ms,
        gpu_ms,
        macs / gpu_ms / 1e6,
        max_abs,
        max_rel
    );
    let ok = max_rel < 1e-3;
    println!("verdict={}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}

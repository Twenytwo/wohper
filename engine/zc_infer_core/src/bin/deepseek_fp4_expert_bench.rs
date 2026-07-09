use std::time::Instant;

use zc_infer_core::deepseek_v4::{
    deepseek_v4_fp4_expert_forward_scalar, deepseek_v4_fp4_expert_scratch_len,
    DeepSeekV4Fp4Expert, Fp4Matvec, DEEPSEEK_V4_EXPERT_INTERMEDIATE_SIZE,
    DEEPSEEK_V4_HIDDEN_SIZE,
};

fn arg_usize(name: &str, default: usize) -> usize {
    let needle = format!("--{name}");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == needle {
            return args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(default);
        }
    }
    default
}

fn fill_packed(seed: u8, bytes: &mut [u8]) {
    for (idx, byte) in bytes.iter_mut().enumerate() {
        let lo = ((idx as u8).wrapping_add(seed)) & 0x07;
        let hi = ((idx as u8).wrapping_mul(3).wrapping_add(seed + 1)) & 0x07;
        *byte = lo | (hi << 4);
    }
}

fn checksum(values: &[f32]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| (*value as f64) * ((idx % 17 + 1) as f64))
        .sum()
}

fn main() {
    let hidden = arg_usize("hidden", DEEPSEEK_V4_HIDDEN_SIZE);
    let intermediate = arg_usize("intermediate", DEEPSEEK_V4_EXPERT_INTERMEDIATE_SIZE);
    let iterations = arg_usize("iterations", 1);
    let scale_cols = arg_usize("scale-cols", 128);
    if hidden == 0 || intermediate == 0 || scale_cols == 0 || hidden % scale_cols != 0 {
        eprintln!("invalid shape: hidden={hidden} intermediate={intermediate} scale_cols={scale_cols}");
        std::process::exit(2);
    }

    let w1_bytes = intermediate * ((hidden + 1) / 2);
    let w3_bytes = w1_bytes;
    let w2_bytes = hidden * ((intermediate + 1) / 2);
    let mut w1 = vec![0u8; w1_bytes];
    let mut w3 = vec![0u8; w3_bytes];
    let mut w2 = vec![0u8; w2_bytes];
    fill_packed(1, &mut w1);
    fill_packed(3, &mut w3);
    fill_packed(5, &mut w2);

    let w1_scales = vec![127u8; intermediate * scale_cols];
    let w3_scales = vec![127u8; intermediate * scale_cols];
    let w2_scale_cols = scale_cols.min(intermediate);
    if intermediate % w2_scale_cols != 0 {
        eprintln!(
            "invalid w2 scale shape: intermediate={intermediate} w2_scale_cols={w2_scale_cols}"
        );
        std::process::exit(2);
    }
    let w2_scales = vec![127u8; hidden * w2_scale_cols];
    let input: Vec<f32> = (0..hidden)
        .map(|idx| ((idx % 97) as f32 - 48.0) / 97.0)
        .collect();
    let mut output = vec![0.0f32; hidden];
    let mut scratch = vec![0.0f32; deepseek_v4_fp4_expert_scratch_len(intermediate)];

    let expert = DeepSeekV4Fp4Expert {
        w1: Fp4Matvec {
            packed_weight: &w1,
            ue8m0_scales: &w1_scales,
            rows: intermediate,
            cols: hidden,
            scale_cols,
        },
        w3: Fp4Matvec {
            packed_weight: &w3,
            ue8m0_scales: &w3_scales,
            rows: intermediate,
            cols: hidden,
            scale_cols,
        },
        w2: Fp4Matvec {
            packed_weight: &w2,
            ue8m0_scales: &w2_scales,
            rows: hidden,
            cols: intermediate,
            scale_cols: w2_scale_cols,
        },
    };

    let started = Instant::now();
    for _ in 0..iterations {
        deepseek_v4_fp4_expert_forward_scalar(expert, &input, &mut output, &mut scratch)
            .expect("fp4 expert forward");
    }
    let elapsed = started.elapsed().as_secs_f64();
    let weight_bytes = w1_bytes + w3_bytes + w2_bytes;
    let scale_bytes = w1_scales.len() + w3_scales.len() + w2_scales.len();
    println!(
        "{{\"format\":\"deepseek-fp4-expert-bench\",\"kernel\":\"scalar\",\"hidden\":{hidden},\"intermediate\":{intermediate},\"iterations\":{iterations},\"elapsed_seconds\":{elapsed:.6},\"seconds_per_forward\":{seconds_per_forward:.6},\"weight_bytes\":{weight_bytes},\"scale_bytes\":{scale_bytes},\"scratch_bytes\":{scratch_bytes},\"checksum\":{checksum:.6}}}",
        seconds_per_forward = elapsed / iterations.max(1) as f64,
        scratch_bytes = scratch.len() * std::mem::size_of::<f32>(),
        checksum = checksum(&output),
    );
}

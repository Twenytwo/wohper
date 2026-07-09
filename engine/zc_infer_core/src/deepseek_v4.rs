//! DeepSeek-V4-Flash adapter contracts.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepSeekV4TensorRole {
    Embed,
    LmHead,
    Norm,
    AttentionQ,
    AttentionKv,
    AttentionO,
    AttentionSink,
    Compressor,
    Indexer,
    Router,
    Expert,
    SharedExpert,
    MhcAttention,
    MhcFfn,
    MhcHead,
    Mtp,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepSeekV4StorageKind {
    Fp8Dense,
    Fp4Expert,
    Bf16OrFp32Aux,
    Scale,
    Unknown,
}

pub const DEEPSEEK_V4_VOCAB_SIZE: usize = 129_280;
pub const DEEPSEEK_V4_HIDDEN_SIZE: usize = 4_096;
pub const DEEPSEEK_V4_EXPERT_INTERMEDIATE_SIZE: usize = 2_048;
pub const DEEPSEEK_V4_LAYER_COUNT: usize = 43;
pub const DEEPSEEK_V4_ROUTED_EXPERTS: usize = 256;
pub const DEEPSEEK_V4_ACTIVE_EXPERTS: usize = 6;
pub const DEEPSEEK_V4_INDEX_TOPK: usize = 512;
pub const DEEPSEEK_V4_SLIDING_WINDOW: usize = 128;
pub const QUANT_DEEPSEEK_RAW_MIXED: u16 = 2400;
pub const QUANT_DEEPSEEK_FP8_E4M3: u16 = 2401;
pub const QUANT_DEEPSEEK_FP4_E2M1_PACKED: u16 = 2402;
pub const QUANT_DEEPSEEK_UE8M0_SCALE: u16 = 2403;
pub const QUANT_DEEPSEEK_BF16_AUX: u16 = 2404;
pub const QUANT_DEEPSEEK_F32_AUX: u16 = 2405;
pub const QUANT_DEEPSEEK_I64_AUX: u16 = 2406;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeepSeekV4AttentionPlan {
    pub sliding_window: usize,
    pub index_topk: usize,
    pub compress_ratio: usize,
    pub uses_indexer: bool,
    pub uses_compressor: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeepSeekV4MhcPlan {
    pub hc_mult: usize,
    pub sinkhorn_iters: usize,
    pub has_attention_hc: bool,
    pub has_ffn_hc: bool,
    pub has_head_hc: bool,
}

pub fn classify_tensor_name(name: &str) -> DeepSeekV4TensorRole {
    let lowered = name.to_ascii_lowercase();
    if lowered == "embed.weight" || lowered.contains("embed_tokens") {
        return DeepSeekV4TensorRole::Embed;
    }
    if lowered == "head.weight" || lowered.contains("lm_head") {
        return DeepSeekV4TensorRole::LmHead;
    }
    if lowered == "norm.weight" {
        return DeepSeekV4TensorRole::Norm;
    }
    if lowered.starts_with("mtp.") {
        return DeepSeekV4TensorRole::Mtp;
    }
    if lowered.starts_with("hc_head_") {
        return DeepSeekV4TensorRole::MhcHead;
    }
    if lowered.contains(".hc_attn_") {
        return DeepSeekV4TensorRole::MhcAttention;
    }
    if lowered.contains(".hc_ffn_") {
        return DeepSeekV4TensorRole::MhcFfn;
    }
    if lowered.ends_with(".norm.weight") || lowered.contains("_norm.weight") {
        return DeepSeekV4TensorRole::Norm;
    }
    if lowered.contains(".compressor.") {
        return DeepSeekV4TensorRole::Compressor;
    }
    if lowered.contains(".indexer.") {
        return DeepSeekV4TensorRole::Indexer;
    }
    if lowered.contains(".attn.") {
        if lowered.contains("attn_sink") {
            return DeepSeekV4TensorRole::AttentionSink;
        }
        if lowered.contains(".wq_a.") || lowered.contains(".wq_b.") {
            return DeepSeekV4TensorRole::AttentionQ;
        }
        if lowered.contains(".wkv.") {
            return DeepSeekV4TensorRole::AttentionKv;
        }
        if lowered.contains(".wo_a.") || lowered.contains(".wo_b.") {
            return DeepSeekV4TensorRole::AttentionO;
        }
    }
    if lowered.contains(".experts.") {
        return DeepSeekV4TensorRole::Expert;
    }
    if lowered.contains("shared_expert") || lowered.contains("shared_experts") {
        return DeepSeekV4TensorRole::SharedExpert;
    }
    if lowered.contains(".gate.") || lowered.contains("router") || lowered.contains(".tid2eid") {
        return DeepSeekV4TensorRole::Router;
    }
    DeepSeekV4TensorRole::Unknown
}

pub fn storage_kind_for(role: DeepSeekV4TensorRole, name: &str) -> DeepSeekV4StorageKind {
    if name.to_ascii_lowercase().ends_with(".scale") {
        return DeepSeekV4StorageKind::Scale;
    }
    match role {
        DeepSeekV4TensorRole::Expert | DeepSeekV4TensorRole::SharedExpert => {
            DeepSeekV4StorageKind::Fp4Expert
        }
        DeepSeekV4TensorRole::AttentionQ
        | DeepSeekV4TensorRole::AttentionKv
        | DeepSeekV4TensorRole::AttentionO
        | DeepSeekV4TensorRole::Router
        | DeepSeekV4TensorRole::Compressor
        | DeepSeekV4TensorRole::Indexer
        | DeepSeekV4TensorRole::Embed
        | DeepSeekV4TensorRole::LmHead => DeepSeekV4StorageKind::Fp8Dense,
        DeepSeekV4TensorRole::Norm
        | DeepSeekV4TensorRole::AttentionSink
        | DeepSeekV4TensorRole::MhcAttention
        | DeepSeekV4TensorRole::MhcFfn
        | DeepSeekV4TensorRole::MhcHead
        | DeepSeekV4TensorRole::Mtp => DeepSeekV4StorageKind::Bf16OrFp32Aux,
        DeepSeekV4TensorRole::Unknown => DeepSeekV4StorageKind::Unknown,
    }
}

pub fn decode_fp8_e4m3(byte: u8, scale: f32) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 && mantissa == 0 {
        return 0.0;
    }
    let value = if exponent == 0 {
        (mantissa as f32 / 8.0) * 2f32.powi(-6)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
    };
    sign * value * scale
}

pub fn decode_ue8m0_scale(byte: u8) -> f32 {
    if byte == 0 {
        0.0
    } else {
        2f32.powi(byte as i32 - 127)
    }
}

/// Base values (scale = 1.0) of FP4 E2M1 nibbles. Must stay bit-identical to
/// `decode_fp4_e2m1(n, 1.0)` for every nibble (enforced by unit test); note
/// the negative-zero nibble (8) decodes to 0.0 like the reference decoder.
pub const FP4_E2M1_BASE_LUT: [f32; 16] = [
    0.0, 0.125, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.125, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 256-entry LUT for `decode_ue8m0_scale` (hot-path replacement for powi).
pub fn ue8m0_scale_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for (byte, slot) in lut.iter_mut().enumerate() {
            *slot = decode_ue8m0_scale(byte as u8);
        }
        lut
    })
}

/// 256-entry LUT for `decode_fp8_e4m3(byte, 1.0)` (hot-path base values).
pub fn fp8_e4m3_base_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for (byte, slot) in lut.iter_mut().enumerate() {
            *slot = decode_fp8_e4m3(byte as u8, 1.0);
        }
        lut
    })
}

/// Thread count for row-parallel kernels: ZC_COMPUTE_THREADS override, else
/// available_parallelism minus 2 (leaves room for the io_uring SQPOLL thread
/// and the async runtime), minimum 1.
pub fn compute_thread_count() -> usize {
    static COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *COUNT.get_or_init(|| {
        if let Ok(value) = std::env::var("ZC_COMPUTE_THREADS") {
            if let Ok(parsed) = value.trim().parse::<usize>() {
                return parsed.max(1);
            }
        }
        std::thread::available_parallelism()
            .map(|count| count.get().saturating_sub(2).max(1))
            .unwrap_or(1)
    })
}

/// Runs `work(row_offset, chunk)` over disjoint contiguous chunks of `output`
/// on scoped threads. Falls back to inline execution for small outputs or a
/// single-thread configuration. The first error wins; output rows written by
/// failed chunks are unspecified (caller returns the error anyway).
pub fn parallel_rows_f32<E: Send>(
    output: &mut [f32],
    min_rows_per_thread: usize,
    work: &(dyn Fn(usize, &mut [f32]) -> Result<(), E> + Sync),
) -> Result<(), E> {
    let threads = compute_thread_count();
    let min_rows = min_rows_per_thread.max(1);
    if threads <= 1 || output.len() < min_rows * 2 {
        return work(0, output);
    }
    let chunk_rows = output.len().div_ceil(threads).max(min_rows);
    let first_error: std::sync::Mutex<Option<E>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        let mut rest: &mut [f32] = output;
        let mut offset = 0usize;
        while !rest.is_empty() {
            let take = chunk_rows.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            rest = tail;
            let error_slot = &first_error;
            scope.spawn(move || {
                if let Err(error) = work(offset, head) {
                    let mut slot = error_slot.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(error);
                    }
                }
            });
            offset += take;
        }
    });
    match first_error.into_inner().unwrap() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Runtime AVX2+FMA availability, cached (used by the SIMD kernel dispatch).
pub fn simd_avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
mod fp4_avx2 {
    use std::arch::x86_64::*;

    /// FP4 E2M1 base values times 8 as exact i8 for `_mm_shuffle_epi8` LUT
    /// decode; the vector group sum is rescaled by scale/8.
    const FP4_BASE_X8_I8: [i8; 16] = [
        0, 1, 8, 12, 16, 24, 32, 48, 0, -1, -8, -12, -16, -24, -32, -48,
    ];

    /// One FP4 GEMV row: 16 packed bytes (32 columns) per iteration via
    /// shuffle-LUT nibble decode + FMA. Caller guarantees:
    /// weight_row.len() == scale_row.len() * group_bytes,
    /// input.len() >= scale_row.len() * group_cols, group_cols == 2*group_bytes.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_row(
        weight_row: &[u8],
        scale_row: &[u8],
        group_bytes: usize,
        group_cols: usize,
        input: &[f32],
        scale_lut: &[f32; 256],
    ) -> f32 {
        let lut = _mm_loadu_si128(FP4_BASE_X8_I8.as_ptr() as *const __m128i);
        let low_mask = _mm_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        let mut tail = 0.0f32;
        for (group, &scale_byte) in scale_row.iter().enumerate() {
            let scale = scale_lut[scale_byte as usize];
            let bytes = &weight_row[group * group_bytes..(group + 1) * group_bytes];
            let cols = &input[group * group_cols..(group + 1) * group_cols];
            let chunks = group_bytes / 16;
            let mut group_acc = _mm256_setzero_ps();
            for chunk in 0..chunks {
                let w = _mm_loadu_si128(bytes.as_ptr().add(chunk * 16) as *const __m128i);
                let lo = _mm_and_si128(w, low_mask);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(w), low_mask);
                let vlo = _mm_shuffle_epi8(lut, lo);
                let vhi = _mm_shuffle_epi8(lut, hi);
                // Interleave low/high nibble values back into column order.
                let il = _mm_unpacklo_epi8(vlo, vhi); // columns 0..16
                let ih = _mm_unpackhi_epi8(vlo, vhi); // columns 16..32
                let base = cols.as_ptr().add(chunk * 32);
                let c0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(il));
                let c1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(il)));
                let c2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ih));
                let c3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(ih)));
                group_acc = _mm256_fmadd_ps(c0, _mm256_loadu_ps(base), group_acc);
                group_acc = _mm256_fmadd_ps(c1, _mm256_loadu_ps(base.add(8)), group_acc);
                group_acc = _mm256_fmadd_ps(c2, _mm256_loadu_ps(base.add(16)), group_acc);
                group_acc = _mm256_fmadd_ps(c3, _mm256_loadu_ps(base.add(24)), group_acc);
            }
            // Vector lanes hold base*8 values: fold scale/8 into the group sum.
            acc = _mm256_fmadd_ps(group_acc, _mm256_set1_ps(scale * 0.125), acc);
            // Remainder bytes of the group (group_bytes % 16) via base LUT.
            let mut rem = 0.0f32;
            for index in chunks * 16..group_bytes {
                let byte = bytes[index];
                rem += super::FP4_E2M1_BASE_LUT[(byte & 0x0f) as usize] * cols[2 * index];
                rem += super::FP4_E2M1_BASE_LUT[(byte >> 4) as usize] * cols[2 * index + 1];
            }
            tail += rem * scale;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let vector_sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        vector_sum + tail
    }
}

pub fn unpack_fp8_e4m3_slice(bytes: &[u8], scale: f32, out: &mut [f32]) -> usize {
    let count = bytes.len().min(out.len());
    for (dst, &src) in out.iter_mut().zip(bytes.iter()).take(count) {
        *dst = decode_fp8_e4m3(src, scale);
    }
    count
}

pub fn decode_fp4_e2m1(nibble: u8, scale: f32) -> f32 {
    let nibble = nibble & 0x0f;
    let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
    let exponent = (nibble >> 1) & 0x03;
    let mantissa = nibble & 0x01;
    if exponent == 0 && mantissa == 0 {
        return 0.0;
    }
    let value = if exponent == 0 {
        (mantissa as f32 / 2.0) * 2f32.powi(-2)
    } else {
        (1.0 + mantissa as f32 / 2.0) * 2f32.powi(exponent as i32 - 1)
    };
    sign * value * scale
}

pub fn unpack_fp4_pair(byte: u8, scale: f32) -> (f32, f32) {
    (
        decode_fp4_e2m1(byte & 0x0f, scale),
        decode_fp4_e2m1(byte >> 4, scale),
    )
}

pub fn unpack_fp4_e2m1_packed_slice(bytes: &[u8], scale: f32, out: &mut [f32]) -> usize {
    let mut written = 0;
    for &byte in bytes {
        if written >= out.len() {
            break;
        }
        out[written] = decode_fp4_e2m1(byte & 0x0f, scale);
        written += 1;
        if written >= out.len() {
            break;
        }
        out[written] = decode_fp4_e2m1(byte >> 4, scale);
        written += 1;
    }
    written
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepSeekV4KernelError {
    EmptyShape,
    InvalidShape,
    PackedWeightTooSmall,
    ScaleTooSmall,
    InputTooSmall,
    OutputTooSmall,
    ScratchTooSmall,
    NonFiniteOutput,
}

#[derive(Clone, Copy, Debug)]
pub struct Fp4Matvec<'a> {
    pub packed_weight: &'a [u8],
    pub ue8m0_scales: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub scale_cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DeepSeekV4Fp4Expert<'a> {
    pub w1: Fp4Matvec<'a>,
    pub w3: Fp4Matvec<'a>,
    pub w2: Fp4Matvec<'a>,
}

pub const fn deepseek_v4_fp4_expert_scratch_len(intermediate_size: usize) -> usize {
    intermediate_size * 2
}

fn packed_row_bytes(cols: usize) -> usize {
    (cols + 1) / 2
}

fn validate_fp4_matvec(spec: Fp4Matvec<'_>, input_len: usize, output_len: usize) -> Result<(), DeepSeekV4KernelError> {
    if spec.rows == 0 || spec.cols == 0 || spec.scale_cols == 0 {
        return Err(DeepSeekV4KernelError::EmptyShape);
    }
    if input_len < spec.cols {
        return Err(DeepSeekV4KernelError::InputTooSmall);
    }
    if output_len < spec.rows {
        return Err(DeepSeekV4KernelError::OutputTooSmall);
    }
    if spec.cols % spec.scale_cols != 0 {
        return Err(DeepSeekV4KernelError::InvalidShape);
    }
    let required_weight = spec
        .rows
        .checked_mul(packed_row_bytes(spec.cols))
        .ok_or(DeepSeekV4KernelError::InvalidShape)?;
    if spec.packed_weight.len() < required_weight {
        return Err(DeepSeekV4KernelError::PackedWeightTooSmall);
    }
    let required_scales = spec
        .rows
        .checked_mul(spec.scale_cols)
        .ok_or(DeepSeekV4KernelError::InvalidShape)?;
    if spec.ue8m0_scales.len() < required_scales {
        return Err(DeepSeekV4KernelError::ScaleTooSmall);
    }
    Ok(())
}

pub fn fp4_e2m1_ue8m0_matvec_scalar(
    spec: Fp4Matvec<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), DeepSeekV4KernelError> {
    validate_fp4_matvec(spec, input.len(), output.len())?;
    let row_bytes = packed_row_bytes(spec.cols);
    let group_cols = spec.cols / spec.scale_cols;
    if spec.cols % 2 == 0 && group_cols % 2 == 0 {
        // Group-hoisted fast path: one scale decode per group, LUT nibble
        // decode, two accumulators (low/high nibble lanes). Summation is
        // grouped, so results can differ from the per-element path only by
        // f32 reassociation noise.
        let scale_lut = ue8m0_scale_lut();
        let group_bytes = group_cols / 2;
        #[cfg(target_arch = "x86_64")]
        {
            if simd_avx2_fma_available() {
                let rows = spec.rows;
                return parallel_rows_f32(&mut output[..rows], 64, &|row_offset, chunk| {
                    for (local, slot) in chunk.iter_mut().enumerate() {
                        let row = row_offset + local;
                        let weight_row =
                            &spec.packed_weight[row * row_bytes..row * row_bytes + row_bytes];
                        let scale_row = &spec.ue8m0_scales
                            [row * spec.scale_cols..row * spec.scale_cols + spec.scale_cols];
                        let acc = unsafe {
                            fp4_avx2::dot_row(
                                weight_row,
                                scale_row,
                                group_bytes,
                                group_cols,
                                &input[..spec.cols],
                                scale_lut,
                            )
                        };
                        if !acc.is_finite() {
                            return Err(DeepSeekV4KernelError::NonFiniteOutput);
                        }
                        *slot = acc;
                    }
                    Ok(())
                });
            }
        }
        for row in 0..spec.rows {
            let weight_row = &spec.packed_weight[row * row_bytes..row * row_bytes + row_bytes];
            let scale_row =
                &spec.ue8m0_scales[row * spec.scale_cols..row * spec.scale_cols + spec.scale_cols];
            let mut acc = 0.0f32;
            for (group, &scale_byte) in scale_row.iter().enumerate() {
                let scale = scale_lut[scale_byte as usize];
                let bytes = &weight_row[group * group_bytes..(group + 1) * group_bytes];
                let cols = &input[group * group_cols..(group + 1) * group_cols];
                let mut low = 0.0f32;
                let mut high = 0.0f32;
                for (index, &byte) in bytes.iter().enumerate() {
                    low += FP4_E2M1_BASE_LUT[(byte & 0x0f) as usize] * cols[2 * index];
                    high += FP4_E2M1_BASE_LUT[(byte >> 4) as usize] * cols[2 * index + 1];
                }
                acc += scale * (low + high);
            }
            if !acc.is_finite() {
                return Err(DeepSeekV4KernelError::NonFiniteOutput);
            }
            output[row] = acc;
        }
        return Ok(());
    }
    for row in 0..spec.rows {
        let row_offset = row * row_bytes;
        let scale_offset = row * spec.scale_cols;
        let mut acc = 0.0f32;
        for col in 0..spec.cols {
            let packed = spec.packed_weight[row_offset + col / 2];
            let nibble = if col & 1 == 0 { packed & 0x0f } else { packed >> 4 };
            let scale = decode_ue8m0_scale(spec.ue8m0_scales[scale_offset + col / group_cols]);
            acc += decode_fp4_e2m1(nibble, scale) * input[col];
        }
        if !acc.is_finite() {
            return Err(DeepSeekV4KernelError::NonFiniteOutput);
        }
        output[row] = acc;
    }
    Ok(())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// DeepSeek-V4-Flash SwiGLU clamp (config `swiglu_limit`, 10.0): the gate
/// activation is clamped from above, the up activation on both sides,
/// BEFORE silu/product (reference: inference/model.py `Expert.forward`).
pub const DEEPSEEK_V4_SWIGLU_LIMIT: f32 = 10.0;

pub fn deepseek_v4_fp4_expert_forward_scalar(
    expert: DeepSeekV4Fp4Expert<'_>,
    input: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), DeepSeekV4KernelError> {
    if expert.w1.cols != input.len()
        || expert.w3.cols != input.len()
        || expert.w1.rows != expert.w3.rows
        || expert.w2.cols != expert.w1.rows
    {
        return Err(DeepSeekV4KernelError::InvalidShape);
    }
    if output.len() < expert.w2.rows {
        return Err(DeepSeekV4KernelError::OutputTooSmall);
    }
    let required_scratch = deepseek_v4_fp4_expert_scratch_len(expert.w1.rows);
    if scratch.len() < required_scratch {
        return Err(DeepSeekV4KernelError::ScratchTooSmall);
    }
    let (gate, rest) = scratch.split_at_mut(expert.w1.rows);
    let up = &mut rest[..expert.w1.rows];
    // QAT act quant: the reference fp4_matvec_from_records quant-dequants
    // the FP8 activation before every fp4 matvec (w1/w3 on x, w2 on the
    // swiglu hidden).
    let mut input_q = input[..expert.w1.cols].to_vec();
    crate::compute::fp8_act_quant_dequant_in_place(&mut input_q, 64);
    fp4_e2m1_ue8m0_matvec_scalar(expert.w1, &input_q, gate)?;
    fp4_e2m1_ue8m0_matvec_scalar(expert.w3, &input_q, up)?;
    for idx in 0..expert.w1.rows {
        let gated = gate[idx].min(DEEPSEEK_V4_SWIGLU_LIMIT);
        let upped = up[idx].clamp(-DEEPSEEK_V4_SWIGLU_LIMIT, DEEPSEEK_V4_SWIGLU_LIMIT);
        gate[idx] = silu(gated) * upped;
    }
    crate::compute::fp8_act_quant_dequant_in_place(gate, 64);
    fp4_e2m1_ue8m0_matvec_scalar(expert.w2, gate, output)?;
    if output[..expert.w2.rows].iter().any(|value| !value.is_finite()) {
        return Err(DeepSeekV4KernelError::NonFiniteOutput);
    }
    Ok(())
}

pub fn deepseek_quant_format_name(format: u16) -> &'static str {
    match format {
        QUANT_DEEPSEEK_RAW_MIXED => "deepseek_raw_mixed",
        QUANT_DEEPSEEK_FP8_E4M3 => "deepseek_fp8_e4m3",
        QUANT_DEEPSEEK_FP4_E2M1_PACKED => "deepseek_fp4_e2m1_packed",
        QUANT_DEEPSEEK_UE8M0_SCALE => "deepseek_ue8m0_scale",
        QUANT_DEEPSEEK_BF16_AUX => "deepseek_bf16_aux",
        QUANT_DEEPSEEK_F32_AUX => "deepseek_f32_aux",
        QUANT_DEEPSEEK_I64_AUX => "deepseek_i64_aux",
        _ => "unknown",
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn deepseek_sqrtsoftplus_route(
    logits: &[f32],
    bias: Option<&[f32]>,
    topk: usize,
    route_scale: f32,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| {
            let weight_score = softplus(logit).sqrt();
            let select_score = weight_score + bias.map(|b| b[idx]).unwrap_or(0.0);
            (idx, select_score, weight_score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(topk.min(scored.len()));
    let denom: f32 = scored.iter().map(|(_, _, weight)| *weight).sum();
    if denom <= f32::EPSILON {
        return scored
            .into_iter()
            .map(|(idx, _, _)| (idx, route_scale / topk.max(1) as f32))
            .collect();
    }
    scored
        .into_iter()
        .map(|(idx, _, weight)| (idx, weight / denom * route_scale))
        .collect()
}

pub fn attention_plan_for_compress_ratio(compress_ratio: usize) -> DeepSeekV4AttentionPlan {
    DeepSeekV4AttentionPlan {
        sliding_window: DEEPSEEK_V4_SLIDING_WINDOW,
        index_topk: DEEPSEEK_V4_INDEX_TOPK,
        compress_ratio,
        uses_indexer: compress_ratio == 4,
        uses_compressor: compress_ratio > 0,
    }
}

pub fn mhc_plan() -> DeepSeekV4MhcPlan {
    DeepSeekV4MhcPlan {
        hc_mult: 4,
        sinkhorn_iters: 20,
        has_attention_hc: true,
        has_ffn_hc: true,
        has_head_hc: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_real_deepseek_v4_names() {
        assert_eq!(classify_tensor_name("embed.weight"), DeepSeekV4TensorRole::Embed);
        assert_eq!(classify_tensor_name("head.weight"), DeepSeekV4TensorRole::LmHead);
        assert_eq!(
            classify_tensor_name("layers.0.attn.wq_a.weight"),
            DeepSeekV4TensorRole::AttentionQ
        );
        assert_eq!(
            classify_tensor_name("layers.0.attn.wkv.scale"),
            DeepSeekV4TensorRole::AttentionKv
        );
        assert_eq!(
            classify_tensor_name("layers.0.mlp.experts.12.w1.weight"),
            DeepSeekV4TensorRole::Expert
        );
        assert_eq!(
            classify_tensor_name("layers.0.hc_attn_scale"),
            DeepSeekV4TensorRole::MhcAttention
        );
        assert_eq!(
            classify_tensor_name("mtp.0.e_proj.weight"),
            DeepSeekV4TensorRole::Mtp
        );
    }

    #[test]
    fn separates_fp4_expert_from_scale_and_aux() {
        assert_eq!(
            storage_kind_for(DeepSeekV4TensorRole::Expert, "layers.0.mlp.experts.0.w1.weight"),
            DeepSeekV4StorageKind::Fp4Expert
        );
        assert_eq!(
            storage_kind_for(DeepSeekV4TensorRole::Expert, "layers.0.mlp.experts.0.w1.scale"),
            DeepSeekV4StorageKind::Scale
        );
        assert_eq!(
            storage_kind_for(DeepSeekV4TensorRole::MhcHead, "hc_head_scale"),
            DeepSeekV4StorageKind::Bf16OrFp32Aux
        );
    }

    #[test]
    fn fp8_and_fp4_decode_zero_and_signed_values() {
        assert_eq!(decode_fp8_e4m3(0x00, 1.0), 0.0);
        assert_eq!(decode_fp4_e2m1(0x00, 1.0), 0.0);
        assert_eq!(decode_ue8m0_scale(0), 0.0);
        assert_eq!(decode_ue8m0_scale(120), 0.0078125);
        assert_eq!(decode_ue8m0_scale(116), 0.00048828125);
        assert!(decode_fp8_e4m3(0x38, 1.0) > 0.0);
        assert!(decode_fp8_e4m3(0xb8, 1.0) < 0.0);
        let (lo, hi) = unpack_fp4_pair(0xa2, 2.0);
        assert!(lo > 0.0);
        assert!(hi < 0.0);
    }

    #[test]
    fn vector_unpackers_preserve_order_and_bounds() {
        let mut fp8 = [0.0; 3];
        assert_eq!(unpack_fp8_e4m3_slice(&[0x38, 0xb8, 0x00, 0x38], 1.0, &mut fp8), 3);
        assert!(fp8[0] > 0.0);
        assert!(fp8[1] < 0.0);
        assert_eq!(fp8[2], 0.0);

        let mut fp4 = [0.0; 3];
        assert_eq!(unpack_fp4_e2m1_packed_slice(&[0xa2, 0x11], 2.0, &mut fp4), 3);
        assert!(fp4[0] > 0.0);
        assert!(fp4[1] < 0.0);
        assert!(fp4[2] > 0.0);
    }

    #[test]
    fn fp4_lut_matches_decoder_bit_exact() {
        for nibble in 0..16u8 {
            assert_eq!(
                FP4_E2M1_BASE_LUT[nibble as usize].to_bits(),
                decode_fp4_e2m1(nibble, 1.0).to_bits(),
                "nibble {nibble}"
            );
        }
    }

    #[test]
    fn fp8_and_ue8m0_luts_match_decoders_bit_exact() {
        for byte in 0..=255u8 {
            assert_eq!(
                fp8_e4m3_base_lut()[byte as usize].to_bits(),
                decode_fp8_e4m3(byte, 1.0).to_bits(),
                "fp8 byte {byte}"
            );
            assert_eq!(
                ue8m0_scale_lut()[byte as usize].to_bits(),
                decode_ue8m0_scale(byte).to_bits(),
                "ue8m0 byte {byte}"
            );
        }
    }

    #[test]
    fn fp4_matvec_fast_path_matches_per_element_reference() {
        // Even shape (cols 32, group 8) takes the group fast path; compare
        // against a per-element reference accumulation.
        let rows = 3usize;
        let cols = 32usize;
        let scale_cols = 4usize;
        let mut state = 0x12345678u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let packed: Vec<u8> = (0..rows * cols / 2).map(|_| (next() >> 24) as u8).collect();
        let scales: Vec<u8> = (0..rows * scale_cols).map(|_| 120 + (next() >> 29) as u8).collect();
        let input: Vec<f32> = (0..cols).map(|_| ((next() >> 16) as f32 / 65536.0) - 0.5).collect();
        let spec = Fp4Matvec {
            packed_weight: &packed,
            ue8m0_scales: &scales,
            rows,
            cols,
            scale_cols,
        };
        let mut output = vec![0.0f32; rows];
        fp4_e2m1_ue8m0_matvec_scalar(spec, &input, &mut output).unwrap();
        let group_cols = cols / scale_cols;
        for row in 0..rows {
            let mut acc = 0.0f32;
            for col in 0..cols {
                let byte = packed[row * cols / 2 + col / 2];
                let nibble = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let scale = decode_ue8m0_scale(scales[row * scale_cols + col / group_cols]);
                acc += decode_fp4_e2m1(nibble, scale) * input[col];
            }
            assert!(
                (output[row] - acc).abs() <= 1e-4 * acc.abs().max(1.0),
                "row {row}: fast {} vs ref {acc}",
                output[row]
            );
        }
    }

    #[test]
    fn fp4_ue8m0_matvec_uses_low_nibble_first_and_row_scales() {
        let spec = Fp4Matvec {
            packed_weight: &[0x02, 0x00, 0x20, 0x00],
            ue8m0_scales: &[127, 127],
            rows: 2,
            cols: 4,
            scale_cols: 1,
        };
        let input = [2.0, 3.0, 5.0, 7.0];
        let mut output = [0.0; 2];
        fp4_e2m1_ue8m0_matvec_scalar(spec, &input, &mut output).unwrap();
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn fp4_expert_forward_uses_caller_scratch() {
        let expert = DeepSeekV4Fp4Expert {
            w1: Fp4Matvec {
                packed_weight: &[0x02],
                ue8m0_scales: &[127],
                rows: 1,
                cols: 2,
                scale_cols: 1,
            },
            w3: Fp4Matvec {
                packed_weight: &[0x20],
                ue8m0_scales: &[127],
                rows: 1,
                cols: 2,
                scale_cols: 1,
            },
            w2: Fp4Matvec {
                packed_weight: &[0x02, 0x04],
                ue8m0_scales: &[127, 127],
                rows: 2,
                cols: 1,
                scale_cols: 1,
            },
        };
        let input = [2.0, 3.0];
        let mut output = [0.0; 2];
        let mut scratch = [0.0; 2];
        deepseek_v4_fp4_expert_forward_scalar(expert, &input, &mut output, &mut scratch).unwrap();
        // The swiglu hidden goes through the QAT act quant before w2.
        let mut hidden = [2.0 / (1.0 + (-2.0f32).exp()) * 3.0];
        crate::compute::fp8_act_quant_dequant_in_place(&mut hidden, 64);
        let hidden = hidden[0];
        assert!((output[0] - hidden).abs() < 1e-5);
        assert!((output[1] - 2.0 * hidden).abs() < 1e-5);
    }

    #[test]
    fn fp4_expert_forward_rejects_small_scratch() {
        let expert = DeepSeekV4Fp4Expert {
            w1: Fp4Matvec {
                packed_weight: &[0x02],
                ue8m0_scales: &[127],
                rows: 1,
                cols: 2,
                scale_cols: 1,
            },
            w3: Fp4Matvec {
                packed_weight: &[0x20],
                ue8m0_scales: &[127],
                rows: 1,
                cols: 2,
                scale_cols: 1,
            },
            w2: Fp4Matvec {
                packed_weight: &[0x02, 0x04],
                ue8m0_scales: &[127, 127],
                rows: 2,
                cols: 1,
                scale_cols: 1,
            },
        };
        let mut output = [0.0; 2];
        let mut scratch = [0.0; 1];
        assert_eq!(
            deepseek_v4_fp4_expert_forward_scalar(expert, &[2.0, 3.0], &mut output, &mut scratch),
            Err(DeepSeekV4KernelError::ScratchTooSmall)
        );
    }

    #[test]
    fn names_deepseek_raw_quant_formats() {
        assert_eq!(deepseek_quant_format_name(2401), "deepseek_fp8_e4m3");
        assert_eq!(deepseek_quant_format_name(2402), "deepseek_fp4_e2m1_packed");
        assert_eq!(deepseek_quant_format_name(777), "unknown");
    }

    #[test]
    fn sqrtsoftplus_router_selects_topk_and_normalizes() {
        let routes = deepseek_sqrtsoftplus_route(&[0.0, 3.0, 1.0, -2.0], None, 2, 1.5);
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].0, 1);
        let total: f32 = routes.iter().map(|(_, weight)| *weight).sum();
        assert!((total - 1.5).abs() < 1e-5);
    }

    #[test]
    fn attention_and_mhc_plans_match_deepseek_config() {
        let ratio4 = attention_plan_for_compress_ratio(4);
        assert!(ratio4.uses_compressor);
        assert!(ratio4.uses_indexer);
        assert_eq!(ratio4.index_topk, 512);
        let ratio128 = attention_plan_for_compress_ratio(128);
        assert!(ratio128.uses_compressor);
        assert!(!ratio128.uses_indexer);
        let hc = mhc_plan();
        assert_eq!(hc.hc_mult, 4);
        assert_eq!(hc.sinkhorn_iters, 20);
        assert!(hc.has_attention_hc && hc.has_ffn_hc && hc.has_head_hc);
    }
}

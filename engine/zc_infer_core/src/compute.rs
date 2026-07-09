use crate::direct_io::BlockKind;
use crate::scheduler::ExpertRoute;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use crate::model_format::{
    QUANT_BLOCK_HEADER_SIZE, QUANT_BLOCK_MAGIC, QUANT_BLOCK_VERSION, QUANT_TENSOR_RECORD_SIZE,
};
use crate::deepseek_v4::{
    decode_fp8_e4m3, deepseek_v4_fp4_expert_forward_scalar,
    DeepSeekV4Fp4Expert, Fp4Matvec, DEEPSEEK_V4_EXPERT_INTERMEDIATE_SIZE,
    DEEPSEEK_V4_HIDDEN_SIZE, QUANT_DEEPSEEK_BF16_AUX, QUANT_DEEPSEEK_FP4_E2M1_PACKED,
    QUANT_DEEPSEEK_FP8_E4M3, QUANT_DEEPSEEK_UE8M0_SCALE,
};
use std::fmt;

const GLM52_INDEX_TOPK: usize = 2048;

fn glm52_indexer_bypass_status(context_len: usize) -> (&'static str, bool) {
    if context_len <= GLM52_INDEX_TOPK {
        ("not_required_context_le_topk", true)
    } else {
        ("bypassed_long_context", false)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiskQuantFormat {
    Int4Symmetric,
    Int4Affine,
    Int8Symmetric,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuKernel {
    Scalar,
    Avx2,
    Avx512,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantBlockParams<'a> {
    pub format: DiskQuantFormat,
    pub group_size: usize,
    pub scales: &'a [f32],
    pub zero_points: &'a [i8],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DenseMathProfile {
    pub has_attention: bool,
    pub has_router: bool,
    pub has_shared_expert: bool,
    pub has_indexer: bool,
    pub norm_tensors: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertCacheConfig {
    pub cache_dir: PathBuf,
    pub max_bytes: u64,
    pub remote_endpoint: Option<String>,
}

impl Default for ExpertCacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("cache/experts"),
            max_bytes: 100 * 1024 * 1024 * 1024,
            remote_endpoint: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExpertCacheEntry {
    pub layer_id: u32,
    pub expert_id: u32,
    pub path: PathBuf,
    pub bytes: u64,
    pub last_used_epoch_ms: u128,
}

#[derive(Debug)]
pub enum ExpertCacheError {
    Io(std::io::Error),
    InvalidRemoteEndpoint(String),
    HttpStatus { status: u16, url: String },
    InvalidHttpResponse(String),
    MissingRemoteHook { layer_id: u32, expert_id: u32 },
}

impl std::fmt::Display for ExpertCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "expert cache I/O error: {err}"),
            Self::InvalidRemoteEndpoint(endpoint) => {
                write!(f, "invalid expert remote endpoint: {endpoint}")
            }
            Self::HttpStatus { status, url } => {
                write!(f, "expert remote returned HTTP {status} for {url}")
            }
            Self::InvalidHttpResponse(reason) => write!(f, "invalid expert HTTP response: {reason}"),
            Self::MissingRemoteHook { layer_id, expert_id } => write!(
                f,
                "expert {layer_id}/{expert_id} is missing locally and no remote fetch hook is configured"
            ),
        }
    }
}

impl std::error::Error for ExpertCacheError {}

impl From<std::io::Error> for ExpertCacheError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub struct ExpertLruCache {
    config: ExpertCacheConfig,
    entries: HashMap<(u32, u32), ExpertCacheEntry>,
}

impl ExpertLruCache {
    pub fn new(config: ExpertCacheConfig) -> Result<Self, ExpertCacheError> {
        fs::create_dir_all(&config.cache_dir)?;
        Ok(Self {
            config,
            entries: HashMap::new(),
        })
    }

    pub fn ensure_expert(
        &mut self,
        layer_id: u32,
        expert_id: u32,
        source_path: Option<&Path>,
        expected_bytes: Option<u64>,
    ) -> Result<PathBuf, ExpertCacheError> {
        let path = self.cache_path(layer_id, expert_id);
        if path.exists() {
            let actual = fs::metadata(&path)?.len();
            if expected_bytes.map(|expected| expected == actual).unwrap_or(true) {
                self.touch(layer_id, expert_id, path.clone())?;
                return Ok(path);
            }
            fs::remove_file(&path)?;
        }

        if let Some(source) = source_path {
            copy_file_atomic(source, &path)?;
            self.touch(layer_id, expert_id, path.clone())?;
            self.prune_if_needed()?;
            return Ok(path);
        }

        self.fetch_remote(layer_id, expert_id, &path)?;
        self.touch(layer_id, expert_id, path.clone())?;
        self.prune_if_needed()?;
        Ok(path)
    }

    fn cache_path(&self, layer_id: u32, expert_id: u32) -> PathBuf {
        self.config
            .cache_dir
            .join(format!("layer{layer_id}_expert{expert_id}.zcblk"))
    }

    fn touch(
        &mut self,
        layer_id: u32,
        expert_id: u32,
        path: PathBuf,
    ) -> Result<(), ExpertCacheError> {
        let bytes = fs::metadata(&path)?.len();
        self.entries.insert(
            (layer_id, expert_id),
            ExpertCacheEntry {
                layer_id,
                expert_id,
                path,
                bytes,
                last_used_epoch_ms: now_ms(),
            },
        );
        Ok(())
    }

    fn total_bytes(&self) -> u64 {
        self.entries.values().map(|entry| entry.bytes).sum()
    }

    fn prune_if_needed(&mut self) -> Result<(), ExpertCacheError> {
        while self.total_bytes() > self.config.max_bytes {
            let Some((&key, victim)) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_epoch_ms)
            else {
                break;
            };
            let path = victim.path.clone();
            if path.exists() {
                fs::remove_file(&path)?;
            }
            self.entries.remove(&key);
        }
        Ok(())
    }

    fn fetch_remote(
        &self,
        layer_id: u32,
        expert_id: u32,
        target: &Path,
    ) -> Result<(), ExpertCacheError> {
        let Some(endpoint) = &self.config.remote_endpoint else {
            return Err(ExpertCacheError::MissingRemoteHook { layer_id, expert_id });
        };
        let name = format!("layer{layer_id}_expert{expert_id}.zcblk");
        let url = format!("{}/experts/{}", endpoint.trim_end_matches('/'), name);
        http_get_to_file(&url, target)
    }
}

fn copy_file_atomic(source: &Path, target: &Path) -> Result<(), ExpertCacheError> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = target.with_extension("zcblk.part");
    fs::copy(source, &tmp)?;
    fs::rename(tmp, target)?;
    Ok(())
}

#[derive(Debug)]
struct ParsedHttpUrl {
    host: String,
    port: u16,
    path: String,
}

fn parse_http_url(url: &str) -> Result<ParsedHttpUrl, ExpertCacheError> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| ExpertCacheError::InvalidRemoteEndpoint(url.to_string()))?;
    let (authority, path) = match rest.split_once('/') {
        Some((authority, path)) => (authority, format!("/{path}")),
        None => (rest, "/".to_string()),
    };
    if authority.is_empty() {
        return Err(ExpertCacheError::InvalidRemoteEndpoint(url.to_string()));
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !host.is_empty() => {
            let port = port
                .parse::<u16>()
                .map_err(|_| ExpertCacheError::InvalidRemoteEndpoint(url.to_string()))?;
            (host.to_string(), port)
        }
        _ => (authority.to_string(), 80),
    };
    Ok(ParsedHttpUrl { host, port, path })
}

fn http_get_to_file(url: &str, target: &Path) -> Result<(), ExpertCacheError> {
    let parsed = parse_http_url(url)?;
    let mut stream = TcpStream::connect((parsed.host.as_str(), parsed.port))?;
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAccept: application/octet-stream\r\n\r\n",
        parsed.path, parsed.host
    );
    stream.write_all(request.as_bytes())?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| ExpertCacheError::InvalidHttpResponse("missing status line".to_string()))?;

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(ExpertCacheError::InvalidHttpResponse(
                "missing header terminator".to_string(),
            ));
        }
        if line == "\r\n" {
            break;
        }
    }

    if status != 200 {
        return Err(ExpertCacheError::HttpStatus {
            status,
            url: url.to_string(),
        });
    }

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = target.with_extension("zcblk.part");
    {
        let mut file = fs::File::create(&tmp)?;
        std::io::copy(&mut reader, &mut file)?;
        file.sync_all()?;
    }
    fs::rename(tmp, target)?;
    Ok(())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[derive(Debug)]
pub enum ComputeError {
    EmptyGroupSize,
    InvalidQuantBlockMagic([u8; 8]),
    UnsupportedQuantBlockVersion(u32),
    TruncatedQuantBlock(&'static str),
    InvalidQuantBlock(&'static str),
    TensorIndexOutOfRange { index: usize, count: usize },
    InvalidPointer(&'static str),
    InvalidShape(&'static str),
    OutputTooSmall { required: usize, actual: usize },
    ScratchTooSmall { required: usize, actual: usize },
    MissingScale { group: usize },
    MissingZeroPoint { group: usize },
    UnsupportedQuantFormat(DiskQuantFormat),
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGroupSize => write!(f, "quant group_size must be greater than zero"),
            Self::InvalidQuantBlockMagic(magic) => write!(f, "invalid ZCBLK magic: {magic:?}"),
            Self::UnsupportedQuantBlockVersion(version) => {
                write!(f, "unsupported ZCBLK version: {version}")
            }
            Self::TruncatedQuantBlock(field) => write!(f, "truncated ZCBLK field: {field}"),
            Self::InvalidQuantBlock(reason) => write!(f, "invalid ZCBLK: {reason}"),
            Self::TensorIndexOutOfRange { index, count } => {
                write!(f, "tensor index out of range: index={index}, count={count}")
            }
            Self::InvalidPointer(name) => write!(f, "invalid null pointer: {name}"),
            Self::InvalidShape(name) => write!(f, "invalid compute shape: {name}"),
            Self::OutputTooSmall { required, actual } => {
                write!(f, "output buffer too small: required={required}, actual={actual}")
            }
            Self::ScratchTooSmall { required, actual } => {
                write!(f, "scratch buffer too small: required={required}, actual={actual}")
            }
            Self::MissingScale { group } => write!(f, "missing quant scale for group {group}"),
            Self::MissingZeroPoint { group } => {
                write!(f, "missing quant zero point for group {group}")
            }
            Self::UnsupportedQuantFormat(format) => {
                write!(f, "unsupported quant format: {format:?}")
            }
        }
    }
}

impl std::error::Error for ComputeError {}

pub struct ComputeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub rope_theta: f32,
    pub prefill_chunk_tokens: usize,
    pub quant_group_size: usize,
    pub preferred_kernel: CpuKernel,
    pub router_math: RouterMath,
    pub route_scale: f32,
    pub num_hash_layers: usize,
    pub attention_kind: AttentionKind,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    /// Hyper-connections width (DeepSeek mHC): number of residual stream
    /// copies. 1 disables the mHC path.
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
}

/// Router scoring semantics per model family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouterMath {
    /// Legacy behavior: raw gate logits, weights normalized downstream.
    RawLogits,
    /// DeepSeek-V4-Flash: scores = sqrt(softplus(logits));
    /// `gate.bias` (e_score_correction) shifts scores for top-k selection
    /// only; final weights are the original scores of the selected experts,
    /// normalized to sum 1 and multiplied by `route_scale`.
    DeepSeekV4SqrtSoftplus,
}

/// Attention math per model family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    /// Legacy GLM-derived probe path (q_a/q_b, kv_a/kv_b expansion).
    GlmDsaProbe,
    /// DeepSeek-V4-Flash MLA: single 512-dim latent KV shared as key and
    /// value, per-head parameter-free q RMS, attn_sink softmax bias,
    /// inverse RoPE on the attention output, grouped low-rank wo_a/wo_b
    /// output projection.
    DeepSeekV4Mla,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantBlockHeader {
    pub tensor_count: u32,
    pub quant_format: u32,
    pub flags: u32,
    pub record_table_offset: usize,
    pub names_offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantTensorLayout<'a> {
    pub dtype_original: u16,
    pub quant_format: u16,
    pub rank: u32,
    pub flags: u32,
    pub name: &'a [u8],
    pub shape: ShapeLayout<'a>,
    pub data: &'a [u8],
    pub scale: f32,
    pub zero_point: f32,
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorRole {
    Unknown = 0,
    QkvProj = 1,
    QProj = 2,
    KProj = 3,
    VProj = 4,
    OProj = 5,
    GateProj = 6,
    UpProj = 7,
    DownProj = 8,
    Router = 9,
    Norm = 10,
    Embed = 11,
    LmHead = 12,
    SharedExpert = 13,
    KvProj = 14,
}

impl TensorRole {
    pub const fn from_code(code: u16) -> Self {
        match code {
            1 => Self::QkvProj,
            2 => Self::QProj,
            3 => Self::KProj,
            4 => Self::VProj,
            5 => Self::OProj,
            6 => Self::GateProj,
            7 => Self::UpProj,
            8 => Self::DownProj,
            9 => Self::Router,
            10 => Self::Norm,
            11 => Self::Embed,
            12 => Self::LmHead,
            13 => Self::SharedExpert,
            14 => Self::KvProj,
            _ => Self::Unknown,
        }
    }

    pub const fn code(self) -> u16 {
        self as u16
    }
}

impl QuantTensorLayout<'_> {
    pub fn role(&self) -> TensorRole {
        TensorRole::from_code((self.flags & 0xffff) as u16)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShapeLayout<'a> {
    data: &'a [u8],
    rank: usize,
}

impl ShapeLayout<'_> {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn dim(&self, index: usize) -> Result<usize, ComputeError> {
        if index >= self.rank {
            return Err(ComputeError::InvalidQuantBlock("shape dim index out of range"));
        }
        read_u64_le(self.data, 4 + index * 8, "shape.dim").map(|value| value as usize)
    }

    pub fn element_count(&self) -> Result<usize, ComputeError> {
        let mut count = 1usize;
        for index in 0..self.rank {
            count = count
                .checked_mul(self.dim(index)?)
                .ok_or(ComputeError::InvalidQuantBlock("shape element count overflow"))?;
        }
        Ok(count)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuantBlockLayout<'a> {
    buffer: &'a [u8],
    pub header: QuantBlockHeader,
}

impl<'a> QuantBlockLayout<'a> {
    pub fn tensor_count(&self) -> usize {
        self.header.tensor_count as usize
    }

    pub fn tensor(&self, index: usize) -> Result<QuantTensorLayout<'a>, ComputeError> {
        let count = self.tensor_count();
        if index >= count {
            return Err(ComputeError::TensorIndexOutOfRange { index, count });
        }

        let record_offset = self
            .header
            .record_table_offset
            .checked_add(index * QUANT_TENSOR_RECORD_SIZE)
            .ok_or(ComputeError::InvalidQuantBlock("record offset overflow"))?;
        let record = self
            .buffer
            .get(record_offset..record_offset + QUANT_TENSOR_RECORD_SIZE)
            .ok_or(ComputeError::TruncatedQuantBlock("tensor record"))?;

        let dtype_original = read_u16_le(record, 0, "record.dtype_original")?;
        let quant_format = read_u16_le(record, 2, "record.quant_format")?;
        let rank = read_u32_le(record, 4, "record.rank")?;
        let flags = read_u32_le(record, 8, "record.flags")?;
        let name_offset = read_u64_le(record, 12, "record.name_offset")? as usize;
        let shape_offset = read_u64_le(record, 20, "record.shape_offset")? as usize;
        let data_offset = read_u64_le(record, 28, "record.data_offset")? as usize;
        let data_bytes = read_u64_le(record, 36, "record.data_bytes")? as usize;
        let scale = read_f32_le(record, 44, "record.scale")?;
        let zero_point = read_f32_le(record, 48, "record.zero_point")?;

        let name = read_name_blob(self.buffer, self.header.names_offset, name_offset)?;
        let shape = read_shape_layout(self.buffer, shape_offset, rank as usize)?;
        let data_end = data_offset
            .checked_add(data_bytes)
            .ok_or(ComputeError::InvalidQuantBlock("tensor data range overflow"))?;
        let data = self
            .buffer
            .get(data_offset..data_end)
            .ok_or(ComputeError::TruncatedQuantBlock("tensor data"))?;

        Ok(QuantTensorLayout {
            dtype_original,
            quant_format,
            rank,
            flags,
            name,
            shape,
            data,
            scale,
            zero_point,
        })
    }

    pub fn first_tensor(&self) -> Result<QuantTensorLayout<'a>, ComputeError> {
        self.tensor(0)
    }

    pub fn tensor_by_role(
        &self,
        role: TensorRole,
    ) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
        for index in 0..self.tensor_count() {
            let tensor = self.tensor(index)?;
            if tensor.role() == role {
                return Ok(Some(tensor));
            }
        }
        Ok(None)
    }

    pub fn first_tensor_by_roles(
        &self,
        roles: &[TensorRole],
    ) -> Result<QuantTensorLayout<'a>, ComputeError> {
        for role in roles {
            if let Some(tensor) = self.tensor_by_role(*role)? {
                return Ok(tensor);
            }
        }
        self.first_tensor()
    }
}

impl ComputeConfig {
    pub fn glm52_like(hidden_size: usize) -> Self {
        let num_attention_heads = if hidden_size >= 6144 { 64 } else { 1 };
        let qk_rope_head_dim = if hidden_size >= 6144 { 64 } else { hidden_size.min(64) };
        let qk_nope_head_dim = if hidden_size >= 6144 { 192 } else { hidden_size.saturating_sub(qk_rope_head_dim) };
        let v_head_dim = if hidden_size >= 6144 { 256 } else { hidden_size };
        let q_lora_rank = if hidden_size >= 6144 { 2048 } else { hidden_size };
        let kv_lora_rank = if hidden_size >= 6144 { 512 } else { hidden_size };
        Self {
            hidden_size,
            intermediate_size: hidden_size.saturating_mul(4),
            num_attention_heads,
            num_kv_heads: num_attention_heads,
            qk_rope_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            rope_theta: 8_000_000.0,
            prefill_chunk_tokens: 16,
            quant_group_size: 128,
            preferred_kernel: detect_best_kernel(),
            router_math: RouterMath::RawLogits,
            route_scale: 1.0,
            num_hash_layers: 0,
            attention_kind: AttentionKind::GlmDsaProbe,
            o_groups: 0,
            o_lora_rank: 0,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        }
    }

    pub fn deepseek_v4_flash() -> Self {
        Self {
            hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
            intermediate_size: DEEPSEEK_V4_EXPERT_INTERMEDIATE_SIZE,
            num_attention_heads: 64,
            num_kv_heads: 1,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 448,
            v_head_dim: 512,
            q_lora_rank: 1024,
            kv_lora_rank: 448,
            rope_theta: 10_000.0,
            prefill_chunk_tokens: 16,
            quant_group_size: 128,
            preferred_kernel: detect_best_kernel(),
            router_math: RouterMath::DeepSeekV4SqrtSoftplus,
            route_scale: 1.5,
            num_hash_layers: 3,
            attention_kind: AttentionKind::DeepSeekV4Mla,
            o_groups: 8,
            o_lora_rank: 1024,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        }
    }

    pub fn prefill_scratch_f32(&self) -> usize {
        let qk_head_dim = self.qk_nope_head_dim.saturating_add(self.qk_rope_head_dim);
        let q_full = self.num_attention_heads.saturating_mul(qk_head_dim);
        let value_concat = self.num_kv_heads.saturating_mul(self.v_head_dim);
        let q_or_value = q_full.max(value_concat);
        let kv_a = self.kv_lora_rank.saturating_add(self.qk_rope_head_dim);
        let kv_full = self
            .num_kv_heads
            .saturating_mul(self.qk_nope_head_dim.saturating_add(self.v_head_dim));
        // DeepSeek MLA path: q_lora | q | kv | attn_out | o_mid
        let mla = if self.attention_kind == AttentionKind::DeepSeekV4Mla {
            self.q_lora_rank
                .saturating_add(q_full)
                .saturating_add(qk_head_dim)
                .saturating_add(q_full)
                .saturating_add(self.o_groups.saturating_mul(self.o_lora_rank))
        } else {
            0
        };
        // mHC path: residual copies + working stream + sub-block scratch
        // (MLA attention or MoE, whichever is larger).
        let mla = if self.hc_mult > 1 {
            let moe = self
                .hidden_size
                .saturating_mul(4)
                .saturating_add(self.intermediate_size.saturating_mul(2));
            self.hc_mult
                .saturating_mul(self.hidden_size)
                .saturating_add(self.hidden_size)
                .saturating_add(mla.max(moe))
        } else {
            mla
        };
        self.hidden_size
            .max(self.q_lora_rank.saturating_add(q_or_value))
            .max(
                self.q_lora_rank
                    .saturating_add(q_or_value)
                    .saturating_add(kv_a)
                    .saturating_add(kv_full),
            )
            .max(mla)
            .saturating_add(self.hidden_size)
            .max(
                self.hidden_size
                    .saturating_mul(4)
                    .saturating_add(self.intermediate_size.saturating_mul(2)),
            )
            .max(1)
    }
}

pub struct ComputeScratch<'a> {
    /// Tile-sized decode scratch supplied by the caller. This is deliberately
    /// outside the kernel so the hot path never allocates while computing.
    pub dequant_tile_f32: &'a mut [f32],
}

pub struct KVCache<'a> {
    storage: &'a mut [f32],
    pub num_layers: usize,
    pub max_tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub cursor: usize,
}

impl<'a> KVCache<'a> {
    pub fn required_f32(
        num_layers: usize,
        max_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<usize, ComputeError> {
        num_layers
            .checked_mul(max_tokens)
            .and_then(|value| value.checked_mul(num_kv_heads))
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(2))
            .ok_or(ComputeError::InvalidShape("KV cache size overflow"))
    }

    pub fn from_scratch(
        storage: &'a mut [f32],
        num_layers: usize,
        max_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ComputeError> {
        let required = Self::required_f32(num_layers, max_tokens, num_kv_heads, head_dim)?;
        if storage.len() < required {
            return Err(ComputeError::ScratchTooSmall {
                required,
                actual: storage.len(),
            });
        }
        storage[..required].fill(0.0);
        Ok(Self {
            storage: &mut storage[..required],
            num_layers,
            max_tokens,
            num_kv_heads,
            head_dim,
            cursor: 0,
        })
    }

    pub fn append(
        &mut self,
        layer_id: usize,
        token_pos: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), ComputeError> {
        let vector_len = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or(ComputeError::InvalidShape("KV vector size overflow"))?;
        if key.len() < vector_len || value.len() < vector_len {
            return Err(ComputeError::InvalidShape("KV append vector too small"));
        }
        if layer_id >= self.num_layers || token_pos >= self.max_tokens {
            return Err(ComputeError::InvalidShape("KV append index out of range"));
        }

        let base = self.slot_offset(layer_id, token_pos, vector_len)?;
        let half = self.storage.len() / 2;
        let (keys, values) = self.storage.split_at_mut(half);
        keys[base..base + vector_len].copy_from_slice(&key[..vector_len]);
        values[base..base + vector_len].copy_from_slice(&value[..vector_len]);
        self.cursor = self.cursor.max(token_pos + 1);
        Ok(())
    }

    /// Raw view of the whole cache storage (keys+values halves) for
    /// session save/restore. The layout is only meaningful for an
    /// identical (num_layers, max_tokens, num_kv_heads, head_dim) shape.
    pub fn raw_storage(&self) -> &[f32] {
        self.storage
    }

    /// Restores a previously saved raw storage (same shape required).
    pub fn restore_raw(&mut self, data: &[f32]) -> Result<(), ComputeError> {
        if data.len() != self.storage.len() {
            return Err(ComputeError::InvalidShape("KV restore shape mismatch"));
        }
        self.storage.copy_from_slice(data);
        Ok(())
    }

    pub fn key_slice(&self, layer_id: usize, token_pos: usize) -> Result<&[f32], ComputeError> {
        self.slot_slice(layer_id, token_pos, false)
    }

    pub fn value_slice(&self, layer_id: usize, token_pos: usize) -> Result<&[f32], ComputeError> {
        self.slot_slice(layer_id, token_pos, true)
    }

    pub fn key_value_slices_mut(
        &mut self,
        layer_id: usize,
        token_pos: usize,
    ) -> Result<(&mut [f32], &mut [f32]), ComputeError> {
        let vector_len = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or(ComputeError::InvalidShape("KV vector size overflow"))?;
        let base = self.slot_offset(layer_id, token_pos, vector_len)?;
        let half = self.storage.len() / 2;
        let (keys, values) = self.storage.split_at_mut(half);
        self.cursor = self.cursor.max(token_pos + 1);
        Ok((
            &mut keys[base..base + vector_len],
            &mut values[base..base + vector_len],
        ))
    }

    fn slot_slice(
        &self,
        layer_id: usize,
        token_pos: usize,
        values: bool,
    ) -> Result<&[f32], ComputeError> {
        let vector_len = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or(ComputeError::InvalidShape("KV vector size overflow"))?;
        let base = self.slot_offset(layer_id, token_pos, vector_len)?;
        let half = self.storage.len() / 2;
        let start = if values { half + base } else { base };
        Ok(&self.storage[start..start + vector_len])
    }

    fn slot_offset(
        &self,
        layer_id: usize,
        token_pos: usize,
        vector_len: usize,
    ) -> Result<usize, ComputeError> {
        if layer_id >= self.num_layers || token_pos >= self.max_tokens {
            return Err(ComputeError::InvalidShape("KV cache index out of range"));
        }
        layer_id
            .checked_mul(self.max_tokens)
            .and_then(|value| value.checked_add(token_pos))
            .and_then(|value| value.checked_mul(vector_len))
            .ok_or(ComputeError::InvalidShape("KV cache offset overflow"))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IoBlockPtr {
    pub kind: BlockKind,
    pub layer_id: u32,
    pub expert_id: u32,
    pub route_weight: f32,
    pub ptr: *const u8,
    pub len: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LayerComputeStats {
    pub dense_bytes: usize,
    pub expert_bytes: usize,
    pub experts: usize,
    pub dequantized_values: usize,
}

pub trait GemmKernel {
    /// # Safety contract
    ///
    /// `a`, `b`, and `c` must point to valid contiguous buffers large enough for
    /// the matrix shape. Implementations may assume alignment and non-aliasing
    /// only when their concrete kernel documents it.
    unsafe fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
    ) -> Result<(), ComputeError>;

    /// Fused token GEMV for output-major INT4 weights.
    ///
    /// Computes `out[n] = dot(input[0..k], dequant(weights[n][0..k]))`.
    /// Each output row stores two 4-bit weights per byte, low nibble first.
    /// `scales` and `zero_points` are grouped by `group_size` inside each row.
    unsafe fn gemv_i4_affine(
        &self,
        n: usize,
        k: usize,
        input: *const f32,
        packed_weights: *const u8,
        scales: *const f32,
        zero_points: *const i8,
        group_size: usize,
        output: *mut f32,
    ) -> Result<CpuKernel, ComputeError>;

    unsafe fn gemv_i4_affine_tensorwise(
        &self,
        n: usize,
        k: usize,
        input: *const f32,
        packed_weights: *const u8,
        scale: f32,
        zero_point: f32,
        output: *mut f32,
    ) -> Result<CpuKernel, ComputeError>;
}

pub struct NativeTileGemm;

impl GemmKernel for NativeTileGemm {
    unsafe fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
    ) -> Result<(), ComputeError> {
        if a.is_null() {
            return Err(ComputeError::InvalidPointer("sgemm.a"));
        }
        if b.is_null() {
            return Err(ComputeError::InvalidPointer("sgemm.b"));
        }
        if c.is_null() {
            return Err(ComputeError::InvalidPointer("sgemm.c"));
        }
        if m == 0 || n == 0 || k == 0 {
            return Err(ComputeError::InvalidShape("sgemm dimensions must be non-zero"));
        }

        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for inner in 0..k {
                    acc += *a.add(row * k + inner) * *b.add(inner * n + col);
                }
                *c.add(row * n + col) = acc;
            }
        }
        Ok(())
    }

    unsafe fn gemv_i4_affine(
        &self,
        n: usize,
        k: usize,
        input: *const f32,
        packed_weights: *const u8,
        scales: *const f32,
        zero_points: *const i8,
        group_size: usize,
        output: *mut f32,
    ) -> Result<CpuKernel, ComputeError> {
        fused_gemv_i4_affine(
            n,
            k,
            input,
            packed_weights,
            scales,
            zero_points,
            group_size,
            output,
        )
    }

    unsafe fn gemv_i4_affine_tensorwise(
        &self,
        n: usize,
        k: usize,
        input: *const f32,
        packed_weights: *const u8,
        scale: f32,
        zero_point: f32,
        output: *mut f32,
    ) -> Result<CpuKernel, ComputeError> {
        fused_gemv_i4_affine_tensorwise(n, k, input, packed_weights, scale, zero_point, output)
    }
}

pub struct FusedInt4Gemm;

impl GemmKernel for FusedInt4Gemm {
    unsafe fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
    ) -> Result<(), ComputeError> {
        let native = NativeTileGemm;
        native.sgemm(m, n, k, a, b, c)
    }

    unsafe fn gemv_i4_affine(
        &self,
        n: usize,
        k: usize,
        input: *const f32,
        packed_weights: *const u8,
        scales: *const f32,
        zero_points: *const i8,
        group_size: usize,
        output: *mut f32,
    ) -> Result<CpuKernel, ComputeError> {
        fused_gemv_i4_affine(
            n,
            k,
            input,
            packed_weights,
            scales,
            zero_points,
            group_size,
            output,
        )
    }

    unsafe fn gemv_i4_affine_tensorwise(
        &self,
        n: usize,
        k: usize,
        input: *const f32,
        packed_weights: *const u8,
        scale: f32,
        zero_point: f32,
        output: *mut f32,
    ) -> Result<CpuKernel, ComputeError> {
        fused_gemv_i4_affine_tensorwise(n, k, input, packed_weights, scale, zero_point, output)
    }
}

pub fn detect_best_kernel() -> CpuKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return CpuKernel::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return CpuKernel::Avx2;
        }
    }
    CpuKernel::Scalar
}

pub fn parse_quant_block(buffer: &[u8]) -> Result<QuantBlockLayout<'_>, ComputeError> {
    if buffer.len() < QUANT_BLOCK_HEADER_SIZE {
        return Err(ComputeError::TruncatedQuantBlock("block header"));
    }

    let mut magic = [0u8; 8];
    magic.copy_from_slice(
        buffer
            .get(0..8)
            .ok_or(ComputeError::TruncatedQuantBlock("block magic"))?,
    );
    if magic != QUANT_BLOCK_MAGIC {
        return Err(ComputeError::InvalidQuantBlockMagic(magic));
    }

    let version = read_u32_le(buffer, 8, "block.version")?;
    if version != QUANT_BLOCK_VERSION {
        return Err(ComputeError::UnsupportedQuantBlockVersion(version));
    }

    let tensor_count = read_u32_le(buffer, 12, "block.tensor_count")?;
    let quant_format = read_u32_le(buffer, 16, "block.quant_format")?;
    let flags = read_u32_le(buffer, 20, "block.flags")?;
    let record_table_offset = read_u64_le(buffer, 24, "block.record_table_offset")? as usize;
    let names_offset = read_u64_le(buffer, 32, "block.names_offset")? as usize;

    let record_bytes = (tensor_count as usize)
        .checked_mul(QUANT_TENSOR_RECORD_SIZE)
        .ok_or(ComputeError::InvalidQuantBlock("record table size overflow"))?;
    let record_end = record_table_offset
        .checked_add(record_bytes)
        .ok_or(ComputeError::InvalidQuantBlock("record table range overflow"))?;
    if record_table_offset < QUANT_BLOCK_HEADER_SIZE || record_end > buffer.len() {
        return Err(ComputeError::TruncatedQuantBlock("record table"));
    }
    if names_offset > buffer.len() {
        return Err(ComputeError::TruncatedQuantBlock("names blob"));
    }

    Ok(QuantBlockLayout {
        buffer,
        header: QuantBlockHeader {
            tensor_count,
            quant_format,
            flags,
            record_table_offset,
            names_offset,
        },
    })
}

fn find_tensor_by_suffix<'a>(
    layout: &'a QuantBlockLayout<'a>,
    suffix: &[u8],
) -> Result<QuantTensorLayout<'a>, ComputeError> {
    let mut found = None;
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.name.ends_with(suffix) {
            if found.is_some() {
                return Err(ComputeError::InvalidQuantBlock(
                    "duplicate DeepSeek expert tensor suffix",
                ));
            }
            found = Some(tensor);
        }
    }
    found.ok_or(ComputeError::InvalidQuantBlock(
        "missing DeepSeek expert tensor suffix",
    ))
}

fn find_optional_tensor_by_name<'a>(
    layout: &'a QuantBlockLayout<'a>,
    name: &str,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.name == name.as_bytes() {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

fn fp8_scale_for_weight<'a>(
    layout: &'a QuantBlockLayout<'a>,
    weight: QuantTensorLayout<'a>,
) -> Result<QuantTensorLayout<'a>, ComputeError> {
    let name = std::str::from_utf8(weight.name)
        .map_err(|_| ComputeError::InvalidQuantBlock("DeepSeek FP8 tensor name is not UTF-8"))?;
    let scale_name = name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.scale"))
        .ok_or(ComputeError::InvalidQuantBlock(
            "DeepSeek FP8 weight name does not end with .weight",
        ))?;
    find_optional_tensor_by_name(layout, &scale_name)?.ok_or(ComputeError::InvalidQuantBlock(
        "missing DeepSeek FP8 scale tensor",
    ))
}

fn deepseek_fp4_matvec_from_tensors<'a>(
    weight: QuantTensorLayout<'a>,
    scale: QuantTensorLayout<'a>,
) -> Result<Fp4Matvec<'a>, ComputeError> {
    if weight.quant_format != QUANT_DEEPSEEK_FP4_E2M1_PACKED {
        return Err(ComputeError::InvalidQuantBlock(
            "DeepSeek expert weight is not FP4 E2M1 packed",
        ));
    }
    if scale.quant_format != QUANT_DEEPSEEK_UE8M0_SCALE {
        return Err(ComputeError::InvalidQuantBlock(
            "DeepSeek expert scale is not UE8M0",
        ));
    }
    if weight.shape.rank() != 2 || scale.shape.rank() != 2 {
        return Err(ComputeError::InvalidShape("DeepSeek FP4 expert rank"));
    }
    let rows = weight.shape.dim(0)?;
    let packed_cols = weight.shape.dim(1)?;
    let scale_rows = scale.shape.dim(0)?;
    let scale_cols = scale.shape.dim(1)?;
    if rows == 0 || packed_cols == 0 || scale_cols == 0 || scale_rows != rows {
        return Err(ComputeError::InvalidShape("DeepSeek FP4 expert shape"));
    }
    let cols = packed_cols
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP4 expert cols overflow"))?;
    let required_weight = rows
        .checked_mul(packed_cols)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP4 expert weight overflow"))?;
    if weight.data.len() < required_weight {
        return Err(ComputeError::TruncatedQuantBlock("DeepSeek FP4 expert weight"));
    }
    let required_scales = rows
        .checked_mul(scale_cols)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP4 expert scale overflow"))?;
    if scale.data.len() < required_scales {
        return Err(ComputeError::TruncatedQuantBlock("DeepSeek FP4 expert scale"));
    }
    Ok(Fp4Matvec {
        packed_weight: &weight.data[..required_weight],
        ue8m0_scales: &scale.data[..required_scales],
        rows,
        cols,
        scale_cols,
    })
}

pub fn deepseek_v4_fp4_expert_from_quant_block<'a>(
    layout: &'a QuantBlockLayout<'a>,
) -> Result<DeepSeekV4Fp4Expert<'a>, ComputeError> {
    let w1 = deepseek_fp4_matvec_from_tensors(
        find_tensor_by_suffix(layout, b".w1.weight")?,
        find_tensor_by_suffix(layout, b".w1.scale")?,
    )?;
    let w3 = deepseek_fp4_matvec_from_tensors(
        find_tensor_by_suffix(layout, b".w3.weight")?,
        find_tensor_by_suffix(layout, b".w3.scale")?,
    )?;
    let w2 = deepseek_fp4_matvec_from_tensors(
        find_tensor_by_suffix(layout, b".w2.weight")?,
        find_tensor_by_suffix(layout, b".w2.scale")?,
    )?;
    if w1.cols != w3.cols || w1.rows != w3.rows || w2.cols != w1.rows {
        return Err(ComputeError::InvalidShape("DeepSeek FP4 expert projection"));
    }
    Ok(DeepSeekV4Fp4Expert { w1, w3, w2 })
}

pub fn dequantize_block_simd(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    out_f32: &mut [f32],
) -> Result<CpuKernel, ComputeError> {
    if params.group_size == 0 {
        return Err(ComputeError::EmptyGroupSize);
    }

    match params.format {
        DiskQuantFormat::Int4Symmetric | DiskQuantFormat::Int4Affine => {
            let required = packed_weights.len().saturating_mul(2);
            if out_f32.len() < required {
                return Err(ComputeError::OutputTooSmall {
                    required,
                    actual: out_f32.len(),
                });
            }

            let kernel = detect_best_kernel();
            match kernel {
                CpuKernel::Avx512 => unsafe {
                    dequantize_i4_avx512(packed_weights, params, &mut out_f32[..required])?
                },
                CpuKernel::Avx2 => unsafe {
                    dequantize_i4_avx2(packed_weights, params, &mut out_f32[..required])?
                },
                CpuKernel::Scalar => {
                    dequantize_i4_scalar(packed_weights, params, &mut out_f32[..required])?
                }
            }
            Ok(kernel)
        }
        DiskQuantFormat::Int8Symmetric => {
            if out_f32.len() < packed_weights.len() {
                return Err(ComputeError::OutputTooSmall {
                    required: packed_weights.len(),
                    actual: out_f32.len(),
                });
            }
            dequantize_i8_scalar(packed_weights, params, &mut out_f32[..packed_weights.len()])?;
            Ok(CpuKernel::Scalar)
        }
    }
}

pub unsafe fn compute_layer(
    layer_index: usize,
    token_pos: usize,
    dense_ptr: *const u8,
    dense_len: usize,
    active_experts_ptrs: &[IoBlockPtr],
    hidden_states: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
) -> Result<LayerComputeStats, ComputeError> {
    if dense_ptr.is_null() {
        return Err(ComputeError::InvalidPointer("dense_ptr"));
    }
    for expert in active_experts_ptrs {
        if expert.ptr.is_null() {
            return Err(ComputeError::InvalidPointer("expert.ptr"));
        }
    }

    let dense = std::slice::from_raw_parts(dense_ptr, dense_len);
    let mut stats = LayerComputeStats {
        dense_bytes: dense.len(),
        expert_bytes: 0,
        experts: active_experts_ptrs.len(),
        dequantized_values: 0,
    };

    if is_non_compute_dense_block(dense)? {
        return Ok(stats);
    }

    if config.attention_kind == AttentionKind::DeepSeekV4Mla && config.hc_mult > 1 {
        compute_layer_deepseek_mhc(
            dense,
            layer_index,
            token_pos,
            active_experts_ptrs,
            hidden_states,
            kv_cache,
            scratch,
            config,
            gemm,
            &mut stats,
        )?;
        return Ok(stats);
    }

    compute_attention_with_residual(
        dense,
        layer_index,
        token_pos,
        hidden_states,
        kv_cache,
        scratch,
        config,
        gemm,
        &mut stats,
    )?;

    let hidden = hidden_states.len().min(config.hidden_size);
    let profile = dense_math_profile(dense)?;
    let moe_applied = profile.has_shared_expert || !active_experts_ptrs.is_empty();
    if moe_applied && hidden > 0 {
        let required = hidden
            .checked_add(1)
            .ok_or(ComputeError::InvalidShape("MLP residual scratch overflow"))?;
        if scratch.dequant_tile_f32.len() < required {
            return Err(ComputeError::ScratchTooSmall {
                required,
                actual: scratch.dequant_tile_f32.len(),
            });
        }
        let (mlp_residual, moe_scratch_buf) = scratch.dequant_tile_f32.split_at_mut(hidden);
        mlp_residual.copy_from_slice(&hidden_states[..hidden]);
        let mut moe_scratch = ComputeScratch {
            dequant_tile_f32: moe_scratch_buf,
        };

        if !apply_named_layer_norm_in_place(
            dense,
            "post_attention_layernorm",
            hidden_states,
            &mut moe_scratch,
        )? {
            rms_norm_in_place(hidden_states);
        }

        if profile.has_shared_expert {
            let shared_applied = compute_active_and_shared_experts(
                dense,
                active_experts_ptrs,
                hidden_states,
                &mut moe_scratch,
                config,
                gemm,
                &mut stats,
            )?;
            crate::vlog!(
                "math_fidelity layer={} component=shared_expert present=true applied={}",
                layer_index, shared_applied
            );
        } else {
            compute_active_experts(
                active_experts_ptrs,
                hidden_states,
                &mut moe_scratch,
                config,
                gemm,
                &mut stats,
            )?;
        }

        for (slot, residual) in hidden_states.iter_mut().take(hidden).zip(mlp_residual.iter()) {
            *slot += *residual;
        }
    } else {
        rms_norm_in_place(hidden_states);
    }

    Ok(stats)
}

/// DeepSeek-V4 block with mHC hyper-connections. `hidden_states` holds
/// hc_mult contiguous copies of the hidden state ([hc, hidden] flattened).
///
/// Reference `Block.forward`:
///   residual = x; y, post, comb = hc_pre(x, hc_attn_*)
///   y = attn(attn_norm(y)); x = hc_post(y, residual, post, comb)
///   residual = x; y, post, comb = hc_pre(x, hc_ffn_*)
///   y = moe(ffn_norm(y));  x = hc_post(y, residual, post, comb)
unsafe fn compute_layer_deepseek_mhc(
    dense: &[u8],
    layer_index: usize,
    token_pos: usize,
    active_experts_ptrs: &[IoBlockPtr],
    hidden_states: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    let mut carry = compute_layer_deepseek_mhc_phase1(
        dense,
        layer_index,
        token_pos,
        hidden_states,
        kv_cache,
        scratch,
        config,
    )?;
    compute_layer_deepseek_mhc_phase2(
        dense,
        layer_index,
        active_experts_ptrs,
        &mut carry,
        hidden_states,
        scratch,
        config,
        gemm,
        stats,
    )
}

/// State carried between the mHC attention phase and the MoE phase so the
/// caller can run the router on the REAL gate input (post-attention,
/// post-ffn_norm) and only then issue the expert I/O - matching the
/// reference ordering instead of the pre-layer prefetch approximation.
pub struct DeepSeekMhcCarry {
    /// hc_pre(ffn) + ffn_norm output: the exact router/MoE input.
    pub gate_input: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
    residual: Vec<f32>,
    values: usize,
}

/// Phase 1 of the DeepSeek mHC block: attention sub-block (hc_pre ->
/// attn_norm -> MLA -> hc_post) plus the ffn hc_pre + ffn_norm. Returns the
/// carry holding the gate input for routing and the ffn hc_post state.
pub unsafe fn compute_layer_deepseek_mhc_phase1(
    dense: &[u8],
    layer_index: usize,
    token_pos: usize,
    hidden_states: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
) -> Result<DeepSeekMhcCarry, ComputeError> {
    let hc = config.hc_mult;
    let hidden = config.hidden_size;
    let hc_dim = hc
        .checked_mul(hidden)
        .ok_or(ComputeError::InvalidShape("mHC dim overflow"))?;
    if hidden_states.len() < hc_dim {
        return Err(ComputeError::InvalidShape("mHC hidden state too small"));
    }
    let layout = parse_quant_block(dense)?;
    let Some(hc_attn) = hc_params_from_layout(&layout, "hc_attn")? else {
        return Err(ComputeError::InvalidShape("mHC hc_attn params missing"));
    };
    let Some(hc_ffn) = hc_params_from_layout(&layout, "hc_ffn")? else {
        return Err(ComputeError::InvalidShape("mHC hc_ffn params missing"));
    };

    let required = hc_dim
        .checked_add(hidden)
        .and_then(|value| value.checked_add(1))
        .ok_or(ComputeError::InvalidShape("mHC scratch overflow"))?;
    if scratch.dequant_tile_f32.len() < required {
        return Err(ComputeError::ScratchTooSmall {
            required,
            actual: scratch.dequant_tile_f32.len(),
        });
    }
    let (residual, rest) = scratch.dequant_tile_f32.split_at_mut(hc_dim);
    let (y, sub_scratch_buf) = rest.split_at_mut(hidden);
    let mut sub_scratch = ComputeScratch {
        dequant_tile_f32: sub_scratch_buf,
    };

    if std::env::var("ZC_MHC_DEBUG").is_ok() {
        let input_l2 = hidden_states[..hc_dim]
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            .sqrt();
        eprintln!(
            "mhc_debug layer={} token_pos={} input_l2={:.6}",
            layer_index, token_pos, input_l2
        );
    }

    // --- attention sub-block ---
    residual.copy_from_slice(&hidden_states[..hc_dim]);
    let (post, comb) = hc_pre_into(
        &hc_attn,
        &hidden_states[..hc_dim],
        y,
        hc,
        hidden,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )?;
    if let Some(norm) = norm_tensor_by_marker(&layout, "input_layernorm")? {
        apply_weighted_rms_norm(y, norm, &mut sub_scratch.dequant_tile_f32[..hidden])?;
        let (normed, _) = sub_scratch.dequant_tile_f32.split_at(hidden);
        y.copy_from_slice(normed);
    } else {
        rms_norm_in_place(y);
    }
    let sample_debug = std::env::var("ZC_MHC_DEBUG").is_ok();
    if sample_debug {
        eprintln!(
            "sample_debug layer={} pos={} point=attn_in values={:?}",
            layer_index,
            token_pos,
            &y[..8]
        );
        let row_sums: Vec<f32> = (0..hc)
            .map(|row| (0..hc).map(|col| comb[row * hc + col]).sum())
            .collect();
        let col_sums: Vec<f32> = (0..hc)
            .map(|col| (0..hc).map(|row| comb[row * hc + col]).sum())
            .collect();
        eprintln!(
            "sample_debug layer={} pos={} point=hc_attn post={:?} comb_row_sums={:?} comb_col_sums={:?}",
            layer_index, token_pos, &post, row_sums, col_sums
        );
    }
    let values = compute_deepseek_v4_mla_attention(
        &layout,
        layer_index,
        token_pos,
        y,
        kv_cache,
        &mut sub_scratch,
        config,
    )?;
    if sample_debug {
        eprintln!(
            "sample_debug layer={} pos={} point=attn_out values={:?}",
            layer_index,
            token_pos,
            &y[..8]
        );
    }
    hc_post_into(y, residual, &post, &comb, &mut hidden_states[..hc_dim], hc, hidden)?;

    // --- ffn hc_pre + norm: produce the real gate/MoE input ---
    let ffn_residual = hidden_states[..hc_dim].to_vec();
    let (post, comb) = hc_pre_into(
        &hc_ffn,
        &hidden_states[..hc_dim],
        y,
        hc,
        hidden,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )?;
    if let Some(norm) = norm_tensor_by_marker(&layout, "post_attention_layernorm")? {
        apply_weighted_rms_norm(y, norm, &mut sub_scratch.dequant_tile_f32[..hidden])?;
        let (normed, _) = sub_scratch.dequant_tile_f32.split_at(hidden);
        y.copy_from_slice(normed);
    } else {
        rms_norm_in_place(y);
    }
    if sample_debug {
        eprintln!(
            "sample_debug layer={} pos={} point=ffn_in values={:?}",
            layer_index,
            token_pos,
            &y[..8]
        );
    }
    Ok(DeepSeekMhcCarry {
        gate_input: y.to_vec(),
        post,
        comb,
        residual: ffn_residual,
        values,
    })
}

/// Phase 2 of the DeepSeek mHC block: MoE on the carried gate input, then
/// the ffn hc_post back into the hc copies.
pub unsafe fn compute_layer_deepseek_mhc_phase2(
    dense: &[u8],
    layer_index: usize,
    active_experts_ptrs: &[IoBlockPtr],
    carry: &mut DeepSeekMhcCarry,
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    let hc = config.hc_mult;
    let hidden = config.hidden_size;
    let hc_dim = hc
        .checked_mul(hidden)
        .ok_or(ComputeError::InvalidShape("mHC dim overflow"))?;
    stats.dequantized_values += carry.values;
    carry.values = 0;
    let layout = parse_quant_block(dense)?;
    let profile = dense_math_profile_from_layout(&layout)?;
    let y = &mut carry.gate_input;
    if profile.has_shared_expert {
        let shared_applied = compute_active_and_shared_experts(
            dense,
            active_experts_ptrs,
            y,
            scratch,
            config,
            gemm,
            stats,
        )?;
        crate::vlog!(
            "math_fidelity layer={} component=shared_expert present=true applied={}",
            layer_index, shared_applied
        );
    } else {
        compute_active_experts(active_experts_ptrs, y, scratch, config, gemm, stats)?;
    }
    let sample_debug = std::env::var("ZC_MHC_DEBUG").is_ok();
    if sample_debug {
        eprintln!(
            "sample_debug layer={} point=ffn_out values={:?}",
            layer_index,
            &y[..8]
        );
    }
    hc_post_into(
        y,
        &carry.residual,
        &carry.post,
        &carry.comb,
        &mut hidden_states[..hc_dim],
        hc,
        hidden,
    )?;
    if sample_debug {
        crate::vlog!(
            "sample_debug layer={} point=final_hidden_hc values={:?}",
            layer_index,
            &hidden_states[..8]
        );
    }
    crate::vlog!(
        "math_fidelity layer={} component=mhc_hyper_connections present=true applied=true hc={}",
        layer_index, hc
    );
    Ok(())
}

fn compute_attention_with_residual(
    dense: &[u8],
    layer_index: usize,
    token_pos: usize,
    hidden_states: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    let hidden = hidden_states.len().min(config.hidden_size);
    if hidden == 0 {
        return Err(ComputeError::InvalidShape("hidden_states is empty"));
    }
    let required = hidden
        .checked_add(1)
        .ok_or(ComputeError::InvalidShape("attention residual scratch overflow"))?;
    if scratch.dequant_tile_f32.len() < required {
        return Err(ComputeError::ScratchTooSmall {
            required,
            actual: scratch.dequant_tile_f32.len(),
        });
    }

    let (attention_residual, attention_scratch_buf) = scratch.dequant_tile_f32.split_at_mut(hidden);
    attention_residual.copy_from_slice(&hidden_states[..hidden]);
    let mut attention_scratch = ComputeScratch {
        dequant_tile_f32: attention_scratch_buf,
    };

    apply_named_layer_norm_in_place(
        dense,
        "input_layernorm",
        hidden_states,
        &mut attention_scratch,
    )?;
    compute_attention_block(
        dense,
        layer_index,
        token_pos,
        hidden_states,
        kv_cache,
        &mut attention_scratch,
        config,
        gemm,
        stats,
    )?;
    for (slot, residual) in hidden_states
        .iter_mut()
        .take(hidden)
        .zip(attention_residual.iter())
    {
        *slot += *residual;
    }
    Ok(())
}

/// Batch prefill: accumulate ONE expert's weighted contribution into the
/// routed accumulators of every position that selected it. Inputs are the
/// per-position gate inputs (flat npos x hidden); `targets` lists
/// (position_index, normalized_weight) pairs; `routed_acc_flat` is the flat
/// npos x hidden accumulator. Math identical to compute_active_experts
/// (weight already normalized by the caller), just grouped expert-major so
/// each expert block is read from NVMe once per layer instead of once per
/// position.
pub unsafe fn accumulate_expert_for_positions(
    expert_bytes: &[u8],
    gate_inputs_flat: &[f32],
    hidden: usize,
    targets: &[(usize, f32)],
    routed_acc_flat: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    if scratch.dequant_tile_f32.len() <= hidden {
        return Err(ComputeError::ScratchTooSmall {
            required: hidden + 1,
            actual: scratch.dequant_tile_f32.len(),
        });
    }
    let (work, expert_scratch_buf) = scratch.dequant_tile_f32.split_at_mut(hidden);
    let mut expert_scratch = ComputeScratch {
        dequant_tile_f32: expert_scratch_buf,
    };
    for &(position, weight) in targets {
        work.copy_from_slice(&gate_inputs_flat[position * hidden..(position + 1) * hidden]);
        compute_expert_block(expert_bytes, work, &mut expert_scratch, config, gemm, stats)?;
        let acc = &mut routed_acc_flat[position * hidden..(position + 1) * hidden];
        for (slot, value) in acc.iter_mut().zip(work.iter()) {
            *slot += *value * weight;
        }
    }
    stats.expert_bytes += expert_bytes.len();
    Ok(())
}

/// Batch prefill: finish one position's MoE phase after the routed
/// contributions were accumulated expert-major. Mirrors
/// compute_active_and_shared_experts + hc_post of phase 2: shared expert on
/// the gate input, plus the accumulated routed output, then hc_post back
/// into the position's hc copies.
pub unsafe fn finish_layer_deepseek_mhc_phase2_batch(
    dense: &[u8],
    carry: &mut DeepSeekMhcCarry,
    routed: Option<&[f32]>,
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    let hc = config.hc_mult;
    let hidden = config.hidden_size;
    let hc_dim = hc
        .checked_mul(hidden)
        .ok_or(ComputeError::InvalidShape("mHC dim overflow"))?;
    stats.dequantized_values += carry.values;
    carry.values = 0;
    let layout = parse_quant_block(dense)?;
    let profile = dense_math_profile_from_layout(&layout)?;
    let y = &mut carry.gate_input;
    let original = y.clone();
    let shared_applied = if profile.has_shared_expert {
        compute_shared_expert_from_dense(dense, y, scratch, config, gemm, stats)?
    } else {
        false
    };
    match routed {
        Some(routed_out) => {
            if shared_applied {
                for (slot, value) in y.iter_mut().zip(routed_out.iter()) {
                    *slot += *value;
                }
            } else {
                y[..hidden.min(routed_out.len())]
                    .copy_from_slice(&routed_out[..hidden.min(routed_out.len())]);
            }
        }
        None => {
            if !shared_applied {
                y.copy_from_slice(&original);
            }
        }
    }
    hc_post_into(
        y,
        &carry.residual,
        &carry.post,
        &carry.comb,
        &mut hidden_states[..hc_dim],
        hc,
        hidden,
    )?;
    Ok(())
}

unsafe fn compute_active_and_shared_experts(
    dense: &[u8],
    active_experts_ptrs: &[IoBlockPtr],
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<bool, ComputeError> {
    let hidden = hidden_states.len().min(config.hidden_size);
    if hidden == 0 {
        return Err(ComputeError::InvalidShape("hidden_states is empty"));
    }

    let required = hidden
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("shared expert scratch overflow"))?;
    if scratch.dequant_tile_f32.len() <= required {
        return Err(ComputeError::ScratchTooSmall {
            required: required + 1,
            actual: scratch.dequant_tile_f32.len(),
        });
    }

    let (state_scratch, expert_scratch_buf) = scratch.dequant_tile_f32.split_at_mut(required);
    let (original, routed_output) = state_scratch.split_at_mut(hidden);
    original.copy_from_slice(&hidden_states[..hidden]);

    let mut expert_scratch = ComputeScratch {
        dequant_tile_f32: expert_scratch_buf,
    };

    if active_experts_ptrs.is_empty() {
        let applied = compute_shared_expert_from_dense(
            dense,
            hidden_states,
            &mut expert_scratch,
            config,
            gemm,
            stats,
        )?;
        if !applied {
            hidden_states[..hidden].copy_from_slice(original);
        }
        return Ok(applied);
    }

    compute_active_experts(
        active_experts_ptrs,
        hidden_states,
        &mut expert_scratch,
        config,
        gemm,
        stats,
    )?;
    routed_output.copy_from_slice(&hidden_states[..hidden]);
    hidden_states[..hidden].copy_from_slice(original);

    let applied = compute_shared_expert_from_dense(
        dense,
        hidden_states,
        &mut expert_scratch,
        config,
        gemm,
        stats,
    )?;
    if applied {
        for (slot, routed) in hidden_states.iter_mut().take(hidden).zip(routed_output.iter()) {
            *slot += *routed;
        }
    } else {
        hidden_states[..hidden].copy_from_slice(routed_output);
    }
    Ok(applied)
}

unsafe fn compute_active_experts(
    active_experts_ptrs: &[IoBlockPtr],
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    if active_experts_ptrs.is_empty() {
        return Ok(());
    }
    if active_experts_ptrs.len() == 1 {
        let expert = &active_experts_ptrs[0];
        let expert_bytes = std::slice::from_raw_parts(expert.ptr, expert.len);
        compute_expert_block(expert_bytes, hidden_states, scratch, config, gemm, stats)?;
        stats.expert_bytes += expert.len;
        return Ok(());
    }

    let hidden = hidden_states.len().min(config.hidden_size);
    let required = hidden
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("multi-expert scratch overflow"))?;
    if scratch.dequant_tile_f32.len() <= required {
        return Err(ComputeError::ScratchTooSmall {
            required: required + 1,
            actual: scratch.dequant_tile_f32.len(),
        });
    }

    let (state_scratch, expert_scratch_buf) = scratch.dequant_tile_f32.split_at_mut(required);
    let (original, accum) = state_scratch.split_at_mut(hidden);
    original.copy_from_slice(&hidden_states[..hidden]);
    accum.fill(0.0);
    let weight_sum = valid_route_weight_sum(active_experts_ptrs);
    let uniform_weight = 1.0 / active_experts_ptrs.len() as f32;

    let mut expert_scratch = ComputeScratch {
        dequant_tile_f32: expert_scratch_buf,
    };
    for expert in active_experts_ptrs {
        let weight = weight_sum
            .map(|sum| expert.route_weight / sum)
            .unwrap_or(uniform_weight);
        hidden_states[..hidden].copy_from_slice(original);
        let expert_bytes = std::slice::from_raw_parts(expert.ptr, expert.len);
        compute_expert_block(
            expert_bytes,
            hidden_states,
            &mut expert_scratch,
            config,
            gemm,
            stats,
        )?;
        for (slot, value) in accum.iter_mut().zip(hidden_states.iter().take(hidden)) {
            *slot += *value * weight;
        }
        stats.expert_bytes += expert.len;
    }

    for (slot, value) in hidden_states.iter_mut().take(hidden).zip(accum.iter()) {
        *slot = *value;
    }
    Ok(())
}

fn valid_route_weight_sum(active_experts_ptrs: &[IoBlockPtr]) -> Option<f32> {
    let sum = active_experts_ptrs
        .iter()
        .map(|expert| expert.route_weight)
        .try_fold(0.0f32, |acc, weight| {
            if weight.is_finite() && weight >= 0.0 {
                Some(acc + weight)
            } else {
                None
            }
        })?;
    if sum <= f32::EPSILON {
        None
    } else {
        Some(sum)
    }
}

pub fn is_non_compute_dense_block(dense: &[u8]) -> Result<bool, ComputeError> {
    let layout = parse_quant_block(dense)?;
    if layout.tensor_count() == 0 {
        return Ok(true);
    }

    let mut has_compute_tensor = false;
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        match tensor.role() {
            TensorRole::QkvProj
            | TensorRole::QProj
            | TensorRole::KProj
            | TensorRole::VProj
            | TensorRole::OProj
            | TensorRole::KvProj
            | TensorRole::Router
            | TensorRole::GateProj
            | TensorRole::UpProj
            | TensorRole::DownProj
            | TensorRole::SharedExpert => has_compute_tensor = true,
            TensorRole::Embed | TensorRole::LmHead | TensorRole::Norm | TensorRole::Unknown => {}
        }
    }
    Ok(!has_compute_tensor)
}

pub fn dense_math_profile(dense: &[u8]) -> Result<DenseMathProfile, ComputeError> {
    let layout = parse_quant_block(dense)?;
    dense_math_profile_from_layout(&layout)
}

fn dense_math_profile_from_layout(
    layout: &QuantBlockLayout<'_>,
) -> Result<DenseMathProfile, ComputeError> {
    let mut profile = DenseMathProfile::default();
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        match tensor.role() {
            TensorRole::QkvProj
            | TensorRole::QProj
            | TensorRole::KProj
            | TensorRole::VProj
            | TensorRole::OProj
            | TensorRole::KvProj => profile.has_attention = true,
            TensorRole::Router => profile.has_router = true,
            TensorRole::SharedExpert => profile.has_shared_expert = true,
            TensorRole::Norm => profile.norm_tensors += 1,
            TensorRole::GateProj
            | TensorRole::UpProj
            | TensorRole::DownProj
            | TensorRole::Embed
            | TensorRole::LmHead
            | TensorRole::Unknown => {}
        }
        let name = std::str::from_utf8(tensor.name).unwrap_or("").to_ascii_lowercase();
        if name.contains("indexer.") || name.contains(".indexer") {
            profile.has_indexer = true;
        }
    }
    Ok(profile)
}

/// Options controlling expert routing for one layer.
#[derive(Clone, Copy, Debug)]
pub struct RouteOptions {
    pub math: RouterMath,
    pub route_scale: f32,
    /// Current input token id: enables DeepSeek hash routing (`tid2eid`
    /// lookup) when the dense block carries the static table.
    pub hash_token_id: Option<u32>,
}

impl Default for RouteOptions {
    fn default() -> Self {
        Self {
            math: RouterMath::RawLogits,
            route_scale: 1.0,
            hash_token_id: None,
        }
    }
}

impl RouteOptions {
    pub fn from_config(config: &ComputeConfig, physical_layer_id: u32, token_id: u32) -> Self {
        Self {
            math: config.router_math,
            route_scale: config.route_scale,
            hash_token_id: if (physical_layer_id as usize) < config.num_hash_layers {
                Some(token_id)
            } else {
                None
            },
        }
    }
}

pub fn route_experts_from_dense_block(
    dense: &[u8],
    hidden_states: &[f32],
    active_experts: usize,
    available_experts: &[u32],
    scratch: &mut [f32],
    gemm: &dyn GemmKernel,
    options: RouteOptions,
) -> Result<Option<Vec<ExpertRoute>>, ComputeError> {
    if active_experts == 0 || available_experts.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let layout = parse_quant_block(dense)?;
    let Some(router) = router_gate_tensor(&layout, hidden_states.len())? else {
        return Ok(None);
    };
    let shape = tensor_matrix_shape(router)?;
    if hidden_states.len() < shape.cols {
        return Err(ComputeError::InvalidShape("router input too small"));
    }
    if scratch.len() < shape.rows {
        return Err(ComputeError::ScratchTooSmall {
            required: shape.rows,
            actual: scratch.len(),
        });
    }

    gemv_tensorwise_from_layout(
        &layout,
        router,
        &hidden_states[..shape.cols],
        &mut scratch[..shape.rows],
        gemm,
    )?;

    if options.math == RouterMath::DeepSeekV4SqrtSoftplus {
        if std::env::var("ZC_MHC_DEBUG").is_ok() {
            if let Some(token_id) = options.hash_token_id {
                if let Ok(Some(table)) = router_tid2eid_tensor(&layout) {
                    if let Ok(hashed) = tid2eid_expert_ids(table, token_id) {
                        let logits: Vec<(u32, f32)> = hashed
                            .iter()
                            .map(|&expert| {
                                (expert, scratch.get(expert as usize).copied().unwrap_or(f32::NAN))
                            })
                            .collect();
                        crate::vlog!(
                            "router_hash_logits token_id={} logits={:?}",
                            token_id, logits
                        );
                    }
                }
            }
        }
        for value in scratch[..shape.rows].iter_mut() {
            *value = sqrt_softplus(*value);
        }
    }

    // DeepSeek hash routing: the first num_hash_layers layers select experts
    // through the static tid2eid table instead of score top-k. Weights still
    // come from the gate scores of the selected experts.
    if let Some(token_id) = options.hash_token_id {
        if let Some(table) = router_tid2eid_tensor(&layout)? {
            let hashed = tid2eid_expert_ids(table, token_id)?;
            let mut scored = hashed
                .into_iter()
                .filter(|expert_id| available_experts.contains(expert_id))
                .filter_map(|expert_id| {
                    let score = *scratch.get(expert_id as usize)?;
                    Some(ExpertRoute { expert_id, score })
                })
                .collect::<Vec<_>>();
            scored.truncate(active_experts.min(scored.len()));
            finalize_route_weights(&mut scored, options);
            return Ok(Some(scored));
        }
    }

    // e_score_correction bias shifts scores for selection only.
    let selection_bias = if options.math == RouterMath::DeepSeekV4SqrtSoftplus {
        router_bias_values(&layout, shape.rows)?
    } else {
        None
    };

    let mut selected = available_experts
        .iter()
        .copied()
        .filter_map(|expert_id| {
            let score = *scratch.get(expert_id as usize)?;
            let selection = selection_bias
                .as_ref()
                .and_then(|bias| bias.get(expert_id as usize).copied())
                .map(|bias| score + bias)
                .unwrap_or(score);
            Some((selection, ExpertRoute { expert_id, score }))
        })
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| right.0.total_cmp(&left.0));
    selected.truncate(active_experts.min(selected.len()));
    let mut scored = selected
        .into_iter()
        .map(|(_, route)| route)
        .collect::<Vec<_>>();
    finalize_route_weights(&mut scored, options);
    Ok(Some(scored))
}

fn sqrt_softplus(value: f32) -> f32 {
    // softplus(x) = ln(1 + e^x); numerically stable split.
    let softplus = if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    };
    softplus.max(0.0).sqrt()
}

/// DeepSeek-V4 final routing weights: normalize selected original scores to
/// sum 1, then apply route_scale. Legacy RawLogits keeps raw scores (weights
/// are normalized downstream).
fn finalize_route_weights(routes: &mut [ExpertRoute], options: RouteOptions) {
    if options.math != RouterMath::DeepSeekV4SqrtSoftplus || routes.is_empty() {
        return;
    }
    let sum: f32 = routes.iter().map(|route| route.score).sum();
    if sum > 0.0 && sum.is_finite() {
        for route in routes.iter_mut() {
            route.score = route.score / sum * options.route_scale;
        }
    } else {
        let uniform = options.route_scale / routes.len() as f32;
        for route in routes.iter_mut() {
            route.score = uniform;
        }
    }
}

fn router_tid2eid_tensor<'a>(
    layout: &'a QuantBlockLayout<'a>,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() != TensorRole::Router || tensor.shape.rank() != 2 {
            continue;
        }
        let name = std::str::from_utf8(tensor.name).unwrap_or("");
        if name.to_ascii_lowercase().contains("tid2eid") {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

/// Read the top-k expert ids for `token_id` from the static tid2eid table.
/// The converter stores the table as int64 (dtype 9) or int32 (dtype 7)
/// little-endian rows of shape [vocab, top_k].
fn tid2eid_expert_ids(
    table: QuantTensorLayout<'_>,
    token_id: u32,
) -> Result<Vec<u32>, ComputeError> {
    let shape = tensor_matrix_shape(table)?;
    let top_k = shape.cols;
    let elem_bytes = match table.dtype_original {
        7 => 4usize, // int32
        9 => 8usize, // int64
        _ => {
            return Err(ComputeError::InvalidShape(
                "tid2eid table has unsupported dtype",
            ))
        }
    };
    let row = token_id as usize;
    if row >= shape.rows {
        return Err(ComputeError::InvalidShape("tid2eid token id out of range"));
    }
    let row_bytes = top_k
        .checked_mul(elem_bytes)
        .ok_or(ComputeError::InvalidShape("tid2eid row size overflow"))?;
    let start = row
        .checked_mul(row_bytes)
        .ok_or(ComputeError::InvalidShape("tid2eid row offset overflow"))?;
    let data = table
        .data
        .get(start..start + row_bytes)
        .ok_or(ComputeError::InvalidShape("tid2eid row out of range"))?;
    let mut ids = Vec::with_capacity(top_k);
    for chunk in data.chunks_exact(elem_bytes) {
        let value = match elem_bytes {
            4 => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64,
            _ => i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
        };
        if value < 0 {
            return Err(ComputeError::InvalidShape("tid2eid negative expert id"));
        }
        ids.push(value as u32);
    }
    Ok(ids)
}

/// Read the router selection bias (`gate.bias` / e_score_correction) as f32.
fn router_bias_values(
    layout: &QuantBlockLayout<'_>,
    rows: usize,
) -> Result<Option<Vec<f32>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() != TensorRole::Router || tensor.shape.rank() != 1 {
            continue;
        }
        let name = std::str::from_utf8(tensor.name)
            .unwrap_or("")
            .to_ascii_lowercase();
        if !name.contains("bias") {
            continue;
        }
        let count = tensor.shape.dim(0)? as usize;
        if count < rows {
            continue;
        }
        let mut values = Vec::with_capacity(rows);
        match tensor.dtype_original {
            12 => {
                for i in 0..rows {
                    let offset = i * 4;
                    let bytes = tensor
                        .data
                        .get(offset..offset + 4)
                        .ok_or(ComputeError::InvalidShape("router bias out of range"))?;
                    values.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                }
            }
            11 => {
                for i in 0..rows {
                    values.push(bf16_tensor_value(tensor, i)?);
                }
            }
            _ => continue,
        }
        return Ok(Some(values));
    }
    Ok(None)
}

pub fn router_top_experts_from_dense_block(
    dense: &[u8],
    hidden_states: &[f32],
    top_k: usize,
    scratch: &mut [f32],
    gemm: &dyn GemmKernel,
) -> Result<Option<Vec<ExpertRoute>>, ComputeError> {
    if top_k == 0 {
        return Ok(Some(Vec::new()));
    }
    let layout = parse_quant_block(dense)?;
    let Some(router) = router_gate_tensor(&layout, hidden_states.len())? else {
        return Ok(None);
    };
    let shape = tensor_matrix_shape(router)?;
    if hidden_states.len() < shape.cols {
        return Err(ComputeError::InvalidShape("router input too small"));
    }
    if scratch.len() < shape.rows {
        return Err(ComputeError::ScratchTooSmall {
            required: shape.rows,
            actual: scratch.len(),
        });
    }

    gemv_tensorwise_from_layout(
        &layout,
        router,
        &hidden_states[..shape.cols],
        &mut scratch[..shape.rows],
        gemm,
    )?;

    let mut scored = scratch[..shape.rows]
        .iter()
        .copied()
        .enumerate()
        .map(|(expert_id, score)| ExpertRoute {
            expert_id: expert_id as u32,
            score,
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.score.total_cmp(&left.score));
    scored.truncate(top_k.min(scored.len()));
    Ok(Some(scored))
}

fn first_rank2_tensor_by_role<'a>(
    layout: &'a QuantBlockLayout<'a>,
    role: TensorRole,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() == role && tensor.shape.rank() == 2 {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

/// Select the router gate weight for routing. DeepSeek dense blocks carry
/// additional rank-2 Router-tagged tensors (for example the static
/// `ffn.gate.tid2eid` token-to-expert table with `vocab` rows), so the first
/// rank-2 Router tensor is not necessarily the gate projection. Prefer the
/// Router tensor whose column count matches the hidden width; fall back to
/// the first rank-2 Router tensor for legacy blocks.
fn router_gate_tensor<'a>(
    layout: &'a QuantBlockLayout<'a>,
    hidden_len: usize,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    let mut fallback = None;
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() != TensorRole::Router || tensor.shape.rank() != 2 {
            continue;
        }
        let name = std::str::from_utf8(tensor.name).unwrap_or("");
        if name.to_ascii_lowercase().contains("tid2eid") {
            // Static token-to-expert table, never the gate projection.
            continue;
        }
        let shape = tensor_matrix_shape(tensor)?;
        if shape.cols == hidden_len {
            return Ok(Some(tensor));
        }
        if fallback.is_none() {
            fallback = Some(tensor);
        }
    }
    Ok(fallback)
}

fn tensor_global_row_range(
    tensor: QuantTensorLayout<'_>,
    local_rows: usize,
) -> Result<(usize, usize), ComputeError> {
    let default = (0usize, local_rows);
    let Ok(name) = std::str::from_utf8(tensor.name) else {
        return Ok(default);
    };
    let Some((_, suffix)) = name.rsplit_once(".rows_") else {
        return Ok(default);
    };
    let Some((start_text, end_text)) = suffix.split_once('_') else {
        return Err(ComputeError::InvalidShape("tensor row range suffix missing end"));
    };
    let start = start_text
        .parse::<usize>()
        .map_err(|_| ComputeError::InvalidShape("tensor row range start invalid"))?;
    let end = end_text
        .parse::<usize>()
        .map_err(|_| ComputeError::InvalidShape("tensor row range end invalid"))?;
    if end < start {
        return Err(ComputeError::InvalidShape("tensor row range end before start"));
    }
    if end - start != local_rows {
        return Err(ComputeError::InvalidShape("tensor row range does not match rows"));
    }
    Ok((start, end))
}

/// FP8 E4M3 quantize-dequantize simulation for activations (QAT match).
/// Reference: kernel.py `act_quant(..., round_scale=True, inplace=True)` and
/// the numpy port `quant_dequant_fp8_activation`: per block of
/// `block_size` values, amax-derived power-of-2 scale, nearest E4M3 value
/// (ties toward the lower LUT entry), then dequantized in place.
pub fn fp8_act_quant_dequant_in_place(values: &mut [f32], block_size: usize) {
    if block_size == 0 || values.is_empty() {
        return;
    }
    let lut = fp8_e4m3_sorted_lut();
    for block in values.chunks_mut(block_size) {
        let amax = block
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()))
            .max(1.0e-4);
        let scale = (amax / 448.0).log2().ceil().exp2();
        for value in block.iter_mut() {
            let scaled = (*value / scale).clamp(-448.0, 448.0);
            *value = nearest_in_sorted_lut(lut, scaled) * scale;
        }
    }
}

fn fp8_e4m3_sorted_lut() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut values = [0.0f32; 256];
        for (byte, slot) in values.iter_mut().enumerate() {
            *slot = decode_fp8_e4m3(byte as u8, 1.0);
        }
        values.sort_by(f32::total_cmp);
        values
    })
}

fn nearest_in_sorted_lut(lut: &[f32; 256], target: f32) -> f32 {
    let idx = lut.partition_point(|value| *value < target);
    let lower = lut[idx.saturating_sub(1).min(255)];
    let upper = lut[idx.min(255)];
    // Nearest value; ties resolve to the lower entry (numpy reference).
    if (upper - target).abs() < (lower - target).abs() {
        upper
    } else {
        lower
    }
}

/// Per-layer compress ratios for DeepSeek-V4-Flash (reference
/// `ModelArgs.compress_ratios`): 0 = pure sliding-window attention with
/// plain RoPE theta 10000; >0 = compressed path with YaRN scaling and
/// compress_rope_theta 160000. Index = physical layer id (43 layers + MTP).
pub const DEEPSEEK_V4_COMPRESS_RATIOS: [u32; 44] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
];

/// YaRN rope scaling parameters (reference `precompute_freqs_cis`).
#[derive(Clone, Copy, Debug)]
pub struct YarnRope {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_seq_len: usize,
}

pub const DEEPSEEK_V4_YARN: YarnRope = YarnRope {
    factor: 16.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
    original_seq_len: 65_536,
};

pub const DEEPSEEK_V4_COMPRESS_ROPE_THETA: f32 = 160_000.0;

/// (theta, yarn) selection for a DeepSeek physical layer.
pub fn deepseek_rope_for_layer(physical_layer_id: usize) -> (f32, Option<YarnRope>) {
    let ratio = DEEPSEEK_V4_COMPRESS_RATIOS
        .get(physical_layer_id)
        .copied()
        .unwrap_or(0);
    if ratio > 0 {
        (DEEPSEEK_V4_COMPRESS_ROPE_THETA, Some(DEEPSEEK_V4_YARN))
    } else {
        (10_000.0, None)
    }
}

/// Precompute per-pair inverse frequencies, optionally YaRN-scaled.
/// Matches reference `precompute_freqs_cis`: linear ramp between the
/// beta_fast/beta_slow correction dims blends original and factor-scaled
/// frequencies.
pub fn rope_inv_freqs(rope_dim: usize, theta: f32, yarn: Option<YarnRope>) -> Vec<f32> {
    let pairs = rope_dim / 2;
    let mut freqs = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        freqs.push(theta.powf(-((2 * pair) as f32) / rope_dim as f32));
    }
    let Some(yarn) = yarn else {
        return freqs;
    };
    if yarn.original_seq_len == 0 {
        return freqs;
    }
    let dim = rope_dim as f32;
    let base_ln = theta.ln();
    let find_correction_dim = |num_rotations: f32| -> f32 {
        dim * ((yarn.original_seq_len as f32) / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * base_ln)
    };
    let low = find_correction_dim(yarn.beta_fast).floor().max(0.0);
    let mut high = find_correction_dim(yarn.beta_slow).ceil().min(dim - 1.0);
    if (low - high).abs() < f32::EPSILON {
        high = low + 0.001;
    }
    for (pair, freq) in freqs.iter_mut().enumerate() {
        let ramp = (((pair as f32) - low) / (high - low)).clamp(0.0, 1.0);
        let smooth = 1.0 - ramp;
        *freq = *freq / yarn.factor * (1.0 - smooth) + *freq * smooth;
    }
    freqs
}

/// Interleaved RoPE with explicit per-pair inverse frequencies.
pub fn apply_rope_interleaved_with_freqs(
    values: &mut [f32],
    position: usize,
    inv_freqs: &[f32],
    inverse: bool,
) -> Result<(), ComputeError> {
    let rope_dim = inv_freqs.len() * 2;
    if rope_dim == 0 {
        return Ok(());
    }
    if values.len() < rope_dim {
        return Err(ComputeError::InvalidShape("RoPE input shorter than dim"));
    }
    let position = position as f32;
    for (pair, inv_freq) in inv_freqs.iter().enumerate() {
        let even_index = pair * 2;
        let odd_index = even_index + 1;
        let angle = position * inv_freq;
        let (mut sin, cos) = angle.sin_cos();
        if inverse {
            sin = -sin;
        }
        let even = values[even_index];
        let odd = values[odd_index];
        values[even_index] = even * cos - odd * sin;
        values[odd_index] = even * sin + odd * cos;
    }
    Ok(())
}

pub fn apply_rope_interleaved_in_place(
    values: &mut [f32],
    position: usize,
    rope_dim: usize,
    theta: f32,
) -> Result<(), ComputeError> {
    if rope_dim == 0 {
        return Ok(());
    }
    if rope_dim % 2 != 0 {
        return Err(ComputeError::InvalidShape("RoPE dim must be even"));
    }
    if values.len() < rope_dim {
        return Err(ComputeError::InvalidShape("RoPE input shorter than dim"));
    }
    if theta <= 0.0 {
        return Err(ComputeError::InvalidShape("RoPE theta must be positive"));
    }

    let position = position as f32;
    for pair in 0..(rope_dim / 2) {
        let even_index = pair * 2;
        let odd_index = even_index + 1;
        let inv_freq = theta.powf(-(even_index as f32) / rope_dim as f32);
        let angle = position * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let even = values[even_index];
        let odd = values[odd_index];
        values[even_index] = even * cos - odd * sin;
        values[odd_index] = even * sin + odd * cos;
    }
    Ok(())
}

pub fn apply_rope_to_heads_interleaved(
    values: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    theta: f32,
) -> Result<(), ComputeError> {
    apply_rope_to_heads_interleaved_at(values, num_heads, head_dim, 0, rope_dim, position, theta)
}

pub fn apply_rope_to_heads_interleaved_at(
    values: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rope_offset: usize,
    rope_dim: usize,
    position: usize,
    theta: f32,
) -> Result<(), ComputeError> {
    if num_heads == 0 || head_dim == 0 {
        return Err(ComputeError::InvalidShape("RoPE heads/head_dim must be non-zero"));
    }
    let required = num_heads
        .checked_mul(head_dim)
        .ok_or(ComputeError::InvalidShape("RoPE head span overflow"))?;
    if values.len() < required {
        return Err(ComputeError::InvalidShape("RoPE head input too small"));
    }
    if rope_offset
        .checked_add(rope_dim)
        .ok_or(ComputeError::InvalidShape("RoPE offset overflow"))?
        > head_dim
    {
        return Err(ComputeError::InvalidShape("RoPE dim larger than head_dim"));
    }
    for head in 0..num_heads {
        let start = head * head_dim + rope_offset;
        let end = start + rope_dim;
        apply_rope_interleaved_in_place(&mut values[start..end], position, rope_dim, theta)?;
    }
    Ok(())
}

pub unsafe fn lm_head_argmax_from_block(
    block: &[u8],
    hidden_states: &[f32],
    scratch_logits: &mut [f32],
    gemm: &dyn GemmKernel,
) -> Result<Option<u32>, ComputeError> {
    lm_head_argmax_chunked_from_block(block, hidden_states, scratch_logits, gemm)
}

pub unsafe fn lm_head_argmax_chunked_from_block(
    block: &[u8],
    hidden_states: &[f32],
    scratch: &mut [f32],
    gemm: &dyn GemmKernel,
) -> Result<Option<u32>, ComputeError> {
    lm_head_argmax_score_chunked_from_block(block, hidden_states, scratch, gemm)
        .map(|best| best.map(|(token, _)| token))
}

pub unsafe fn lm_head_argmax_score_chunked_from_block(
    block: &[u8],
    hidden_states: &[f32],
    scratch: &mut [f32],
    gemm: &dyn GemmKernel,
) -> Result<Option<(u32, f32)>, ComputeError> {
    Ok(lm_head_topk_score_chunked_from_block(block, hidden_states, scratch, gemm, 1)?
        .into_iter()
        .next())
}

pub unsafe fn lm_head_topk_score_chunked_from_block(
    block: &[u8],
    hidden_states: &[f32],
    scratch: &mut [f32],
    gemm: &dyn GemmKernel,
    top_k: usize,
) -> Result<Vec<(u32, f32)>, ComputeError> {
    let scratch_len = scratch.len();
    let layout = parse_quant_block(block)?;
    if scratch.is_empty() {
        return Err(ComputeError::ScratchTooSmall {
            required: 1,
            actual: 0,
        });
    }

    let top_k = top_k.max(1);
    let mut best = Vec::with_capacity(top_k.min(64));
    let mut found = false;

    for index in 0..layout.tensor_count() {
        let lm_head = layout.tensor(index)?;
        if lm_head.role() != TensorRole::LmHead || lm_head.shape.rank() != 2 {
            continue;
        }
        found = true;
        let shape = tensor_matrix_shape(lm_head)?;
        if hidden_states.len() < shape.cols {
            return Err(ComputeError::InvalidShape("LM-head input too small"));
        }

        if lm_head.quant_format == QUANT_DEEPSEEK_BF16_AUX && lm_head.dtype_original == 11 {
            let input = &hidden_states[..shape.cols];
            score_bf16_lm_head_topk(lm_head, shape, input, scratch, top_k, &mut best)?;
            continue;
        }

        let (input, logits_scratch): (&[f32], &mut [f32]) =
            match final_norm_tensor(&layout, shape.cols)? {
                Some(norm) if scratch.len() > shape.cols => {
                    let (normed, logits) = scratch.split_at_mut(shape.cols);
                    apply_weighted_rms_norm(hidden_states, norm, normed)?;
                    (&normed[..shape.cols], logits)
                }
                _ => (&hidden_states[..shape.cols], &mut *scratch),
            };
        if logits_scratch.is_empty() {
            return Err(ComputeError::ScratchTooSmall {
                required: shape.cols + 1,
                actual: scratch_len,
            });
        }

        let (row_base, _) = tensor_global_row_range(lm_head, shape.rows)?;
        let row_bytes = packed_i4_row_bytes(shape.cols);
        let max_rows_per_chunk = logits_scratch.len().min(shape.rows).max(1);
        let mut row_start = 0usize;

        while row_start < shape.rows {
            let rows = max_rows_per_chunk.min(shape.rows - row_start);
            let data_offset = row_start
                .checked_mul(row_bytes)
                .ok_or(ComputeError::InvalidShape("LM-head row offset overflow"))?;
            if data_offset >= lm_head.data.len() {
                return Err(ComputeError::InvalidShape("LM-head row offset out of range"));
            }

            gemm.gemv_i4_affine_tensorwise(
                rows,
                shape.cols,
                input.as_ptr(),
                lm_head.data.as_ptr().add(data_offset),
                lm_head.scale,
                lm_head.zero_point,
                logits_scratch.as_mut_ptr(),
            )?;

            for (local_index, logit) in logits_scratch[..rows].iter().copied().enumerate() {
                let token = row_base + row_start + local_index;
                let token = u32::try_from(token)
                    .map_err(|_| ComputeError::InvalidShape("LM-head token overflow"))?;
                push_topk_candidate(&mut best, top_k, token, logit);
            }
            row_start += rows;
        }
    }

    if !found {
        return Ok(Vec::new());
    }
    if best.is_empty() {
        return Err(ComputeError::InvalidShape("LM-head has no rows"));
    }
    best.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    Ok(best)
}

fn score_bf16_lm_head_topk(
    lm_head: QuantTensorLayout<'_>,
    shape: MatrixShape,
    input: &[f32],
    scratch: &mut [f32],
    top_k: usize,
    best: &mut Vec<(u32, f32)>,
) -> Result<(), ComputeError> {
    if scratch.is_empty() {
        return Err(ComputeError::ScratchTooSmall {
            required: 1,
            actual: 0,
        });
    }
    let (row_base, _) = tensor_global_row_range(lm_head, shape.rows)?;
    let row_bytes = shape
        .cols
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("BF16 LM-head row bytes overflow"))?;
    let expected_bytes = row_bytes
        .checked_mul(shape.rows)
        .ok_or(ComputeError::InvalidShape("BF16 LM-head bytes overflow"))?;
    if lm_head.data.len() < expected_bytes {
        return Err(ComputeError::InvalidShape("BF16 LM-head data too small"));
    }

    let max_rows_per_chunk = scratch.len().min(shape.rows).max(1);
    let mut row_start = 0usize;
    while row_start < shape.rows {
        let rows = max_rows_per_chunk.min(shape.rows - row_start);
        for local_row in 0..rows {
            let row = row_start + local_row;
            let row_offset = row
                .checked_mul(row_bytes)
                .ok_or(ComputeError::InvalidShape("BF16 LM-head row offset overflow"))?;
            let row_data = lm_head
                .data
                .get(row_offset..row_offset + row_bytes)
                .ok_or(ComputeError::InvalidShape("BF16 LM-head row out of range"))?;
            scratch[local_row] = dot_bf16_row(row_data, input)?;
        }
        for (local_index, logit) in scratch[..rows].iter().copied().enumerate() {
            let token = row_base + row_start + local_index;
            let token = u32::try_from(token)
                .map_err(|_| ComputeError::InvalidShape("LM-head token overflow"))?;
            push_topk_candidate(best, top_k, token, logit);
        }
        row_start += rows;
    }
    Ok(())
}

pub(crate) fn dot_bf16_row(row_data: &[u8], input: &[f32]) -> Result<f32, ComputeError> {
    if row_data.len() < input.len() * 2 {
        return Err(ComputeError::InvalidShape("BF16 row too small"));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::deepseek_v4::simd_avx2_fma_available() {
            return Ok(unsafe { simd_x86::dot_bf16_row_avx2(row_data, input) });
        }
    }
    // Four independent accumulators break the serial FP dependency chain and
    // let the compiler vectorize; changes summation order vs a single
    // accumulator (f32 reassociation noise only).
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let quads = input.len() / 4;
    for quad in 0..quads {
        let index = quad * 4;
        let offset = index * 2;
        s0 += bf16_to_f32(u16::from_le_bytes([row_data[offset], row_data[offset + 1]])) * input[index];
        s1 += bf16_to_f32(u16::from_le_bytes([row_data[offset + 2], row_data[offset + 3]])) * input[index + 1];
        s2 += bf16_to_f32(u16::from_le_bytes([row_data[offset + 4], row_data[offset + 5]])) * input[index + 2];
        s3 += bf16_to_f32(u16::from_le_bytes([row_data[offset + 6], row_data[offset + 7]])) * input[index + 3];
    }
    for index in quads * 4..input.len() {
        let offset = index * 2;
        s0 += bf16_to_f32(u16::from_le_bytes([row_data[offset], row_data[offset + 1]])) * input[index];
    }
    Ok((s0 + s1) + (s2 + s3))
}

#[inline]
fn bf16_to_f32(raw: u16) -> f32 {
    f32::from_bits((raw as u32) << 16)
}

fn push_topk_candidate(best: &mut Vec<(u32, f32)>, top_k: usize, token: u32, logit: f32) {
    if best.len() < top_k {
        best.push((token, logit));
        return;
    }
    if let Some((worst_index, (_, worst_logit))) = best
        .iter()
        .enumerate()
        .min_by(|(_, (_, left)), (_, (_, right))| left.total_cmp(right))
    {
        if logit > *worst_logit {
            best[worst_index] = (token, logit);
        }
    }
}

pub fn lm_head_row_count_from_block(block: &[u8]) -> Result<Option<usize>, ComputeError> {
    let layout = parse_quant_block(block)?;
    let mut rows = 0usize;
    let mut found = false;
    for index in 0..layout.tensor_count() {
        let lm_head = layout.tensor(index)?;
        if lm_head.role() != TensorRole::LmHead || lm_head.shape.rank() != 2 {
            continue;
        }
        found = true;
        rows = rows
            .checked_add(tensor_matrix_shape(lm_head)?.rows)
            .ok_or(ComputeError::InvalidShape("LM-head row count overflow"))?;
    }
    Ok(found.then_some(rows))
}

pub fn token_embedding_from_block(
    block: &[u8],
    token_id: u32,
    output: &mut [f32],
) -> Result<bool, ComputeError> {
    let layout = parse_quant_block(block)?;
    let token_id = token_id as usize;
    for index in 0..layout.tensor_count() {
        let embed = layout.tensor(index)?;
        if embed.role() != TensorRole::Embed || embed.shape.rank() != 2 {
            continue;
        }
        let shape = tensor_matrix_shape(embed)?;
        let (row_base, row_end) = tensor_global_row_range(embed, shape.rows)?;
        if token_id < row_base || token_id >= row_end {
            continue;
        }
        if output.len() < shape.cols {
            return Err(ComputeError::OutputTooSmall {
                required: shape.cols,
                actual: output.len(),
            });
        }

        let local_token_id = token_id - row_base;
        if embed.quant_format == QUANT_DEEPSEEK_BF16_AUX && embed.dtype_original == 11 {
            let row_bytes = shape
                .cols
                .checked_mul(2)
                .ok_or(ComputeError::InvalidShape("BF16 embedding row bytes overflow"))?;
            let row_offset = local_token_id
                .checked_mul(row_bytes)
                .ok_or(ComputeError::InvalidShape("BF16 embedding row offset overflow"))?;
            let row = embed
                .data
                .get(row_offset..row_offset + row_bytes)
                .ok_or(ComputeError::InvalidShape("BF16 embedding row outside tensor data"))?;
            for (index, slot) in output[..shape.cols].iter_mut().enumerate() {
                let offset = index * 2;
                let raw = u16::from_le_bytes([row[offset], row[offset + 1]]);
                *slot = bf16_to_f32(raw);
            }
            return Ok(true);
        }
        if embed.quant_format as u32 != 4 {
            return Err(ComputeError::UnsupportedQuantFormat(DiskQuantFormat::Int8Symmetric));
        }
        let row_bytes = packed_i4_row_bytes(shape.cols);
        let row_offset = local_token_id
            .checked_mul(row_bytes)
            .ok_or(ComputeError::InvalidShape("embedding row offset overflow"))?;
        if row_offset + row_bytes > embed.data.len() {
            return Err(ComputeError::InvalidShape("embedding row outside tensor data"));
        }
        let row = unsafe { embed.data.as_ptr().add(row_offset) };
        for (index, slot) in output[..shape.cols].iter_mut().enumerate() {
            *slot = unsafe {
                decode_i4_affine_tensorwise_at(row, index, embed.scale, embed.zero_point)
            };
        }
        return Ok(true);
    }
    Ok(false)
}

fn final_norm_tensor<'a>(
    layout: &'a QuantBlockLayout<'a>,
    hidden_size: usize,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() != TensorRole::Norm || tensor.shape.rank() != 1 {
            continue;
        }
        if tensor.shape.dim(0)? != hidden_size {
            continue;
        }
        if is_final_norm_name(tensor.name) {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

fn is_final_norm_name(name: &[u8]) -> bool {
    let Ok(name) = std::str::from_utf8(name) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower == "model.norm.weight"
        || lower.ends_with(".model.norm.weight")
        || lower.ends_with("transformer.norm.weight")
        || lower.ends_with("final_layernorm.weight")
        || lower.ends_with("ln_f.weight")
}

fn apply_weighted_rms_norm(
    hidden_states: &[f32],
    norm: QuantTensorLayout<'_>,
    output: &mut [f32],
) -> Result<(), ComputeError> {
    let hidden = hidden_states.len();
    if output.len() < hidden {
        return Err(ComputeError::OutputTooSmall {
            required: hidden,
            actual: output.len(),
        });
    }
    if norm.shape.rank() != 1 || norm.shape.dim(0)? < hidden {
        return Err(ComputeError::InvalidShape("final norm shape mismatch"));
    }

    let mean_square = hidden_states
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / hidden as f32;
    // eps 1e-6: matches the numpy reference (rms_norm / rms_norm_rows with
    // HC_EPS); 1e-5 here caused a small systematic scale delta on every
    // weighted norm that act_quant then amplified to grid-step flips.
    let inv_rms = 1.0 / (mean_square + 1.0e-6).sqrt();
    for (index, hidden) in hidden_states.iter().copied().enumerate() {
        let weight = if norm.quant_format == QUANT_DEEPSEEK_BF16_AUX && norm.dtype_original == 11 {
            bf16_tensor_value(norm, index)?
        } else if norm.quant_format as u32 == 4 {
            unsafe {
                decode_i4_affine_tensorwise_at(norm.data.as_ptr(), index, norm.scale, norm.zero_point)
            }
        } else {
            return Err(ComputeError::UnsupportedQuantFormat(DiskQuantFormat::Int8Symmetric));
        };
        output[index] = hidden * inv_rms * weight;
    }
    Ok(())
}

fn bf16_tensor_value(tensor: QuantTensorLayout<'_>, index: usize) -> Result<f32, ComputeError> {
    let offset = index
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("BF16 tensor offset overflow"))?;
    let raw = tensor
        .data
        .get(offset..offset + 2)
        .ok_or(ComputeError::InvalidShape("BF16 tensor value out of range"))?;
    Ok(bf16_to_f32(u16::from_le_bytes([raw[0], raw[1]])))
}

fn apply_named_layer_norm_in_place(
    dense: &[u8],
    marker: &str,
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
) -> Result<bool, ComputeError> {
    let layout = parse_quant_block(dense)?;
    let Some(norm) = norm_tensor_by_marker(&layout, marker)? else {
        return Ok(false);
    };
    let hidden = hidden_states.len();
    if scratch.dequant_tile_f32.len() < hidden {
        return Err(ComputeError::ScratchTooSmall {
            required: hidden,
            actual: scratch.dequant_tile_f32.len(),
        });
    }
    apply_weighted_rms_norm(hidden_states, norm, &mut scratch.dequant_tile_f32[..hidden])?;
    hidden_states.copy_from_slice(&scratch.dequant_tile_f32[..hidden]);
    Ok(true)
}

fn norm_tensor_by_marker<'a>(
    layout: &'a QuantBlockLayout<'a>,
    marker: &str,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() != TensorRole::Norm || tensor.shape.rank() != 1 {
            continue;
        }
        let name = std::str::from_utf8(tensor.name).unwrap_or("");
        if norm_name_matches_marker(name, marker) {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

fn norm_name_matches_marker(name: &str, marker: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.contains(marker) {
        return true;
    }
    match marker {
        "input_layernorm" => lower.contains("attn_norm"),
        "post_attention_layernorm" => lower.contains("ffn_norm"),
        "q_a_layernorm" => lower.contains(".q_norm") || lower.ends_with("q_norm.weight"),
        "kv_a_layernorm" => lower.contains(".kv_norm") || lower.ends_with("kv_norm.weight"),
        _ => false,
    }
}

fn compute_attention_block(
    dense: &[u8],
    layer_index: usize,
    token_pos: usize,
    hidden_states: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    let layout = parse_quant_block(dense)?;
    let profile = dense_math_profile_from_layout(&layout)?;
    if profile.has_attention {
        if profile.has_indexer {
            let context_len = token_pos.saturating_add(1);
            let (reason, equivalent) = glm52_indexer_bypass_status(context_len);
            crate::vlog!(
                "math_fidelity layer={} component=indexer present=true applied=false reason={} equivalent_to_full_causal={}",
                layer_index, reason, equivalent
            );
            if !equivalent {
                return Err(ComputeError::InvalidShape(
                    "GLM-DSA indexer required for context above topk",
                ));
            }
        }
        if config.attention_kind == AttentionKind::DeepSeekV4Mla {
            let values = compute_deepseek_v4_mla_attention(
                &layout,
                layer_index,
                token_pos,
                hidden_states,
                kv_cache,
                scratch,
                config,
            )?;
            stats.dequantized_values += values;
            return Ok(());
        }
        let values = compute_glm_dsa_attention_prefill_probe(
            &layout,
            layer_index,
            token_pos,
            hidden_states,
            kv_cache,
            scratch,
            config,
            gemm,
        )?;
        stats.dequantized_values += values;
        return Ok(());
    }
    let values = compute_quant_block_layout_into_hidden(&layout, hidden_states, scratch, config, gemm)?;
    stats.dequantized_values += values;
    Ok(())
}

/// DeepSeek-V4-Flash MLA attention (reference: inference/model.py
/// `Attention.forward`). Differences from the GLM probe path:
/// - single 512-dim latent KV per token (num_kv_heads=1) used as both key
///   and value - there is NO kv_b expansion in this architecture;
/// - kv_norm covers the full head_dim (448 nope + 64 rope);
/// - per-head parameter-free q RMS after wq_b;
/// - `attn_sink` adds exp(sink - max) to the softmax denominator;
/// - inverse RoPE is applied to the attention output rope dims;
/// - output projection is grouped low-rank: o reshaped to o_groups groups,
///   per-group wo_a rows -> o_lora, concatenated, then wo_b -> hidden.
///
/// Sliding-window/compressed sparse selection is not implemented yet: for
/// bounded contexts (token_pos < window 128) attending over all cached
/// positions matches the reference window behavior exactly.
/// One tensor of a synthetic in-RAM ZCBLK01 block (see
/// `build_synthetic_quant_block`).
pub struct SyntheticTensor<'a> {
    pub name: &'a str,
    pub role_code: u32,
    pub shape: &'a [u64],
    pub data: &'a [u8],
    pub dtype_original: u16,
    pub quant_format: u16,
    pub scale: f32,
    pub zero_point: f32,
}

/// Builds a ZCBLK01 compute block in RAM from loose tensors. Used by the
/// MTP module: its dense tensors live inside the giant global block of
/// dense_core (not readable through the block scheduler), so they are read
/// individually via the tensor index and reassembled here, letting the
/// standard `parse_quant_block` + phase1/phase2 compute paths run on them
/// unchanged.
pub fn build_synthetic_quant_block(tensors: &[SyntheticTensor<'_>]) -> Vec<u8> {
    let names_offset = QUANT_BLOCK_HEADER_SIZE + tensors.len() * QUANT_TENSOR_RECORD_SIZE;
    let mut names = Vec::new();
    let mut name_offsets = Vec::new();
    for tensor in tensors {
        name_offsets.push(names.len() as u64);
        names.extend_from_slice(&(tensor.name.len() as u16).to_le_bytes());
        names.extend_from_slice(tensor.name.as_bytes());
    }
    let mut cursor = names_offset + names.len();
    let mut data_offsets = Vec::new();
    let mut shape_offsets = Vec::new();
    for tensor in tensors {
        data_offsets.push(cursor as u64);
        cursor += tensor.data.len();
        shape_offsets.push(cursor as u64);
        cursor += 4 + tensor.shape.len() * 8;
    }

    let mut block = Vec::with_capacity(cursor);
    block.extend_from_slice(&QUANT_BLOCK_MAGIC);
    block.extend_from_slice(&QUANT_BLOCK_VERSION.to_le_bytes());
    block.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    block.extend_from_slice(&4u32.to_le_bytes());
    block.extend_from_slice(&0u32.to_le_bytes());
    block.extend_from_slice(&(QUANT_BLOCK_HEADER_SIZE as u64).to_le_bytes());
    block.extend_from_slice(&(names_offset as u64).to_le_bytes());
    for (index, tensor) in tensors.iter().enumerate() {
        block.extend_from_slice(&tensor.dtype_original.to_le_bytes());
        block.extend_from_slice(&tensor.quant_format.to_le_bytes());
        block.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        block.extend_from_slice(&tensor.role_code.to_le_bytes());
        block.extend_from_slice(&name_offsets[index].to_le_bytes());
        block.extend_from_slice(&shape_offsets[index].to_le_bytes());
        block.extend_from_slice(&data_offsets[index].to_le_bytes());
        block.extend_from_slice(&(tensor.data.len() as u64).to_le_bytes());
        block.extend_from_slice(&tensor.scale.to_le_bytes());
        block.extend_from_slice(&tensor.zero_point.to_le_bytes());
    }
    block.extend_from_slice(&names);
    for tensor in tensors {
        block.extend_from_slice(tensor.data);
        block.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in tensor.shape {
            block.extend_from_slice(&dim.to_le_bytes());
        }
    }
    block
}

/// Role code for a tensor of the synthetic MTP dense block. The converter
/// classifies every `mtp.*` name as the opaque MTP role, so the tensor index
/// carries role 0 for the attention/norm/router tensors - but the compute
/// paths (phase1/phase2) look tensors up by compute role. Re-derive the role
/// from the name exactly like the converter does for the main layers.
pub fn mtp_synthetic_role_code(name: &str, index_role_code: u16) -> u16 {
    let lowered = name.to_ascii_lowercase();
    if lowered.contains(".ffn.shared_experts.") {
        // Weight AND scale: the FP8 shared-expert detection checks both.
        return TensorRole::SharedExpert.code();
    }
    if lowered.contains(".ffn.gate.") {
        return TensorRole::Router.code();
    }
    if lowered.ends_with("norm.weight") {
        return TensorRole::Norm.code();
    }
    if lowered.ends_with(".scale") {
        return index_role_code;
    }
    if lowered.contains(".attn.wq_a.") || lowered.contains(".attn.wq_b.") {
        return TensorRole::QProj.code();
    }
    if lowered.contains(".attn.wkv.") {
        return TensorRole::KvProj.code();
    }
    if lowered.contains(".attn.wo_a.") || lowered.contains(".attn.wo_b.") {
        return TensorRole::OProj.code();
    }
    index_role_code
}

/// MTP draft input (reference `MTPBlock.forward` before `super().forward`):
/// `x = e_proj(enorm(embed(t+1)))  broadcast over the hc copies
///      + h_proj(hnorm(hidden))    per copy`.
/// `main_hidden_hc` is the main model's hc state POST all layers (PRE
/// hc_head pooling); `token_embed` is the raw embedding row of the token
/// sampled at t+1. Writes the hc-wide input of the MTP block into
/// `out_hidden`.
pub fn mtp_prepare_draft_hidden(
    dense: &[u8],
    token_embed: &[f32],
    main_hidden_hc: &[f32],
    out_hidden: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
) -> Result<usize, ComputeError> {
    let hc = config.hc_mult.max(1);
    let hidden = config.hidden_size;
    let hc_dim = hc
        .checked_mul(hidden)
        .ok_or(ComputeError::InvalidShape("MTP hc dim overflow"))?;
    if token_embed.len() < hidden
        || main_hidden_hc.len() < hc_dim
        || out_hidden.len() < hc_dim
    {
        return Err(ComputeError::InvalidShape("MTP draft input sizes"));
    }
    let layout = parse_quant_block(dense)?;
    let enorm = norm_tensor_by_marker(&layout, "enorm")?
        .ok_or(ComputeError::InvalidShape("MTP enorm missing"))?;
    let hnorm = norm_tensor_by_marker(&layout, "hnorm")?
        .ok_or(ComputeError::InvalidShape("MTP hnorm missing"))?;
    let e_proj = tensor_by_name_suffix(&layout, "e_proj.weight")?
        .ok_or(ComputeError::InvalidShape("MTP e_proj missing"))?;
    let h_proj = tensor_by_name_suffix(&layout, "h_proj.weight")?
        .ok_or(ComputeError::InvalidShape("MTP h_proj missing"))?;
    let required = hidden
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("MTP draft scratch overflow"))?;
    if scratch.dequant_tile_f32.len() < required {
        return Err(ComputeError::ScratchTooSmall {
            required,
            actual: scratch.dequant_tile_f32.len(),
        });
    }
    let (work, rest) = scratch.dequant_tile_f32.split_at_mut(hidden);
    let (e_out, _rest) = rest.split_at_mut(hidden);

    let mut values = 0usize;
    // e = e_proj(enorm(embed)); QAT act quant before every fp8 matvec.
    apply_weighted_rms_norm(&token_embed[..hidden], enorm, work)?;
    if e_proj.quant_format == QUANT_DEEPSEEK_FP8_E4M3 {
        fp8_act_quant_dequant_in_place(work, 64);
    }
    values += gemv_tensorwise_from_layout(&layout, e_proj, work, e_out, gemm)?;
    // Per hc copy: h_proj(hnorm(x_c)) + e (broadcast).
    for copy in 0..hc {
        let src = &main_hidden_hc[copy * hidden..(copy + 1) * hidden];
        apply_weighted_rms_norm(src, hnorm, work)?;
        if h_proj.quant_format == QUANT_DEEPSEEK_FP8_E4M3 {
            fp8_act_quant_dequant_in_place(work, 64);
        }
        let dst = &mut out_hidden[copy * hidden..(copy + 1) * hidden];
        values += gemv_tensorwise_from_layout(&layout, h_proj, work, dst, gemm)?;
        for (slot, e_value) in dst.iter_mut().zip(e_out.iter()) {
            *slot += *e_value;
        }
    }
    Ok(values)
}

/// MTP head boundary (reference `MTPBlock.forward` tail): pool the hc
/// copies with the MTP block's OWN `hc_head_fn/scale/base`, then apply its
/// OWN final weighted RMS norm (`final_norm_name`, e.g. "mtp.0.norm.weight").
/// The lm_head itself is shared with the main model and scanned by the
/// caller on `pooled_out`.
pub fn mtp_head_pool_and_norm(
    dense: &[u8],
    final_norm_name: &str,
    hidden_hc: &[f32],
    pooled_out: &mut [f32],
    config: &ComputeConfig,
) -> Result<(), ComputeError> {
    let hc = config.hc_mult.max(1);
    let hidden = config.hidden_size;
    let hc_dim = hc
        .checked_mul(hidden)
        .ok_or(ComputeError::InvalidShape("MTP hc dim overflow"))?;
    if hidden_hc.len() < hc_dim || pooled_out.len() < hidden {
        return Err(ComputeError::InvalidShape("MTP head input sizes"));
    }
    let layout = parse_quant_block(dense)?;
    let params = hc_params_from_layout(&layout, "hc_head")?
        .ok_or(ComputeError::InvalidShape("MTP hc_head params missing"))?;
    if params.scale.is_empty() {
        return Err(ComputeError::InvalidShape("MTP hc_head scale empty"));
    }
    let mut pooled = vec![0.0f32; hidden];
    hc_head_pool(
        params.fn_data,
        params.scale[0],
        &params.base,
        hidden_hc,
        &mut pooled,
        hc,
        hidden,
        config.hc_eps,
    )?;
    let norm = find_optional_tensor_by_name(&layout, final_norm_name)?
        .ok_or(ComputeError::InvalidShape("MTP final norm missing"))?;
    apply_weighted_rms_norm(&pooled, norm, pooled_out)?;
    Ok(())
}

/// Sliding-window size of the DeepSeek sparse attention (reference
/// WINDOW_SIZE): raw KV positions attended beyond the compressed blocks.
const DEEPSEEK_V4_SLIDING_WINDOW_SIZE: usize = 128;

/// Incremental state of the DSA short-context compressor for one layer.
/// Mirrors `compressor_prefill_kv` in the numpy reference, computed
/// position-by-position: pending wkv/wgate projections of the current
/// block, the first-half columns of the previous block (overlap for
/// ratio==4), and the finalized compressed KV rows (head_dim each).
struct CompressorLayerState {
    coff: usize,
    pending_kv: Vec<f32>,
    pending_score: Vec<f32>,
    pending_len: usize,
    prev_first_half: Option<(Vec<f32>, Vec<f32>)>,
    compressed: Vec<f32>,
    missing_logged: bool,
    /// Highest token position already fed. The generation loop runs the
    /// last prompt position twice (prefill then decode); the second pass
    /// must not feed the stateful compressor again.
    last_fed_pos: Option<usize>,
    /// Speculative-decode undo stack (M3): one pre-feed snapshot per
    /// speculative position, so a rejected draft position can be rolled
    /// back exactly (the state is cumulative and a block flush clears the
    /// pending buffers - a simple truncate is not enough).
    spec_undo: Vec<CompressorSnapshot>,
}

/// Pre-feed snapshot of a compressor layer's mutable fields, taken for
/// positions fed while speculation is active.
struct CompressorSnapshot {
    pos: usize,
    pending_kv: Vec<f32>,
    pending_score: Vec<f32>,
    pending_len: usize,
    prev_first_half: Option<(Vec<f32>, Vec<f32>)>,
    compressed_len: usize,
    last_fed_pos: Option<usize>,
}

/// Minimum speculative position while a speculative pass is in flight:
/// feeds at positions >= this value push an undo snapshot first.
fn deepseek_compressor_spec_min() -> &'static std::sync::Mutex<Option<usize>> {
    static SPEC: std::sync::OnceLock<std::sync::Mutex<Option<usize>>> = std::sync::OnceLock::new();
    SPEC.get_or_init(|| std::sync::Mutex::new(None))
}

/// Begin a speculative pass: compressor feeds at positions >= `min_pos`
/// snapshot their layer state first so they can be rolled back.
pub fn deepseek_compressor_speculation_begin(min_pos: usize) {
    *deepseek_compressor_spec_min().lock().unwrap() = Some(min_pos);
}

/// Accept the speculative positions: drop the undo snapshots.
pub fn deepseek_compressor_speculation_commit() {
    *deepseek_compressor_spec_min().lock().unwrap() = None;
    let mut state_map = deepseek_compressor_state().lock().unwrap();
    for state in state_map.values_mut() {
        state.spec_undo.clear();
    }
}

/// Reject the speculative positions: restore every layer to its state
/// right before the first feed at a position >= `from_pos`.
pub fn deepseek_compressor_speculation_rollback(from_pos: usize) {
    *deepseek_compressor_spec_min().lock().unwrap() = None;
    let mut state_map = deepseek_compressor_state().lock().unwrap();
    for state in state_map.values_mut() {
        let mut restore: Option<CompressorSnapshot> = None;
        while state
            .spec_undo
            .last()
            .map_or(false, |snapshot| snapshot.pos >= from_pos)
        {
            // Pop until the OLDEST snapshot >= from_pos is in hand.
            restore = state.spec_undo.pop();
        }
        if let Some(snapshot) = restore {
            state.pending_kv = snapshot.pending_kv;
            state.pending_score = snapshot.pending_score;
            state.pending_len = snapshot.pending_len;
            state.prev_first_half = snapshot.prev_first_half;
            state.compressed.truncate(snapshot.compressed_len);
            state.last_fed_pos = snapshot.last_fed_pos;
        }
    }
}

/// Portable copy of one layer's compressor state, held by a session slot
/// so a restored KV prefix comes back with the compressor state that
/// produced it (the live map below is process-global and only matches the
/// LAST request).
struct CompressorLayerExport {
    coff: usize,
    pending_kv: Vec<f32>,
    pending_score: Vec<f32>,
    pending_len: usize,
    prev_first_half: Option<(Vec<f32>, Vec<f32>)>,
    compressed: Vec<f32>,
    last_fed_pos: Option<usize>,
}

/// Whole-map compressor snapshot for session save/restore (multi-slot
/// session cache). Opaque outside this module.
pub struct CompressorSessionState {
    layers: Vec<(usize, CompressorLayerExport)>,
}

/// Snapshots the global compressor state at the end of a request. Never
/// call mid-speculation (undo stacks are intentionally not exported; the
/// generation loop commits or rolls back before storing a session).
pub fn deepseek_compressor_state_export() -> CompressorSessionState {
    let state_map = deepseek_compressor_state().lock().unwrap();
    let mut layers: Vec<(usize, CompressorLayerExport)> = state_map
        .iter()
        .map(|(&layer, state)| {
            (
                layer,
                CompressorLayerExport {
                    coff: state.coff,
                    pending_kv: state.pending_kv.clone(),
                    pending_score: state.pending_score.clone(),
                    pending_len: state.pending_len,
                    prev_first_half: state.prev_first_half.clone(),
                    compressed: state.compressed.clone(),
                    last_fed_pos: state.last_fed_pos,
                },
            )
        })
        .collect();
    layers.sort_by_key(|(layer, _)| *layer);
    CompressorSessionState { layers }
}

/// Replaces the global compressor state with a snapshot, right after the
/// matching KV prefix has been restored into the request's cache.
pub fn deepseek_compressor_state_import(snapshot: &CompressorSessionState) {
    let mut state_map = deepseek_compressor_state().lock().unwrap();
    state_map.clear();
    for (layer, export) in &snapshot.layers {
        state_map.insert(
            *layer,
            CompressorLayerState {
                coff: export.coff,
                pending_kv: export.pending_kv.clone(),
                pending_score: export.pending_score.clone(),
                pending_len: export.pending_len,
                prev_first_half: export.prev_first_half.clone(),
                compressed: export.compressed.clone(),
                missing_logged: false,
                last_fed_pos: export.last_fed_pos,
                spec_undo: Vec::new(),
            },
        );
    }
}

/// Process-global compressor state, one entry per physical layer. The
/// generation server runs one request at a time; a layer entry resets when
/// it sees token_pos == 0 (start of a new prefill). Session slots carry
/// their own snapshot (export/import above) so a restored prefix does not
/// depend on which request ran last.
fn deepseek_compressor_state(
) -> &'static std::sync::Mutex<std::collections::HashMap<usize, CompressorLayerState>> {
    static STATE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, CompressorLayerState>>,
    > = std::sync::OnceLock::new();
    STATE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn tensor_by_name_suffix<'a>(
    layout: &'a QuantBlockLayout<'a>,
    suffix: &str,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        let name = std::str::from_utf8(tensor.name).unwrap_or("");
        if name.ends_with(suffix) {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

fn read_f32_tensor_values(tensor: &QuantTensorLayout<'_>, count: usize) -> Option<Vec<f32>> {
    if tensor.data.len() < count * 4 {
        return None;
    }
    Some(
        tensor.data[..count * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

/// Feeds this position's attention input into the layer's compressor and
/// returns a snapshot of the compressed KV rows visible at `token_pos`
/// (flat, `head_dim` per row). Empty when the layer has no compressor
/// (ratio 0 or tensors absent) or no complete block is available yet.
fn deepseek_compressor_feed_and_snapshot(
    layout: &QuantBlockLayout<'_>,
    physical_layer: usize,
    token_pos: usize,
    x_attn: &[f32],
    inv_freqs: &[f32],
    head_dim: usize,
    nope_dim: usize,
) -> Result<Vec<f32>, ComputeError> {
    let ratio = *DEEPSEEK_V4_COMPRESS_RATIOS
        .get(physical_layer)
        .unwrap_or(&0) as usize;
    if ratio == 0 {
        return Ok(Vec::new());
    }
    let coff = if ratio == 4 { 2 } else { 1 };
    let proj_dim = coff * head_dim;

    // Read the speculation flag BEFORE locking the state map (fixed lock
    // order: spec flag first, state map second - no nesting).
    let spec_min_pos: Option<usize> = *deepseek_compressor_spec_min().lock().unwrap();

    let mut state_map = deepseek_compressor_state().lock().unwrap();
    let state = state_map
        .entry(physical_layer)
        .or_insert_with(|| CompressorLayerState {
            coff,
            pending_kv: Vec::new(),
            pending_score: Vec::new(),
            pending_len: 0,
            prev_first_half: None,
            compressed: Vec::new(),
            missing_logged: false,
            last_fed_pos: None,
            spec_undo: Vec::new(),
        });
    if token_pos == 0 {
        state.pending_kv.clear();
        state.pending_score.clear();
        state.pending_len = 0;
        state.prev_first_half = None;
        state.compressed.clear();
        state.last_fed_pos = None;
        state.spec_undo.clear();
    }
    // Idempotent per position: a re-run of an already-fed position (the
    // generation loop re-executes the last prompt position when decoding)
    // only reads the snapshot, never feeds again.
    if state.last_fed_pos.map_or(false, |fed| token_pos <= fed) {
        let blocks = state.compressed.len() / head_dim;
        let available = ((token_pos + 1) / ratio).min(blocks).min(512);
        return Ok(state.compressed[..available * head_dim].to_vec());
    }

    let (Some(wkv), Some(wgate), Some(ape), Some(norm)) = (
        tensor_by_name_suffix(layout, "attn.compressor.wkv.weight")?,
        tensor_by_name_suffix(layout, "attn.compressor.wgate.weight")?,
        tensor_by_name_suffix(layout, "attn.compressor.ape")?,
        tensor_by_name_suffix(layout, "attn.compressor.norm.weight")?,
    ) else {
        if !state.missing_logged {
            state.missing_logged = true;
            crate::vlog!(
                "compressor layer={} mode=missing_tensors ratio={}",
                physical_layer, ratio
            );
        }
        return Ok(Vec::new());
    };

    // Project this position: kv = wkv @ x, score = wgate @ x (+ ape row).
    let mut kv_row = vec![0.0f32; proj_dim];
    let mut score_row = vec![0.0f32; proj_dim];
    let gemm_fp8 = FusedInt4Gemm;
    let mut project = |tensor: &QuantTensorLayout<'_>,
                       out: &mut [f32]|
     -> Result<(), ComputeError> {
        if tensor.quant_format == QUANT_DEEPSEEK_FP8_E4M3 {
            let scale = fp8_scale_for_weight(layout, *tensor)?;
            gemv_fp8_e4m3_ue8m0_tensorwise(*tensor, scale, x_attn, out)?;
        } else if tensor.quant_format == QUANT_DEEPSEEK_BF16_AUX {
            gemv_bf16_tensorwise(*tensor, x_attn, out)?;
        } else {
            gemv_tensorwise_from_layout(layout, *tensor, x_attn, out, &gemm_fp8)?;
        }
        Ok(())
    };
    project(&wkv, &mut kv_row)?;
    project(&wgate, &mut score_row)?;
    let ape_values = read_f32_tensor_values(&ape, ratio * proj_dim)
        .ok_or(ComputeError::InvalidShape("compressor ape too small"))?;
    if token_pos == 0 && std::env::var("ZC_MHC_DEBUG").is_ok() {
        crate::vlog!(
            "compressor_tensors layer={} wkv_fmt={} wgate_fmt={} ape_fmt={} norm_fmt={} ape4={:?} kv_row4={:?} score_row4={:?}",
            physical_layer,
            wkv.quant_format,
            wgate.quant_format,
            ape.quant_format,
            norm.quant_format,
            &ape_values[..4],
            &kv_row[..4],
            &score_row[..4]
        );
    }
    // M3 speculative feed: snapshot the layer state before the first
    // mutation for this position, so a rejected draft rolls back exactly.
    if let Some(min_pos) = spec_min_pos {
        if token_pos >= min_pos
            && state
                .spec_undo
                .last()
                .map_or(true, |snapshot| snapshot.pos != token_pos)
        {
            state.spec_undo.push(CompressorSnapshot {
                pos: token_pos,
                pending_kv: state.pending_kv.clone(),
                pending_score: state.pending_score.clone(),
                pending_len: state.pending_len,
                prev_first_half: state.prev_first_half.clone(),
                compressed_len: state.compressed.len(),
                last_fed_pos: state.last_fed_pos,
            });
        }
    }

    let pos_in_block = state.pending_len;
    for (slot, ape_value) in score_row
        .iter_mut()
        .zip(&ape_values[pos_in_block * proj_dim..(pos_in_block + 1) * proj_dim])
    {
        *slot += *ape_value;
    }
    state.pending_kv.extend_from_slice(&kv_row);
    state.pending_score.extend_from_slice(&score_row);
    state.pending_len += 1;
    state.last_fed_pos = Some(token_pos);

    if state.pending_len == ratio {
        // Finalize the block: per-dim softmax over the pool of slots
        // (previous block first-half + current block second-half when
        // overlapping, else the current block itself), weighted sum,
        // weighted RMS norm, RoPE tail at the block start position, then
        // FP8-sim act quant on the non-rope dims.
        let block_index = state.compressed.len() / head_dim;
        let mut compressed = vec![0.0f32; head_dim];
        for dim in 0..head_dim {
            let mut slot_scores: Vec<f32> = Vec::with_capacity(2 * ratio);
            let mut slot_values: Vec<f32> = Vec::with_capacity(2 * ratio);
            if coff == 2 {
                if let Some((prev_kv, prev_score)) = state.prev_first_half.as_ref() {
                    for pos in 0..ratio {
                        slot_scores.push(prev_score[pos * head_dim + dim]);
                        slot_values.push(prev_kv[pos * head_dim + dim]);
                    }
                }
                for pos in 0..ratio {
                    slot_scores.push(state.pending_score[pos * proj_dim + head_dim + dim]);
                    slot_values.push(state.pending_kv[pos * proj_dim + head_dim + dim]);
                }
            } else {
                for pos in 0..ratio {
                    slot_scores.push(state.pending_score[pos * proj_dim + dim]);
                    slot_values.push(state.pending_kv[pos * proj_dim + dim]);
                }
            }
            let max_score = slot_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut acc = 0.0f32;
            for (score, value) in slot_scores.iter().zip(&slot_values) {
                let weight = (score - max_score).exp();
                denom += weight;
                acc += weight * value;
            }
            compressed[dim] = if denom > 0.0 { acc / denom } else { 0.0 };
        }
        let mut normed = vec![0.0f32; head_dim];
        apply_weighted_rms_norm(&compressed, norm, &mut normed)?;
        apply_rope_interleaved_with_freqs(
            &mut normed[nope_dim..head_dim],
            block_index * ratio,
            inv_freqs,
            false,
        )?;
        fp8_act_quant_dequant_in_place(&mut normed[..nope_dim], 64);
        state.compressed.extend_from_slice(&normed);
        if coff == 2 {
            let mut prev_kv = vec![0.0f32; ratio * head_dim];
            let mut prev_score = vec![0.0f32; ratio * head_dim];
            for pos in 0..ratio {
                prev_kv[pos * head_dim..(pos + 1) * head_dim]
                    .copy_from_slice(&state.pending_kv[pos * proj_dim..pos * proj_dim + head_dim]);
                prev_score[pos * head_dim..(pos + 1) * head_dim].copy_from_slice(
                    &state.pending_score[pos * proj_dim..pos * proj_dim + head_dim],
                );
            }
            state.prev_first_half = Some((prev_kv, prev_score));
        }
        state.pending_kv.clear();
        state.pending_score.clear();
        state.pending_len = 0;
        let block_l2 = state.compressed[block_index * head_dim..]
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            .sqrt();
        crate::vlog!(
            "compressor layer={} mode=short_context_prefill block_index={} ratio={} overlap={} l2={:.6} sample={:?}",
            physical_layer,
            block_index,
            ratio,
            coff == 2,
            block_l2,
            &state.compressed[block_index * head_dim..block_index * head_dim + head_dim.min(8)]
        );
    }

    let blocks = state.compressed.len() / head_dim;
    let available = ((token_pos + 1) / ratio).min(blocks).min(512);
    Ok(state.compressed[..available * head_dim].to_vec())
}

/// Fine-grained phase1 profiling counters (ZC_PROF=1): nanoseconds per
/// sub-phase of the MLA attention path, accumulated process-wide and dumped
/// by the pass profiler in generation.rs.
pub mod phase1_prof {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static Q_NS: AtomicU64 = AtomicU64::new(0);
    pub static KV_NS: AtomicU64 = AtomicU64::new(0);
    pub static COMPRESSOR_NS: AtomicU64 = AtomicU64::new(0);
    pub static WINDOW_NS: AtomicU64 = AtomicU64::new(0);
    pub static O_NS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("ZC_PROF").map(|v| v == "1").unwrap_or(false))
    }

    pub fn add(counter: &AtomicU64, started: std::time::Instant) {
        counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Returns the per-phase totals in ms and resets the counters.
    pub fn dump_and_reset() -> String {
        let take = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed) / 1_000_000;
        format!(
            "q_ms={} kv_ms={} compressor_ms={} window_ms={} o_ms={}",
            take(&Q_NS),
            take(&KV_NS),
            take(&COMPRESSOR_NS),
            take(&WINDOW_NS),
            take(&O_NS)
        )
    }
}

fn compute_deepseek_v4_mla_attention(
    layout: &QuantBlockLayout<'_>,
    layer_index: usize,
    token_pos: usize,
    hidden_states: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
) -> Result<usize, ComputeError> {
    let heads = config.num_attention_heads;
    let head_dim = config
        .qk_nope_head_dim
        .checked_add(config.qk_rope_head_dim)
        .ok_or(ComputeError::InvalidShape("MLA head dim overflow"))?;
    let rope_dim = config.qk_rope_head_dim;
    let nope_dim = config.qk_nope_head_dim;
    let hidden = config.hidden_size;
    let q_full_len = heads
        .checked_mul(head_dim)
        .ok_or(ComputeError::InvalidShape("MLA q size overflow"))?;
    let groups = config.o_groups;
    let o_lora = config.o_lora_rank;
    if groups == 0 || o_lora == 0 || q_full_len % groups != 0 {
        return Err(ComputeError::InvalidShape("MLA o_groups configuration"));
    }
    let group_in = q_full_len / groups;
    let o_mid_len = groups
        .checked_mul(o_lora)
        .ok_or(ComputeError::InvalidShape("MLA o mid size overflow"))?;

    // scratch: q_lora | q | kv | attn_out | o_mid
    let required = config
        .q_lora_rank
        .checked_add(q_full_len)
        .and_then(|value| value.checked_add(head_dim))
        .and_then(|value| value.checked_add(q_full_len))
        .and_then(|value| value.checked_add(o_mid_len))
        .ok_or(ComputeError::InvalidShape("MLA scratch size overflow"))?;
    if scratch.dequant_tile_f32.len() < required {
        return Err(ComputeError::ScratchTooSmall {
            required,
            actual: scratch.dequant_tile_f32.len(),
        });
    }
    let (q_lora, rest) = scratch.dequant_tile_f32.split_at_mut(config.q_lora_rank);
    let (q, rest) = rest.split_at_mut(q_full_len);
    let (kv, rest) = rest.split_at_mut(head_dim);
    let (attn_out, o_mid_full) = rest.split_at_mut(q_full_len);
    let o_mid = &mut o_mid_full[..o_mid_len];

    let mut values = 0usize;
    let gemm_fp8 = FusedInt4Gemm;

    // Debug: ZC_DUMP_ATTN=<dir> dumps raw f32 buffers at layer 0 / pos 3
    // for bit-level comparison against the numpy reference formula.
    let dump_dir = std::env::var("ZC_DUMP_ATTN")
        .ok()
        .filter(|_| layer_index == 0 && token_pos == 3);
    let dump_f32 = |name: &str, values: &[f32]| {
        if let Some(dir) = dump_dir.as_ref() {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(format!("{dir}/{name}.bin"), bytes);
        }
    };

    let prof_on = phase1_prof::enabled();
    // q path: wq_a -> q_norm -> wq_b -> per-head RMS -> RoPE
    let q_started = Instant::now();
    dump_f32("attn_in_x", &hidden_states[..hidden]);
    // QAT: the reference quant-dequants the FP8 activation before EVERY
    // fp8 matvec (fp8_matvec: x = quant_dequant_fp8_activation(x)). The
    // original x stays untouched (the compressor consumes it in bf16).
    let mut x_fp8 = hidden_states[..hidden].to_vec();
    fp8_act_quant_dequant_in_place(&mut x_fp8, 64);
    let Some(wq_a) = tensor_by_role_with_cols(layout, TensorRole::QProj, hidden)? else {
        return Err(ComputeError::InvalidShape("MLA wq_a missing"));
    };
    values += gemv_tensorwise_from_layout(layout, wq_a, &x_fp8, q_lora, &gemm_fp8)?;
    dump_f32("q_lora_pre_norm", q_lora);
    if let Some(q_norm) = norm_tensor_by_marker(layout, "q_a_layernorm")? {
        apply_weighted_rms_norm(q_lora, q_norm, &mut attn_out[..config.q_lora_rank])?;
        q_lora.copy_from_slice(&attn_out[..config.q_lora_rank]);
    } else {
        rms_norm_in_place(q_lora);
    }
    let Some(wq_b) = tensor_by_role_with_cols(layout, TensorRole::QProj, config.q_lora_rank)?
    else {
        return Err(ComputeError::InvalidShape("MLA wq_b missing"));
    };
    fp8_act_quant_dequant_in_place(q_lora, 64);
    values += gemv_tensorwise_from_layout(layout, wq_b, q_lora, q, &gemm_fp8)?;
    dump_f32("q_full_pre_head_norm", &q[..q_full_len]);
    for head in 0..heads {
        let slice = &mut q[head * head_dim..(head + 1) * head_dim];
        rms_norm_in_place(slice);
    }
    // Per-layer rope: plain theta 10000 on pure-window layers, YaRN with
    // compress_rope_theta on compressed layers (reference behavior).
    let (layer_theta, yarn) = deepseek_rope_for_layer(layer_index);
    let inv_freqs = rope_inv_freqs(rope_dim, layer_theta, yarn);
    for head in 0..heads {
        let start = head * head_dim + nope_dim;
        apply_rope_interleaved_with_freqs(
            &mut q[start..start + rope_dim],
            token_pos,
            &inv_freqs,
            false,
        )?;
    }

    if prof_on {
        phase1_prof::add(&phase1_prof::Q_NS, q_started);
    }
    let kv_started = Instant::now();
    // kv path: wkv -> kv_norm (full head_dim) -> RoPE on rope dims
    let Some(wkv) = tensor_by_role_with_cols(layout, TensorRole::KvProj, hidden)? else {
        return Err(ComputeError::InvalidShape("MLA wkv missing"));
    };
    values += gemv_tensorwise_from_layout(layout, wkv, &x_fp8, kv, &gemm_fp8)?;
    if let Some(kv_norm) = norm_tensor_by_marker(layout, "kv_a_layernorm")? {
        apply_weighted_rms_norm(kv, kv_norm, &mut attn_out[..head_dim])?;
        kv.copy_from_slice(&attn_out[..head_dim]);
        crate::vlog!(
            "math_fidelity layer={} component=kv_norm_full_head_dim present=true applied=true",
            layer_index
        );
    } else {
        rms_norm_in_place(kv);
    }
    apply_rope_interleaved_with_freqs(&mut kv[nope_dim..head_dim], token_pos, &inv_freqs, false)?;
    // FP8-simulated activation quant on the non-rope kv dims (QAT match;
    // reference: act_quant(kv[..., :-rope], 64, round_scale, inplace)).
    fp8_act_quant_dequant_in_place(&mut kv[..nope_dim], 64);

    // cache the latent kv for this position (key slot doubles as value)
    if kv_cache.num_kv_heads != 1 || kv_cache.head_dim < head_dim {
        return Err(ComputeError::InvalidShape("MLA KV cache layout mismatch"));
    }
    {
        let (key_slot, _value_slot) = kv_cache.key_value_slices_mut(layer_index, token_pos)?;
        key_slot[..head_dim].copy_from_slice(kv);
    }

    dump_f32("q_post_rope", &q[..q_full_len]);
    if dump_dir.is_some() {
        let mut all_kv = Vec::new();
        for pos in 0..=token_pos {
            all_kv.extend_from_slice(&kv_cache.key_slice(layer_index, pos)?[..head_dim]);
        }
        dump_f32("kv_cache_rows", &all_kv);
    }

    if prof_on {
        phase1_prof::add(&phase1_prof::KV_NS, kv_started);
    }
    // attn_sink per head (fp32/bf16 aux tensor)
    let sink = attention_sink_values(layout, heads)?;
    let scale = (head_dim as f32).powf(-0.5);

    // DSA short-context compressor: feed this position's attention input
    // and take the compressed KV rows visible at this position. The
    // attention window is then raw KV (sliding window) + compressed blocks
    // (reference: window_parts = [raw tail, compressed[:available]]).
    let compressor_started = Instant::now();
    let compressed = deepseek_compressor_feed_and_snapshot(
        layout,
        layer_index,
        token_pos,
        &hidden_states[..hidden],
        &inv_freqs,
        head_dim,
        nope_dim,
    )?;
    let compressed_count = compressed.len() / head_dim;
    if prof_on {
        phase1_prof::add(&phase1_prof::COMPRESSOR_NS, compressor_started);
    }
    let window_started = Instant::now();

    // two-pass softmax over the window: raw cached positions in the
    // sliding window plus the visible compressed blocks
    attn_out.fill(0.0);
    let context = token_pos + 1;
    let window_start = context.saturating_sub(DEEPSEEK_V4_SLIDING_WINDOW_SIZE);
    let debug_scores = std::env::var("ZC_MHC_DEBUG").is_ok();
    let mut scores_sumsq = 0.0f64;
    let mut weights_sumsq = 0.0f64;
    for head in 0..heads {
        let q_head = &q[head * head_dim..(head + 1) * head_dim];
        let mut max_score = f32::NEG_INFINITY;
        for pos in window_start..context {
            let key = kv_cache.key_slice(layer_index, pos)?;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_head[d] * key[d];
            }
            score *= scale;
            if debug_scores {
                scores_sumsq += (score as f64) * (score as f64);
            }
            if score > max_score {
                max_score = score;
            }
        }
        for block in 0..compressed_count {
            let key = &compressed[block * head_dim..(block + 1) * head_dim];
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_head[d] * key[d];
            }
            score *= scale;
            if debug_scores {
                scores_sumsq += (score as f64) * (score as f64);
            }
            if score > max_score {
                max_score = score;
            }
        }
        let mut denom = 0.0f32;
        if let Some(sink) = sink.as_ref() {
            denom += (sink[head] - max_score).exp();
        }
        let out_head = &mut attn_out[head * head_dim..(head + 1) * head_dim];
        let mut weights_sum = 0.0f32;
        for pos in window_start..context {
            let key = kv_cache.key_slice(layer_index, pos)?;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_head[d] * key[d];
            }
            let weight = ((score * scale) - max_score).exp();
            if debug_scores {
                weights_sumsq += (weight as f64) * (weight as f64);
            }
            weights_sum += weight;
            for d in 0..head_dim {
                out_head[d] += weight * key[d];
            }
        }
        for block in 0..compressed_count {
            let key = &compressed[block * head_dim..(block + 1) * head_dim];
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_head[d] * key[d];
            }
            let weight = ((score * scale) - max_score).exp();
            if debug_scores {
                weights_sumsq += (weight as f64) * (weight as f64);
            }
            weights_sum += weight;
            for d in 0..head_dim {
                out_head[d] += weight * key[d];
            }
        }
        denom += weights_sum;
        if denom > 0.0 {
            for value in out_head.iter_mut() {
                *value /= denom;
            }
        }
        // inverse RoPE on the output rope dims at this position
        apply_rope_interleaved_with_freqs(
            &mut out_head[nope_dim..head_dim],
            token_pos,
            &inv_freqs,
            true,
        )?;
    }

    if debug_scores {
        let l2 = |slice: &[f32]| -> f64 {
            slice
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                .sqrt()
        };
        eprintln!(
            "mla_debug layer={} pos={} q_l2={:.6} kv_l2={:.6} attn_out_l2={:.6} sink={} scores_l2={:.6} weights_l2={:.6} window={} compressed={}",
            layer_index,
            token_pos,
            l2(&q[..q_full_len]),
            l2(kv),
            l2(&attn_out[..q_full_len]),
            sink.is_some(),
            scores_sumsq.sqrt(),
            weights_sumsq.sqrt(),
            context - window_start,
            compressed_count
        );
    }

    if prof_on {
        phase1_prof::add(&phase1_prof::WINDOW_NS, window_started);
    }
    let o_started = Instant::now();
    dump_f32("attn_pre_wo", &attn_out[..q_full_len]);

    // QAT act quant before the grouped wo_a (reference fp8_grouped_wo_a
    // quantizes each 4096-wide group; block 64 never crosses a group
    // boundary so quantizing the whole vector is equivalent).
    fp8_act_quant_dequant_in_place(&mut attn_out[..q_full_len], 64);

    // grouped low-rank output projection: wo_a per group, then wo_b
    let Some(wo_a) = tensor_by_role_with_cols(layout, TensorRole::OProj, group_in)? else {
        return Err(ComputeError::InvalidShape("MLA wo_a missing"));
    };
    let wo_a_shape = tensor_matrix_shape(wo_a)?;
    if wo_a_shape.rows != o_mid_len {
        return Err(ComputeError::InvalidShape("MLA wo_a rows mismatch"));
    }
    let wo_a_scale = if wo_a.quant_format == QUANT_DEEPSEEK_FP8_E4M3 {
        Some(fp8_scale_for_weight(layout, wo_a)?)
    } else {
        None
    };
    for group in 0..groups {
        let input = &attn_out[group * group_in..(group + 1) * group_in];
        let output = &mut o_mid[group * o_lora..(group + 1) * o_lora];
        values += match wo_a_scale {
            Some(scale) => gemv_fp8_e4m3_ue8m0_rows(
                wo_a,
                scale,
                group * o_lora,
                (group + 1) * o_lora,
                input,
                output,
            )?,
            None => gemv_bf16_rows(wo_a, group * o_lora, (group + 1) * o_lora, input, output)?,
        };
    }
    dump_f32("o_mid", &o_mid[..o_mid_len]);
    let Some(wo_b) = tensor_by_role_with_cols(layout, TensorRole::OProj, o_mid_len)? else {
        return Err(ComputeError::InvalidShape("MLA wo_b missing"));
    };
    // QAT act quant before wo_b (reference fp8_matvec on the wo_a output).
    fp8_act_quant_dequant_in_place(o_mid, 64);
    values += gemv_tensorwise_from_layout(layout, wo_b, o_mid, hidden_states, &gemm_fp8)?;
    dump_f32("attn_post_wo", &hidden_states[..hidden]);
    if std::env::var("ZC_MHC_DEBUG").is_ok() {
        let l2 = |slice: &[f32]| -> f64 {
            slice
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                .sqrt()
        };
        eprintln!(
            "mla_debug layer={} pos={} o_mid_l2={:.6} attn_final_l2={:.6}",
            layer_index,
            token_pos,
            l2(o_mid),
            l2(&hidden_states[..hidden])
        );
    }
    if prof_on {
        phase1_prof::add(&phase1_prof::O_NS, o_started);
    }
    crate::vlog!(
        "math_fidelity layer={} component=deepseek_mla_attention present=true applied=true sink={} context={}",
        layer_index,
        sink.is_some(),
        context
    );
    Ok(values)
}

/// Batched MLA attention (T2): the projection matrices (wq_a, wq_b, wkv,
/// wo_a, wo_b) are streamed ONCE per layer pass and applied to every
/// position, instead of once per position - phase1 was measured
/// memory-bound on exactly those weights (~133MB/position/layer). All
/// per-position math (norms, rope, compressor feed order, window softmax,
/// activation quant) is unchanged; each gemv (row, input) dot runs the
/// same kernel as the single path, so outputs are bit-identical.
///
/// `ys` holds npos × hidden attention inputs and is overwritten with the
/// attention outputs. Falls back to the single-position path when a debug
/// env is set (those flows compare intermediates position-by-position),
/// when a projection tensor is not FP8, or with ZC_MLA_BATCH=0.
fn compute_deepseek_v4_mla_attention_batch(
    layout: &QuantBlockLayout<'_>,
    layer_index: usize,
    positions: &[usize],
    ys: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
) -> Result<usize, ComputeError> {
    let npos = positions.len();
    let hidden = config.hidden_size;
    if npos == 0 || ys.len() < npos * hidden {
        return Err(ComputeError::InvalidShape("MLA batch ys size"));
    }
    let batch_enabled = std::env::var("ZC_MLA_BATCH").map(|v| v != "0").unwrap_or(true);
    let debug_flow = std::env::var("ZC_DUMP_ATTN").is_ok() || std::env::var("ZC_MHC_DEBUG").is_ok();
    let heads = config.num_attention_heads;
    let head_dim = config
        .qk_nope_head_dim
        .checked_add(config.qk_rope_head_dim)
        .ok_or(ComputeError::InvalidShape("MLA head dim overflow"))?;
    let rope_dim = config.qk_rope_head_dim;
    let nope_dim = config.qk_nope_head_dim;
    let q_full_len = heads
        .checked_mul(head_dim)
        .ok_or(ComputeError::InvalidShape("MLA q size overflow"))?;
    let groups = config.o_groups;
    let o_lora = config.o_lora_rank;
    if groups == 0 || o_lora == 0 || q_full_len % groups != 0 {
        return Err(ComputeError::InvalidShape("MLA o_groups configuration"));
    }
    let group_in = q_full_len / groups;
    let o_mid_len = groups
        .checked_mul(o_lora)
        .ok_or(ComputeError::InvalidShape("MLA o mid size overflow"))?;

    let Some(wq_a) = tensor_by_role_with_cols(layout, TensorRole::QProj, hidden)? else {
        return Err(ComputeError::InvalidShape("MLA wq_a missing"));
    };
    let Some(wq_b) = tensor_by_role_with_cols(layout, TensorRole::QProj, config.q_lora_rank)?
    else {
        return Err(ComputeError::InvalidShape("MLA wq_b missing"));
    };
    let Some(wkv) = tensor_by_role_with_cols(layout, TensorRole::KvProj, hidden)? else {
        return Err(ComputeError::InvalidShape("MLA wkv missing"));
    };
    let Some(wo_a) = tensor_by_role_with_cols(layout, TensorRole::OProj, group_in)? else {
        return Err(ComputeError::InvalidShape("MLA wo_a missing"));
    };
    let Some(wo_b) = tensor_by_role_with_cols(layout, TensorRole::OProj, o_mid_len)? else {
        return Err(ComputeError::InvalidShape("MLA wo_b missing"));
    };
    let all_fp8 = [&wq_a, &wq_b, &wkv, &wo_a, &wo_b]
        .iter()
        .all(|tensor| tensor.quant_format == QUANT_DEEPSEEK_FP8_E4M3);
    if npos == 1 || !batch_enabled || debug_flow || !all_fp8 {
        let mut values = 0usize;
        for (index, &token_pos) in positions.iter().enumerate() {
            values += compute_deepseek_v4_mla_attention(
                layout,
                layer_index,
                token_pos,
                &mut ys[index * hidden..(index + 1) * hidden],
                kv_cache,
                scratch,
                config,
            )?;
        }
        return Ok(values);
    }
    debug_assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    let wq_a_scale = fp8_scale_for_weight(layout, wq_a)?;
    let wq_b_scale = fp8_scale_for_weight(layout, wq_b)?;
    let wkv_scale = fp8_scale_for_weight(layout, wkv)?;
    let wo_a_scale = fp8_scale_for_weight(layout, wo_a)?;
    let wo_b_scale = fp8_scale_for_weight(layout, wo_b)?;
    let wq_a_shape = tensor_matrix_shape(wq_a)?;
    let wq_b_shape = tensor_matrix_shape(wq_b)?;
    let wkv_shape = tensor_matrix_shape(wkv)?;
    let wo_a_shape = tensor_matrix_shape(wo_a)?;
    if wo_a_shape.rows != o_mid_len {
        return Err(ComputeError::InvalidShape("MLA wo_a rows mismatch"));
    }
    let wo_b_shape = tensor_matrix_shape(wo_b)?;
    if kv_cache.num_kv_heads != 1 || kv_cache.head_dim < head_dim {
        return Err(ComputeError::InvalidShape("MLA KV cache layout mismatch"));
    }

    let mut values = 0usize;
    let mut x_fp8_all = vec![0.0f32; npos * hidden];
    let mut q_lora_all = vec![0.0f32; npos * config.q_lora_rank];
    let mut q_all = vec![0.0f32; npos * q_full_len];
    let mut kv_all = vec![0.0f32; npos * head_dim];
    let mut attn_out_all = vec![0.0f32; npos * q_full_len];
    let mut o_mid_all = vec![0.0f32; npos * o_mid_len];
    let mut norm_tmp = vec![0.0f32; hidden.max(config.q_lora_rank).max(head_dim)];

    let prof_on = phase1_prof::enabled();
    let q_started = Instant::now();
    // q path stage 1: QAT act quant of x (the reference quantizes the
    // activation before every fp8 matvec; the original x stays untouched -
    // the compressor consumes it in bf16).
    for index in 0..npos {
        let x_fp8 = &mut x_fp8_all[index * hidden..(index + 1) * hidden];
        x_fp8.copy_from_slice(&ys[index * hidden..(index + 1) * hidden]);
        fp8_act_quant_dequant_in_place(x_fp8, 64);
    }
    {
        let inputs: Vec<&[f32]> = (0..npos)
            .map(|index| &x_fp8_all[index * hidden..(index + 1) * hidden])
            .collect();
        let mut outputs: Vec<&mut [f32]> = q_lora_all
            .chunks_mut(config.q_lora_rank)
            .collect();
        values +=
            gemv_fp8_e4m3_ue8m0_rows_multi(wq_a, wq_a_scale, 0, wq_a_shape.rows, &inputs, &mut outputs)?;
    }
    let q_norm = norm_tensor_by_marker(layout, "q_a_layernorm")?;
    for index in 0..npos {
        let q_lora = &mut q_lora_all[index * config.q_lora_rank..(index + 1) * config.q_lora_rank];
        if let Some(norm) = q_norm.as_ref() {
            apply_weighted_rms_norm(q_lora, *norm, &mut norm_tmp[..config.q_lora_rank])?;
            q_lora.copy_from_slice(&norm_tmp[..config.q_lora_rank]);
        } else {
            rms_norm_in_place(q_lora);
        }
        fp8_act_quant_dequant_in_place(q_lora, 64);
    }
    {
        let inputs: Vec<&[f32]> = q_lora_all.chunks(config.q_lora_rank).collect();
        let mut outputs: Vec<&mut [f32]> = q_all.chunks_mut(q_full_len).collect();
        values +=
            gemv_fp8_e4m3_ue8m0_rows_multi(wq_b, wq_b_scale, 0, wq_b_shape.rows, &inputs, &mut outputs)?;
    }
    let (layer_theta, yarn) = deepseek_rope_for_layer(layer_index);
    let inv_freqs = rope_inv_freqs(rope_dim, layer_theta, yarn);
    for (index, &token_pos) in positions.iter().enumerate() {
        let q = &mut q_all[index * q_full_len..(index + 1) * q_full_len];
        for head in 0..heads {
            let slice = &mut q[head * head_dim..(head + 1) * head_dim];
            rms_norm_in_place(slice);
        }
        for head in 0..heads {
            let start = head * head_dim + nope_dim;
            apply_rope_interleaved_with_freqs(
                &mut q[start..start + rope_dim],
                token_pos,
                &inv_freqs,
                false,
            )?;
        }
    }

    if prof_on {
        phase1_prof::add(&phase1_prof::Q_NS, q_started);
    }
    let kv_started = Instant::now();
    // kv path: batched wkv, then per-position norm + rope + cache write.
    {
        let inputs: Vec<&[f32]> = (0..npos)
            .map(|index| &x_fp8_all[index * hidden..(index + 1) * hidden])
            .collect();
        let mut outputs: Vec<&mut [f32]> = kv_all.chunks_mut(head_dim).collect();
        values +=
            gemv_fp8_e4m3_ue8m0_rows_multi(wkv, wkv_scale, 0, wkv_shape.rows, &inputs, &mut outputs)?;
    }
    let kv_norm = norm_tensor_by_marker(layout, "kv_a_layernorm")?;
    for (index, &token_pos) in positions.iter().enumerate() {
        let kv = &mut kv_all[index * head_dim..(index + 1) * head_dim];
        if let Some(norm) = kv_norm.as_ref() {
            apply_weighted_rms_norm(kv, *norm, &mut norm_tmp[..head_dim])?;
            kv.copy_from_slice(&norm_tmp[..head_dim]);
        } else {
            rms_norm_in_place(kv);
        }
        apply_rope_interleaved_with_freqs(&mut kv[nope_dim..head_dim], token_pos, &inv_freqs, false)?;
        fp8_act_quant_dequant_in_place(&mut kv[..nope_dim], 64);
        let (key_slot, _value_slot) = kv_cache.key_value_slices_mut(layer_index, token_pos)?;
        key_slot[..head_dim].copy_from_slice(kv);
    }

    if prof_on {
        phase1_prof::add(&phase1_prof::KV_NS, kv_started);
    }
    // Compressor feeds (stateful, ascending order preserved) + window
    // softmax per position. Later positions' KV rows are already written,
    // but the window loop only reads rows <= its own position - the values
    // read are identical to the sequential path.
    let sink = attention_sink_values(layout, heads)?;
    let scale = (head_dim as f32).powf(-0.5);
    for (index, &token_pos) in positions.iter().enumerate() {
        let compressor_started = Instant::now();
        let compressed = deepseek_compressor_feed_and_snapshot(
            layout,
            layer_index,
            token_pos,
            &ys[index * hidden..index * hidden + hidden],
            &inv_freqs,
            head_dim,
            nope_dim,
        )?;
        let compressed_count = compressed.len() / head_dim;
        if prof_on {
            phase1_prof::add(&phase1_prof::COMPRESSOR_NS, compressor_started);
        }
        let window_started = Instant::now();
        let q = &q_all[index * q_full_len..(index + 1) * q_full_len];
        let attn_out = &mut attn_out_all[index * q_full_len..(index + 1) * q_full_len];
        attn_out.fill(0.0);
        let context = token_pos + 1;
        let window_start = context.saturating_sub(DEEPSEEK_V4_SLIDING_WINDOW_SIZE);
        for head in 0..heads {
            let q_head = &q[head * head_dim..(head + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for pos in window_start..context {
                let key = kv_cache.key_slice(layer_index, pos)?;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * key[d];
                }
                score *= scale;
                if score > max_score {
                    max_score = score;
                }
            }
            for block in 0..compressed_count {
                let key = &compressed[block * head_dim..(block + 1) * head_dim];
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * key[d];
                }
                score *= scale;
                if score > max_score {
                    max_score = score;
                }
            }
            let mut denom = 0.0f32;
            if let Some(sink) = sink.as_ref() {
                denom += (sink[head] - max_score).exp();
            }
            let out_head = &mut attn_out[head * head_dim..(head + 1) * head_dim];
            let mut weights_sum = 0.0f32;
            for pos in window_start..context {
                let key = kv_cache.key_slice(layer_index, pos)?;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * key[d];
                }
                let weight = ((score * scale) - max_score).exp();
                weights_sum += weight;
                for d in 0..head_dim {
                    out_head[d] += weight * key[d];
                }
            }
            for block in 0..compressed_count {
                let key = &compressed[block * head_dim..(block + 1) * head_dim];
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * key[d];
                }
                let weight = ((score * scale) - max_score).exp();
                weights_sum += weight;
                for d in 0..head_dim {
                    out_head[d] += weight * key[d];
                }
            }
            denom += weights_sum;
            if denom > 0.0 {
                for value in out_head.iter_mut() {
                    *value /= denom;
                }
            }
            apply_rope_interleaved_with_freqs(
                &mut out_head[nope_dim..head_dim],
                token_pos,
                &inv_freqs,
                true,
            )?;
        }
        if prof_on {
            phase1_prof::add(&phase1_prof::WINDOW_NS, window_started);
        }
    }

    // o path: act quant per position, then batched grouped wo_a and wo_b.
    let o_started = Instant::now();
    for index in 0..npos {
        fp8_act_quant_dequant_in_place(
            &mut attn_out_all[index * q_full_len..(index + 1) * q_full_len],
            64,
        );
    }
    for group in 0..groups {
        let inputs: Vec<&[f32]> = (0..npos)
            .map(|index| {
                &attn_out_all
                    [index * q_full_len + group * group_in..index * q_full_len + (group + 1) * group_in]
            })
            .collect();
        let mut outputs: Vec<&mut [f32]> = Vec::with_capacity(npos);
        let mut rest: &mut [f32] = &mut o_mid_all;
        for _ in 0..npos {
            let (chunk, tail) = rest.split_at_mut(o_mid_len);
            let (_, group_tail) = chunk.split_at_mut(group * o_lora);
            let (group_slice, _) = group_tail.split_at_mut(o_lora);
            outputs.push(group_slice);
            rest = tail;
        }
        values += gemv_fp8_e4m3_ue8m0_rows_multi(
            wo_a,
            wo_a_scale,
            group * o_lora,
            (group + 1) * o_lora,
            &inputs,
            &mut outputs,
        )?;
    }
    for index in 0..npos {
        fp8_act_quant_dequant_in_place(&mut o_mid_all[index * o_mid_len..(index + 1) * o_mid_len], 64);
    }
    {
        let inputs: Vec<&[f32]> = o_mid_all.chunks(o_mid_len).collect();
        let mut outputs: Vec<&mut [f32]> = ys[..npos * hidden].chunks_mut(hidden).collect();
        values +=
            gemv_fp8_e4m3_ue8m0_rows_multi(wo_b, wo_b_scale, 0, wo_b_shape.rows, &inputs, &mut outputs)?;
    }
    if prof_on {
        phase1_prof::add(&phase1_prof::O_NS, o_started);
    }
    crate::vlog!(
        "math_fidelity layer={} component=deepseek_mla_attention_batch present=true applied=true npos={}",
        layer_index,
        npos
    );
    Ok(values)
}

/// Batched phase 1 of the DeepSeek mHC block (T2): per-position hc_pre /
/// norms / hc_post are unchanged; the MLA attention in the middle streams
/// each projection matrix once for ALL positions. Positions must be
/// ascending (compressor feed order). Returns one carry per position,
/// identical to calling `compute_layer_deepseek_mhc_phase1` per position.
pub unsafe fn compute_layer_deepseek_mhc_phase1_batch(
    dense: &[u8],
    layer_index: usize,
    positions: &[usize],
    hiddens: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
) -> Result<Vec<DeepSeekMhcCarry>, ComputeError> {
    let npos = positions.len();
    let hc = config.hc_mult;
    let hidden = config.hidden_size;
    let hc_dim = hc
        .checked_mul(hidden)
        .ok_or(ComputeError::InvalidShape("mHC dim overflow"))?;
    if hiddens.len() < npos * hc_dim {
        return Err(ComputeError::InvalidShape("mHC batch hidden state too small"));
    }
    let layout = parse_quant_block(dense)?;
    let Some(hc_attn) = hc_params_from_layout(&layout, "hc_attn")? else {
        return Err(ComputeError::InvalidShape("mHC hc_attn params missing"));
    };
    let Some(hc_ffn) = hc_params_from_layout(&layout, "hc_ffn")? else {
        return Err(ComputeError::InvalidShape("mHC hc_ffn params missing"));
    };

    let input_norm = norm_tensor_by_marker(&layout, "input_layernorm")?;
    let post_norm = norm_tensor_by_marker(&layout, "post_attention_layernorm")?;
    let mut norm_tmp = vec![0.0f32; hidden];

    // --- attention sub-block, batched ---
    let mut ys = vec![0.0f32; npos * hidden];
    let mut residuals = vec![0.0f32; npos * hc_dim];
    let mut attn_posts: Vec<Vec<f32>> = Vec::with_capacity(npos);
    let mut attn_combs: Vec<Vec<f32>> = Vec::with_capacity(npos);
    for index in 0..npos {
        let hidden_states = &hiddens[index * hc_dim..(index + 1) * hc_dim];
        residuals[index * hc_dim..(index + 1) * hc_dim].copy_from_slice(hidden_states);
        let y = &mut ys[index * hidden..(index + 1) * hidden];
        let (post, comb) = hc_pre_into(
            &hc_attn,
            hidden_states,
            y,
            hc,
            hidden,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )?;
        if let Some(norm) = input_norm.as_ref() {
            apply_weighted_rms_norm(y, *norm, &mut norm_tmp)?;
            y.copy_from_slice(&norm_tmp);
        } else {
            rms_norm_in_place(y);
        }
        attn_posts.push(post);
        attn_combs.push(comb);
    }
    let values = compute_deepseek_v4_mla_attention_batch(
        &layout,
        layer_index,
        positions,
        &mut ys,
        kv_cache,
        scratch,
        config,
    )?;

    // --- hc_post + ffn hc_pre + norm per position ---
    let mut carries = Vec::with_capacity(npos);
    for index in 0..npos {
        let hidden_states = &mut hiddens[index * hc_dim..(index + 1) * hc_dim];
        let y = &mut ys[index * hidden..(index + 1) * hidden];
        hc_post_into(
            y,
            &residuals[index * hc_dim..(index + 1) * hc_dim],
            &attn_posts[index],
            &attn_combs[index],
            hidden_states,
            hc,
            hidden,
        )?;
        let ffn_residual = hidden_states.to_vec();
        let (post, comb) = hc_pre_into(
            &hc_ffn,
            hidden_states,
            y,
            hc,
            hidden,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )?;
        if let Some(norm) = post_norm.as_ref() {
            apply_weighted_rms_norm(y, *norm, &mut norm_tmp)?;
            y.copy_from_slice(&norm_tmp);
        } else {
            rms_norm_in_place(y);
        }
        carries.push(DeepSeekMhcCarry {
            gate_input: y.to_vec(),
            post,
            comb,
            residual: ffn_residual,
            // Stats-only field: split the batch total evenly so the summed
            // dequantized_values matches the sequential path.
            values: values / npos,
        });
    }
    Ok(carries)
}

/// Read the per-head `attn_sink` values (fp32 or bf16 rank-1 aux tensor).
fn attention_sink_values(
    layout: &QuantBlockLayout<'_>,
    heads: usize,
) -> Result<Option<Vec<f32>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.shape.rank() != 1 {
            continue;
        }
        let name = std::str::from_utf8(tensor.name)
            .unwrap_or("")
            .to_ascii_lowercase();
        if !name.contains("attn_sink") {
            continue;
        }
        let count = tensor.shape.dim(0)? as usize;
        if count < heads {
            continue;
        }
        let mut sink = Vec::with_capacity(heads);
        match tensor.dtype_original {
            12 => {
                for i in 0..heads {
                    let offset = i * 4;
                    let bytes = tensor
                        .data
                        .get(offset..offset + 4)
                        .ok_or(ComputeError::InvalidShape("attn_sink out of range"))?;
                    sink.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                }
            }
            11 => {
                for i in 0..heads {
                    sink.push(bf16_tensor_value(tensor, i)?);
                }
            }
            _ => continue,
        }
        return Ok(Some(sink));
    }
    Ok(None)
}

/// Inverse interleaved RoPE (rotation by -position); matches the reference
/// `apply_rotary_emb(..., inverse=True)` applied to attention outputs.
pub fn apply_rope_interleaved_inverse_in_place(
    values: &mut [f32],
    position: usize,
    rope_dim: usize,
    theta: f32,
) -> Result<(), ComputeError> {
    if rope_dim == 0 {
        return Ok(());
    }
    if rope_dim % 2 != 0 {
        return Err(ComputeError::InvalidShape("RoPE dim must be even"));
    }
    if values.len() < rope_dim {
        return Err(ComputeError::InvalidShape("RoPE input shorter than dim"));
    }
    if theta <= 0.0 {
        return Err(ComputeError::InvalidShape("RoPE theta must be positive"));
    }
    let position = position as f32;
    for pair in 0..(rope_dim / 2) {
        let even_index = pair * 2;
        let odd_index = even_index + 1;
        let inv_freq = theta.powf(-(even_index as f32) / rope_dim as f32);
        let angle = position * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let even = values[even_index];
        let odd = values[odd_index];
        // rotation by -angle
        values[even_index] = even * cos + odd * sin;
        values[odd_index] = -even * sin + odd * cos;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DeepSeek-V4 Hyper-Connections (mHC) residual path.
// Reference: inference/model.py `Block.hc_pre` / `Block.hc_post` and
// kernel.py `hc_split_sinkhorn_kernel`. The residual stream holds `hc_mult`
// copies of the hidden state; each sub-block (attn, ffn) reduces the copies
// to one working stream via learned pre-weights, then redistributes its
// output across the copies via post-weights and a doubly-normalized
// (Sinkhorn) combination matrix.
// ---------------------------------------------------------------------------

/// Borrowed hc parameter tensors from a dense block layout.
pub struct HcParams<'a> {
    /// f32le rows: [mix_hc, hc_mult * hidden]
    pub fn_data: &'a [u8],
    pub scale: Vec<f32>,
    pub base: Vec<f32>,
    pub mix_hc: usize,
    pub hc_dim: usize,
}

fn f32_tensor_values(tensor: QuantTensorLayout<'_>, count: usize) -> Result<Vec<f32>, ComputeError> {
    if tensor.dtype_original != 12 {
        return Err(ComputeError::InvalidShape("expected fp32 tensor"));
    }
    let bytes = count
        .checked_mul(4)
        .ok_or(ComputeError::InvalidShape("fp32 tensor size overflow"))?;
    let data = tensor
        .data
        .get(..bytes)
        .ok_or(ComputeError::InvalidShape("fp32 tensor data too small"))?;
    Ok(data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn dot_f32_row(row: &[u8], input: &[f32]) -> Result<f32, ComputeError> {
    if row.len() < input.len() * 4 {
        return Err(ComputeError::InvalidShape("f32 row shorter than input"));
    }
    let mut acc = 0.0f32;
    for (chunk, value) in row.chunks_exact(4).zip(input.iter()) {
        acc += f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) * value;
    }
    Ok(acc)
}

/// Locate `<prefix>_fn` / `<prefix>_scale` / `<prefix>_base` in a dense
/// block (e.g. prefix "hc_attn" matches `layers.3.hc_attn_fn`).
pub fn hc_params_from_layout<'a>(
    layout: &'a QuantBlockLayout<'a>,
    prefix: &str,
) -> Result<Option<HcParams<'a>>, ComputeError> {
    let mut fn_tensor = None;
    let mut scale_tensor = None;
    let mut base_tensor = None;
    let fn_suffix = format!("{prefix}_fn");
    let scale_suffix = format!("{prefix}_scale");
    let base_suffix = format!("{prefix}_base");
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        let name = std::str::from_utf8(tensor.name).unwrap_or("");
        if name.ends_with(&fn_suffix) {
            fn_tensor = Some(tensor);
        } else if name.ends_with(&scale_suffix) {
            scale_tensor = Some(tensor);
        } else if name.ends_with(&base_suffix) {
            base_tensor = Some(tensor);
        }
    }
    let (Some(fn_tensor), Some(scale_tensor), Some(base_tensor)) =
        (fn_tensor, scale_tensor, base_tensor)
    else {
        return Ok(None);
    };
    if fn_tensor.shape.rank() != 2 {
        return Err(ComputeError::InvalidShape("hc fn tensor rank"));
    }
    let mix_hc = fn_tensor.shape.dim(0)? as usize;
    let hc_dim = fn_tensor.shape.dim(1)? as usize;
    let scale_len = scale_tensor.shape.dim(0)? as usize;
    let base = f32_tensor_values(base_tensor, mix_hc)?;
    let scale = f32_tensor_values(scale_tensor, scale_len)?;
    if fn_tensor.dtype_original != 12 {
        return Err(ComputeError::InvalidShape("hc fn tensor is not fp32"));
    }
    let fn_bytes = mix_hc
        .checked_mul(hc_dim)
        .and_then(|value| value.checked_mul(4))
        .ok_or(ComputeError::InvalidShape("hc fn size overflow"))?;
    let fn_data = fn_tensor
        .data
        .get(..fn_bytes)
        .ok_or(ComputeError::InvalidShape("hc fn data too small"))?;
    Ok(Some(HcParams {
        fn_data,
        scale,
        base,
        mix_hc,
        hc_dim,
    }))
}

/// Scalar port of `hc_split_sinkhorn_kernel`: split the mixed projections
/// into pre[hc], post[hc] and a doubly-normalized comb[hc][hc].
pub fn hc_split_sinkhorn_scalar(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    hc: usize,
    sinkhorn_iters: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), ComputeError> {
    let mix_hc = (2 + hc) * hc;
    if mixes.len() < mix_hc || base.len() < mix_hc || scale.len() < 3 {
        return Err(ComputeError::InvalidShape("hc sinkhorn input sizes"));
    }
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    let mut comb = vec![0.0f32; hc * hc];
    for j in 0..hc {
        pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
        post[j] = 2.0 * sigmoid(mixes[j + hc] * scale[1] + base[j + hc]);
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] = mixes[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
        }
    }
    // comb = softmax(comb, dim=-1) + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in row.iter_mut() {
            *value = *value / sum + eps;
        }
    }
    // first column normalization, then (iters - 1) row/col rounds
    normalize_comb_cols(&mut comb, hc, eps);
    for _ in 1..sinkhorn_iters {
        normalize_comb_rows(&mut comb, hc, eps);
        normalize_comb_cols(&mut comb, hc, eps);
    }
    Ok((pre, post, comb))
}

fn normalize_comb_rows(comb: &mut [f32], hc: usize, eps: f32) {
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let sum: f32 = row.iter().sum();
        for value in row.iter_mut() {
            *value /= sum + eps;
        }
    }
}

fn normalize_comb_cols(comb: &mut [f32], hc: usize, eps: f32) {
    for k in 0..hc {
        let mut sum = 0.0f32;
        for j in 0..hc {
            sum += comb[j * hc + k];
        }
        for j in 0..hc {
            comb[j * hc + k] /= sum + eps;
        }
    }
}

/// hc_pre: reduce hc copies to a single working stream.
/// Returns (post, comb) for the matching hc_post call.
pub fn hc_pre_into(
    params: &HcParams<'_>,
    x_flat: &[f32],
    y: &mut [f32],
    hc: usize,
    hidden: usize,
    sinkhorn_iters: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    let hc_dim = hc * hidden;
    if x_flat.len() < hc_dim || y.len() < hidden || params.hc_dim != hc_dim {
        return Err(ComputeError::InvalidShape("hc_pre sizes"));
    }
    let mean_square = x_flat[..hc_dim]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / hc_dim as f32;
    let inv_rms = 1.0 / (mean_square + eps).sqrt();
    let mut mixes = vec![0.0f32; params.mix_hc];
    let row_bytes = hc_dim * 4;
    for (j, mix) in mixes.iter_mut().enumerate() {
        let row = &params.fn_data[j * row_bytes..(j + 1) * row_bytes];
        *mix = dot_f32_row(row, &x_flat[..hc_dim])? * inv_rms;
    }
    let (pre, post, comb) =
        hc_split_sinkhorn_scalar(&mixes, &params.scale, &params.base, hc, sinkhorn_iters, eps)?;
    for d in 0..hidden {
        let mut acc = 0.0f32;
        for i in 0..hc {
            acc += pre[i] * x_flat[i * hidden + d];
        }
        y[d] = acc;
    }
    Ok((post, comb))
}

/// hc_post: redistribute the sub-block output across the hc copies.
/// out[k] = post[k] * y + sum_i comb[i][k] * residual[i]
pub fn hc_post_into(
    y: &[f32],
    residual_flat: &[f32],
    post: &[f32],
    comb: &[f32],
    out_flat: &mut [f32],
    hc: usize,
    hidden: usize,
) -> Result<(), ComputeError> {
    let hc_dim = hc * hidden;
    if y.len() < hidden
        || residual_flat.len() < hc_dim
        || out_flat.len() < hc_dim
        || post.len() < hc
        || comb.len() < hc * hc
    {
        return Err(ComputeError::InvalidShape("hc_post sizes"));
    }
    for k in 0..hc {
        for d in 0..hidden {
            let mut acc = post[k] * y[d];
            for i in 0..hc {
                // Reference: einsum("jk,kd->jd", comb, residual) -
                // out[k] = sum_i comb[k][i] * residual[i]. Using the
                // transposed comb here is nearly invisible at short context
                // (the hc residual copies start identical) but scrambles the
                // copies as they diverge.
                acc += comb[k * hc + i] * residual_flat[i * hidden + d];
            }
            out_flat[k * hidden + d] = acc;
        }
    }
    Ok(())
}

/// hc_head: pool the hc copies into one stream at the LM boundary
/// (sigmoid gating, no sinkhorn). fn rows: [hc, hc_dim] f32le.
pub fn hc_head_pool(
    fn_rows: &[u8],
    scale: f32,
    base: &[f32],
    x_flat: &[f32],
    pooled: &mut [f32],
    hc: usize,
    hidden: usize,
    eps: f32,
) -> Result<(), ComputeError> {
    let hc_dim = hc * hidden;
    if x_flat.len() < hc_dim || pooled.len() < hidden || base.len() < hc {
        return Err(ComputeError::InvalidShape("hc_head sizes"));
    }
    if fn_rows.len() < hc * hc_dim * 4 {
        return Err(ComputeError::InvalidShape("hc_head fn data too small"));
    }
    let mean_square = x_flat[..hc_dim]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / hc_dim as f32;
    let inv_rms = 1.0 / (mean_square + eps).sqrt();
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let row_bytes = hc_dim * 4;
    let mut pre = vec![0.0f32; hc];
    for (j, weight) in pre.iter_mut().enumerate() {
        let row = &fn_rows[j * row_bytes..(j + 1) * row_bytes];
        let mix = dot_f32_row(row, &x_flat[..hc_dim])? * inv_rms;
        *weight = sigmoid(mix * scale + base[j]) + eps;
    }
    for d in 0..hidden {
        let mut acc = 0.0f32;
        for i in 0..hc {
            acc += pre[i] * x_flat[i * hidden + d];
        }
        pooled[d] = acc;
    }
    Ok(())
}

fn compute_glm_dsa_attention_prefill_probe(
    layout: &QuantBlockLayout<'_>,
    layer_index: usize,
    token_pos: usize,
    hidden_states: &mut [f32],
    kv_cache: &mut KVCache<'_>,
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
) -> Result<usize, ComputeError> {
    let qk_head_dim = config
        .qk_nope_head_dim
        .checked_add(config.qk_rope_head_dim)
        .ok_or(ComputeError::InvalidShape("qk head dim overflow"))?;
    let q_full_len = config
        .num_attention_heads
        .checked_mul(qk_head_dim)
        .ok_or(ComputeError::InvalidShape("q full size overflow"))?;
    let value_concat_len = config
        .num_kv_heads
        .checked_mul(config.v_head_dim)
        .ok_or(ComputeError::InvalidShape("value concat size overflow"))?;
    let q_or_value_len = q_full_len.max(value_concat_len);
    let kv_a_len = config
        .kv_lora_rank
        .checked_add(config.qk_rope_head_dim)
        .ok_or(ComputeError::InvalidShape("kv_a size overflow"))?;
    let kv_full_len = config
        .num_kv_heads
        .checked_mul(config.qk_nope_head_dim.saturating_add(config.v_head_dim))
        .ok_or(ComputeError::InvalidShape("kv full size overflow"))?;
    let required = config
        .q_lora_rank
        .checked_add(q_or_value_len)
        .and_then(|value| value.checked_add(kv_a_len))
        .and_then(|value| value.checked_add(kv_full_len))
        .ok_or(ComputeError::InvalidShape("attention scratch size overflow"))?;
    if scratch.dequant_tile_f32.len() < required {
        return Err(ComputeError::ScratchTooSmall {
            required,
            actual: scratch.dequant_tile_f32.len(),
        });
    }

    let (q_lora, rest) = scratch.dequant_tile_f32.split_at_mut(config.q_lora_rank);
    let (q_or_value, rest) = rest.split_at_mut(q_or_value_len);
    let (kv_a, kv_full) = rest.split_at_mut(kv_a_len);

    let mut values = 0usize;
    if let Some(q_a) = tensor_by_role_with_cols(layout, TensorRole::QProj, config.hidden_size)? {
        values += gemv_tensorwise_from_layout(layout, q_a, hidden_states, q_lora, gemm)?;
        if let Some(q_norm) = norm_tensor_by_marker(layout, "q_a_layernorm")? {
            apply_weighted_rms_norm(q_lora, q_norm, &mut q_or_value[..config.q_lora_rank])?;
            q_lora.copy_from_slice(&q_or_value[..config.q_lora_rank]);
            crate::vlog!(
                "math_fidelity layer={} component=q_a_layernorm present=true applied=true",
                layer_index
            );
        } else {
            rms_norm_in_place(q_lora);
        }
        if let Some(q_b) = tensor_by_role_with_cols(layout, TensorRole::QProj, config.q_lora_rank)? {
            values += gemv_tensorwise_from_layout(
                layout,
                q_b,
                q_lora,
                &mut q_or_value[..q_full_len],
                gemm,
            )?;
            apply_rope_to_heads_interleaved_at(
                &mut q_or_value[..q_full_len],
                config.num_attention_heads,
                qk_head_dim,
                config.qk_nope_head_dim,
                config.qk_rope_head_dim,
                token_pos,
                config.rope_theta,
            )?;
        }
    }

    if let Some(kv_a_proj) = tensor_by_role_with_cols(layout, TensorRole::KvProj, config.hidden_size)? {
        values += gemv_tensorwise_from_layout(layout, kv_a_proj, hidden_states, kv_a, gemm)?;
        if let Some(kv_norm) = norm_tensor_by_marker(layout, "kv_a_layernorm")? {
            apply_weighted_rms_norm(
                &kv_a[..config.kv_lora_rank],
                kv_norm,
                &mut kv_full[..config.kv_lora_rank],
            )?;
            kv_a[..config.kv_lora_rank].copy_from_slice(&kv_full[..config.kv_lora_rank]);
            crate::vlog!(
                "math_fidelity layer={} component=kv_a_layernorm present=true applied=true",
                layer_index
            );
        } else {
            rms_norm_in_place(&mut kv_a[..config.kv_lora_rank]);
        }
        let rope_start = config.kv_lora_rank;
        let rope_end = rope_start + config.qk_rope_head_dim;
        if rope_end <= kv_a.len() {
            apply_rope_interleaved_in_place(
                &mut kv_a[rope_start..rope_end],
                token_pos,
                config.qk_rope_head_dim,
                config.rope_theta,
            )?;
        }
        if let Some(kv_b) = tensor_by_role_with_cols(layout, TensorRole::KvProj, config.kv_lora_rank)? {
            values += gemv_tensorwise_from_layout(
                layout,
                kv_b,
                &kv_a[..config.kv_lora_rank],
                kv_full,
                gemm,
            )?;
        }
    }

    let kv_head_stride = config.qk_nope_head_dim.saturating_add(config.v_head_dim);
    if kv_head_stride > 0 && value_concat_len <= q_or_value.len() {
        append_glm_dsa_kv_cache(
            layer_index,
            token_pos,
            kv_a,
            kv_full,
            kv_cache,
            config,
        )?;
        causal_attention_from_kv_cache(
            layer_index,
            token_pos,
            &q_or_value[..q_full_len],
            kv_cache,
            &mut kv_full[..value_concat_len],
            config,
        )?;

        if let Some(o_proj) = tensor_by_role_with_cols(layout, TensorRole::OProj, value_concat_len)? {
            values += gemv_tensorwise_from_layout(
                layout,
                o_proj,
                &kv_full[..value_concat_len],
                hidden_states,
                gemm,
            )?;
        }
    }

    Ok(values)
}

fn append_glm_dsa_kv_cache(
    layer_index: usize,
    token_pos: usize,
    kv_a: &[f32],
    kv_full: &[f32],
    kv_cache: &mut KVCache<'_>,
    config: &ComputeConfig,
) -> Result<(), ComputeError> {
    let qk_head_dim = config
        .qk_nope_head_dim
        .checked_add(config.qk_rope_head_dim)
        .ok_or(ComputeError::InvalidShape("qk head dim overflow"))?;
    if kv_cache.num_kv_heads != config.num_kv_heads || kv_cache.head_dim < qk_head_dim {
        return Err(ComputeError::InvalidShape("KV cache layout does not match attention config"));
    }
    if kv_cache.head_dim < config.v_head_dim {
        return Err(ComputeError::InvalidShape("KV cache value head dim too small"));
    }
    let cache_head_dim = kv_cache.head_dim;

    let kv_head_stride = config
        .qk_nope_head_dim
        .checked_add(config.v_head_dim)
        .ok_or(ComputeError::InvalidShape("kv head stride overflow"))?;
    let rope_start = config.kv_lora_rank;
    let rope_end = rope_start
        .checked_add(config.qk_rope_head_dim)
        .ok_or(ComputeError::InvalidShape("kv rope span overflow"))?;
    let shared_rope = kv_a
        .get(rope_start..rope_end)
        .ok_or(ComputeError::InvalidShape("kv_a rope span out of range"))?;

    let (key_slot, value_slot) = kv_cache.key_value_slices_mut(layer_index, token_pos)?;
    key_slot.fill(0.0);
    value_slot.fill(0.0);

    for head in 0..config.num_kv_heads {
        let kv_start = head
            .checked_mul(kv_head_stride)
            .ok_or(ComputeError::InvalidShape("kv_full head offset overflow"))?;
        let key_nope_end = kv_start
            .checked_add(config.qk_nope_head_dim)
            .ok_or(ComputeError::InvalidShape("kv_full key end overflow"))?;
        let value_start = key_nope_end;
        let value_end = value_start
            .checked_add(config.v_head_dim)
            .ok_or(ComputeError::InvalidShape("kv_full value end overflow"))?;
        let key_out = head
            .checked_mul(cache_head_dim)
            .ok_or(ComputeError::InvalidShape("KV key slot offset overflow"))?;
        let rope_out = key_out
            .checked_add(config.qk_nope_head_dim)
            .ok_or(ComputeError::InvalidShape("KV rope slot offset overflow"))?;
        let value_out = head
            .checked_mul(cache_head_dim)
            .ok_or(ComputeError::InvalidShape("KV value slot offset overflow"))?;

        key_slot[key_out..key_out + config.qk_nope_head_dim]
            .copy_from_slice(kv_full.get(kv_start..key_nope_end).ok_or(
                ComputeError::InvalidShape("kv_full key span out of range"),
            )?);
        key_slot[rope_out..rope_out + config.qk_rope_head_dim].copy_from_slice(shared_rope);
        value_slot[value_out..value_out + config.v_head_dim]
            .copy_from_slice(kv_full.get(value_start..value_end).ok_or(
                ComputeError::InvalidShape("kv_full value span out of range"),
            )?);
    }
    Ok(())
}

fn causal_attention_from_kv_cache(
    layer_index: usize,
    token_pos: usize,
    query_heads: &[f32],
    kv_cache: &KVCache<'_>,
    value_concat_out: &mut [f32],
    config: &ComputeConfig,
) -> Result<(), ComputeError> {
    let qk_head_dim = config
        .qk_nope_head_dim
        .checked_add(config.qk_rope_head_dim)
        .ok_or(ComputeError::InvalidShape("qk head dim overflow"))?;
    let query_required = config
        .num_attention_heads
        .checked_mul(qk_head_dim)
        .ok_or(ComputeError::InvalidShape("query heads size overflow"))?;
    let value_required = config
        .num_kv_heads
        .checked_mul(config.v_head_dim)
        .ok_or(ComputeError::InvalidShape("value concat size overflow"))?;
    if query_heads.len() < query_required || value_concat_out.len() < value_required {
        return Err(ComputeError::InvalidShape("attention input/output too small"));
    }
    if kv_cache.num_kv_heads != config.num_kv_heads || kv_cache.head_dim < qk_head_dim {
        return Err(ComputeError::InvalidShape("KV cache layout does not match attention config"));
    }
    if token_pos >= kv_cache.max_tokens {
        return Err(ComputeError::InvalidShape("attention token_pos outside KV cache"));
    }

    let scale = 1.0f32 / (qk_head_dim as f32).sqrt();
    let history_len = token_pos + 1;
    value_concat_out[..value_required].fill(0.0);

    let output_heads = value_required / config.v_head_dim.max(1);
    let heads_to_compute = config.num_attention_heads.min(output_heads);
    for head in 0..heads_to_compute {
        let q_start = head
            .checked_mul(qk_head_dim)
            .ok_or(ComputeError::InvalidShape("query head offset overflow"))?;
        let q = &query_heads[q_start..q_start + qk_head_dim];
        let kv_head = if config.num_kv_heads == 0 {
            return Err(ComputeError::InvalidShape("num_kv_heads is zero"));
        } else if config.num_kv_heads == config.num_attention_heads {
            head
        } else {
            head % config.num_kv_heads
        };

        let mut max_score = f32::NEG_INFINITY;
        for pos in 0..history_len {
            let key = kv_cache.key_slice(layer_index, pos)?;
            let head_key_start = kv_head
                .checked_mul(kv_cache.head_dim)
                .ok_or(ComputeError::InvalidShape("key head offset overflow"))?;
            let score = dot(&key[head_key_start..head_key_start + qk_head_dim], q) * scale;
            max_score = max_score.max(score);
        }

        let mut exp_sum = 0.0f32;
        let out_start = head
            .checked_mul(config.v_head_dim)
            .ok_or(ComputeError::InvalidShape("attention value offset overflow"))?;
        let out = &mut value_concat_out[out_start..out_start + config.v_head_dim];
        for pos in 0..history_len {
            let key = kv_cache.key_slice(layer_index, pos)?;
            let value = kv_cache.value_slice(layer_index, pos)?;
            let head_key_start = kv_head
                .checked_mul(kv_cache.head_dim)
                .ok_or(ComputeError::InvalidShape("key head offset overflow"))?;
            let score = dot(&key[head_key_start..head_key_start + qk_head_dim], q) * scale;
            let weight = (score - max_score).exp();
            exp_sum += weight;

            let value_start = kv_head
                .checked_mul(kv_cache.head_dim)
                .ok_or(ComputeError::InvalidShape("value head offset overflow"))?;
            let value_head = &value[value_start..value_start + config.v_head_dim];
            for (acc, value) in out.iter_mut().zip(value_head.iter()) {
                *acc += weight * *value;
            }
        }

        if exp_sum > 0.0 && exp_sum.is_finite() {
            for value in out.iter_mut() {
                *value /= exp_sum;
            }
        }
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn tensor_by_role_with_cols<'a>(
    layout: &'a QuantBlockLayout<'a>,
    role: TensorRole,
    cols: usize,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() == role && tensor.shape.rank() == 2 && tensor.shape.dim(1)? == cols {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

fn compute_expert_block(
    expert: &[u8],
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<(), ComputeError> {
    let layout = parse_quant_block(expert)?;
    if quant_block_contains_deepseek_fp4_expert(&layout)? {
        let values = compute_deepseek_fp4_expert_into_hidden(&layout, hidden_states, scratch, config)?;
        stats.dequantized_values += values;
        return Ok(());
    }
    if let (Some(gate), Some(up), Some(down)) = (
        first_rank2_tensor_by_role(&layout, TensorRole::GateProj)?,
        first_rank2_tensor_by_role(&layout, TensorRole::UpProj)?,
        first_rank2_tensor_by_role(&layout, TensorRole::DownProj)?,
    ) {
        let values =
            compute_glm_moe_expert_into_hidden(gate, up, down, hidden_states, scratch, config, gemm)?;
        stats.dequantized_values += values;
        return Ok(());
    }

    silu_gate_in_place(hidden_states);
    let values = compute_quant_block_layout_into_hidden(&layout, hidden_states, scratch, config, gemm)?;
    stats.dequantized_values += values;
    Ok(())
}

fn compute_shared_expert_from_dense(
    dense: &[u8],
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
    stats: &mut LayerComputeStats,
) -> Result<bool, ComputeError> {
    let layout = parse_quant_block(dense)?;
    if dense_block_contains_deepseek_fp8_shared_expert(&layout)? {
        let values =
            compute_deepseek_fp8_shared_expert_into_hidden(&layout, hidden_states, scratch, config, gemm)?;
        stats.dequantized_values += values;
        return Ok(true);
    }
    let gate = shared_expert_tensor_by_marker(&layout, "gate_proj")?;
    let up = shared_expert_tensor_by_marker(&layout, "up_proj")?;
    let down = shared_expert_tensor_by_marker(&layout, "down_proj")?;
    let (Some(gate), Some(up), Some(down)) = (gate, up, down) else {
        return Ok(false);
    };

    let values =
        compute_glm_moe_expert_into_hidden(gate, up, down, hidden_states, scratch, config, gemm)?;
    stats.dequantized_values += values;
    Ok(true)
}

fn dense_block_contains_deepseek_fp8_shared_expert(
    layout: &QuantBlockLayout<'_>,
) -> Result<bool, ComputeError> {
    let mut has_weight = false;
    let mut has_scale = false;
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() != TensorRole::SharedExpert {
            continue;
        }
        if tensor.quant_format == QUANT_DEEPSEEK_FP8_E4M3 {
            has_weight = true;
        } else if tensor.quant_format == QUANT_DEEPSEEK_UE8M0_SCALE {
            has_scale = true;
        }
    }
    Ok(has_weight && has_scale)
}

fn compute_deepseek_fp8_shared_expert_into_hidden(
    layout: &QuantBlockLayout<'_>,
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
) -> Result<usize, ComputeError> {
    let w1 = find_tensor_by_suffix(layout, b".w1.weight")?;
    let w3 = find_tensor_by_suffix(layout, b".w3.weight")?;
    let w2 = find_tensor_by_suffix(layout, b".w2.weight")?;
    let w1_shape = tensor_matrix_shape(w1)?;
    let w3_shape = tensor_matrix_shape(w3)?;
    let w2_shape = tensor_matrix_shape(w2)?;
    let hidden = hidden_states.len().min(config.hidden_size);
    if w1_shape.cols != hidden || w3_shape.cols != hidden {
        return Err(ComputeError::InvalidShape("DeepSeek shared expert input dim"));
    }
    if w1_shape.rows != w3_shape.rows || w2_shape.cols != w1_shape.rows {
        return Err(ComputeError::InvalidShape("DeepSeek shared expert projection shape"));
    }
    if w2_shape.rows > hidden_states.len() {
        return Err(ComputeError::OutputTooSmall {
            required: w2_shape.rows,
            actual: hidden_states.len(),
        });
    }

    let intermediate = w1_shape.rows;
    let required_scratch = hidden
        .checked_add(intermediate.checked_mul(2).ok_or(ComputeError::InvalidShape(
            "DeepSeek shared expert scratch overflow",
        ))?)
        .ok_or(ComputeError::InvalidShape("DeepSeek shared expert scratch overflow"))?;
    if scratch.dequant_tile_f32.len() < required_scratch {
        return Err(ComputeError::ScratchTooSmall {
            required: required_scratch,
            actual: scratch.dequant_tile_f32.len(),
        });
    }

    let (input_scratch, rest) = scratch.dequant_tile_f32.split_at_mut(hidden);
    let (gate, up) = rest.split_at_mut(intermediate);
    input_scratch.copy_from_slice(&hidden_states[..hidden]);
    // QAT act quant: the reference fp8_matvec quant-dequants the input
    // activation before every fp8 matvec (w1/w3 on x, w2 on the swiglu
    // hidden).
    fp8_act_quant_dequant_in_place(input_scratch, 64);
    let mut values = 0usize;
    values += gemv_tensorwise_from_layout(layout, w1, input_scratch, gate, gemm)?;
    values += gemv_tensorwise_from_layout(layout, w3, input_scratch, up, gemm)?;
    for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
        // DeepSeek-V4 swiglu_limit clamp (10.0): gate clamped from above,
        // up clamped on both sides, before silu/product.
        let gated = gate_value.min(crate::deepseek_v4::DEEPSEEK_V4_SWIGLU_LIMIT);
        let upped = up_value.clamp(
            -crate::deepseek_v4::DEEPSEEK_V4_SWIGLU_LIMIT,
            crate::deepseek_v4::DEEPSEEK_V4_SWIGLU_LIMIT,
        );
        *gate_value = silu(gated) * upped;
    }
    fp8_act_quant_dequant_in_place(gate, 64);
    values += gemv_tensorwise_from_layout(
        layout,
        w2,
        gate,
        &mut hidden_states[..w2_shape.rows],
        gemm,
    )?;
    Ok(values)
}

fn quant_block_contains_deepseek_fp4_expert(
    layout: &QuantBlockLayout<'_>,
) -> Result<bool, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.quant_format == QUANT_DEEPSEEK_FP4_E2M1_PACKED
            || tensor.quant_format == QUANT_DEEPSEEK_UE8M0_SCALE
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn compute_deepseek_fp4_expert_into_hidden(
    layout: &QuantBlockLayout<'_>,
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
) -> Result<usize, ComputeError> {
    let expert = deepseek_v4_fp4_expert_from_quant_block(layout)?;
    let hidden = hidden_states.len().min(config.hidden_size);
    if hidden < expert.w1.cols {
        return Err(ComputeError::InvalidShape("DeepSeek FP4 expert input dim"));
    }
    if expert.w2.rows > hidden_states.len() {
        return Err(ComputeError::OutputTooSmall {
            required: expert.w2.rows,
            actual: hidden_states.len(),
        });
    }
    let forward_scratch = expert
        .w1
        .rows
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP4 expert scratch overflow"))?;
    let required_scratch = expert
        .w1
        .cols
        .checked_add(forward_scratch)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP4 expert scratch overflow"))?;
    if scratch.dequant_tile_f32.len() < required_scratch {
        return Err(ComputeError::ScratchTooSmall {
            required: required_scratch,
            actual: scratch.dequant_tile_f32.len(),
        });
    }

    let (input_scratch, forward_scratch_buf) =
        scratch.dequant_tile_f32.split_at_mut(expert.w1.cols);
    input_scratch.copy_from_slice(&hidden_states[..expert.w1.cols]);
    deepseek_v4_fp4_expert_forward_scalar(
        expert,
        input_scratch,
        &mut hidden_states[..expert.w2.rows],
        forward_scratch_buf,
    )
    .map_err(|_| ComputeError::InvalidShape("DeepSeek FP4 expert forward"))?;

    fp4_expert_logical_values(expert)
}

fn fp4_expert_logical_values(expert: DeepSeekV4Fp4Expert<'_>) -> Result<usize, ComputeError> {
    let w1 = fp4_matvec_logical_values(expert.w1)?;
    let w3 = fp4_matvec_logical_values(expert.w3)?;
    let w2 = fp4_matvec_logical_values(expert.w2)?;
    w1.checked_add(w3)
        .and_then(|value| value.checked_add(w2))
        .ok_or(ComputeError::InvalidShape("DeepSeek FP4 expert value count overflow"))
}

fn fp4_matvec_logical_values(spec: Fp4Matvec<'_>) -> Result<usize, ComputeError> {
    spec.rows
        .checked_mul(spec.cols)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP4 matvec value count overflow"))
}

fn shared_expert_tensor_by_marker<'a>(
    layout: &'a QuantBlockLayout<'a>,
    marker: &str,
) -> Result<Option<QuantTensorLayout<'a>>, ComputeError> {
    for index in 0..layout.tensor_count() {
        let tensor = layout.tensor(index)?;
        if tensor.role() != TensorRole::SharedExpert || tensor.shape.rank() != 2 {
            continue;
        }
        let name = std::str::from_utf8(tensor.name).unwrap_or("");
        if name.to_ascii_lowercase().contains(marker) {
            return Ok(Some(tensor));
        }
    }
    Ok(None)
}

fn compute_quant_block_layout_into_hidden(
    layout: &QuantBlockLayout<'_>,
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
) -> Result<usize, ComputeError> {
    let tensor = layout.first_tensor_by_roles(&[
        TensorRole::QkvProj,
        TensorRole::QProj,
        TensorRole::KvProj,
        TensorRole::KProj,
        TensorRole::VProj,
        TensorRole::OProj,
        TensorRole::GateProj,
        TensorRole::UpProj,
        TensorRole::DownProj,
        TensorRole::LmHead,
        TensorRole::Unknown,
    ])?;
    let k = hidden_states.len().min(config.hidden_size);
    if k == 0 {
        return Err(ComputeError::InvalidShape("hidden_states is empty"));
    }
    let elements = tensor.shape.element_count()?;
    if elements < k {
        return Err(ComputeError::InvalidShape(
            "quant tensor has fewer values than hidden size",
        ));
    }
    let n = (elements / k).min(hidden_states.len()).min(scratch.dequant_tile_f32.len());
    if n == 0 {
        return Err(ComputeError::InvalidShape("inferred output rows is zero"));
    }

    if tensor.quant_format == QUANT_DEEPSEEK_FP8_E4M3
        || (tensor.quant_format == QUANT_DEEPSEEK_BF16_AUX && tensor.dtype_original == 11)
    {
        gemv_tensorwise_from_layout(
            layout,
            tensor,
            &hidden_states[..k],
            &mut scratch.dequant_tile_f32[..n],
            gemm,
        )?;
    } else if tensor.quant_format as u32 == 4 {
        unsafe {
        gemm.gemv_i4_affine_tensorwise(
            n,
            k,
            hidden_states.as_ptr(),
            tensor.data.as_ptr(),
            tensor.scale,
            tensor.zero_point,
            scratch.dequant_tile_f32.as_mut_ptr(),
        )?;
        }
    } else {
        return Err(ComputeError::UnsupportedQuantFormat(DiskQuantFormat::Int8Symmetric));
    }

    hidden_states[..n].copy_from_slice(&scratch.dequant_tile_f32[..n]);
    Ok(elements)
}

fn compute_glm_moe_expert_into_hidden(
    gate: QuantTensorLayout<'_>,
    up: QuantTensorLayout<'_>,
    down: QuantTensorLayout<'_>,
    hidden_states: &mut [f32],
    scratch: &mut ComputeScratch<'_>,
    config: &ComputeConfig,
    gemm: &dyn GemmKernel,
) -> Result<usize, ComputeError> {
    let hidden = hidden_states.len().min(config.hidden_size);
    if hidden == 0 {
        return Err(ComputeError::InvalidShape("hidden_states is empty"));
    }

    let gate_shape = tensor_matrix_shape(gate)?;
    let up_shape = tensor_matrix_shape(up)?;
    let down_shape = tensor_matrix_shape(down)?;
    if gate_shape.cols != hidden || up_shape.cols != hidden {
        return Err(ComputeError::InvalidShape("expert gate/up input dim mismatch"));
    }
    if gate_shape.rows != up_shape.rows {
        return Err(ComputeError::InvalidShape("expert gate/up row mismatch"));
    }
    if down_shape.cols != gate_shape.rows {
        return Err(ComputeError::InvalidShape("expert down input dim mismatch"));
    }
    if down_shape.rows > hidden_states.len() {
        return Err(ComputeError::OutputTooSmall {
            required: down_shape.rows,
            actual: hidden_states.len(),
        });
    }

    let intermediate = gate_shape.rows;
    let required_scratch = intermediate
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("expert scratch overflow"))?;
    if scratch.dequant_tile_f32.len() < required_scratch {
        return Err(ComputeError::ScratchTooSmall {
            required: required_scratch,
            actual: scratch.dequant_tile_f32.len(),
        });
    }

    let (gate_buf, rest) = scratch.dequant_tile_f32.split_at_mut(intermediate);
    let up_buf = &mut rest[..intermediate];

    unsafe {
        gemv_tensorwise(gate, &hidden_states[..hidden], gate_buf, gemm)?;
        gemv_tensorwise(up, &hidden_states[..hidden], up_buf, gemm)?;
    }
    for (gate_value, up_value) in gate_buf.iter_mut().zip(up_buf.iter()) {
        *gate_value = silu(*gate_value) * *up_value;
    }
    unsafe {
        gemv_tensorwise(down, gate_buf, &mut hidden_states[..down_shape.rows], gemm)?;
    }

    let gate_elements = gate_shape.elements()?;
    let up_elements = up_shape.elements()?;
    let down_elements = down_shape.elements()?;
    gate_elements
        .checked_add(up_elements)
        .and_then(|value| value.checked_add(down_elements))
        .ok_or(ComputeError::InvalidShape("expert element count overflow"))
}

#[derive(Clone, Copy)]
struct MatrixShape {
    rows: usize,
    cols: usize,
}

impl MatrixShape {
    fn elements(self) -> Result<usize, ComputeError> {
        self.rows
            .checked_mul(self.cols)
            .ok_or(ComputeError::InvalidShape("matrix element count overflow"))
    }
}

fn tensor_matrix_shape(tensor: QuantTensorLayout<'_>) -> Result<MatrixShape, ComputeError> {
    if tensor.shape.rank() != 2 {
        let name = std::str::from_utf8(tensor.name).unwrap_or("<non-utf8>");
        eprintln!(
            "invalid_matrix_tensor role={:?} rank={} name={}",
            tensor.role(),
            tensor.shape.rank(),
            name
        );
        return Err(ComputeError::InvalidShape("quant tensor must be rank-2 matrix"));
    }
    Ok(MatrixShape {
        rows: tensor.shape.dim(0)?,
        cols: tensor.shape.dim(1)?,
    })
}

unsafe fn gemv_tensorwise(
    tensor: QuantTensorLayout<'_>,
    input: &[f32],
    output: &mut [f32],
    gemm: &dyn GemmKernel,
) -> Result<usize, ComputeError> {
    if tensor.quant_format == QUANT_DEEPSEEK_BF16_AUX && tensor.dtype_original == 11 {
        return gemv_bf16_tensorwise(tensor, input, output);
    }
    if tensor.quant_format as u32 != 4 {
        return Err(ComputeError::UnsupportedQuantFormat(DiskQuantFormat::Int8Symmetric));
    }
    let shape = tensor_matrix_shape(tensor)?;
    if input.len() < shape.cols {
        return Err(ComputeError::InvalidShape("tensorwise GEMV input too small"));
    }
    if output.len() < shape.rows {
        return Err(ComputeError::OutputTooSmall {
            required: shape.rows,
            actual: output.len(),
        });
    }
    gemm.gemv_i4_affine_tensorwise(
        shape.rows,
        shape.cols,
        input.as_ptr(),
        tensor.data.as_ptr(),
        tensor.scale,
        tensor.zero_point,
        output.as_mut_ptr(),
    )?;
    shape.elements()
}

fn gemv_tensorwise_from_layout(
    layout: &QuantBlockLayout<'_>,
    tensor: QuantTensorLayout<'_>,
    input: &[f32],
    output: &mut [f32],
    gemm: &dyn GemmKernel,
) -> Result<usize, ComputeError> {
    if tensor.quant_format == QUANT_DEEPSEEK_FP8_E4M3 {
        let scale = fp8_scale_for_weight(layout, tensor)?;
        return gemv_fp8_e4m3_ue8m0_tensorwise(tensor, scale, input, output);
    }
    unsafe { gemv_tensorwise(tensor, input, output, gemm) }
}

/// One FP8-E4M3 GEMV row with UE8M0 group scales: LUT byte decode, scale
/// hoisted per column group, four accumulators. Summation is group-wise and
/// unrolled, so it differs from per-element accumulation only by f32
/// reassociation noise.
#[inline]
fn fp8_ue8m0_dot_row(weight_row: &[u8], scale_row: &[u8], col_group: usize, input: &[f32]) -> f32 {
    let fp8_lut = crate::deepseek_v4::fp8_e4m3_base_lut();
    let scale_lut = crate::deepseek_v4::ue8m0_scale_lut();
    #[cfg(target_arch = "x86_64")]
    {
        if crate::deepseek_v4::simd_avx2_fma_available() {
            return unsafe {
                simd_x86::fp8_ue8m0_dot_row_avx2(weight_row, scale_row, col_group, input, fp8_lut, scale_lut)
            };
        }
    }
    let mut acc = 0.0f32;
    for (group, &scale_byte) in scale_row.iter().enumerate() {
        let scale = scale_lut[scale_byte as usize];
        let start = group * col_group;
        let bytes = &weight_row[start..start + col_group];
        let cols = &input[start..start + col_group];
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let quads = col_group / 4;
        for quad in 0..quads {
            let index = quad * 4;
            s0 += fp8_lut[bytes[index] as usize] * cols[index];
            s1 += fp8_lut[bytes[index + 1] as usize] * cols[index + 1];
            s2 += fp8_lut[bytes[index + 2] as usize] * cols[index + 2];
            s3 += fp8_lut[bytes[index + 3] as usize] * cols[index + 3];
        }
        for index in quads * 4..col_group {
            s0 += fp8_lut[bytes[index] as usize] * cols[index];
        }
        acc += scale * ((s0 + s1) + (s2 + s3));
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), v);
        ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    }

    /// FP8 GEMV row: 8-byte gather LUT decode + FMA, two accumulators in
    /// flight to hide gather latency; scale folded per column group.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fp8_ue8m0_dot_row_avx2(
        weight_row: &[u8],
        scale_row: &[u8],
        col_group: usize,
        input: &[f32],
        fp8_lut: &[f32; 256],
        scale_lut: &[f32; 256],
    ) -> f32 {
        let lut_ptr = fp8_lut.as_ptr();
        let mut acc = _mm256_setzero_ps();
        let mut tail = 0.0f32;
        for (group, &scale_byte) in scale_row.iter().enumerate() {
            let scale = scale_lut[scale_byte as usize];
            let start = group * col_group;
            let bytes = &weight_row[start..start + col_group];
            let cols = &input[start..start + col_group];
            let mut g0 = _mm256_setzero_ps();
            let mut g1 = _mm256_setzero_ps();
            let chunks = col_group / 16;
            for chunk in 0..chunks {
                let base = chunk * 16;
                let idx0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    bytes.as_ptr().add(base) as *const __m128i
                ));
                let idx1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    bytes.as_ptr().add(base + 8) as *const __m128i,
                ));
                let w0 = _mm256_i32gather_ps::<4>(lut_ptr, idx0);
                let w1 = _mm256_i32gather_ps::<4>(lut_ptr, idx1);
                g0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(cols.as_ptr().add(base)), g0);
                g1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(cols.as_ptr().add(base + 8)), g1);
            }
            let mut rem = 0.0f32;
            for index in chunks * 16..col_group {
                rem += fp8_lut[bytes[index] as usize] * cols[index];
            }
            acc = _mm256_fmadd_ps(_mm256_add_ps(g0, g1), _mm256_set1_ps(scale), acc);
            tail += rem * scale;
        }
        hsum256(acc) + tail
    }

    /// Same as `fp8_ue8m0_dot_row_avx2`, but ALSO stores the LUT-decoded
    /// (unscaled) weight values into `dequant_out` - the batch GEMV runs
    /// this for the first input and the plain-load dot for the rest, so
    /// the gather cost is paid once per row instead of once per input.
    /// The accumulation path is untouched: bit-identical to the plain
    /// kernel for this input, and the stored values are exactly what the
    /// scalar LUT decode would produce.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fp8_ue8m0_dot_row_avx2_store_dequant(
        weight_row: &[u8],
        scale_row: &[u8],
        col_group: usize,
        input: &[f32],
        fp8_lut: &[f32; 256],
        scale_lut: &[f32; 256],
        dequant_out: &mut [f32],
    ) -> f32 {
        let lut_ptr = fp8_lut.as_ptr();
        let mut acc = _mm256_setzero_ps();
        let mut tail = 0.0f32;
        for (group, &scale_byte) in scale_row.iter().enumerate() {
            let scale = scale_lut[scale_byte as usize];
            let start = group * col_group;
            let bytes = &weight_row[start..start + col_group];
            let cols = &input[start..start + col_group];
            let out = &mut dequant_out[start..start + col_group];
            let mut g0 = _mm256_setzero_ps();
            let mut g1 = _mm256_setzero_ps();
            let chunks = col_group / 16;
            for chunk in 0..chunks {
                let base = chunk * 16;
                let idx0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    bytes.as_ptr().add(base) as *const __m128i
                ));
                let idx1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    bytes.as_ptr().add(base + 8) as *const __m128i,
                ));
                let w0 = _mm256_i32gather_ps::<4>(lut_ptr, idx0);
                let w1 = _mm256_i32gather_ps::<4>(lut_ptr, idx1);
                _mm256_storeu_ps(out.as_mut_ptr().add(base), w0);
                _mm256_storeu_ps(out.as_mut_ptr().add(base + 8), w1);
                g0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(cols.as_ptr().add(base)), g0);
                g1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(cols.as_ptr().add(base + 8)), g1);
            }
            let mut rem = 0.0f32;
            for index in chunks * 16..col_group {
                let value = fp8_lut[bytes[index] as usize];
                out[index] = value;
                rem += value * cols[index];
            }
            acc = _mm256_fmadd_ps(_mm256_add_ps(g0, g1), _mm256_set1_ps(scale), acc);
            tail += rem * scale;
        }
        hsum256(acc) + tail
    }

    /// Dot on a pre-decoded (LUT f32) FP8 row: identical group reduction
    /// to `fp8_ue8m0_dot_row_avx2` (two accumulators, scale folded per
    /// group) with plain loads instead of gathers - bit-identical output.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fp8_ue8m0_dot_dequant_row_avx2(
        dequant_row: &[f32],
        scale_row: &[u8],
        col_group: usize,
        input: &[f32],
        scale_lut: &[f32; 256],
    ) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut tail = 0.0f32;
        for (group, &scale_byte) in scale_row.iter().enumerate() {
            let scale = scale_lut[scale_byte as usize];
            let start = group * col_group;
            let weights = &dequant_row[start..start + col_group];
            let cols = &input[start..start + col_group];
            let mut g0 = _mm256_setzero_ps();
            let mut g1 = _mm256_setzero_ps();
            let chunks = col_group / 16;
            for chunk in 0..chunks {
                let base = chunk * 16;
                let w0 = _mm256_loadu_ps(weights.as_ptr().add(base));
                let w1 = _mm256_loadu_ps(weights.as_ptr().add(base + 8));
                g0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(cols.as_ptr().add(base)), g0);
                g1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(cols.as_ptr().add(base + 8)), g1);
            }
            let mut rem = 0.0f32;
            for index in chunks * 16..col_group {
                rem += weights[index] * cols[index];
            }
            acc = _mm256_fmadd_ps(_mm256_add_ps(g0, g1), _mm256_set1_ps(scale), acc);
            tail += rem * scale;
        }
        hsum256(acc) + tail
    }

    /// BF16 dot: 8x u16 widen + shift-left 16 = f32 bits, then FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_bf16_row_avx2(row_data: &[u8], input: &[f32]) -> f32 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let chunks = input.len() / 16;
        for chunk in 0..chunks {
            let index = chunk * 16;
            let raw0 = _mm_loadu_si128(row_data.as_ptr().add(index * 2) as *const __m128i);
            let raw1 = _mm_loadu_si128(row_data.as_ptr().add(index * 2 + 16) as *const __m128i);
            let w0 = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(raw0)));
            let w1 = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(raw1)));
            acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(input.as_ptr().add(index)), acc0);
            acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(input.as_ptr().add(index + 8)), acc1);
        }
        let mut tail = 0.0f32;
        for index in chunks * 16..input.len() {
            let offset = index * 2;
            let raw = u16::from_le_bytes([row_data[offset], row_data[offset + 1]]);
            tail += f32::from_bits((raw as u32) << 16) * input[index];
        }
        hsum256(_mm256_add_ps(acc0, acc1)) + tail
    }
}

fn gemv_bf16_tensorwise(
    tensor: QuantTensorLayout<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<usize, ComputeError> {
    let shape = tensor_matrix_shape(tensor)?;
    if input.len() < shape.cols {
        return Err(ComputeError::InvalidShape("BF16 GEMV input too small"));
    }
    if output.len() < shape.rows {
        return Err(ComputeError::OutputTooSmall {
            required: shape.rows,
            actual: output.len(),
        });
    }
    let row_bytes = shape
        .cols
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("BF16 GEMV row bytes overflow"))?;
    let required = row_bytes
        .checked_mul(shape.rows)
        .ok_or(ComputeError::InvalidShape("BF16 GEMV bytes overflow"))?;
    if tensor.data.len() < required {
        return Err(ComputeError::InvalidShape("BF16 GEMV tensor data too small"));
    }
    crate::deepseek_v4::parallel_rows_f32(&mut output[..shape.rows], 32, &|row_offset, chunk| {
        for (local, slot) in chunk.iter_mut().enumerate() {
            let byte_offset = (row_offset + local) * row_bytes;
            *slot = dot_bf16_row(&tensor.data[byte_offset..byte_offset + row_bytes], input)?;
        }
        Ok(())
    })?;
    shape.elements()
}

fn gemv_fp8_e4m3_ue8m0_tensorwise(
    weight: QuantTensorLayout<'_>,
    scale: QuantTensorLayout<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<usize, ComputeError> {
    if scale.quant_format != QUANT_DEEPSEEK_UE8M0_SCALE {
        return Err(ComputeError::InvalidQuantBlock("DeepSeek FP8 scale is not UE8M0"));
    }
    let shape = tensor_matrix_shape(weight)?;
    if scale.shape.rank() != 2 {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale rank"));
    }
    let scale_rows = scale.shape.dim(0)?;
    let scale_cols = scale.shape.dim(1)?;
    if shape.rows == 0
        || shape.cols == 0
        || scale_rows == 0
        || scale_cols == 0
        || shape.rows % scale_rows != 0
        || shape.cols % scale_cols != 0
    {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale shape"));
    }
    if input.len() < shape.cols {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 GEMV input too small"));
    }
    if output.len() < shape.rows {
        return Err(ComputeError::OutputTooSmall {
            required: shape.rows,
            actual: output.len(),
        });
    }
    let weight_values = shape.elements()?;
    if weight.data.len() < weight_values {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 weight data too small"));
    }
    let scale_values = scale_rows
        .checked_mul(scale_cols)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP8 scale count overflow"))?;
    if scale.data.len() < scale_values {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale data too small"));
    }

    let row_group = shape.rows / scale_rows;
    let col_group = shape.cols / scale_cols;
    crate::deepseek_v4::parallel_rows_f32(&mut output[..shape.rows], 32, &|row_offset, chunk| {
        for (local, slot) in chunk.iter_mut().enumerate() {
            let row = row_offset + local;
            let weight_row = row * shape.cols;
            let scale_offset = (row / row_group) * scale_cols;
            let acc = fp8_ue8m0_dot_row(
                &weight.data[weight_row..weight_row + shape.cols],
                &scale.data[scale_offset..scale_offset + scale_cols],
                col_group,
                &input[..shape.cols],
            );
            if !acc.is_finite() {
                return Err(ComputeError::InvalidShape("DeepSeek FP8 GEMV non-finite output"));
            }
            *slot = acc;
        }
        Ok(())
    })?;
    Ok(weight_values)
}

/// Row-range variant of the DeepSeek FP8 GEMV: computes output rows
/// [row_start, row_end) only. Scale indexing uses the GLOBAL row so a
/// sub-range keeps the same 128x128 block scales as the full matrix.
fn gemv_fp8_e4m3_ue8m0_rows(
    weight: QuantTensorLayout<'_>,
    scale: QuantTensorLayout<'_>,
    row_start: usize,
    row_end: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<usize, ComputeError> {
    if scale.quant_format != QUANT_DEEPSEEK_UE8M0_SCALE {
        return Err(ComputeError::InvalidQuantBlock("DeepSeek FP8 scale is not UE8M0"));
    }
    let shape = tensor_matrix_shape(weight)?;
    if row_start >= row_end || row_end > shape.rows {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 row range"));
    }
    if scale.shape.rank() != 2 {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale rank"));
    }
    let scale_rows = scale.shape.dim(0)?;
    let scale_cols = scale.shape.dim(1)?;
    if shape.rows == 0
        || shape.cols == 0
        || scale_rows == 0
        || scale_cols == 0
        || shape.rows % scale_rows != 0
        || shape.cols % scale_cols != 0
    {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale shape"));
    }
    let rows = row_end - row_start;
    if input.len() < shape.cols {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 GEMV input too small"));
    }
    if output.len() < rows {
        return Err(ComputeError::OutputTooSmall {
            required: rows,
            actual: output.len(),
        });
    }
    let required = shape
        .cols
        .checked_mul(row_end)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP8 bytes overflow"))?;
    if weight.data.len() < required {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 weight data too small"));
    }

    let scale_values = scale_rows
        .checked_mul(scale_cols)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP8 scale count overflow"))?;
    if scale.data.len() < scale_values {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale data too small"));
    }

    let row_group = shape.rows / scale_rows;
    let col_group = shape.cols / scale_cols;
    crate::deepseek_v4::parallel_rows_f32(&mut output[..rows], 32, &|row_offset, chunk| {
        for (local, slot) in chunk.iter_mut().enumerate() {
            let row = row_start + row_offset + local;
            let weight_row = row * shape.cols;
            let scale_offset = (row / row_group) * scale_cols;
            let acc = fp8_ue8m0_dot_row(
                &weight.data[weight_row..weight_row + shape.cols],
                &scale.data[scale_offset..scale_offset + scale_cols],
                col_group,
                &input[..shape.cols],
            );
            if !acc.is_finite() {
                return Err(ComputeError::InvalidShape("DeepSeek FP8 GEMV non-finite output"));
            }
            *slot = acc;
        }
        Ok(())
    })?;
    rows
        .checked_mul(shape.cols)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP8 values overflow"))
}

/// Dot of an already-LUT-decoded FP8 row against one input, with the SAME
/// group-wise reduction structure as `fp8_ue8m0_dot_row` (and its AVX2
/// twin): two in-flight accumulators per group, scale folded once per
/// group. Same adds in the same order on the same values = bit-identical
/// to the gathering kernel; only the weight load is a plain f32 read.
#[inline]
fn fp8_ue8m0_dot_dequant_row(
    dequant_row: &[f32],
    scale_row: &[u8],
    col_group: usize,
    input: &[f32],
) -> f32 {
    let scale_lut = crate::deepseek_v4::ue8m0_scale_lut();
    #[cfg(target_arch = "x86_64")]
    {
        if crate::deepseek_v4::simd_avx2_fma_available() {
            return unsafe {
                simd_x86::fp8_ue8m0_dot_dequant_row_avx2(
                    dequant_row,
                    scale_row,
                    col_group,
                    input,
                    scale_lut,
                )
            };
        }
    }
    let mut acc = 0.0f32;
    for (group, &scale_byte) in scale_row.iter().enumerate() {
        let scale = scale_lut[scale_byte as usize];
        let start = group * col_group;
        let weights = &dequant_row[start..start + col_group];
        let cols = &input[start..start + col_group];
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let quads = col_group / 4;
        for quad in 0..quads {
            let index = quad * 4;
            s0 += weights[index] * cols[index];
            s1 += weights[index + 1] * cols[index + 1];
            s2 += weights[index + 2] * cols[index + 2];
            s3 += weights[index + 3] * cols[index + 3];
        }
        for index in quads * 4..col_group {
            s0 += weights[index] * cols[index];
        }
        acc += scale * ((s0 + s1) + (s2 + s3));
    }
    acc
}

/// Multi-input FP8 GEMV over a row range (T2 batch phase1): one weight-row
/// read serves EVERY input, instead of re-streaming the whole matrix once
/// per position. Each (row, input) dot runs the same row kernel as the
/// single-input path, so every output value is bit-identical - only the
/// order in which (row, input) pairs execute changes.
///
/// `outputs[i][row - row_start]` receives input i's row. All inputs must
/// cover `shape.cols`; all outputs must cover the row range.
fn gemv_fp8_e4m3_ue8m0_rows_multi(
    weight: QuantTensorLayout<'_>,
    scale: QuantTensorLayout<'_>,
    row_start: usize,
    row_end: usize,
    inputs: &[&[f32]],
    outputs: &mut [&mut [f32]],
) -> Result<usize, ComputeError> {
    if scale.quant_format != QUANT_DEEPSEEK_UE8M0_SCALE {
        return Err(ComputeError::InvalidQuantBlock("DeepSeek FP8 scale is not UE8M0"));
    }
    let count = inputs.len();
    if count == 0 || outputs.len() != count {
        return Err(ComputeError::InvalidShape("FP8 multi GEMV input/output count"));
    }
    let shape = tensor_matrix_shape(weight)?;
    if row_start >= row_end || row_end > shape.rows {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 row range"));
    }
    if scale.shape.rank() != 2 {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale rank"));
    }
    let scale_rows = scale.shape.dim(0)?;
    let scale_cols = scale.shape.dim(1)?;
    if shape.rows == 0
        || shape.cols == 0
        || scale_rows == 0
        || scale_cols == 0
        || shape.rows % scale_rows != 0
        || shape.cols % scale_cols != 0
    {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale shape"));
    }
    let rows = row_end - row_start;
    for input in inputs {
        if input.len() < shape.cols {
            return Err(ComputeError::InvalidShape("DeepSeek FP8 GEMV input too small"));
        }
    }
    for output in outputs.iter() {
        if output.len() < rows {
            return Err(ComputeError::OutputTooSmall {
                required: rows,
                actual: output.len(),
            });
        }
    }
    let required = shape
        .cols
        .checked_mul(row_end)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP8 bytes overflow"))?;
    if weight.data.len() < required {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 weight data too small"));
    }
    let scale_values = scale_rows
        .checked_mul(scale_cols)
        .ok_or(ComputeError::InvalidShape("DeepSeek FP8 scale count overflow"))?;
    if scale.data.len() < scale_values {
        return Err(ComputeError::InvalidShape("DeepSeek FP8 scale data too small"));
    }

    let row_group = shape.rows / scale_rows;
    let col_group = shape.cols / scale_cols;
    // Flat work array, one element per (row, input) pair, row-major. Each
    // worker owns whole rows: it LUT-decodes the weight row to f32 ONCE
    // (the gather is the single-input kernel's bottleneck), then runs one
    // plain-FMA dot per input with the same group-wise reduction order as
    // fp8_ue8m0_dot_row - bit-identical results, gather cost amortized
    // over the batch.
    let mut flat = vec![0.0f32; rows * count];
    {
        // Row-aligned chunks (multiples of `count`, unlike the generic
        // parallel_rows_f32 splitter) so a row never spans two workers.
        let threads = crate::deepseek_v4::compute_thread_count().max(1);
        let chunk_rows = rows.div_ceil(threads).max(8);
        let first_error: std::sync::Mutex<Option<ComputeError>> = std::sync::Mutex::new(None);
        let worker = |chunk_row_start: usize, chunk: &mut [f32]| -> Result<(), ComputeError> {
            let fp8_lut = crate::deepseek_v4::fp8_e4m3_base_lut();
            let scale_lut = crate::deepseek_v4::ue8m0_scale_lut();
            #[cfg(target_arch = "x86_64")]
            let use_avx2 = crate::deepseek_v4::simd_avx2_fma_available();
            #[cfg(not(target_arch = "x86_64"))]
            let use_avx2 = false;
            let mut dequant_row = vec![0.0f32; shape.cols];
            for (local_row, row_slots) in chunk.chunks_mut(count).enumerate() {
                let row = row_start + chunk_row_start + local_row;
                let weight_row = &weight.data[row * shape.cols..(row + 1) * shape.cols];
                let scale_offset = (row / row_group) * scale_cols;
                let scale_row = &scale.data[scale_offset..scale_offset + scale_cols];
                // First input: gather-dot that also stores the decoded row
                // (one gather pass per ROW). Remaining inputs: plain-load
                // dot on the stored values. Same reduction order for every
                // input as the single-position kernel = bit-identical.
                for (index, slot) in row_slots.iter_mut().enumerate() {
                    #[cfg(target_arch = "x86_64")]
                    let acc = if use_avx2 {
                        if index == 0 {
                            unsafe {
                                simd_x86::fp8_ue8m0_dot_row_avx2_store_dequant(
                                    weight_row,
                                    scale_row,
                                    col_group,
                                    &inputs[index][..shape.cols],
                                    fp8_lut,
                                    scale_lut,
                                    &mut dequant_row,
                                )
                            }
                        } else {
                            unsafe {
                                simd_x86::fp8_ue8m0_dot_dequant_row_avx2(
                                    &dequant_row,
                                    scale_row,
                                    col_group,
                                    &inputs[index][..shape.cols],
                                    scale_lut,
                                )
                            }
                        }
                    } else {
                        if index == 0 {
                            for (value, &byte) in dequant_row.iter_mut().zip(weight_row) {
                                *value = fp8_lut[byte as usize];
                            }
                        }
                        fp8_ue8m0_dot_dequant_row(
                            &dequant_row,
                            scale_row,
                            col_group,
                            &inputs[index][..shape.cols],
                        )
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let acc = {
                        if index == 0 {
                            for (value, &byte) in dequant_row.iter_mut().zip(weight_row) {
                                *value = fp8_lut[byte as usize];
                            }
                        }
                        fp8_ue8m0_dot_dequant_row(
                            &dequant_row,
                            scale_row,
                            col_group,
                            &inputs[index][..shape.cols],
                        )
                    };
                    if !acc.is_finite() {
                        return Err(ComputeError::InvalidShape(
                            "DeepSeek FP8 GEMV non-finite output",
                        ));
                    }
                    *slot = acc;
                }
            }
            Ok(())
        };
        if threads <= 1 || rows <= chunk_rows {
            worker(0, &mut flat)?;
        } else {
            std::thread::scope(|scope| {
                let mut rest: &mut [f32] = &mut flat;
                let mut row_offset = 0usize;
                while !rest.is_empty() {
                    let take_rows = chunk_rows.min(rest.len() / count);
                    let (head, tail) = rest.split_at_mut(take_rows * count);
                    rest = tail;
                    let error_slot = &first_error;
                    let worker_ref = &worker;
                    scope.spawn(move || {
                        if let Err(error) = worker_ref(row_offset, head) {
                            let mut slot = error_slot.lock().unwrap();
                            if slot.is_none() {
                                *slot = Some(error);
                            }
                        }
                    });
                    row_offset += take_rows;
                }
            });
            if let Some(error) = first_error.into_inner().unwrap() {
                return Err(error);
            }
        }
    }
    for (index, output) in outputs.iter_mut().enumerate() {
        for row in 0..rows {
            output[row] = flat[row * count + index];
        }
    }
    rows
        .checked_mul(shape.cols)
        .and_then(|value| value.checked_mul(count))
        .ok_or(ComputeError::InvalidShape("DeepSeek FP8 values overflow"))
}

/// Row-range BF16 GEMV: computes output rows [row_start, row_end) only.
fn gemv_bf16_rows(
    tensor: QuantTensorLayout<'_>,
    row_start: usize,
    row_end: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<usize, ComputeError> {
    let shape = tensor_matrix_shape(tensor)?;
    if row_start >= row_end || row_end > shape.rows {
        return Err(ComputeError::InvalidShape("BF16 GEMV row range"));
    }
    let rows = row_end - row_start;
    if input.len() < shape.cols {
        return Err(ComputeError::InvalidShape("BF16 GEMV input too small"));
    }
    if output.len() < rows {
        return Err(ComputeError::OutputTooSmall {
            required: rows,
            actual: output.len(),
        });
    }
    let row_bytes = shape
        .cols
        .checked_mul(2)
        .ok_or(ComputeError::InvalidShape("BF16 GEMV row bytes overflow"))?;
    let required = row_bytes
        .checked_mul(row_end)
        .ok_or(ComputeError::InvalidShape("BF16 GEMV bytes overflow"))?;
    if tensor.data.len() < required {
        return Err(ComputeError::InvalidShape("BF16 GEMV tensor data too small"));
    }
    crate::deepseek_v4::parallel_rows_f32(&mut output[..rows], 32, &|row_offset, chunk| {
        for (local, slot) in chunk.iter_mut().enumerate() {
            let byte_offset = (row_start + row_offset + local) * row_bytes;
            *slot = dot_bf16_row(&tensor.data[byte_offset..byte_offset + row_bytes], input)?;
        }
        Ok(())
    })?;
    rows
        .checked_mul(shape.cols)
        .ok_or(ComputeError::InvalidShape("BF16 GEMV values overflow"))
}

pub unsafe fn fused_gemv_i4_affine(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scales: *const f32,
    zero_points: *const i8,
    group_size: usize,
    output: *mut f32,
) -> Result<CpuKernel, ComputeError> {
    validate_fused_gemv_args(n, k, input, packed_weights, scales, zero_points, group_size, output)?;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            fused_gemv_i4_affine_avx512(
                n,
                k,
                input,
                packed_weights,
                scales,
                zero_points,
                group_size,
                output,
            )?;
            return Ok(CpuKernel::Avx512);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            fused_gemv_i4_affine_avx2(
                n,
                k,
                input,
                packed_weights,
                scales,
                zero_points,
                group_size,
                output,
            )?;
            return Ok(CpuKernel::Avx2);
        }
    }

    fused_gemv_i4_affine_scalar(
        n,
        k,
        input,
        packed_weights,
        scales,
        zero_points,
        group_size,
        output,
    )?;
    Ok(CpuKernel::Scalar)
}

pub unsafe fn fused_gemv_i4_affine_tensorwise(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scale: f32,
    zero_point: f32,
    output: *mut f32,
) -> Result<CpuKernel, ComputeError> {
    validate_fused_gemv_tensorwise_args(n, k, input, packed_weights, scale, zero_point, output)?;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            fused_gemv_i4_affine_tensorwise_avx512(
                n,
                k,
                input,
                packed_weights,
                scale,
                zero_point,
                output,
            )?;
            return Ok(CpuKernel::Avx512);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            fused_gemv_i4_affine_tensorwise_avx2(
                n,
                k,
                input,
                packed_weights,
                scale,
                zero_point,
                output,
            )?;
            return Ok(CpuKernel::Avx2);
        }
    }

    fused_gemv_i4_affine_tensorwise_scalar(n, k, input, packed_weights, scale, zero_point, output)?;
    Ok(CpuKernel::Scalar)
}

fn validate_fused_gemv_args(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scales: *const f32,
    zero_points: *const i8,
    group_size: usize,
    output: *mut f32,
) -> Result<(), ComputeError> {
    if n == 0 || k == 0 {
        return Err(ComputeError::InvalidShape("gemv dimensions must be non-zero"));
    }
    if group_size == 0 {
        return Err(ComputeError::EmptyGroupSize);
    }
    if input.is_null() {
        return Err(ComputeError::InvalidPointer("gemv.input"));
    }
    if packed_weights.is_null() {
        return Err(ComputeError::InvalidPointer("gemv.packed_weights"));
    }
    if scales.is_null() {
        return Err(ComputeError::InvalidPointer("gemv.scales"));
    }
    if zero_points.is_null() {
        return Err(ComputeError::InvalidPointer("gemv.zero_points"));
    }
    if output.is_null() {
        return Err(ComputeError::InvalidPointer("gemv.output"));
    }
    Ok(())
}

fn validate_fused_gemv_tensorwise_args(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scale: f32,
    zero_point: f32,
    output: *mut f32,
) -> Result<(), ComputeError> {
    if n == 0 || k == 0 {
        return Err(ComputeError::InvalidShape("tensorwise GEMV dimensions must be non-zero"));
    }
    if input.is_null() {
        return Err(ComputeError::InvalidPointer("tensorwise.input"));
    }
    if packed_weights.is_null() {
        return Err(ComputeError::InvalidPointer("tensorwise.packed_weights"));
    }
    if output.is_null() {
        return Err(ComputeError::InvalidPointer("tensorwise.output"));
    }
    if !scale.is_finite() || !zero_point.is_finite() {
        return Err(ComputeError::InvalidQuantBlock("non-finite quant metadata"));
    }
    Ok(())
}

unsafe fn fused_gemv_i4_affine_tensorwise_scalar(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scale: f32,
    zero_point: f32,
    output: *mut f32,
) -> Result<(), ComputeError> {
    let row_bytes = packed_i4_row_bytes(k);

    for row in 0..n {
        let weight_row = packed_weights.add(row * row_bytes);
        let mut acc = 0.0f32;
        for inner in 0..k {
            let weight = decode_i4_affine_tensorwise_at(weight_row, inner, scale, zero_point);
            acc += *input.add(inner) * weight;
        }
        *output.add(row) = acc;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fused_gemv_i4_affine_tensorwise_avx512(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scale: f32,
    zero_point: f32,
    output: *mut f32,
) -> Result<(), ComputeError> {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_set_ps,
        _mm512_storeu_ps,
    };

    let row_bytes = packed_i4_row_bytes(k);
    for row in 0..n {
        let weight_row = packed_weights.add(row * row_bytes);
        let mut acc = _mm512_set1_ps(0.0);
        let mut inner = 0usize;

        while inner + 16 <= k {
            let x = _mm512_loadu_ps(input.add(inner));
            let w = _mm512_set_ps(
                decode_i4_affine_tensorwise_at(weight_row, inner + 15, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 14, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 13, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 12, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 11, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 10, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 9, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 8, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 7, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 6, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 5, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 4, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 3, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 2, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 1, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner, scale, zero_point),
            );
            acc = _mm512_add_ps(acc, _mm512_mul_ps(x, w));
            inner += 16;
        }

        let mut lanes = [0.0f32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum = lanes.iter().sum::<f32>();
        while inner < k {
            sum += *input.add(inner)
                * decode_i4_affine_tensorwise_at(weight_row, inner, scale, zero_point);
            inner += 1;
        }
        *output.add(row) = sum;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn fused_gemv_i4_affine_tensorwise_avx2(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scale: f32,
    zero_point: f32,
    output: *mut f32,
) -> Result<(), ComputeError> {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_set_ps, _mm256_storeu_ps,
    };

    let row_bytes = packed_i4_row_bytes(k);
    for row in 0..n {
        let weight_row = packed_weights.add(row * row_bytes);
        let mut acc = _mm256_set1_ps(0.0);
        let mut inner = 0usize;

        while inner + 8 <= k {
            let x = _mm256_loadu_ps(input.add(inner));
            let w = _mm256_set_ps(
                decode_i4_affine_tensorwise_at(weight_row, inner + 7, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 6, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 5, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 4, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 3, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 2, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner + 1, scale, zero_point),
                decode_i4_affine_tensorwise_at(weight_row, inner, scale, zero_point),
            );
            acc = _mm256_fmadd_ps(x, w, acc);
            inner += 8;
        }

        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum = lanes.iter().sum::<f32>();
        while inner < k {
            sum += *input.add(inner)
                * decode_i4_affine_tensorwise_at(weight_row, inner, scale, zero_point);
            inner += 1;
        }
        *output.add(row) = sum;
    }
    Ok(())
}

unsafe fn fused_gemv_i4_affine_scalar(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scales: *const f32,
    zero_points: *const i8,
    group_size: usize,
    output: *mut f32,
) -> Result<(), ComputeError> {
    let row_bytes = packed_i4_row_bytes(k);
    let groups_per_row = grouped_count(k, group_size);

    for row in 0..n {
        let weight_row = packed_weights.add(row * row_bytes);
        let scale_row = scales.add(row * groups_per_row);
        let zero_row = zero_points.add(row * groups_per_row);
        let mut acc = 0.0f32;

        for inner in 0..k {
            let weight = decode_i4_affine_at(weight_row, scale_row, zero_row, inner, group_size);
            acc += *input.add(inner) * weight;
        }

        *output.add(row) = acc;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fused_gemv_i4_affine_avx512(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scales: *const f32,
    zero_points: *const i8,
    group_size: usize,
    output: *mut f32,
) -> Result<(), ComputeError> {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_set_ps,
        _mm512_storeu_ps,
    };

    let row_bytes = packed_i4_row_bytes(k);
    let groups_per_row = grouped_count(k, group_size);

    for row in 0..n {
        let weight_row = packed_weights.add(row * row_bytes);
        let scale_row = scales.add(row * groups_per_row);
        let zero_row = zero_points.add(row * groups_per_row);
        let mut acc = _mm512_set1_ps(0.0);
        let mut inner = 0usize;

        while inner + 16 <= k {
            let x = _mm512_loadu_ps(input.add(inner));
            let w = _mm512_set_ps(
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 15, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 14, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 13, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 12, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 11, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 10, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 9, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 8, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 7, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 6, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 5, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 4, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 3, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 2, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 1, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner, group_size),
            );
            acc = _mm512_add_ps(acc, _mm512_mul_ps(x, w));
            inner += 16;
        }

        let mut lanes = [0.0f32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum = lanes.iter().sum::<f32>();
        while inner < k {
            let weight = decode_i4_affine_at(weight_row, scale_row, zero_row, inner, group_size);
            sum += *input.add(inner) * weight;
            inner += 1;
        }
        *output.add(row) = sum;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn fused_gemv_i4_affine_avx2(
    n: usize,
    k: usize,
    input: *const f32,
    packed_weights: *const u8,
    scales: *const f32,
    zero_points: *const i8,
    group_size: usize,
    output: *mut f32,
) -> Result<(), ComputeError> {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_set_ps, _mm256_storeu_ps,
    };

    let row_bytes = packed_i4_row_bytes(k);
    let groups_per_row = grouped_count(k, group_size);

    for row in 0..n {
        let weight_row = packed_weights.add(row * row_bytes);
        let scale_row = scales.add(row * groups_per_row);
        let zero_row = zero_points.add(row * groups_per_row);
        let mut acc = _mm256_set1_ps(0.0);
        let mut inner = 0usize;

        while inner + 8 <= k {
            let x = _mm256_loadu_ps(input.add(inner));
            let w = _mm256_set_ps(
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 7, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 6, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 5, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 4, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 3, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 2, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner + 1, group_size),
                decode_i4_affine_at(weight_row, scale_row, zero_row, inner, group_size),
            );
            acc = _mm256_fmadd_ps(x, w, acc);
            inner += 8;
        }

        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum = lanes.iter().sum::<f32>();
        while inner < k {
            let weight = decode_i4_affine_at(weight_row, scale_row, zero_row, inner, group_size);
            sum += *input.add(inner) * weight;
            inner += 1;
        }
        *output.add(row) = sum;
    }
    Ok(())
}

#[inline]
const fn packed_i4_row_bytes(k: usize) -> usize {
    (k + 1) / 2
}

#[inline]
const fn grouped_count(len: usize, group_size: usize) -> usize {
    (len + group_size - 1) / group_size
}

#[inline]
unsafe fn decode_i4_affine_at(
    weight_row: *const u8,
    scale_row: *const f32,
    zero_row: *const i8,
    inner: usize,
    group_size: usize,
) -> f32 {
    let byte = *weight_row.add(inner / 2);
    let nibble = if inner & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let group = inner / group_size;
    let scale = *scale_row.add(group);
    let zero = *zero_row.add(group);
    (nibble as i8 - zero) as f32 * scale
}

#[inline]
unsafe fn decode_i4_affine_tensorwise_at(
    weight_row: *const u8,
    inner: usize,
    scale: f32,
    zero_point: f32,
) -> f32 {
    let byte = *weight_row.add(inner / 2);
    let nibble = if inner & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    (nibble as f32 - zero_point) * scale
}

fn rms_norm_in_place(hidden_states: &mut [f32]) {
    if hidden_states.is_empty() {
        return;
    }
    let mean_square = hidden_states
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / hidden_states.len() as f32;
    let inv_rms = 1.0 / (mean_square + 1.0e-6).sqrt();
    for value in hidden_states {
        *value *= inv_rms;
    }
}

fn silu_gate_in_place(hidden_states: &mut [f32]) {
    for value in hidden_states {
        *value = silu(*value);
    }
}

#[inline]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn read_name_blob<'a>(
    buffer: &'a [u8],
    names_offset: usize,
    name_offset: usize,
) -> Result<&'a [u8], ComputeError> {
    let absolute = names_offset
        .checked_add(name_offset)
        .ok_or(ComputeError::InvalidQuantBlock("name offset overflow"))?;
    let len = read_u16_le(buffer, absolute, "name.len")? as usize;
    let start = absolute
        .checked_add(2)
        .ok_or(ComputeError::InvalidQuantBlock("name start overflow"))?;
    let end = start
        .checked_add(len)
        .ok_or(ComputeError::InvalidQuantBlock("name end overflow"))?;
    buffer
        .get(start..end)
        .ok_or(ComputeError::TruncatedQuantBlock("name blob"))
}

fn read_shape_layout<'a>(
    buffer: &'a [u8],
    shape_offset: usize,
    expected_rank: usize,
) -> Result<ShapeLayout<'a>, ComputeError> {
    let rank = read_u32_le(buffer, shape_offset, "shape.rank")? as usize;
    if rank != expected_rank {
        return Err(ComputeError::InvalidQuantBlock("shape rank mismatch"));
    }
    let byte_len = 4usize
        .checked_add(
            rank.checked_mul(8)
                .ok_or(ComputeError::InvalidQuantBlock("shape size overflow"))?,
        )
        .ok_or(ComputeError::InvalidQuantBlock("shape range overflow"))?;
    let end = shape_offset
        .checked_add(byte_len)
        .ok_or(ComputeError::InvalidQuantBlock("shape end overflow"))?;
    let data = buffer
        .get(shape_offset..end)
        .ok_or(ComputeError::TruncatedQuantBlock("shape blob"))?;
    Ok(ShapeLayout { data, rank })
}

fn read_u16_le(buffer: &[u8], offset: usize, field: &'static str) -> Result<u16, ComputeError> {
    let bytes = read_le_bytes::<2>(buffer, offset, field)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32_le(buffer: &[u8], offset: usize, field: &'static str) -> Result<u32, ComputeError> {
    let bytes = read_le_bytes::<4>(buffer, offset, field)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_le(buffer: &[u8], offset: usize, field: &'static str) -> Result<u64, ComputeError> {
    let bytes = read_le_bytes::<8>(buffer, offset, field)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32_le(buffer: &[u8], offset: usize, field: &'static str) -> Result<f32, ComputeError> {
    let bytes = read_le_bytes::<4>(buffer, offset, field)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_le_bytes<const N: usize>(
    buffer: &[u8],
    offset: usize,
    field: &'static str,
) -> Result<[u8; N], ComputeError> {
    let end = offset
        .checked_add(N)
        .ok_or(ComputeError::InvalidQuantBlock("read range overflow"))?;
    let slice = buffer
        .get(offset..end)
        .ok_or(ComputeError::TruncatedQuantBlock(field))?;
    let mut bytes = [0u8; N];
    bytes.copy_from_slice(slice);
    Ok(bytes)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dequantize_i4_avx512(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    out_f32: &mut [f32],
) -> Result<(), ComputeError> {
    // AVX-512 dispatch boundary. The first version intentionally reuses the
    // AVX2 decoder when available until the packed GEMM micro-kernel lands.
    if std::is_x86_feature_detected!("avx2") {
        return dequantize_i4_avx2(packed_weights, params, out_f32);
    }
    dequantize_i4_scalar(packed_weights, params, out_f32)
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dequantize_i4_avx512(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    out_f32: &mut [f32],
) -> Result<(), ComputeError> {
    dequantize_i4_scalar(packed_weights, params, out_f32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_i4_avx2(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    out_f32: &mut [f32],
) -> Result<(), ComputeError> {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_srli_epi16,
        _mm256_storeu_si256,
    };

    let mask = _mm256_set1_epi8(0x0f);
    let mut input_index = 0usize;
    let mut output_index = 0usize;

    while input_index + 32 <= packed_weights.len() {
        let packed =
            _mm256_loadu_si256(packed_weights.as_ptr().add(input_index).cast::<__m256i>());
        let low = _mm256_and_si256(packed, mask);
        let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);

        let mut lows = [0u8; 32];
        let mut highs = [0u8; 32];
        _mm256_storeu_si256(lows.as_mut_ptr().cast::<__m256i>(), low);
        _mm256_storeu_si256(highs.as_mut_ptr().cast::<__m256i>(), high);

        for lane in 0..32 {
            write_i4_value(
                lows[lane],
                output_index,
                params,
                &mut out_f32[output_index..],
            )?;
            output_index += 1;
            write_i4_value(
                highs[lane],
                output_index,
                params,
                &mut out_f32[output_index..],
            )?;
            output_index += 1;
        }

        input_index += 32;
    }

    dequantize_i4_scalar_from(
        &packed_weights[input_index..],
        params,
        output_index,
        &mut out_f32[output_index..],
    )
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dequantize_i4_avx2(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    out_f32: &mut [f32],
) -> Result<(), ComputeError> {
    dequantize_i4_scalar(packed_weights, params, out_f32)
}

fn dequantize_i4_scalar(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    out_f32: &mut [f32],
) -> Result<(), ComputeError> {
    dequantize_i4_scalar_from(packed_weights, params, 0, out_f32)
}

fn dequantize_i4_scalar_from(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    absolute_output_offset: usize,
    out_f32: &mut [f32],
) -> Result<(), ComputeError> {
    let mut output_index = 0usize;
    for byte in packed_weights {
        write_i4_value(
            byte & 0x0f,
            absolute_output_offset + output_index,
            params,
            &mut out_f32[output_index..],
        )?;
        output_index += 1;
        write_i4_value(
            byte >> 4,
            absolute_output_offset + output_index,
            params,
            &mut out_f32[output_index..],
        )?;
        output_index += 1;
    }
    Ok(())
}

fn dequantize_i8_scalar(
    packed_weights: &[u8],
    params: QuantBlockParams<'_>,
    out_f32: &mut [f32],
) -> Result<(), ComputeError> {
    for (index, byte) in packed_weights.iter().enumerate() {
        let group = index / params.group_size;
        let scale = *params
            .scales
            .get(group)
            .or_else(|| params.scales.last())
            .ok_or(ComputeError::MissingScale { group })?;
        out_f32[index] = (*byte as i8 as f32) * scale;
    }
    Ok(())
}

fn write_i4_value(
    nibble: u8,
    absolute_index: usize,
    params: QuantBlockParams<'_>,
    out: &mut [f32],
) -> Result<(), ComputeError> {
    let group = absolute_index / params.group_size;
    let scale = *params
        .scales
        .get(group)
        .or_else(|| params.scales.last())
        .ok_or(ComputeError::MissingScale { group })?;

    let value = match params.format {
        DiskQuantFormat::Int4Symmetric => {
            let signed = if nibble >= 8 {
                nibble as i8 - 16
            } else {
                nibble as i8
            };
            signed as f32 * scale
        }
        DiskQuantFormat::Int4Affine => {
            let zero = *params
                .zero_points
                .get(group)
                .or_else(|| params.zero_points.last())
                .ok_or(ComputeError::MissingZeroPoint { group })?;
            (nibble as i8 - zero) as f32 * scale
        }
        other => return Err(ComputeError::UnsupportedQuantFormat(other)),
    };

    out[0] = value;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_bytes(seed: u32, count: usize) -> Vec<u8> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect()
    }

    fn lcg_f32(seed: u32, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65536.0) - 0.5
            })
            .collect()
    }

    #[test]
    fn fp8_dot_row_matches_per_element_reference() {
        let cols = 200usize; // non multiplo di 16: esercita anche il tail
        let groups = 4usize;
        let col_group = cols / groups; // 50
        let weight = lcg_bytes(0xbeef, cols);
        let scales: Vec<u8> = (0..groups).map(|g| 120 + g as u8).collect();
        let input = lcg_f32(0xcafe, cols);
        let fast = fp8_ue8m0_dot_row(&weight, &scales, col_group, &input);
        let mut reference = 0.0f64;
        for col in 0..cols {
            let scale = crate::deepseek_v4::decode_ue8m0_scale(scales[col / col_group]);
            reference +=
                (crate::deepseek_v4::decode_fp8_e4m3(weight[col], scale) * input[col]) as f64;
        }
        assert!(
            (fast as f64 - reference).abs() <= 1e-4 * reference.abs().max(1.0),
            "fast {fast} vs reference {reference}"
        );
    }

    #[test]
    fn bf16_dot_row_matches_per_element_reference() {
        let count = 37usize; // non multiplo di 16: esercita anche il tail
        let input = lcg_f32(0x1234, count);
        let values = lcg_f32(0x5678, count);
        let mut row_data = Vec::with_capacity(count * 2);
        let mut reference = 0.0f64;
        for (index, value) in values.iter().enumerate() {
            let raw = (value.to_bits() >> 16) as u16;
            row_data.extend_from_slice(&raw.to_le_bytes());
            reference += (f32::from_bits((raw as u32) << 16) * input[index]) as f64;
        }
        let fast = dot_bf16_row(&row_data, &input).unwrap();
        assert!(
            (fast as f64 - reference).abs() <= 1e-4 * reference.abs().max(1.0),
            "fast {fast} vs reference {reference}"
        );
    }

    #[test]
    fn scalar_int4_symmetric_dequantizes_low_then_high_nibble() {
        let packed = [0x1f, 0x87];
        let params = QuantBlockParams {
            format: DiskQuantFormat::Int4Symmetric,
            group_size: 4,
            scales: &[0.5],
            zero_points: &[],
        };
        let mut out = [0.0f32; 4];

        dequantize_i4_scalar(&packed, params, &mut out).unwrap();

        assert_eq!(out, [-0.5, 0.5, 3.5, -4.0]);
    }

    #[test]
    fn scalar_int4_affine_uses_group_zero_points() {
        let packed = [0x10, 0x32];
        let params = QuantBlockParams {
            format: DiskQuantFormat::Int4Affine,
            group_size: 2,
            scales: &[1.0, 2.0],
            zero_points: &[1, 2],
        };
        let mut out = [0.0f32; 4];

        dequantize_i4_scalar(&packed, params, &mut out).unwrap();

        assert_eq!(out, [-1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn fused_scalar_i4_affine_gemv_matches_expected_dot_products() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let weights = pack_i4_rows(&[&[1, 2, 3, 4], &[4, 3, 2, 1]]);
        let scales = [0.5f32, 1.0];
        let zeros = [0i8, 2];
        let mut output = [0.0f32; 2];

        unsafe {
            fused_gemv_i4_affine_scalar(
                2,
                4,
                input.as_ptr(),
                weights.as_ptr(),
                scales.as_ptr(),
                zeros.as_ptr(),
                4,
                output.as_mut_ptr(),
            )
            .unwrap();
        }

        assert_close(output[0], 15.0);
        assert_close(output[1], 0.0);
    }

    #[test]
    fn fused_kernel_dispatch_matches_scalar_reference() {
        let input = [1.0f32, -2.0, 3.5, 0.5, -1.0, 2.0, 1.5, -0.5, 4.0];
        let weights = pack_i4_rows(&[&[1, 2, 3, 4, 5, 6, 7, 8], &[8, 7, 6, 5, 4, 3, 2, 1]]);
        let scales = [0.25f32, 0.5, 0.125, 0.75];
        let zeros = [1i8, 2, 3, 4];
        let mut scalar = [0.0f32; 2];
        let mut dispatched = [0.0f32; 2];

        unsafe {
            fused_gemv_i4_affine_scalar(
                2,
                8,
                input.as_ptr(),
                weights.as_ptr(),
                scales.as_ptr(),
                zeros.as_ptr(),
                4,
                scalar.as_mut_ptr(),
            )
            .unwrap();
            fused_gemv_i4_affine(
                2,
                8,
                input.as_ptr(),
                weights.as_ptr(),
                scales.as_ptr(),
                zeros.as_ptr(),
                4,
                dispatched.as_mut_ptr(),
            )
            .unwrap();
        }

        assert_close(dispatched[0], scalar[0]);
        assert_close(dispatched[1], scalar[1]);
    }

    #[test]
    fn parse_quant_block_reads_zcblk01_tensor_without_allocating() {
        let block = make_test_zcblk01();
        let layout = parse_quant_block(&block).unwrap();
        let tensor = layout.first_tensor().unwrap();

        assert_eq!(layout.tensor_count(), 1);
        assert_eq!(tensor.quant_format, 4);
        assert_eq!(tensor.name, b"expert.up_proj.weight");
        assert_eq!(tensor.role(), TensorRole::UpProj);
        assert_eq!(tensor.shape.rank(), 2);
        assert_eq!(tensor.shape.dim(0).unwrap(), 2);
        assert_eq!(tensor.shape.dim(1).unwrap(), 4);
        assert_eq!(tensor.data, &[0x21, 0x43, 0x34, 0x12]);
        assert_close(tensor.scale, 0.5);
        assert_close(tensor.zero_point, 0.0);
    }

    #[test]
    fn compute_quant_block_uses_zcblk01_metadata_for_tensorwise_gemv() {
        let block = make_test_zcblk01();
        let mut hidden = [1.0f32, 2.0, 3.0, 4.0];
        let mut scratch_buf = [0.0f32; 8];
        let mut scratch = ComputeScratch {
            dequant_tile_f32: &mut scratch_buf,
        };
        let config = ComputeConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 1,
            num_kv_heads: 1,
            qk_rope_head_dim: 4,
            qk_nope_head_dim: 0,
            v_head_dim: 4,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            rope_theta: 10_000.0,
            prefill_chunk_tokens: 1,
            quant_group_size: 4,
            preferred_kernel: CpuKernel::Scalar,
            router_math: RouterMath::RawLogits,
            route_scale: 1.0,
            num_hash_layers: 0,
            attention_kind: AttentionKind::GlmDsaProbe,
            o_groups: 0,
            o_lora_rank: 0,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        };
        let gemm = FusedInt4Gemm;

        let layout = parse_quant_block(&block).unwrap();
        compute_quant_block_layout_into_hidden(&layout, &mut hidden, &mut scratch, &config, &gemm)
            .unwrap();

        assert_close(hidden[0], 15.0);
        assert_close(hidden[1], 10.0);
    }

    #[test]
    fn kv_cache_uses_caller_provided_scratch() {
        let required = KVCache::required_f32(2, 3, 2, 4).unwrap();
        let mut storage = vec![0.0f32; required];
        let mut cache = KVCache::from_scratch(&mut storage, 2, 3, 2, 4).unwrap();
        let key = [1.0f32; 8];
        let value = [2.0f32; 8];

        cache.append(1, 2, &key, &value).unwrap();

        assert_eq!(cache.key_slice(1, 2).unwrap(), &key);
        assert_eq!(cache.value_slice(1, 2).unwrap(), &value);
        assert_eq!(cache.cursor, 3);
    }

    #[test]
    fn causal_attention_reads_history_from_kv_cache() {
        let config = ComputeConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 1,
            num_kv_heads: 1,
            qk_rope_head_dim: 2,
            qk_nope_head_dim: 2,
            v_head_dim: 4,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            rope_theta: 10_000.0,
            prefill_chunk_tokens: 1,
            quant_group_size: 4,
            preferred_kernel: CpuKernel::Scalar,
            router_math: RouterMath::RawLogits,
            route_scale: 1.0,
            num_hash_layers: 0,
            attention_kind: AttentionKind::GlmDsaProbe,
            o_groups: 0,
            o_lora_rank: 0,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        };
        let required = KVCache::required_f32(1, 2, 1, 4).unwrap();
        let mut storage = vec![0.0f32; required];
        let mut cache = KVCache::from_scratch(&mut storage, 1, 2, 1, 4).unwrap();
        cache
            .append(0, 0, &[1.0, 0.0, 0.0, 0.0], &[10.0, 0.0, 0.0, 0.0])
            .unwrap();
        cache
            .append(0, 1, &[0.0, 8.0, 0.0, 0.0], &[0.0, 20.0, 0.0, 0.0])
            .unwrap();

        let query = [0.0f32, 8.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        causal_attention_from_kv_cache(0, 1, &query, &cache, &mut out, &config).unwrap();

        assert!(out[1] > 19.0, "expected current token value to dominate: {out:?}");
        assert!(out[0] < 1.0, "expected previous token value to be suppressed: {out:?}");
    }

    #[test]
    fn glm_dsa_append_expands_shared_rope_key_per_head() {
        let config = ComputeConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_kv_heads: 2,
            qk_rope_head_dim: 2,
            qk_nope_head_dim: 2,
            v_head_dim: 4,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            rope_theta: 10_000.0,
            prefill_chunk_tokens: 1,
            quant_group_size: 4,
            preferred_kernel: CpuKernel::Scalar,
            router_math: RouterMath::RawLogits,
            route_scale: 1.0,
            num_hash_layers: 0,
            attention_kind: AttentionKind::GlmDsaProbe,
            o_groups: 0,
            o_lora_rank: 0,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        };
        let required = KVCache::required_f32(1, 1, 2, 4).unwrap();
        let mut storage = vec![0.0f32; required];
        let mut cache = KVCache::from_scratch(&mut storage, 1, 1, 2, 4).unwrap();
        let kv_a = [0.0f32, 0.0, 0.0, 0.0, 7.0, 9.0];
        let kv_full = [
            1.0f32, 2.0, 10.0, 11.0, 12.0, 13.0, 3.0, 4.0, 20.0, 21.0, 22.0, 23.0,
        ];

        append_glm_dsa_kv_cache(0, 0, &kv_a, &kv_full, &mut cache, &config).unwrap();

        assert_eq!(
            cache.key_slice(0, 0).unwrap(),
            &[1.0, 2.0, 7.0, 9.0, 3.0, 4.0, 7.0, 9.0]
        );
        assert_eq!(
            cache.value_slice(0, 0).unwrap(),
            &[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]
        );
    }

    #[test]
    fn glm52_prefill_scratch_covers_q_kv_and_value_buffers() {
        let config = ComputeConfig::glm52_like(6144);
        let qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
        let q_full = config.num_attention_heads * qk_head_dim;
        let value_concat = config.num_kv_heads * config.v_head_dim;
        let kv_a = config.kv_lora_rank + config.qk_rope_head_dim;
        let kv_full = config.num_kv_heads * (config.qk_nope_head_dim + config.v_head_dim);
        let required = config.q_lora_rank + q_full.max(value_concat) + kv_a + kv_full;
        assert_eq!(required, 47_680);
        assert!(config.prefill_scratch_f32() >= required);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope_interleaved_in_place(&mut values, 0, 4, 10_000.0).unwrap();
        assert_close(values[0], 1.0);
        assert_close(values[1], 2.0);
        assert_close(values[2], 3.0);
        assert_close(values[3], 4.0);
    }

    #[test]
    fn rope_interleaved_rotates_each_head_prefix() {
        let mut values = vec![1.0, 0.0, 9.0, 9.0, 0.0, 1.0, 8.0, 8.0];
        apply_rope_to_heads_interleaved(&mut values, 2, 4, 2, 1, 10_000.0).unwrap();
        let (sin, cos) = 1.0f32.sin_cos();
        assert_close(values[0], cos);
        assert_close(values[1], sin);
        assert_close(values[2], 9.0);
        assert_close(values[3], 9.0);
        assert_close(values[4], -sin);
        assert_close(values[5], cos);
        assert_close(values[6], 8.0);
        assert_close(values[7], 8.0);
    }

    #[test]
    fn lm_head_argmax_reads_zcblk01_logits() {
        let block = make_single_tensor_zcblk01(
            b"lm_head.weight",
            TensorRole::LmHead,
            &[2, 2],
            &[0x01, 0x20],
            1.0,
            0.0,
        );
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;
        let token = unsafe { lm_head_argmax_from_block(&block, &hidden, &mut scratch, &gemm) }
            .unwrap()
            .unwrap();
        assert_eq!(token, 1);
        assert_close(scratch[0], 1.0);
        assert_close(scratch[1], 4.0);
    }

    #[test]
    fn lm_head_argmax_streams_rows_in_chunks() {
        let block = make_single_tensor_zcblk01(
            b"lm_head.weight",
            TensorRole::LmHead,
            &[4, 2],
            &pack_i4_rows(&[&[1, 0], &[0, 2], &[3, 3], &[2, 1]]),
            1.0,
            0.0,
        );
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;

        let token =
            unsafe { lm_head_argmax_chunked_from_block(&block, &hidden, &mut scratch, &gemm) }
                .unwrap()
                .unwrap();

        assert_eq!(token, 2);
    }

    #[test]
    fn lm_head_topk_streams_rows_in_chunks() {
        let block = make_single_tensor_zcblk01(
            b"lm_head.weight",
            TensorRole::LmHead,
            &[4, 2],
            &pack_i4_rows(&[&[1, 0], &[0, 2], &[3, 3], &[2, 1]]),
            1.0,
            0.0,
        );
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;

        let top =
            unsafe { lm_head_topk_score_chunked_from_block(&block, &hidden, &mut scratch, &gemm, 3) }
                .unwrap();

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 2);
        assert_close(top[0].1, 9.0);
        assert_eq!(top[1].0, 3);
        assert_close(top[1].1, 4.0);
        assert_eq!(top[2].0, 1);
        assert_close(top[2].1, 4.0);
    }

    #[test]
    fn lm_head_topk_reads_deepseek_bf16_head() {
        let block = make_single_tensor_zcblk01_with_format(
            b"head.weight",
            TensorRole::LmHead,
            &[3, 2],
            &bf16_rows(&[&[1.0, 0.0], &[0.0, 2.0], &[2.0, 2.0]]),
            11,
            QUANT_DEEPSEEK_BF16_AUX,
            1.0,
            0.0,
        );
        let hidden = [1.0f32, 1.0];
        let mut scratch = [0.0f32; 1];
        let gemm = FusedInt4Gemm;

        let top =
            unsafe { lm_head_topk_score_chunked_from_block(&block, &hidden, &mut scratch, &gemm, 2) }
                .unwrap();

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 2);
        assert_close(top[0].1, 4.0);
        assert_eq!(top[1].0, 1);
        assert_close(top[1].1, 2.0);
    }

    #[test]
    fn lm_head_argmax_returns_global_row_for_shard() {
        let block = make_single_tensor_zcblk01(
            b"lm_head.weight.rows_100_104",
            TensorRole::LmHead,
            &[4, 2],
            &pack_i4_rows(&[&[1, 0], &[0, 2], &[3, 3], &[2, 1]]),
            1.0,
            0.0,
        );
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;

        let token =
            unsafe { lm_head_argmax_chunked_from_block(&block, &hidden, &mut scratch, &gemm) }
                .unwrap()
                .unwrap();

        assert_eq!(token, 102);
    }

    #[test]
    fn global_embed_block_is_not_a_compute_layer() {
        let block = make_single_tensor_zcblk01(
            b"model.embed_tokens.weight",
            TensorRole::Embed,
            &[2, 4],
            &[0x10, 0x32, 0x54, 0x76],
            1.0,
            0.0,
        );

        assert!(is_non_compute_dense_block(&block).unwrap());
    }

    #[test]
    fn dense_math_profile_detects_shared_expert_gap_inputs() {
        let block = make_two_tensor_zcblk01(
            (
                b"model.layers.3.self_attn.q_a_proj.weight",
                TensorRole::QProj,
                &[2, 2],
                &pack_i4_rows(&[&[1, 0], &[0, 1]]),
                1.0,
                0.0,
            ),
            (
                b"model.layers.3.mlp.shared_experts.gate_proj.weight",
                TensorRole::SharedExpert,
                &[2, 2],
                &pack_i4_rows(&[&[2, 0], &[0, 2]]),
                1.0,
                0.0,
            ),
        );

        let profile = dense_math_profile(&block).unwrap();

        assert!(profile.has_attention);
        assert!(profile.has_shared_expert);
        assert!(!profile.has_router);
        assert!(!profile.has_indexer);
        assert_eq!(profile.norm_tensors, 0);
        assert!(!is_non_compute_dense_block(&block).unwrap());
    }

    #[test]
    fn dense_math_profile_detects_indexer_for_explicit_bypass() {
        let block = make_single_tensor_zcblk01(
            b"model.layers.3.self_attn.indexer.weight",
            TensorRole::Unknown,
            &[2, 2],
            &pack_i4_rows(&[&[1, 0], &[0, 1]]),
            1.0,
            0.0,
        );

        let profile = dense_math_profile(&block).unwrap();

        assert!(profile.has_indexer);
    }

    #[test]
    fn glm52_indexer_bypass_is_equivalent_until_topk() {
        assert_eq!(
            glm52_indexer_bypass_status(GLM52_INDEX_TOPK),
            ("not_required_context_le_topk", true)
        );
        assert_eq!(
            glm52_indexer_bypass_status(GLM52_INDEX_TOPK + 1),
            ("bypassed_long_context", false)
        );
    }

    #[test]
    fn shared_expert_dense_branch_computes_gate_up_down() {
        let dense = make_three_tensor_zcblk01(
            (
                b"model.layers.3.mlp.shared_experts.gate_proj.weight",
                TensorRole::SharedExpert,
                &[2, 2],
                &pack_i4_rows(&[&[1, 0], &[0, 1]]),
                1.0,
                0.0,
            ),
            (
                b"model.layers.3.mlp.shared_experts.up_proj.weight",
                TensorRole::SharedExpert,
                &[2, 2],
                &pack_i4_rows(&[&[1, 0], &[0, 1]]),
                1.0,
                0.0,
            ),
            (
                b"model.layers.3.mlp.shared_experts.down_proj.weight",
                TensorRole::SharedExpert,
                &[2, 2],
                &pack_i4_rows(&[&[1, 0], &[0, 1]]),
                1.0,
                0.0,
            ),
        );
        let mut hidden = [1.0f32, 1.0];
        let mut scratch_buf = [0.0f32; 16];
        let mut scratch = ComputeScratch {
            dequant_tile_f32: &mut scratch_buf,
        };
        let config = ComputeConfig {
            hidden_size: 2,
            intermediate_size: 2,
            num_attention_heads: 1,
            num_kv_heads: 1,
            qk_rope_head_dim: 2,
            qk_nope_head_dim: 0,
            v_head_dim: 2,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            rope_theta: 10_000.0,
            prefill_chunk_tokens: 1,
            quant_group_size: 4,
            preferred_kernel: CpuKernel::Scalar,
            router_math: RouterMath::RawLogits,
            route_scale: 1.0,
            num_hash_layers: 0,
            attention_kind: AttentionKind::GlmDsaProbe,
            o_groups: 0,
            o_lora_rank: 0,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        };
        let gemm = FusedInt4Gemm;
        let mut stats = LayerComputeStats::default();

        let applied = compute_shared_expert_from_dense(
            &dense,
            &mut hidden,
            &mut scratch,
            &config,
            &gemm,
            &mut stats,
        )
        .unwrap();

        let silu_one = 1.0 / (1.0 + (-1.0f32).exp());
        assert!(applied);
        assert_close(hidden[0], silu_one);
        assert_close(hidden[1], silu_one);
        assert!(stats.dequantized_values > 0);
    }

    #[test]
    fn router_selection_skips_rank1_gate_bias() {
        let block = make_two_tensor_zcblk01(
            (
                b"model.layers.3.mlp.gate.e_score_correction_bias",
                TensorRole::Router,
                &[2],
                &[0x10],
                1.0,
                0.0,
            ),
            (
                b"model.layers.3.mlp.gate.weight",
                TensorRole::Router,
                &[2, 2],
                &pack_i4_rows(&[&[1, 0], &[0, 2]]),
                1.0,
                0.0,
            ),
        );
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;

        let routes = route_experts_from_dense_block(&block, &hidden, 1, &[0, 1], &mut scratch, &gemm, RouteOptions::default())
            .unwrap()
            .unwrap();

        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].expert_id, 1);
    }

    #[test]
    fn router_selection_skips_tid2eid_table_with_non_hidden_cols() {
        // DeepSeek blocks place the static `ffn.gate.tid2eid` table
        // ([vocab, top_k], Router role) before the real gate projection.
        // Routing must pick the gate whose cols match the hidden width.
        let block = make_two_tensor_zcblk01(
            (
                b"layers.3.ffn.gate.tid2eid",
                TensorRole::Router,
                &[4, 3],
                &pack_i4_rows(&[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1], &[1, 1, 1]]),
                1.0,
                0.0,
            ),
            (
                b"layers.3.ffn.gate.weight",
                TensorRole::Router,
                &[2, 2],
                &pack_i4_rows(&[&[1, 0], &[0, 2]]),
                1.0,
                0.0,
            ),
        );
        let hidden = [1.0f32, 2.0];
        // Scratch sized for the real gate rows (2), NOT for the tid2eid rows
        // (4): the old first-rank2 selection would fail with ScratchTooSmall.
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;

        let routes = route_experts_from_dense_block(&block, &hidden, 1, &[0, 1], &mut scratch, &gemm, RouteOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].expert_id, 1);

        let probe = router_top_experts_from_dense_block(&block, &hidden, 2, &mut scratch, &gemm)
            .unwrap()
            .unwrap();
        assert_eq!(probe.len(), 2);
        assert_eq!(probe[0].expert_id, 1);
    }

    #[test]
    fn router_probe_scores_all_router_rows() {
        let block = make_single_tensor_zcblk01(
            b"model.layers.3.mlp.gate.weight",
            TensorRole::Router,
            &[3, 2],
            &pack_i4_rows(&[&[1, 0], &[0, 2], &[3, 3]]),
            1.0,
            0.0,
        );
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 3];
        let gemm = FusedInt4Gemm;

        let routes =
            router_top_experts_from_dense_block(&block, &hidden, 2, &mut scratch, &gemm)
                .unwrap()
                .unwrap();

        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].expert_id, 2);
        assert_eq!(routes[1].expert_id, 1);
    }

    fn deepseek_route_options() -> RouteOptions {
        RouteOptions {
            math: RouterMath::DeepSeekV4SqrtSoftplus,
            route_scale: 1.5,
            hash_token_id: None,
        }
    }

    fn i64_rows(rows: &[&[i64]]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rows {
            for value in *row {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out
    }

    fn f32_values(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn deepseek_router_weights_are_sqrtsoftplus_sum_normalized_and_scaled() {
        // gate rows: e0 = [1, 0] -> logit 1; e1 = [0, 2] -> logit 4
        let block = make_single_tensor_zcblk01(
            b"layers.3.ffn.gate.weight",
            TensorRole::Router,
            &[2, 2],
            &pack_i4_rows(&[&[1, 0], &[0, 2]]),
            1.0,
            0.0,
        );
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;

        let routes = route_experts_from_dense_block(
            &block,
            &hidden,
            2,
            &[0, 1],
            &mut scratch,
            &gemm,
            deepseek_route_options(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].expert_id, 1);
        assert_eq!(routes[1].expert_id, 0);
        // scores: sqrt(softplus(4)) = 2.00452, sqrt(softplus(1)) = 1.14601
        // weights: score / sum * 1.5
        let s0 = (1f32.exp().ln_1p()).sqrt();
        let s1 = (4f32.exp().ln_1p()).sqrt();
        let sum = s0 + s1;
        assert_close(routes[0].score, s1 / sum * 1.5);
        assert_close(routes[1].score, s0 / sum * 1.5);
        let total: f32 = routes.iter().map(|route| route.score).sum();
        assert_close(total, 1.5);
    }

    #[test]
    fn deepseek_router_bias_shifts_selection_but_not_weights() {
        // Same gate as above, plus a large fp32 bias on expert 0: selection
        // order flips to [0, 1] but the weights still come from the
        // unbiased scores.
        let gate_data = pack_i4_rows(&[&[1, 0], &[0, 2]]);
        let bias_data = f32_values(&[10.0, 0.0]);
        let tensors = [
            (
                &b"layers.3.ffn.gate.weight"[..],
                TensorRole::Router,
                &[2u64, 2][..],
                &gate_data[..],
                12u16,
                4u16,
                1.0f32,
                0.0f32,
            ),
            (
                &b"layers.3.ffn.gate.bias"[..],
                TensorRole::Router,
                &[2u64][..],
                &bias_data[..],
                12u16,
                2405u16,
                1.0f32,
                0.0f32,
            ),
        ];
        let block = make_multi_tensor_zcblk01_with_formats(&tensors);
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;

        let routes = route_experts_from_dense_block(
            &block,
            &hidden,
            1,
            &[0, 1],
            &mut scratch,
            &gemm,
            deepseek_route_options(),
        )
        .unwrap()
        .unwrap();

        // Bias pushes expert 0 to the top of the selection despite the
        // lower gate score.
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].expert_id, 0);
        // Single selected expert: weight normalizes to route_scale.
        assert_close(routes[0].score, 1.5);
    }

    #[test]
    fn deepseek_hash_layer_routes_via_tid2eid_table() {
        let table = i64_rows(&[&[0, 0], &[0, 0], &[1, 0], &[0, 0]]);
        let tensors = [
            (
                &b"layers.0.ffn.gate.tid2eid"[..],
                TensorRole::Router,
                &[4u64, 2][..],
                &table[..],
                9u16,
                2405u16,
                1.0f32,
                0.0f32,
            ),
            (
                &b"layers.0.ffn.gate.weight"[..],
                TensorRole::Router,
                &[2u64, 2][..],
                &pack_i4_rows(&[&[1, 0], &[0, 2]])[..],
                12u16,
                4u16,
                1.0f32,
                0.0f32,
            ),
        ];
        let block = make_multi_tensor_zcblk01_with_formats(&tensors);
        let hidden = [1.0f32, 2.0];
        let mut scratch = [0.0f32; 2];
        let gemm = FusedInt4Gemm;
        let options = RouteOptions {
            math: RouterMath::DeepSeekV4SqrtSoftplus,
            route_scale: 1.5,
            hash_token_id: Some(2),
        };

        let routes = route_experts_from_dense_block(
            &block,
            &hidden,
            2,
            &[0, 1],
            &mut scratch,
            &gemm,
            options,
        )
        .unwrap()
        .unwrap();

        // Token 2 hashes to experts [1, 0] regardless of score order.
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].expert_id, 1);
        assert_eq!(routes[1].expert_id, 0);
        let total: f32 = routes.iter().map(|route| route.score).sum();
        assert_close(total, 1.5);
    }

    #[test]
    fn deepseek_mla_attention_single_token_uses_latent_kv_and_grouped_wo() {
        // Mini MLA config: hidden=4, 2 heads x head_dim 4 (nope 2 + rope 2),
        // q_lora=2, o_groups=2, o_lora=3 -> group_in=4, o_mid=6.
        // At token_pos=0 the softmax over one position gives weight 1 (no
        // sink tensor), RoPE is identity, so each head output equals the
        // normalized latent kv. wo_a copies kv[0..3] per group, wo_b picks
        // [kv0, kv1, kv2, kv0].
        let identity4 = bf16_rows(&[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
        ]);
        let wq_a = bf16_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let wq_b = bf16_rows(&[
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
        ]);
        let wo_a = bf16_rows(&[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
        ]);
        let wo_b = bf16_rows(&[
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        let tensors = [
            (
                &b"layers.0.attn.wq_a.weight"[..],
                TensorRole::QProj,
                &[2u64, 4][..],
                &wq_a[..],
                11u16,
                QUANT_DEEPSEEK_BF16_AUX as u16,
                1.0f32,
                0.0f32,
            ),
            (
                &b"layers.0.attn.wq_b.weight"[..],
                TensorRole::QProj,
                &[8u64, 2][..],
                &wq_b[..],
                11u16,
                QUANT_DEEPSEEK_BF16_AUX as u16,
                1.0f32,
                0.0f32,
            ),
            (
                &b"layers.0.attn.wkv.weight"[..],
                TensorRole::KvProj,
                &[4u64, 4][..],
                &identity4[..],
                11u16,
                QUANT_DEEPSEEK_BF16_AUX as u16,
                1.0f32,
                0.0f32,
            ),
            (
                &b"layers.0.attn.wo_a.weight"[..],
                TensorRole::OProj,
                &[6u64, 4][..],
                &wo_a[..],
                11u16,
                QUANT_DEEPSEEK_BF16_AUX as u16,
                1.0f32,
                0.0f32,
            ),
            (
                &b"layers.0.attn.wo_b.weight"[..],
                TensorRole::OProj,
                &[4u64, 6][..],
                &wo_b[..],
                11u16,
                QUANT_DEEPSEEK_BF16_AUX as u16,
                1.0f32,
                0.0f32,
            ),
        ];
        let block = make_multi_tensor_zcblk01_with_formats(&tensors);
        let layout = parse_quant_block(&block).unwrap();

        let config = ComputeConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_kv_heads: 1,
            qk_rope_head_dim: 2,
            qk_nope_head_dim: 2,
            v_head_dim: 4,
            q_lora_rank: 2,
            rope_theta: 10_000.0,
            kv_lora_rank: 2,
            prefill_chunk_tokens: 1,
            quant_group_size: 4,
            preferred_kernel: CpuKernel::Scalar,
            router_math: RouterMath::DeepSeekV4SqrtSoftplus,
            route_scale: 1.5,
            num_hash_layers: 3,
            attention_kind: AttentionKind::DeepSeekV4Mla,
            o_groups: 2,
            o_lora_rank: 3,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        };
        let mut hidden = [1.0f32, 2.0, 3.0, 4.0];
        let mut scratch_buf = vec![0.0f32; config.prefill_scratch_f32()];
        let mut scratch = ComputeScratch {
            dequant_tile_f32: &mut scratch_buf,
        };
        let mut kv_storage = vec![0.0f32; KVCache::required_f32(1, 2, 1, 4).unwrap()];
        let mut kv_cache = KVCache::from_scratch(&mut kv_storage, 1, 2, 1, 4).unwrap();

        let values = compute_deepseek_v4_mla_attention(
            &layout,
            0,
            0,
            &mut hidden,
            &mut kv_cache,
            &mut scratch,
            &config,
        )
        .unwrap();
        assert!(values > 0);

        // Expected latent kv: rms-normalized [1, 2, 3, 4], then the non-rope
        // dims (first nope_dim=2) go through the FP8 act quant simulation.
        let mean_square = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv_rms = 1.0 / (mean_square + 1.0e-6).sqrt();
        let mut kv = [1.0 * inv_rms, 2.0 * inv_rms, 3.0 * inv_rms, 4.0 * inv_rms];
        fp8_act_quant_dequant_in_place(&mut kv[..2], 64);
        // The attention output passes the QAT act quant before wo_a and
        // before wo_b (identity projections here), so the expected values
        // are the quantized attention outputs.
        let mut expected_out = [kv[0], kv[1], kv[2], kv[0]];
        fp8_act_quant_dequant_in_place(&mut expected_out, 64);
        assert_close(hidden[0], expected_out[0]);
        assert_close(hidden[1], expected_out[1]);
        assert_close(hidden[2], expected_out[2]);
        assert_close(hidden[3], expected_out[3]);
        // KV cache must hold the latent vector for position 0.
        let cached = kv_cache.key_slice(0, 0).unwrap();
        assert_close(cached[0], kv[0]);
        assert_close(cached[3], kv[3]);
    }

    #[test]
    fn hc_sinkhorn_split_produces_doubly_normalized_comb() {
        let hc = 4;
        let mix_hc = (2 + hc) * hc;
        let mixes: Vec<f32> = (0..mix_hc).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let scale = [0.5f32, 0.7, 0.9];
        let base: Vec<f32> = (0..mix_hc).map(|i| (i as f32) * 0.01).collect();

        let (pre, post, comb) =
            hc_split_sinkhorn_scalar(&mixes, &scale, &base, hc, 20, 1.0e-6).unwrap();

        assert_eq!(pre.len(), hc);
        assert_eq!(post.len(), hc);
        assert_eq!(comb.len(), hc * hc);
        for value in &pre {
            assert!(*value > 0.0 && *value < 1.0 + 1.0e-5);
        }
        for value in &post {
            assert!(*value > 0.0 && *value < 2.0);
        }
        // After 20 sinkhorn rounds rows and columns must both sum to ~1.
        for j in 0..hc {
            let row_sum: f32 = comb[j * hc..(j + 1) * hc].iter().sum();
            assert!((row_sum - 1.0).abs() < 1.0e-3, "row {j} sum {row_sum}");
        }
        for k in 0..hc {
            let col_sum: f32 = (0..hc).map(|j| comb[j * hc + k]).sum();
            assert!((col_sum - 1.0).abs() < 1.0e-3, "col {k} sum {col_sum}");
        }
    }

    #[test]
    fn hc_pre_and_post_roundtrip_shapes() {
        let hc = 2;
        let hidden = 3;
        let hc_dim = hc * hidden;
        let mix_hc = (2 + hc) * hc;
        // fn rows chosen small; values arbitrary
        let mut fn_data = Vec::new();
        for j in 0..mix_hc {
            for d in 0..hc_dim {
                fn_data.extend_from_slice(&(((j + d) as f32) * 0.05).to_le_bytes());
            }
        }
        let params = HcParams {
            fn_data: &fn_data,
            scale: vec![0.4, 0.6, 0.8],
            base: (0..mix_hc).map(|i| i as f32 * 0.02).collect(),
            mix_hc,
            hc_dim,
        };
        let x: Vec<f32> = (0..hc_dim).map(|i| (i as f32) * 0.3 - 0.5).collect();
        let mut y = vec![0.0f32; hidden];

        let (post, comb) = hc_pre_into(&params, &x, &mut y, hc, hidden, 20, 1.0e-6).unwrap();
        assert!(y.iter().all(|value| value.is_finite()));

        let mut out = vec![0.0f32; hc_dim];
        hc_post_into(&y, &x, &post, &comb, &mut out, hc, hidden).unwrap();
        assert!(out.iter().all(|value| value.is_finite()));
        // out[k] = post[k]*y + sum_i comb[i][k]*x_i - check one element.
        let expected = post[1] * y[0] + comb[1] * x[0] + comb[hc + 1] * x[hidden];
        assert_close(out[hidden], expected);
    }

    #[test]
    fn inverse_rope_undoes_forward_rope() {
        let mut values = [0.3f32, -1.2, 0.7, 2.5];
        let original = values;
        apply_rope_interleaved_in_place(&mut values, 5, 4, 10_000.0).unwrap();
        apply_rope_interleaved_inverse_in_place(&mut values, 5, 4, 10_000.0).unwrap();
        for (actual, expected) in values.iter().zip(original.iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn fp8_act_quant_dequant_preserves_representable_values() {
        // amax = 2.0 -> scale = 2^ceil(log2(2/448)) = 2^-7 = 0.0078125.
        // 2.0 / scale = 256, exactly representable in E4M3 -> unchanged.
        let mut values = [2.0f32, -1.0, 0.5, 0.25];
        fp8_act_quant_dequant_in_place(&mut values, 4);
        assert_close(values[0], 2.0);
        assert_close(values[1], -1.0);
        assert_close(values[2], 0.5);
        assert_close(values[3], 0.25);
    }

    #[test]
    fn fp8_act_quant_dequant_rounds_to_e4m3_grid() {
        // With amax 448 the scale is exactly 1.0; 300.0 is not representable
        // in E4M3 (step 32 in [256, 448]) and must round to a grid point.
        let mut values = [448.0f32, 300.0, 0.0, 0.0];
        fp8_act_quant_dequant_in_place(&mut values, 4);
        assert_close(values[0], 448.0);
        assert!((values[1] - 288.0).abs() < 1.0e-3 || (values[1] - 320.0).abs() < 1.0e-3);
        assert!(values[1] != 300.0);
    }

    #[test]
    fn yarn_scales_low_frequencies_and_keeps_high_ones() {
        let plain = rope_inv_freqs(64, DEEPSEEK_V4_COMPRESS_ROPE_THETA, None);
        let yarn = rope_inv_freqs(64, DEEPSEEK_V4_COMPRESS_ROPE_THETA, Some(DEEPSEEK_V4_YARN));
        assert_eq!(plain.len(), 32);
        assert_eq!(yarn.len(), 32);
        // Highest-frequency pair (index 0) is below the correction range:
        // unchanged.
        assert_close(yarn[0], plain[0]);
        // Lowest-frequency pair (index 31) is above the range: scaled by
        // 1/factor.
        assert_close(yarn[31], plain[31] / DEEPSEEK_V4_YARN.factor);
        // In-between pairs are a blend: strictly between the two extremes.
        let mid = 20usize;
        assert!(yarn[mid] < plain[mid]);
        assert!(yarn[mid] > plain[mid] / DEEPSEEK_V4_YARN.factor);
    }

    #[test]
    fn deepseek_layer_rope_selection_follows_compress_ratios() {
        assert_eq!(deepseek_rope_for_layer(0).0, 10_000.0);
        assert!(deepseek_rope_for_layer(0).1.is_none());
        assert_eq!(deepseek_rope_for_layer(1).0, 10_000.0);
        assert_eq!(deepseek_rope_for_layer(2).0, DEEPSEEK_V4_COMPRESS_ROPE_THETA);
        assert!(deepseek_rope_for_layer(2).1.is_some());
        assert_eq!(deepseek_rope_for_layer(42).0, DEEPSEEK_V4_COMPRESS_ROPE_THETA);
        assert_eq!(deepseek_rope_for_layer(43).0, 10_000.0);
    }

    #[test]
    fn rope_with_freqs_matches_plain_rope_and_inverts() {
        let inv = rope_inv_freqs(4, 10_000.0, None);
        let mut with_freqs = [0.3f32, -1.2, 0.7, 2.5];
        let mut plain = with_freqs;
        apply_rope_interleaved_with_freqs(&mut with_freqs, 7, &inv, false).unwrap();
        apply_rope_interleaved_in_place(&mut plain, 7, 4, 10_000.0).unwrap();
        for (a, b) in with_freqs.iter().zip(plain.iter()) {
            assert_close(*a, *b);
        }
        apply_rope_interleaved_with_freqs(&mut with_freqs, 7, &inv, true).unwrap();
        assert_close(with_freqs[0], 0.3);
        assert_close(with_freqs[3], 2.5);
    }

    #[test]
    fn deepseek_fp4_expert_bridge_reads_zcblk_tensors() {
        let block = make_deepseek_fp4_test_expert_zcblk();
        let layout = parse_quant_block(&block).unwrap();
        let expert = deepseek_v4_fp4_expert_from_quant_block(&layout).unwrap();

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        let mut scratch = [0.0f32; 4];
        crate::deepseek_v4::deepseek_v4_fp4_expert_forward_scalar(
            expert,
            &input,
            &mut output,
            &mut scratch,
        )
        .unwrap();

        // The swiglu hidden goes through the QAT act quant before w2.
        let mut hidden_q = [2.0 / (1.0 + (-2.0f32).exp()) * 3.0];
        fp8_act_quant_dequant_in_place(&mut hidden_q, 64);
        let hidden = hidden_q[0];
        assert_close(output[0], hidden);
        assert_close(output[1], 2.0 * hidden);
    }

    #[test]
    fn deepseek_fp4_expert_dispatch_runs_from_compute_expert_block() {
        let block = make_deepseek_fp4_test_expert_zcblk();
        let mut hidden = [2.0f32, 3.0];
        let mut scratch_buf = [0.0f32; 6];
        let mut scratch = ComputeScratch {
            dequant_tile_f32: &mut scratch_buf,
        };
        let config = ComputeConfig {
            hidden_size: 2,
            intermediate_size: 1,
            num_attention_heads: 1,
            num_kv_heads: 1,
            qk_rope_head_dim: 2,
            qk_nope_head_dim: 0,
            v_head_dim: 2,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            rope_theta: 10_000.0,
            prefill_chunk_tokens: 1,
            quant_group_size: 4,
            preferred_kernel: CpuKernel::Scalar,
            router_math: RouterMath::RawLogits,
            route_scale: 1.0,
            num_hash_layers: 0,
            attention_kind: AttentionKind::GlmDsaProbe,
            o_groups: 0,
            o_lora_rank: 0,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        };
        let gemm = FusedInt4Gemm;
        let mut stats = LayerComputeStats::default();

        compute_expert_block(&block, &mut hidden, &mut scratch, &config, &gemm, &mut stats)
            .unwrap();

        // The swiglu hidden goes through the QAT act quant before w2.
        let mut expected_q = [2.0 / (1.0 + (-2.0f32).exp()) * 3.0];
        fp8_act_quant_dequant_in_place(&mut expected_q, 64);
        let expected = expected_q[0];
        assert_close(hidden[0], expected);
        assert_close(hidden[1], 2.0 * expected);
        assert_eq!(stats.dequantized_values, 12);
    }

    #[test]
    fn deepseek_norm_names_match_glm_compat_markers() {
        assert!(norm_name_matches_marker(
            "layers.0.attn_norm.weight",
            "input_layernorm"
        ));
        assert!(norm_name_matches_marker(
            "layers.0.ffn_norm.weight",
            "post_attention_layernorm"
        ));
        assert!(norm_name_matches_marker(
            "layers.0.attn.q_norm.weight",
            "q_a_layernorm"
        ));
        assert!(norm_name_matches_marker(
            "layers.0.attn.kv_norm.weight",
            "kv_a_layernorm"
        ));
    }

    #[test]
    fn multi_expert_aggregation_uses_route_weights() {
        let first = make_single_tensor_zcblk01(
            b"expert0.up_proj.weight",
            TensorRole::UpProj,
            &[2, 2],
            &pack_i4_rows(&[&[1, 0], &[0, 0]]),
            1.0,
            0.0,
        );
        let second = make_single_tensor_zcblk01(
            b"expert1.up_proj.weight",
            TensorRole::UpProj,
            &[2, 2],
            &pack_i4_rows(&[&[0, 0], &[0, 4]]),
            1.0,
            0.0,
        );
        let experts = [
            IoBlockPtr {
                kind: BlockKind::Expert,
                layer_id: 0,
                expert_id: 0,
                route_weight: 0.75,
                ptr: first.as_ptr(),
                len: first.len(),
            },
            IoBlockPtr {
                kind: BlockKind::Expert,
                layer_id: 0,
                expert_id: 1,
                route_weight: 0.25,
                ptr: second.as_ptr(),
                len: second.len(),
            },
        ];
        let mut hidden = [1.0f32, 1.0];
        let mut scratch_buf = [0.0f32; 16];
        let mut scratch = ComputeScratch {
            dequant_tile_f32: &mut scratch_buf,
        };
        let config = ComputeConfig {
            hidden_size: 2,
            intermediate_size: 4,
            num_attention_heads: 1,
            num_kv_heads: 1,
            qk_rope_head_dim: 2,
            qk_nope_head_dim: 0,
            v_head_dim: 2,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            rope_theta: 10_000.0,
            prefill_chunk_tokens: 1,
            quant_group_size: 4,
            preferred_kernel: CpuKernel::Scalar,
            router_math: RouterMath::RawLogits,
            route_scale: 1.0,
            num_hash_layers: 0,
            attention_kind: AttentionKind::GlmDsaProbe,
            o_groups: 0,
            o_lora_rank: 0,
            hc_mult: 1,
            hc_sinkhorn_iters: 20,
            hc_eps: 1.0e-6,
        };
        let gemm = FusedInt4Gemm;
        let mut stats = LayerComputeStats::default();

        unsafe {
            compute_active_experts(
                &experts,
                &mut hidden,
                &mut scratch,
                &config,
                &gemm,
                &mut stats,
            )
        }
        .unwrap();

        let silu_one = 1.0 / (1.0 + (-1.0f32).exp());
        assert_close(hidden[0], silu_one * 0.75);
        assert_close(hidden[1], silu_one * 4.0 * 0.25);
    }

    #[test]
    fn token_embedding_reads_requested_row() {
        let block = make_single_tensor_zcblk01(
            b"model.embed_tokens.weight",
            TensorRole::Embed,
            &[2, 4],
            &pack_i4_rows(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]),
            0.5,
            0.0,
        );
        let mut hidden = [0.0f32; 4];

        let found = token_embedding_from_block(&block, 1, &mut hidden).unwrap();

        assert!(found);
        assert_close(hidden[0], 2.5);
        assert_close(hidden[1], 3.0);
        assert_close(hidden[2], 3.5);
        assert_close(hidden[3], 4.0);
    }

    #[test]
    fn token_embedding_reads_deepseek_bf16_row() {
        let block = make_single_tensor_zcblk01_with_format(
            b"embed.weight",
            TensorRole::Embed,
            &[2, 4],
            &bf16_rows(&[&[1.0, -2.0, 3.5, 4.0], &[0.25, 0.5, -0.75, 1.25]]),
            11,
            QUANT_DEEPSEEK_BF16_AUX,
            1.0,
            0.0,
        );
        let mut hidden = [0.0f32; 4];

        let found = token_embedding_from_block(&block, 1, &mut hidden).unwrap();

        assert!(found);
        assert_close(hidden[0], 0.25);
        assert_close(hidden[1], 0.5);
        assert_close(hidden[2], -0.75);
        assert_close(hidden[3], 1.25);
    }

    #[test]
    fn token_embedding_skips_rank1_embed_metadata() {
        let block = make_two_tensor_zcblk01(
            (
                b"model.embed_tokens.bias_like_metadata",
                TensorRole::Embed,
                &[2],
                &[0x10],
                1.0,
                0.0,
            ),
            (
                b"model.embed_tokens.weight",
                TensorRole::Embed,
                &[2, 4],
                &pack_i4_rows(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]),
                0.5,
                0.0,
            ),
        );
        let mut hidden = [0.0f32; 4];

        let found = token_embedding_from_block(&block, 1, &mut hidden).unwrap();

        assert!(found);
        assert_close(hidden[0], 2.5);
        assert_close(hidden[3], 4.0);
    }

    #[test]
    fn token_embedding_out_of_vocab_falls_back_cleanly() {
        let block = make_single_tensor_zcblk01(
            b"model.embed_tokens.weight",
            TensorRole::Embed,
            &[2, 4],
            &pack_i4_rows(&[&[1, 2, 3, 4], &[5, 6, 7, 8]]),
            0.5,
            0.0,
        );
        let mut hidden = [9.0f32; 4];

        let found = token_embedding_from_block(&block, 154826, &mut hidden).unwrap();

        assert!(!found);
        assert_close(hidden[0], 9.0);
        assert_close(hidden[3], 9.0);
    }

    #[test]
    fn token_embedding_reads_global_row_shard() {
        let block = make_single_tensor_zcblk01(
            b"model.embed_tokens.weight.rows_154824_154826",
            TensorRole::Embed,
            &[2, 2],
            &pack_i4_rows(&[&[1, 2], &[3, 4]]),
            1.0,
            0.0,
        );
        let mut hidden = [0.0f32; 2];

        assert!(!token_embedding_from_block(&block, 154823, &mut hidden).unwrap());
        assert!(token_embedding_from_block(&block, 154825, &mut hidden).unwrap());
        assert_close(hidden[0], 3.0);
        assert_close(hidden[1], 4.0);
    }

    #[test]
    fn expert_cache_replaces_truncated_local_file() {
        let base = std::env::temp_dir().join(format!(
            "zc_expert_cache_replace_{}_{}",
            std::process::id(),
            now_ms()
        ));
        let source_dir = base.join("source");
        let cache_dir = base.join("cache");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::create_dir_all(&cache_dir).unwrap();
        let source = source_dir.join("layer34_expert1.zcblk");
        let cached = cache_dir.join("layer34_expert1.zcblk");
        std::fs::write(&source, vec![7u8; 4096]).unwrap();
        std::fs::write(&cached, vec![3u8; 1024]).unwrap();

        let mut cache = ExpertLruCache::new(ExpertCacheConfig {
            cache_dir: cache_dir.clone(),
            max_bytes: 1 << 20,
            remote_endpoint: None,
        })
        .unwrap();

        let path = cache
            .ensure_expert(34, 1, Some(source.as_path()), Some(4096))
            .unwrap();
        let metadata = std::fs::metadata(path).unwrap();
        assert_eq!(metadata.len(), 4096);

        std::fs::remove_dir_all(base).unwrap();
    }

    fn pack_i4_rows(rows: &[&[u8]]) -> Vec<u8> {
        let mut packed = Vec::new();
        for row in rows {
            let mut index = 0usize;
            while index < row.len() {
                let low = row[index] & 0x0f;
                let high = row.get(index + 1).copied().unwrap_or(0) & 0x0f;
                packed.push(low | (high << 4));
                index += 2;
            }
        }
        packed
    }

    fn assert_close(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1.0e-5,
            "actual={actual}, expected={expected}, diff={diff}"
        );
    }

    fn make_test_zcblk01() -> Vec<u8> {
        make_single_tensor_zcblk01(
            b"expert.up_proj.weight",
            TensorRole::UpProj,
            &[2, 4],
            &[0x21, 0x43, 0x34, 0x12],
            0.5,
            0.0,
        )
    }

    fn make_single_tensor_zcblk01(
        name: &[u8],
        role: TensorRole,
        shape: &[u64],
        data: &[u8],
        scale: f32,
        zero_point: f32,
    ) -> Vec<u8> {
        make_single_tensor_zcblk01_with_format(name, role, shape, data, 12, 4, scale, zero_point)
    }

    fn make_single_tensor_zcblk01_with_format(
        name: &[u8],
        role: TensorRole,
        shape: &[u64],
        data: &[u8],
        dtype_original: u16,
        quant_format: u16,
        scale: f32,
        zero_point: f32,
    ) -> Vec<u8> {
        let name_offset = 0u64;
        let names_offset = QUANT_BLOCK_HEADER_SIZE + QUANT_TENSOR_RECORD_SIZE;
        let data_offset = names_offset + 2 + name.len();
        let shape_offset = data_offset + data.len();

        let mut block = Vec::new();
        block.extend_from_slice(&QUANT_BLOCK_MAGIC);
        push_u32(&mut block, QUANT_BLOCK_VERSION);
        push_u32(&mut block, 1);
        push_u32(&mut block, quant_format as u32);
        push_u32(&mut block, 0);
        push_u64(&mut block, QUANT_BLOCK_HEADER_SIZE as u64);
        push_u64(&mut block, names_offset as u64);

        push_u16(&mut block, dtype_original);
        push_u16(&mut block, quant_format);
        push_u32(&mut block, shape.len() as u32);
        push_u32(&mut block, role.code() as u32);
        push_u64(&mut block, name_offset);
        push_u64(&mut block, shape_offset as u64);
        push_u64(&mut block, data_offset as u64);
        push_u64(&mut block, data.len() as u64);
        push_f32(&mut block, scale);
        push_f32(&mut block, zero_point);

        push_u16(&mut block, name.len() as u16);
        block.extend_from_slice(name);
        block.extend_from_slice(data);
        push_u32(&mut block, shape.len() as u32);
        for dim in shape {
            push_u64(&mut block, *dim);
        }

        block
    }

    fn make_deepseek_fp4_test_expert_zcblk() -> Vec<u8> {
        let w_shape = [2u64, 1];
        let w2_shape = [2u64, 1];
        let s_shape = [2u64, 1];
        let s2_shape = [2u64, 1];
        let w1_weight = [0x02u8, 0x00u8];
        let w1_scale = [127u8, 127u8];
        let w3_weight = [0x20u8, 0x00u8];
        let w3_scale = [127u8, 127u8];
        let w2_weight = [0x02u8, 0x04u8];
        let w2_scale = [127u8, 127u8];
        let tensors = [
            (
                &b"model.layers.0.mlp.experts.0.w1.weight"[..],
                TensorRole::UpProj,
                &w_shape[..],
                &w1_weight[..],
                12,
                QUANT_DEEPSEEK_FP4_E2M1_PACKED,
                1.0,
                0.0,
            ),
            (
                &b"model.layers.0.mlp.experts.0.w1.scale"[..],
                TensorRole::UpProj,
                &s_shape[..],
                &w1_scale[..],
                1,
                QUANT_DEEPSEEK_UE8M0_SCALE,
                1.0,
                0.0,
            ),
            (
                &b"model.layers.0.mlp.experts.0.w3.weight"[..],
                TensorRole::GateProj,
                &w_shape[..],
                &w3_weight[..],
                12,
                QUANT_DEEPSEEK_FP4_E2M1_PACKED,
                1.0,
                0.0,
            ),
            (
                &b"model.layers.0.mlp.experts.0.w3.scale"[..],
                TensorRole::GateProj,
                &s_shape[..],
                &w3_scale[..],
                1,
                QUANT_DEEPSEEK_UE8M0_SCALE,
                1.0,
                0.0,
            ),
            (
                &b"model.layers.0.mlp.experts.0.w2.weight"[..],
                TensorRole::DownProj,
                &w2_shape[..],
                &w2_weight[..],
                12,
                QUANT_DEEPSEEK_FP4_E2M1_PACKED,
                1.0,
                0.0,
            ),
            (
                &b"model.layers.0.mlp.experts.0.w2.scale"[..],
                TensorRole::DownProj,
                &s2_shape[..],
                &w2_scale[..],
                1,
                QUANT_DEEPSEEK_UE8M0_SCALE,
                1.0,
                0.0,
            ),
        ];
        make_multi_tensor_zcblk01_with_formats(&tensors)
    }

    fn bf16_rows(rows: &[&[f32]]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rows {
            for value in *row {
                let raw = (value.to_bits() >> 16) as u16;
                out.extend_from_slice(&raw.to_le_bytes());
            }
        }
        out
    }

    type TestTensor<'a> = (&'a [u8], TensorRole, &'a [u64], &'a [u8], f32, f32);
    type TestTensorWithFormat<'a> = (
        &'a [u8],
        TensorRole,
        &'a [u64],
        &'a [u8],
        u16,
        u16,
        f32,
        f32,
    );

    fn make_two_tensor_zcblk01(first: TestTensor<'_>, second: TestTensor<'_>) -> Vec<u8> {
        let tensors = [first, second];
        make_multi_tensor_zcblk01(&tensors)
    }

    fn make_three_tensor_zcblk01(
        first: TestTensor<'_>,
        second: TestTensor<'_>,
        third: TestTensor<'_>,
    ) -> Vec<u8> {
        let tensors = [first, second, third];
        make_multi_tensor_zcblk01(&tensors)
    }

    fn make_multi_tensor_zcblk01(tensors: &[TestTensor<'_>]) -> Vec<u8> {
        let formatted = tensors
            .iter()
            .map(|(name, role, shape, data, scale, zero_point)| {
                (
                    *name,
                    *role,
                    *shape,
                    *data,
                    12,
                    4,
                    *scale,
                    *zero_point,
                )
            })
            .collect::<Vec<_>>();
        make_multi_tensor_zcblk01_with_formats(&formatted)
    }

    fn make_multi_tensor_zcblk01_with_formats(tensors: &[TestTensorWithFormat<'_>]) -> Vec<u8> {
        let names_offset = QUANT_BLOCK_HEADER_SIZE + tensors.len() * QUANT_TENSOR_RECORD_SIZE;
        let mut names = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _, _, _, _, _, _, _) in tensors {
            name_offsets.push(names.len() as u64);
            push_u16(&mut names, name.len() as u16);
            names.extend_from_slice(name);
        }
        let mut cursor = names_offset + names.len();
        let mut data_offsets = Vec::new();
        let mut shape_offsets = Vec::new();
        for (_, _, shape, data, _, _, _, _) in tensors {
            data_offsets.push(cursor as u64);
            cursor += data.len();
            shape_offsets.push(cursor as u64);
            cursor += 4 + shape.len() * 8;
        }

        let mut block = Vec::new();
        block.extend_from_slice(&QUANT_BLOCK_MAGIC);
        push_u32(&mut block, QUANT_BLOCK_VERSION);
        push_u32(&mut block, tensors.len() as u32);
        push_u32(&mut block, 4);
        push_u32(&mut block, 0);
        push_u64(&mut block, QUANT_BLOCK_HEADER_SIZE as u64);
        push_u64(&mut block, names_offset as u64);

        for (
            index,
            (name, role, shape, data, dtype_original, quant_format, scale, zero_point),
        ) in tensors.iter().enumerate()
        {
            let _ = name;
            push_u16(&mut block, *dtype_original);
            push_u16(&mut block, *quant_format);
            push_u32(&mut block, shape.len() as u32);
            push_u32(&mut block, role.code() as u32);
            push_u64(&mut block, name_offsets[index]);
            push_u64(&mut block, shape_offsets[index]);
            push_u64(&mut block, data_offsets[index]);
            push_u64(&mut block, data.len() as u64);
            push_f32(&mut block, *scale);
            push_f32(&mut block, *zero_point);
        }
        block.extend_from_slice(&names);
        for (_, _, shape, data, _, _, _, _) in tensors {
            block.extend_from_slice(data);
            push_u32(&mut block, shape.len() as u32);
            for dim in *shape {
                push_u64(&mut block, *dim);
            }
        }
        block
    }

    fn mtp_test_config(hidden: usize, hc: usize) -> ComputeConfig {
        let mut config = ComputeConfig::deepseek_v4_flash();
        config.hidden_size = hidden;
        config.hc_mult = hc;
        config
    }

    fn synth<'a>(
        name: &'a str,
        role_code: u32,
        shape: &'a [u64],
        data: &'a [u8],
        dtype: u16,
        quant: u16,
    ) -> SyntheticTensor<'a> {
        SyntheticTensor {
            name,
            role_code,
            shape,
            data,
            dtype_original: dtype,
            quant_format: quant,
            scale: 1.0,
            zero_point: 0.0,
        }
    }

    #[test]
    fn mtp_synthetic_role_codes_follow_main_layer_conventions() {
        let cases: &[(&str, u16)] = &[
            ("mtp.0.attn.wq_a.weight", TensorRole::QProj.code()),
            ("mtp.0.attn.wq_b.weight", TensorRole::QProj.code()),
            ("mtp.0.attn.wkv.weight", TensorRole::KvProj.code()),
            ("mtp.0.attn.wo_a.weight", TensorRole::OProj.code()),
            ("mtp.0.attn.wo_b.weight", TensorRole::OProj.code()),
            ("mtp.0.attn.wq_a.scale", 0),
            ("mtp.0.attn_norm.weight", TensorRole::Norm.code()),
            ("mtp.0.ffn_norm.weight", TensorRole::Norm.code()),
            ("mtp.0.attn.q_norm.weight", TensorRole::Norm.code()),
            ("mtp.0.attn.kv_norm.weight", TensorRole::Norm.code()),
            ("mtp.0.enorm.weight", TensorRole::Norm.code()),
            ("mtp.0.hnorm.weight", TensorRole::Norm.code()),
            ("mtp.0.norm.weight", TensorRole::Norm.code()),
            ("mtp.0.ffn.gate.weight", TensorRole::Router.code()),
            ("mtp.0.ffn.gate.bias", TensorRole::Router.code()),
            (
                "mtp.0.ffn.shared_experts.w1.weight",
                TensorRole::SharedExpert.code(),
            ),
            (
                "mtp.0.ffn.shared_experts.w2.scale",
                TensorRole::SharedExpert.code(),
            ),
            ("mtp.0.hc_attn_fn", 0),
            ("mtp.0.hc_head_scale", 0),
            ("mtp.0.attn.attn_sink", 0),
            ("mtp.0.e_proj.weight", 0),
            ("mtp.0.h_proj.weight", 0),
        ];
        for (name, expected) in cases {
            assert_eq!(
                mtp_synthetic_role_code(name, 0),
                *expected,
                "role for {name}"
            );
        }
    }

    #[test]
    fn mtp_prepare_draft_hidden_matches_reference_math() {
        // hidden=4, hc=2; bf16 weights so no FP8 act-quant noise. e_proj and
        // h_proj are identity matrices: out_c = enorm(embed) + hnorm(x_c).
        let hidden = 4usize;
        let identity = bf16_rows(&[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
        ]);
        let enorm = bf16_rows(&[&[1.0, 1.0, 1.0, 1.0]]);
        let hnorm = bf16_rows(&[&[2.0, 2.0, 2.0, 2.0]]);
        let w_shape: [u64; 2] = [4, 4];
        let n_shape: [u64; 1] = [4];
        let tensors = [
            synth("mtp.0.enorm.weight", 10, &n_shape, &enorm, 11, QUANT_DEEPSEEK_BF16_AUX),
            synth("mtp.0.hnorm.weight", 10, &n_shape, &hnorm, 11, QUANT_DEEPSEEK_BF16_AUX),
            synth("mtp.0.e_proj.weight", 0, &w_shape, &identity, 11, QUANT_DEEPSEEK_BF16_AUX),
            synth("mtp.0.h_proj.weight", 0, &w_shape, &identity, 11, QUANT_DEEPSEEK_BF16_AUX),
        ];
        let block = build_synthetic_quant_block(&tensors);
        let config = mtp_test_config(hidden, 2);

        let embed = [1.0f32, 2.0, 3.0, 4.0];
        let main_hidden = [
            1.0f32, 0.0, 0.0, 0.0, // copy 0
            0.0, 2.0, 0.0, 0.0, // copy 1
        ];
        let mut out = [0.0f32; 8];
        let mut scratch_buf = vec![0.0f32; 64];
        let mut scratch = ComputeScratch {
            dequant_tile_f32: &mut scratch_buf,
        };
        mtp_prepare_draft_hidden(
            &block,
            &embed,
            &main_hidden,
            &mut out,
            &mut scratch,
            &config,
            &FusedInt4Gemm,
        )
        .unwrap();

        let rms = |values: &[f32]| {
            let mean = values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32;
            1.0 / (mean + 1.0e-6).sqrt()
        };
        let inv_e = rms(&embed);
        let e_out: Vec<f32> = embed.iter().map(|v| v * inv_e).collect();
        for copy in 0..2 {
            let src = &main_hidden[copy * hidden..(copy + 1) * hidden];
            let inv_h = rms(src);
            for d in 0..hidden {
                let expected = e_out[d] + src[d] * inv_h * 2.0;
                let actual = out[copy * hidden + d];
                assert!(
                    (actual - expected).abs() < 2.0e-2,
                    "copy={copy} d={d} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn mtp_head_pool_and_norm_pools_with_block_params() {
        // hidden=2, hc=2. hc_head_fn = 0 rows -> pre-weights =
        // sigmoid(base) + eps; final norm weight all ones -> plain RMS norm
        // of the pooled vector.
        let hidden = 2usize;
        let hc = 2usize;
        let fn_rows = vec![0u8; hc * hc * hidden * 4];
        let scale_raw = 1.0f32.to_le_bytes().to_vec();
        let base_raw: Vec<u8> = [0.0f32, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let norm = bf16_rows(&[&[1.0, 1.0]]);
        let fn_shape: [u64; 2] = [2, 4];
        let one_shape: [u64; 1] = [1];
        let base_shape: [u64; 1] = [2];
        let norm_shape: [u64; 1] = [2];
        let tensors = [
            synth(
                "mtp.0.hc_head_fn",
                0,
                &fn_shape,
                &fn_rows,
                12,
                crate::deepseek_v4::QUANT_DEEPSEEK_F32_AUX,
            ),
            synth(
                "mtp.0.hc_head_scale",
                0,
                &one_shape,
                &scale_raw,
                12,
                crate::deepseek_v4::QUANT_DEEPSEEK_F32_AUX,
            ),
            synth(
                "mtp.0.hc_head_base",
                0,
                &base_shape,
                &base_raw,
                12,
                crate::deepseek_v4::QUANT_DEEPSEEK_F32_AUX,
            ),
            synth("mtp.0.norm.weight", 10, &norm_shape, &norm, 11, QUANT_DEEPSEEK_BF16_AUX),
        ];
        let block = build_synthetic_quant_block(&tensors);
        let config = mtp_test_config(hidden, hc);

        let hidden_hc = [1.0f32, 2.0, 3.0, 4.0];
        let mut pooled = [0.0f32; 2];
        mtp_head_pool_and_norm(&block, "mtp.0.norm.weight", &hidden_hc, &mut pooled, &config)
            .unwrap();

        let eps = config.hc_eps;
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let w0 = sigmoid(0.0) + eps;
        let w1 = sigmoid(1.0) + eps;
        let raw = [
            w0 * hidden_hc[0] + w1 * hidden_hc[2],
            w0 * hidden_hc[1] + w1 * hidden_hc[3],
        ];
        let mean = (raw[0] * raw[0] + raw[1] * raw[1]) / 2.0;
        let inv_rms = 1.0 / (mean + 1.0e-6).sqrt();
        for d in 0..2 {
            let expected = raw[d] * inv_rms;
            assert!(
                (pooled[d] - expected).abs() < 1.0e-4,
                "d={d} actual={} expected={expected}",
                pooled[d]
            );
        }
    }

    fn make_compressor_test_block() -> Vec<u8> {
        // head_dim=4, nope_dim=2, ratio=4 (layer 40), coff=2 -> proj_dim=8.
        let mut wkv_rows: Vec<Vec<f32>> = Vec::new();
        let mut wgate_rows: Vec<Vec<f32>> = Vec::new();
        for row in 0..8 {
            let r = row as f32;
            wkv_rows.push(vec![0.5, -0.25, r * 0.125, 1.0]);
            wgate_rows.push(vec![-0.5, r * 0.0625, 0.25, -1.0]);
        }
        let wkv: Vec<&[f32]> = wkv_rows.iter().map(|row| row.as_slice()).collect();
        let wgate: Vec<&[f32]> = wgate_rows.iter().map(|row| row.as_slice()).collect();
        let wkv_data = bf16_rows(&wkv);
        let wgate_data = bf16_rows(&wgate);
        let ape: Vec<u8> = (0..32u32)
            .flat_map(|index| (0.03125f32 * index as f32 - 0.5).to_le_bytes())
            .collect();
        let norm = bf16_rows(&[&[1.0, 0.5, 2.0, 1.0]]);
        let w_shape: [u64; 2] = [8, 4];
        let ape_shape: [u64; 1] = [32];
        let norm_shape: [u64; 1] = [4];
        let tensors = [
            synth(
                "layers.40.attn.compressor.wkv.weight",
                0,
                &w_shape,
                &wkv_data,
                11,
                QUANT_DEEPSEEK_BF16_AUX,
            ),
            synth(
                "layers.40.attn.compressor.wgate.weight",
                0,
                &w_shape,
                &wgate_data,
                11,
                QUANT_DEEPSEEK_BF16_AUX,
            ),
            synth(
                "layers.40.attn.compressor.ape",
                0,
                &ape_shape,
                &ape,
                12,
                crate::deepseek_v4::QUANT_DEEPSEEK_F32_AUX,
            ),
            synth(
                "layers.40.attn.compressor.norm.weight",
                10,
                &norm_shape,
                &norm,
                11,
                QUANT_DEEPSEEK_BF16_AUX,
            ),
        ];
        build_synthetic_quant_block(&tensors)
    }

    fn compressor_state_capture(
        layer: usize,
    ) -> (Vec<f32>, Vec<f32>, usize, Option<(Vec<f32>, Vec<f32>)>, Vec<f32>, Option<usize>) {
        let map = deepseek_compressor_state().lock().unwrap();
        let state = map.get(&layer).expect("compressor state present");
        (
            state.pending_kv.clone(),
            state.pending_score.clone(),
            state.pending_len,
            state.prev_first_half.clone(),
            state.compressed.clone(),
            state.last_fed_pos,
        )
    }

    #[test]
    fn compressor_speculation_rollback_is_exact_across_flush() {
        let block = make_compressor_test_block();
        let layout = parse_quant_block(&block).unwrap();
        let inv_freqs = rope_inv_freqs(2, 10_000.0, None);
        let x = |pos: usize| {
            vec![
                0.5 * pos as f32 + 0.25,
                -0.5,
                0.125 * pos as f32,
                0.0625,
            ]
        };
        let feed = |pos: usize| {
            deepseek_compressor_feed_and_snapshot(&layout, 40, pos, &x(pos), &inv_freqs, 4, 2)
                .unwrap()
        };

        // Linear feed 0..=2 (pos 0 resets the layer state).
        for pos in 0..3 {
            feed(pos);
        }
        // Speculative feed of pos 3: this flushes the first block (ratio 4),
        // so rollback must restore pending buffers, prev_first_half,
        // compressed length and last_fed_pos exactly.
        deepseek_compressor_speculation_begin(3);
        let snapshot_spec = feed(3);
        let state_spec = compressor_state_capture(40);
        deepseek_compressor_speculation_rollback(3);
        let state_rolled = compressor_state_capture(40);
        assert_eq!(state_rolled.2, 3, "pending_len back to 3");
        assert_eq!(state_rolled.5, Some(2), "last_fed_pos back to 2");
        assert!(state_rolled.4.is_empty(), "compressed truncated");
        // Re-feed the same position non-speculatively: state and snapshot
        // must be bit-identical to the speculative pass.
        let snapshot_replay = feed(3);
        let state_replay = compressor_state_capture(40);
        assert_eq!(snapshot_spec, snapshot_replay);
        assert_eq!(state_spec.0, state_replay.0);
        assert_eq!(state_spec.1, state_replay.1);
        assert_eq!(state_spec.2, state_replay.2);
        assert_eq!(state_spec.3, state_replay.3);
        assert_eq!(state_spec.4, state_replay.4);
        assert_eq!(state_spec.5, state_replay.5);

        // Commit path: speculative feed accepted -> undo dropped, state kept,
        // a later rollback is a no-op.
        deepseek_compressor_speculation_begin(4);
        feed(4);
        deepseek_compressor_speculation_commit();
        let state_committed = compressor_state_capture(40);
        deepseek_compressor_speculation_rollback(4);
        let state_after_noop = compressor_state_capture(40);
        assert_eq!(state_committed.0, state_after_noop.0);
        assert_eq!(state_committed.2, state_after_noop.2);
        assert_eq!(state_committed.4, state_after_noop.4);
        assert_eq!(state_committed.5, state_after_noop.5);
        assert_eq!(state_after_noop.2, 1, "pos 4 kept after commit");
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_f32(out: &mut Vec<u8>, value: f32) {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

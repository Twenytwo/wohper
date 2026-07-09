use crate::compute::{
    accumulate_expert_for_positions, compute_layer, compute_layer_deepseek_mhc_phase1,
    compute_layer_deepseek_mhc_phase1_batch, compute_layer_deepseek_mhc_phase2,
    finish_layer_deepseek_mhc_phase2_batch,
    is_non_compute_dense_block, lm_head_row_count_from_block,
    lm_head_topk_score_chunked_from_block, route_experts_from_dense_block,
    router_top_experts_from_dense_block,
    token_embedding_from_block, AttentionKind, ComputeConfig, ComputeScratch, DeepSeekMhcCarry,
    ExpertCacheConfig, FusedInt4Gemm, IoBlockPtr, KVCache, LayerComputeStats, RouteOptions,
    RouterMath,
};
use crate::direct_io::{BlockKind, DirectIoError, DirectIoRuntime};
use crate::model_format::{ModelFormatError, ModelManifest, MODEL_FAMILY_DEEPSEEK_V4_FLASH};
use crate::row_index::{RowIndexError, RowTensorIndex};
use crate::scheduler::{ExpertRoute, MoEIoScheduler, ReadyBlock, SchedulerError};
use serde::Serialize;
use std::env;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use thiserror::Error;

pub const GLM52_EOS_TOKEN_IDS: [u32; 3] = [154820, 154827, 154829];
pub const GLM52_PRIMARY_EOS_TOKEN_ID: u32 = GLM52_EOS_TOKEN_IDS[0];

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model format error: {0}")]
    Model(#[from] ModelFormatError),
    #[error("direct I/O error: {0}")]
    DirectIo(#[from] DirectIoError),
    #[error("scheduler error: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("compute error: {0}")]
    Compute(#[from] crate::compute::ComputeError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("row tensor index error: {0}")]
    RowIndex(#[from] RowIndexError),
    #[error("empty token input")]
    EmptyTokenInput,
    #[error("missing dense block for layer {0}")]
    MissingDenseBlock(u32),
    #[error("generation exceeded scheduler wait budget")]
    SchedulerWaitBudgetExceeded,
}

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub model_path: PathBuf,
    pub active_experts: u32,
    pub pipeline_depth: usize,
    pub io_buffer_count: usize,
    pub io_buffer_bytes: usize,
    pub runtime_buffer_bytes: usize,
    pub stop_token_ids: Vec<u32>,
    pub expert_cache_dir: PathBuf,
    pub expert_cache_bytes: u64,
    pub expert_remote_endpoint: Option<String>,
    pub cluster_next_node: Option<SocketAddr>,
    pub local_layer_start: u32,
    pub local_layer_end: Option<u32>,
}

impl GenerationConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            active_experts: 8,
            pipeline_depth: 4,
            io_buffer_count: 12,
            io_buffer_bytes: 128 * 1024 * 1024,
            runtime_buffer_bytes: 0,
            stop_token_ids: GLM52_EOS_TOKEN_IDS.to_vec(),
            expert_cache_dir: PathBuf::from("cache/experts"),
            expert_cache_bytes: 100 * 1024 * 1024 * 1024,
            expert_remote_endpoint: None,
            cluster_next_node: None,
            local_layer_start: 0,
            local_layer_end: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub request_id: String,
    pub token_ids: Vec<u32>,
    pub max_new_tokens: u32,
    pub sampling: SamplingConfig,
    pub route_expert_ids: Vec<u32>,
    pub stop_token_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    pub fn from_options(
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        repetition_penalty: Option<f32>,
        seed: Option<u64>,
    ) -> Self {
        let temperature = temperature.unwrap_or(0.0);
        let top_k = top_k.unwrap_or(1).clamp(1, 128) as usize;
        let top_p = top_p.unwrap_or(1.0).clamp(0.01, 1.0);
        let repetition_penalty = repetition_penalty.unwrap_or(1.0).clamp(0.1, 10.0);
        Self {
            temperature: if temperature.is_finite() {
                temperature.clamp(0.0, 5.0)
            } else {
                0.0
            },
            top_k,
            top_p: if top_p.is_finite() { top_p } else { 1.0 },
            repetition_penalty: if repetition_penalty.is_finite() {
                repetition_penalty
            } else {
                1.0
            },
            seed: seed.unwrap_or(0),
        }
    }

    fn source(self) -> &'static str {
        if self.temperature > 0.0 && self.top_k > 1 {
            "lm_head_topk_sample"
        } else if self.repetition_penalty != 1.0 {
            "lm_head_penalized_argmax"
        } else {
            "lm_head_argmax"
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event")]
pub enum GenerationEvent {
    Started {
        request_id: String,
        input_tokens: usize,
        max_new_tokens: u32,
        model_layers: usize,
    },
    Token {
        request_id: String,
        index: u32,
        token_id: u32,
        logit: f32,
        source: &'static str,
        candidates: Vec<TokenCandidate>,
    },
    Finished {
        request_id: String,
        generated_tokens: u32,
        stop_reason: &'static str,
    },
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TokenCandidate {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Clone, Debug)]
struct SampledToken {
    token_id: u32,
    logit: f32,
    candidates: Vec<TokenCandidate>,
}

/// RAM cache of per-layer dense compute blocks. Dense blocks are identical
/// across tokens, so from the second token on a cached layer skips its NVMe
/// read entirely (~143MB/layer). Capacity via ZC_DENSE_CACHE_MB (0 = off);
/// fills first-come until the cap, no eviction (the layer set is fixed).
struct DenseRamCache {
    max_bytes: usize,
    used_bytes: usize,
    blocks: std::collections::HashMap<u32, std::sync::Arc<Vec<u8>>>,
}

impl DenseRamCache {
    fn from_env() -> Self {
        let cache_mb = std::env::var("ZC_DENSE_CACHE_MB")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        Self {
            max_bytes: cache_mb.saturating_mul(1024 * 1024),
            used_bytes: 0,
            blocks: std::collections::HashMap::new(),
        }
    }

    fn get(&self, physical_layer_id: u32) -> Option<std::sync::Arc<Vec<u8>>> {
        self.blocks.get(&physical_layer_id).cloned()
    }

    fn insert(&mut self, physical_layer_id: u32, bytes: &[u8]) {
        if self.max_bytes == 0 || self.blocks.contains_key(&physical_layer_id) {
            return;
        }
        if self.used_bytes.saturating_add(bytes.len()) > self.max_bytes {
            return;
        }
        self.used_bytes += bytes.len();
        self.blocks
            .insert(physical_layer_id, std::sync::Arc::new(bytes.to_vec()));
        crate::vlog!(
            "dense_cache insert layer={} bytes={} used_mb={} cap_mb={}",
            physical_layer_id,
            bytes.len(),
            self.used_bytes / (1024 * 1024),
            self.max_bytes / (1024 * 1024)
        );
    }
}

/// LRU RAM cache of expert blocks keyed by (physical layer, expert id).
/// With the persistent server (E2) hot experts skip their NVMe read on
/// later tokens/turns. Capacity via ZC_EXPERT_RAM_CACHE_MB (0 = off).
struct ExpertRamCache {
    max_bytes: usize,
    used_bytes: usize,
    tick: u64,
    blocks: std::collections::HashMap<(u32, u32), (u64, std::sync::Arc<Vec<u8>>)>,
}

impl ExpertRamCache {
    fn from_env() -> Self {
        let cache_mb = std::env::var("ZC_EXPERT_RAM_CACHE_MB")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        Self {
            max_bytes: cache_mb.saturating_mul(1024 * 1024),
            used_bytes: 0,
            tick: 0,
            blocks: std::collections::HashMap::new(),
        }
    }

    fn get(&mut self, layer: u32, expert: u32) -> Option<std::sync::Arc<Vec<u8>>> {
        self.tick += 1;
        let tick = self.tick;
        self.blocks.get_mut(&(layer, expert)).map(|entry| {
            entry.0 = tick;
            entry.1.clone()
        })
    }

    fn insert(&mut self, layer: u32, expert: u32, bytes: &[u8]) {
        if self.max_bytes == 0
            || bytes.len() > self.max_bytes
            || self.blocks.contains_key(&(layer, expert))
        {
            return;
        }
        while self.used_bytes.saturating_add(bytes.len()) > self.max_bytes {
            // Evict the least recently used entry (linear scan is fine at
            // a few hundred entries).
            let Some((&key, _)) = self
                .blocks
                .iter()
                .min_by_key(|(_, (last_use, _))| *last_use)
            else {
                return;
            };
            if let Some((_, evicted)) = self.blocks.remove(&key) {
                self.used_bytes = self.used_bytes.saturating_sub(evicted.len());
            }
        }
        self.tick += 1;
        self.used_bytes += bytes.len();
        self.blocks
            .insert((layer, expert), (self.tick, std::sync::Arc::new(bytes.to_vec())));
    }
}

/// Physical layer id of the MTP block for rope/compressor selection:
/// index 43 of DEEPSEEK_V4_COMPRESS_RATIOS (ratio 0 -> plain rope theta
/// 10000, no compressor) and above num_hash_layers (real router, no hash).
const MTP_PHYSICAL_LAYER_ID: u32 = 43;

/// MTP module (M1): the dense tensors of the mtp.0.* block reassembled as a
/// synthetic in-RAM ZCBLK01 block, so the standard compute paths can run on
/// it. Routed experts of the MTP block are read per-draft via the index.
struct MtpModule {
    dense_block: Vec<u8>,
    /// Routable expert ids (sorted), parsed from the expert tensor names.
    expert_ids: Vec<u32>,
    /// expert id -> its 6 tensor names (w1/w2/w3 weight+scale).
    expert_tensor_names: std::collections::HashMap<u32, Vec<String>>,
}

/// Expert id from an MTP expert tensor name
/// (e.g. "mtp.0.ffn.experts.17.w1.weight" -> 17).
fn mtp_expert_id_from_name(name: &str) -> Option<u32> {
    let marker = ".experts.";
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let end = rest.find('.')?;
    rest[..end].parse::<u32>().ok()
}

/// KV + token state of one finished conversation turn (V5/V6): when a
/// request's prompt starts with these tokens (multi-turn chat resends the
/// whole history), the cached KV is restored and only the delta is
/// prefilled. Requires an identical fixed KV layout (ZC_KV_SLOTS).
struct SessionCache {
    tokens: Vec<u32>,
    /// Positions 0..valid_positions have valid KV rows (the last emitted
    /// token is never forwarded).
    valid_positions: usize,
    kv_raw: Vec<f32>,
    kv_layers: usize,
    kv_slots: usize,
    /// Compressor state matching the KV (the live compressor map is
    /// process-global and only matches the LAST request; a slot restored
    /// after another conversation ran must bring its own).
    compressor: crate::compute::CompressorSessionState,
    /// LRU stamp (SessionStore.clock at last store/restore).
    last_used: u64,
}

/// V6: multi-slot session cache. One slot per live conversation, so
/// alternating clients stop destroying each other's prefix (single-slot
/// ping-pong: both re-paid the whole prefill every turn).
struct SessionStore {
    slots: Vec<SessionCache>,
    clock: u64,
}

/// Longest prompt prefix covered by a stored session: full-token match
/// over the session's valid positions, always leaving at least the last
/// prompt position to prefill (the sampling hidden must be fresh).
fn session_match_len(
    session_tokens: &[u32],
    valid_positions: usize,
    token_ids: &[u32],
) -> usize {
    let usable = valid_positions
        .min(session_tokens.len())
        .min(token_ids.len().saturating_sub(1));
    if usable == 0 || token_ids[..usable] != session_tokens[..usable] {
        return 0;
    }
    usable
}

/// True when `new_tokens` belongs to the conversation stored in a slot:
/// over the overlap of (stored valid prefix, new prompt) every token
/// agrees - i.e. one is a continuation (next turn) or a truncation
/// (regenerate) of the other. Two DIFFERENT conversations sharing a
/// system prompt diverge at the first question token, so they do NOT
/// alias into one slot - that separation is the point of multi-slot.
fn is_same_conversation(
    session_tokens: &[u32],
    valid_positions: usize,
    new_tokens: &[u32],
) -> bool {
    let valid = valid_positions.min(session_tokens.len());
    let shared = valid.min(new_tokens.len());
    shared > 0 && session_tokens[..shared] == new_tokens[..shared]
}

/// Where a finished request's state lands in the store.
enum StoreSlot {
    Replace(usize),
    Append,
    Evict(usize),
}

/// Pick the slot for a finished request: its own conversation's slot when
/// one exists, else a free slot, else the least-recently-used one.
/// `slots` carries (tokens, valid_positions, last_used) per stored slot.
fn select_store_slot(
    slots: &[(&[u32], usize, u64)],
    new_tokens: &[u32],
    max_slots: usize,
) -> StoreSlot {
    let mut same: Option<(usize, usize)> = None; // (index, valid_positions)
    for (index, (tokens, valid_positions, _)) in slots.iter().enumerate() {
        if is_same_conversation(tokens, *valid_positions, new_tokens) {
            // Longest stored prefix wins if several match (defensive: the
            // replace policy itself keeps one slot per conversation).
            if same.map_or(true, |(_, best)| *valid_positions > best) {
                same = Some((index, *valid_positions));
            }
        }
    }
    if let Some((index, _)) = same {
        return StoreSlot::Replace(index);
    }
    if slots.len() < max_slots.max(1) {
        return StoreSlot::Append;
    }
    let lru = slots
        .iter()
        .enumerate()
        .min_by_key(|(_, (_, _, last_used))| *last_used)
        .map(|(index, _)| index)
        .unwrap_or(0);
    StoreSlot::Evict(lru)
}

/// Speculative draft chain length (V4): how many tokens the MTP module
/// drafts per verify pass. Step 1 uses the main model's hidden; later
/// steps re-feed the MTP block its own output (approximation - accuracy
/// decays with depth, exactness is preserved by the verify pass anyway).
fn mtp_draft_chain_len() -> usize {
    static LEN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LEN.get_or_init(|| {
        std::env::var("ZC_MTP_DRAFT_LEN")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 4)
    })
}

/// How many conversations the session cache keeps (ZC_SESSION_SLOTS,
/// default 2, clamp 1..=4). RAM cost: one full KV copy per slot (~180MB
/// at ZC_KV_SLOTS=1024) plus a small compressor snapshot.
fn session_cache_slots() -> usize {
    static SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SLOTS.get_or_init(|| {
        std::env::var("ZC_SESSION_SLOTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 4)
    })
}

/// Fixed KV slot count for session reuse: every request that fits shares
/// the same cache layout, so the previous turn's KV restores verbatim.
fn kv_session_slots() -> usize {
    static SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SLOTS.get_or_init(|| {
        std::env::var("ZC_KV_SLOTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(512)
            .max(16)
    })
}

pub struct GenerationRuntime {
    config: GenerationConfig,
    manifest: ModelManifest,
    row_index: Option<RowTensorIndex>,
    dense_cache: std::sync::Mutex<DenseRamCache>,
    expert_ram: std::sync::Mutex<ExpertRamCache>,
    mtp: std::sync::Mutex<Option<std::sync::Arc<MtpModule>>>,
    session: std::sync::Mutex<SessionStore>,
}

impl GenerationRuntime {
    pub fn open(config: GenerationConfig) -> Result<Self, GenerationError> {
        let manifest = ModelManifest::load(&config.model_path)?;
        let row_index = RowTensorIndex::load_for_model(&config.model_path)?;
        Ok(Self {
            config,
            manifest,
            row_index,
            dense_cache: std::sync::Mutex::new(DenseRamCache::from_env()),
            expert_ram: std::sync::Mutex::new(ExpertRamCache::from_env()),
            mtp: std::sync::Mutex::new(None),
            session: std::sync::Mutex::new(SessionStore {
                slots: Vec::new(),
                clock: 0,
            }),
        })
    }

    /// V5: restore the previous turn's KV into `kv_cache` when the new
    /// prompt starts with the cached tokens. Returns the number of leading
    /// positions whose KV (and compressor state) is already valid; at least
    /// one prompt position is always left to prefill so the sampling hidden
    /// is fresh.
    fn try_restore_session(
        &self,
        token_ids: &[u32],
        kv_cache: &mut KVCache<'_>,
        kv_layers: usize,
        kv_slots: usize,
    ) -> usize {
        let mut store = self.session.lock().unwrap();
        let mut best: Option<(usize, usize)> = None; // (slot index, usable)
        for (index, session) in store.slots.iter().enumerate() {
            if session.kv_layers != kv_layers || session.kv_slots != kv_slots {
                continue;
            }
            let usable = session_match_len(&session.tokens, session.valid_positions, token_ids);
            if usable > 0 && best.map_or(true, |(_, top)| usable > top) {
                best = Some((index, usable));
            }
        }
        let Some((index, usable)) = best else {
            return 0;
        };
        if kv_cache.restore_raw(&store.slots[index].kv_raw).is_err() {
            return 0;
        }
        // The restored KV is only exact together with the compressor state
        // that produced it (another conversation may have run in between).
        crate::compute::deepseek_compressor_state_import(&store.slots[index].compressor);
        store.clock += 1;
        let clock = store.clock;
        store.slots[index].last_used = clock;
        eprintln!(
            "session_reuse slot={} prefix_positions={} prompt_len={} (skipping their prefill)",
            index,
            usable,
            token_ids.len()
        );
        usable
    }

    /// V5: snapshot the finished request's KV + tokens for the next turn.
    fn store_session(
        &self,
        tokens: &[u32],
        kv_cache: &KVCache<'_>,
        kv_layers: usize,
        kv_slots: usize,
    ) {
        if tokens.len() < 2 {
            return;
        }
        let session = SessionCache {
            tokens: tokens.to_vec(),
            valid_positions: tokens.len() - 1,
            kv_raw: kv_cache.raw_storage().to_vec(),
            kv_layers,
            kv_slots,
            compressor: crate::compute::deepseek_compressor_state_export(),
            last_used: 0,
        };
        let mut store = self.session.lock().unwrap();
        store.clock += 1;
        let clock = store.clock;
        let views: Vec<(&[u32], usize, u64)> = store
            .slots
            .iter()
            .map(|slot| (slot.tokens.as_slice(), slot.valid_positions, slot.last_used))
            .collect();
        let target = select_store_slot(&views, tokens, session_cache_slots());
        drop(views);
        match target {
            StoreSlot::Replace(index) | StoreSlot::Evict(index) => {
                store.slots[index] = session;
                store.slots[index].last_used = clock;
            }
            StoreSlot::Append => {
                let mut session = session;
                session.last_used = clock;
                store.slots.push(session);
            }
        }
    }

    /// Loads (once) the MTP module: every mtp.0.* tensor except the routed
    /// experts is read from the core via the tensor index and reassembled
    /// into a synthetic ZCBLK01 block (~250MB). Returns false when the
    /// slice has no MTP tensors or no tensor index.
    fn ensure_mtp_module(&self) -> Result<bool, GenerationError> {
        {
            let guard = self.mtp.lock().unwrap();
            if guard.is_some() {
                return Ok(true);
            }
        }
        let Some(index) = &self.row_index else {
            return Ok(false);
        };
        let names = index.tensor_names_with_prefix("mtp.0.");
        if names.is_empty() {
            return Ok(false);
        }
        let mut storages: Vec<(String, Vec<u8>, Vec<u64>, u16, u16, u16)> = Vec::new();
        let mut expert_names = Vec::new();
        for name in &names {
            if name.contains(".ffn.experts.") {
                expert_names.push(name.clone());
                continue;
            }
            let record = index
                .tensor(name)
                .ok_or_else(|| RowIndexError::MissingTensor(name.clone()))?;
            let shape: Vec<u64> = record.shape.iter().map(|&dim| dim as u64).collect();
            let dtype = record.dtype_code;
            let quant = record.quant_format;
            // The converter tags every mtp.* tensor with the opaque MTP role,
            // so the index role is 0 for attention/norm/router tensors -
            // re-derive the compute role from the name (M2): without it
            // phase1/phase2 cannot find wq/wkv/wo, the norms, the router
            // gate or the shared expert in the synthetic block.
            let role = crate::compute::mtp_synthetic_role_code(name, record.role_code);
            let bytes = index.read_tensor_bytes(name)?;
            storages.push((name.clone(), bytes, shape, dtype, quant, role));
        }
        let tensors: Vec<crate::compute::SyntheticTensor<'_>> = storages
            .iter()
            .map(|(name, bytes, shape, dtype, quant, role)| crate::compute::SyntheticTensor {
                name,
                role_code: *role as u32,
                shape,
                data: bytes,
                dtype_original: *dtype,
                quant_format: *quant,
                scale: 1.0,
                zero_point: 0.0,
            })
            .collect();
        let block = crate::compute::build_synthetic_quant_block(&tensors);
        let mut expert_tensor_names: std::collections::HashMap<u32, Vec<String>> =
            std::collections::HashMap::new();
        for name in &expert_names {
            if let Some(expert_id) = mtp_expert_id_from_name(name) {
                expert_tensor_names
                    .entry(expert_id)
                    .or_default()
                    .push(name.clone());
            }
        }
        let mut expert_ids: Vec<u32> = expert_tensor_names.keys().copied().collect();
        expert_ids.sort_unstable();
        eprintln!(
            "mtp_module loaded dense_tensors={} experts={} expert_ids={} block_bytes={}",
            storages.len(),
            expert_names.len(),
            expert_ids.len(),
            block.len()
        );
        *self.mtp.lock().unwrap() = Some(std::sync::Arc::new(MtpModule {
            dense_block: block,
            expert_ids,
            expert_tensor_names,
        }));
        Ok(true)
    }

    fn mtp_module(&self) -> Option<std::sync::Arc<MtpModule>> {
        self.mtp.lock().unwrap().clone()
    }

    /// M2 draft step (reference `MTPBlock.forward`, greedy): given the main
    /// model's hc hidden of the last processed position and the token just
    /// sampled (t+1), runs the MTP block once and returns the draft of t+2.
    /// Pure observer: nothing of the main path is mutated. The MTP KV cache
    /// uses LOCAL positions (`mtp_pos`), one per generated token.
    #[allow(clippy::too_many_arguments)]
    fn mtp_draft_next(
        &self,
        module: &MtpModule,
        main_hidden_hc: &[f32],
        next_token_id: u32,
        mtp_pos: usize,
        mtp_kv: &mut KVCache<'_>,
        mtp_hidden: &mut [f32],
        scratch_buf: &mut [f32],
        compute_config: &ComputeConfig,
        gemm: &FusedInt4Gemm,
    ) -> Result<Option<(u32, f32)>, GenerationError> {
        let Some(index) = &self.row_index else {
            return Ok(None);
        };
        let hidden = compute_config.hidden_size;
        let mut embed = vec![0.0f32; hidden];
        if !index.read_bf16_row("embed.weight", next_token_id as usize, &mut embed)? {
            return Ok(None);
        }
        let dense = module.dense_block.as_slice();

        // x = e_proj(enorm(embed(t+1))) + h_proj(hnorm(hidden)) per hc copy.
        crate::compute::mtp_prepare_draft_hidden(
            dense,
            &embed,
            main_hidden_hc,
            mtp_hidden,
            &mut ComputeScratch {
                dequant_tile_f32: &mut *scratch_buf,
            },
            compute_config,
            gemm,
        )?;

        // Full DeepSeek block on the synthetic dense block. layer_index 0
        // indexes the dedicated 1-layer KV cache; compress_ratios[0] == 0 ==
        // compress_ratios[43], so rope (plain theta 10000) and compressor
        // (none) match the MTP block exactly.
        let mut carry = unsafe {
            compute_layer_deepseek_mhc_phase1(
                dense,
                0,
                mtp_pos,
                mtp_hidden,
                mtp_kv,
                &mut ComputeScratch {
                    dequant_tile_f32: &mut *scratch_buf,
                },
                compute_config,
            )?
        };

        // Route on the real gate input with the MTP block's own router
        // (layer id 43 >= num_hash_layers: never hash routing).
        let Some(mut routes) = route_experts_from_dense_block(
            dense,
            &carry.gate_input,
            self.config.active_experts as usize,
            &module.expert_ids,
            scratch_buf,
            gemm,
            RouteOptions::from_config(compute_config, MTP_PHYSICAL_LAYER_ID, next_token_id),
        )?
        else {
            eprintln!("mtp_draft skipped reason=router_missing");
            return Ok(None);
        };
        normalize_route_weights(&mut routes, "router", compute_config);
        eprintln!(
            "mtp_router experts={:?} weights={:?}",
            routes.iter().map(|route| route.expert_id).collect::<Vec<_>>(),
            routes.iter().map(|route| route.score).collect::<Vec<_>>()
        );

        // Targeted preads of the routed experts (6 x ~13MB), assembled as
        // synthetic FP4 expert blocks the standard phase2 consumes natively.
        let mut expert_blocks: Vec<(u32, f32, Vec<u8>)> = Vec::with_capacity(routes.len());
        for route in &routes {
            let Some(names) = module.expert_tensor_names.get(&route.expert_id) else {
                eprintln!(
                    "mtp_draft missing expert tensors expert_id={}",
                    route.expert_id
                );
                continue;
            };
            let mut storages: Vec<(String, Vec<u8>, Vec<u64>, u16, u16, u16)> =
                Vec::with_capacity(names.len());
            for name in names {
                let record = index
                    .tensor(name)
                    .ok_or_else(|| RowIndexError::MissingTensor(name.clone()))?;
                let shape: Vec<u64> = record.shape.iter().map(|&dim| dim as u64).collect();
                let bytes = index.read_tensor_bytes(name)?;
                storages.push((
                    name.clone(),
                    bytes,
                    shape,
                    record.dtype_code,
                    record.quant_format,
                    record.role_code,
                ));
            }
            let tensors: Vec<crate::compute::SyntheticTensor<'_>> = storages
                .iter()
                .map(|(name, bytes, shape, dtype, quant, role)| {
                    crate::compute::SyntheticTensor {
                        name,
                        role_code: *role as u32,
                        shape,
                        data: bytes,
                        dtype_original: *dtype,
                        quant_format: *quant,
                        scale: 1.0,
                        zero_point: 0.0,
                    }
                })
                .collect();
            expert_blocks.push((
                route.expert_id,
                route.score,
                crate::compute::build_synthetic_quant_block(&tensors),
            ));
        }
        let experts: Vec<IoBlockPtr> = expert_blocks
            .iter()
            .map(|(expert_id, score, block)| IoBlockPtr {
                kind: BlockKind::Expert,
                layer_id: MTP_PHYSICAL_LAYER_ID,
                expert_id: *expert_id,
                route_weight: *score,
                ptr: block.as_ptr(),
                len: block.len(),
            })
            .collect();

        let mut stats = LayerComputeStats {
            dense_bytes: dense.len(),
            ..Default::default()
        };
        unsafe {
            compute_layer_deepseek_mhc_phase2(
                dense,
                0,
                &experts,
                &mut carry,
                mtp_hidden,
                &mut ComputeScratch {
                    dequant_tile_f32: &mut *scratch_buf,
                },
                compute_config,
                gemm,
                &mut stats,
            )?;
        }

        // MTP head: pool with the block's own hc_head params + final norm,
        // then greedy-scan the shared lm_head.
        let mut pooled = vec![0.0f32; hidden];
        crate::compute::mtp_head_pool_and_norm(
            dense,
            "mtp.0.norm.weight",
            mtp_hidden,
            &mut pooled,
            compute_config,
        )?;
        let Some(candidates) = index.topk_bf16_lm_head(&pooled, scratch_buf, 1)? else {
            return Ok(None);
        };
        Ok(candidates.first().copied())
    }

    /// Draft wrapper for the speculative loop: logs, advances the local MTP
    /// position, and downgrades errors to "speculation disabled" (never
    /// fatal for the request).
    #[allow(clippy::too_many_arguments)]
    fn mtp_draft_step(
        &self,
        module: &MtpModule,
        main_hidden_hc: &[f32],
        next_token_id: u32,
        mtp_pos: &mut usize,
        mtp_kv: &mut KVCache<'_>,
        mtp_hidden: &mut [f32],
        scratch_buf: &mut [f32],
        compute_config: &ComputeConfig,
        gemm: &FusedInt4Gemm,
        disabled: &mut bool,
    ) -> Option<u32> {
        if *disabled {
            return None;
        }
        let started = Instant::now();
        match self.mtp_draft_next(
            module,
            main_hidden_hc,
            next_token_id,
            *mtp_pos,
            mtp_kv,
            mtp_hidden,
            scratch_buf,
            compute_config,
            gemm,
        ) {
            Ok(Some((draft_token, draft_logit))) => {
                eprintln!(
                    "mtp_draft pos={} token_in={} draft={} logit={:.4} elapsed_ms={}",
                    *mtp_pos,
                    next_token_id,
                    draft_token,
                    draft_logit,
                    started.elapsed().as_millis()
                );
                *mtp_pos += 1;
                Some(draft_token)
            }
            Ok(None) => {
                eprintln!("mtp_draft skipped reason=missing_index_or_embed");
                None
            }
            Err(err) => {
                eprintln!("mtp_draft error={err} - speculation disabled for this request");
                *disabled = true;
                None
            }
        }
    }

    /// V4: draft a CHAIN of up to `chain_len` tokens. The first step reads
    /// the main model's hc hidden; each later step re-feeds the MTP block
    /// its own output hidden together with the previous draft's embedding.
    #[allow(clippy::too_many_arguments)]
    fn mtp_draft_chain(
        &self,
        module: &MtpModule,
        main_hidden_hc: &[f32],
        next_token_id: u32,
        mtp_pos: &mut usize,
        mtp_kv: &mut KVCache<'_>,
        mtp_hidden: &mut [f32],
        chain_input: &mut Vec<f32>,
        scratch_buf: &mut [f32],
        compute_config: &ComputeConfig,
        gemm: &FusedInt4Gemm,
        disabled: &mut bool,
        chain_len: usize,
    ) -> Vec<u32> {
        let mut drafts = Vec::with_capacity(chain_len);
        let mut input_token = next_token_id;
        for step in 0..chain_len {
            let drafted = if step == 0 {
                self.mtp_draft_step(
                    module,
                    main_hidden_hc,
                    input_token,
                    mtp_pos,
                    mtp_kv,
                    mtp_hidden,
                    scratch_buf,
                    compute_config,
                    gemm,
                    disabled,
                )
            } else {
                chain_input.clear();
                chain_input.extend_from_slice(mtp_hidden);
                self.mtp_draft_step(
                    module,
                    chain_input,
                    input_token,
                    mtp_pos,
                    mtp_kv,
                    mtp_hidden,
                    scratch_buf,
                    compute_config,
                    gemm,
                    disabled,
                )
            };
            match drafted {
                Some(token) => {
                    drafts.push(token);
                    input_token = token;
                }
                None => break,
            }
        }
        drafts
    }

    /// Pool the hc state, sample the next token (same seed/sampling
    /// semantics as the sequential loop), push it and emit its Token event.
    #[allow(clippy::too_many_arguments)]
    fn sample_and_emit<W: Write>(
        &self,
        writer: &mut W,
        request: &GenerationRequest,
        scheduler: &mut MoEIoScheduler,
        hidden_hc: &[f32],
        sampling_hidden: &mut [f32],
        scratch_buf: &mut [f32],
        gemm: &FusedInt4Gemm,
        tokens: &mut Vec<u32>,
        index: u32,
        compute_config: &ComputeConfig,
    ) -> Result<u32, GenerationError> {
        self.prepare_sampling_hidden(hidden_hc, sampling_hidden, compute_config)?;
        let sampled = self.sample_next_token_from_lm_head(
            scheduler,
            sampling_hidden,
            scratch_buf,
            gemm,
            tokens,
            &request.sampling,
            stable_sample_seed(&request.request_id, index, request.sampling.seed),
        )?;
        let (token_id, logit, source, candidates) = match sampled {
            Some(sample) => (
                sample.token_id,
                sample.logit,
                request.sampling.source(),
                sample.candidates,
            ),
            None => {
                crate::vlog!("sampling source=hidden_state_placeholder");
                (
                    sample_argmax_token(hidden_hc),
                    f32::NAN,
                    "hidden_state_placeholder",
                    Vec::new(),
                )
            }
        };
        tokens.push(token_id);
        write_event(
            writer,
            &GenerationEvent::Token {
                request_id: request.request_id.clone(),
                index,
                token_id,
                logit,
                source,
                candidates,
            },
        )?;
        Ok(token_id)
    }

    /// M3 exact-greedy speculative decode: after each sampled token t+1 the
    /// MTP module drafts t+2; the next step runs BOTH positions in one
    /// layer-major batched pass (`run_positions_batch`). The real t+2 is
    /// sampled from the first position's hidden; on draft hit the second
    /// position's hidden/KV are already valid and t+3 is sampled with no
    /// extra pass. On miss the speculative position is rolled back
    /// (compressor undo stack; its KV rows are overwritten by the next pass
    /// before any later position attends to them). Emitted tokens are
    /// identical to pure greedy by construction - only wall clock changes.
    #[allow(clippy::too_many_arguments)]
    fn run_decode_speculative<W: Write>(
        &self,
        writer: &mut W,
        request: &GenerationRequest,
        scheduler: &mut MoEIoScheduler,
        tokens: &mut Vec<u32>,
        hidden_states: &mut [f32],
        kv_cache: &mut KVCache<'_>,
        scratch_buf: &mut [f32],
        sampling_hidden: &mut [f32],
        compute_config: &ComputeConfig,
        gemm: &FusedInt4Gemm,
        stop_token_ids: &[u32],
        module: &MtpModule,
        mtp_kv: &mut KVCache<'_>,
        mtp_hidden: &mut [f32],
        session_meta: Option<(usize, usize)>,
    ) -> Result<(), GenerationError> {
        let max_new = request.max_new_tokens;
        let hc_dim = hidden_states.len();
        let mut emitted: u32 = 0;
        let mut mtp_pos = 0usize;
        let mut draft_disabled = false;
        let mut hits = 0u32;
        let mut checked = 0u32;
        let mut batch_hiddens = vec![0.0f32; (1 + mtp_draft_chain_len()) * hc_dim];

        if max_new == 0 {
            return write_speculative_finish(
                writer,
                request.request_id.clone(),
                0,
                "max_new_tokens",
                0,
                0,
            );
        }

        // First token: sampled from the prefill hidden (reference
        // semantics, no extra forward).
        let first = self.sample_and_emit(
            writer,
            request,
            scheduler,
            hidden_states,
            sampling_hidden,
            scratch_buf,
            gemm,
            tokens,
            0,
            compute_config,
        )?;
        emitted = 1;
        if stop_token_ids.contains(&first) {
            if let Some((kv_layers, kv_slots)) = session_meta {
                self.store_session(tokens, kv_cache, kv_layers, kv_slots);
            }
            return write_speculative_finish(
                writer,
                request.request_id.clone(),
                emitted,
                "stop_token",
                hits,
                checked,
            );
        }
        let chain_len_cfg = mtp_draft_chain_len();
        let mut chain_input: Vec<f32> = Vec::with_capacity(hc_dim);
        let mut pending_draft: Vec<u32> = if emitted < max_new {
            self.mtp_draft_chain(
                module,
                hidden_states,
                first,
                &mut mtp_pos,
                mtp_kv,
                mtp_hidden,
                &mut chain_input,
                scratch_buf,
                compute_config,
                gemm,
                &mut draft_disabled,
                chain_len_cfg,
            )
        } else {
            Vec::new()
        };

        while emitted < max_new {
            let pending_token = *tokens.last().expect("tokens non-empty");
            let pending_pos = tokens.len() - 1;
            // Draft positions worth verifying: each accepted draft emits one
            // token and the fully-accepted pass emits one bonus on top.
            let slots_left = (max_new - emitted) as usize;
            let n_draft = pending_draft.len().min(slots_left.saturating_sub(1));
            let chain: Vec<u32> = pending_draft.drain(..n_draft).collect();
            pending_draft.clear();

            if !chain.is_empty() {
                let npos = 1 + chain.len();
                let mut items: Vec<(u32, usize)> = Vec::with_capacity(npos);
                items.push((pending_token, pending_pos));
                for (depth, &draft_token) in chain.iter().enumerate() {
                    items.push((draft_token, pending_pos + 1 + depth));
                }
                let pass_started = Instant::now();
                crate::compute::deepseek_compressor_speculation_begin(pending_pos + 1);
                if let Err(err) = self.run_positions_batch(
                    scheduler,
                    &items,
                    &mut batch_hiddens[..npos * hc_dim],
                    kv_cache,
                    scratch_buf,
                    compute_config,
                    gemm,
                ) {
                    crate::compute::deepseek_compressor_speculation_rollback(pending_pos + 1);
                    return Err(err);
                }
                // Progressive verify: slot i's hidden yields the REAL token
                // at position pending_pos+1+i; it stays valid only while
                // every earlier draft matched.
                let mut accepted = 0usize;
                let mut stopped = false;
                let mut last_token = pending_token;
                for (depth, &draft_token) in chain.iter().enumerate() {
                    let actual = match self.sample_and_emit(
                        writer,
                        request,
                        scheduler,
                        &batch_hiddens[depth * hc_dim..(depth + 1) * hc_dim],
                        sampling_hidden,
                        scratch_buf,
                        gemm,
                        tokens,
                        emitted,
                        compute_config,
                    ) {
                        Ok(token) => token,
                        Err(err) => {
                            crate::compute::deepseek_compressor_speculation_rollback(
                                pending_pos + 1 + accepted,
                            );
                            return Err(err);
                        }
                    };
                    emitted += 1;
                    checked += 1;
                    let hit = actual == draft_token;
                    if hit {
                        hits += 1;
                    }
                    eprintln!(
                        "mtp_verify pos={} depth={} draft={} actual={} accept={} hits={}/{}",
                        pending_pos + 1 + depth,
                        depth,
                        draft_token,
                        actual,
                        hit,
                        hits,
                        checked
                    );
                    last_token = actual;
                    if stop_token_ids.contains(&actual) {
                        if hit {
                            accepted += 1;
                        }
                        stopped = true;
                        break;
                    }
                    if hit {
                        accepted += 1;
                    } else {
                        break;
                    }
                }
                let all_hit = accepted == chain.len();
                if stopped {
                    if all_hit {
                        crate::compute::deepseek_compressor_speculation_commit();
                    } else {
                        crate::compute::deepseek_compressor_speculation_rollback(
                            pending_pos + 1 + accepted,
                        );
                    }
                    if let Some((kv_layers, kv_slots)) = session_meta {
                        self.store_session(tokens, kv_cache, kv_layers, kv_slots);
                    }
                    return write_speculative_finish(
                        writer,
                        request.request_id.clone(),
                        emitted,
                        "stop_token",
                        hits,
                        checked,
                    );
                }
                if all_hit {
                    crate::compute::deepseek_compressor_speculation_commit();
                    // Every draft position is valid: the last one's hidden
                    // yields one more token with no extra pass.
                    hidden_states
                        .copy_from_slice(&batch_hiddens[chain.len() * hc_dim..npos * hc_dim]);
                    if emitted < max_new {
                        let bonus = self.sample_and_emit(
                            writer,
                            request,
                            scheduler,
                            &batch_hiddens[chain.len() * hc_dim..npos * hc_dim],
                            sampling_hidden,
                            scratch_buf,
                            gemm,
                            tokens,
                            emitted,
                            compute_config,
                        )?;
                        emitted += 1;
                        if stop_token_ids.contains(&bonus) {
                            if let Some((kv_layers, kv_slots)) = session_meta {
                                self.store_session(tokens, kv_cache, kv_layers, kv_slots);
                            }
                            return write_speculative_finish(
                                writer,
                                request.request_id.clone(),
                                emitted,
                                "stop_token",
                                hits,
                                checked,
                            );
                        }
                        last_token = bonus;
                    }
                } else {
                    crate::compute::deepseek_compressor_speculation_rollback(
                        pending_pos + 1 + accepted,
                    );
                    // Hidden of the deepest VALID position (slot `accepted`);
                    // dirtied speculative KV rows sit beyond the valid range
                    // and are overwritten by the next pass.
                    hidden_states.copy_from_slice(
                        &batch_hiddens[accepted * hc_dim..(accepted + 1) * hc_dim],
                    );
                }
                eprintln!(
                    "mtp_pass chain={} accepted={} all_hit={} pass_ms={}",
                    chain.len(),
                    accepted,
                    all_hit,
                    pass_started.elapsed().as_millis()
                );
                if emitted < max_new {
                    pending_draft = self.mtp_draft_chain(
                        module,
                        hidden_states,
                        last_token,
                        &mut mtp_pos,
                        mtp_kv,
                        mtp_hidden,
                        &mut chain_input,
                        scratch_buf,
                        compute_config,
                        gemm,
                        &mut draft_disabled,
                        chain_len_cfg,
                    );
                }
            } else {
                // No draft available (or a single slot left): classic
                // sequential forward of the pending token.
                self.fill_hidden_from_embedding_or_fallback(
                    scheduler,
                    pending_token,
                    hidden_states,
                )?;
                self.run_model_for_one_token(
                    scheduler,
                    pending_pos,
                    pending_token,
                    hidden_states,
                    kv_cache,
                    scratch_buf,
                    compute_config,
                    gemm,
                    &[],
                )?;
                let actual = self.sample_and_emit(
                    writer,
                    request,
                    scheduler,
                    hidden_states,
                    sampling_hidden,
                    scratch_buf,
                    gemm,
                    tokens,
                    emitted,
                    compute_config,
                )?;
                emitted += 1;
                if stop_token_ids.contains(&actual) {
                    if let Some((kv_layers, kv_slots)) = session_meta {
                        self.store_session(tokens, kv_cache, kv_layers, kv_slots);
                    }
                    return write_speculative_finish(
                        writer,
                        request.request_id.clone(),
                        emitted,
                        "stop_token",
                        hits,
                        checked,
                    );
                }
                if emitted < max_new {
                    pending_draft = self.mtp_draft_chain(
                        module,
                        hidden_states,
                        actual,
                        &mut mtp_pos,
                        mtp_kv,
                        mtp_hidden,
                        &mut chain_input,
                        scratch_buf,
                        compute_config,
                        gemm,
                        &mut draft_disabled,
                        chain_len_cfg,
                    );
                }
            }
        }

        if let Some((kv_layers, kv_slots)) = session_meta {
            self.store_session(tokens, kv_cache, kv_layers, kv_slots);
        }
        write_speculative_finish(
            writer,
            request.request_id.clone(),
            emitted,
            "max_new_tokens",
            hits,
            checked,
        )
    }

    pub fn generate_stream<W: Write>(
        &self,
        request: GenerationRequest,
        writer: &mut W,
    ) -> Result<(), GenerationError> {
        if request.token_ids.is_empty() {
            return Err(GenerationError::EmptyTokenInput);
        }

        write_event(
            writer,
            &GenerationEvent::Started {
                request_id: request.request_id.clone(),
                input_tokens: request.token_ids.len(),
                max_new_tokens: request.max_new_tokens,
                model_layers: self.manifest.layers.len(),
            },
        )?;

        let io = DirectIoRuntime::open_with_pack(
            &self.config.model_path,
            self.manifest.expert_pack_file(),
            self.config.io_buffer_bytes,
            self.config.runtime_buffer_bytes,
            self.config.io_buffer_count,
        )?;
        // V3: with a zero cache budget and no remote endpoint the sidecar
        // shards are read DIRECTLY (no copy-into-cache, no double storage).
        // The disk cache only pays off when the shards live on a slow or
        // remote medium.
        let mut scheduler = if self.config.expert_cache_bytes == 0
            && self.config.expert_remote_endpoint.is_none()
        {
            eprintln!("expert_cache disabled - sidecar shards read directly");
            MoEIoScheduler::new(io, self.manifest.clone())
        } else {
            MoEIoScheduler::new(io, self.manifest.clone()).with_expert_cache(ExpertCacheConfig {
                cache_dir: self.config.expert_cache_dir.clone(),
                max_bytes: self.config.expert_cache_bytes,
                remote_endpoint: self.config.expert_remote_endpoint.clone(),
            })?
        };
        let compute_config = if self.manifest.header.model_family == MODEL_FAMILY_DEEPSEEK_V4_FLASH {
            ComputeConfig::deepseek_v4_flash()
        } else {
            ComputeConfig::glm52_like(self.manifest.header.hidden_size as usize)
        };
        let hc_mult = compute_config.hc_mult.max(1);
        let mut hidden_states = vec![0.0f32; compute_config.hidden_size * hc_mult];
        let mut scratch_buf = vec![0.0f32; compute_config.prefill_scratch_f32()];
        // Pooled single-stream buffer for the LM boundary when mHC is active.
        let mut sampling_hidden = vec![0.0f32; compute_config.hidden_size];
        let max_tokens = request.token_ids.len() + request.max_new_tokens as usize;
        let head_dim = compute_config
            .qk_nope_head_dim
            .saturating_add(compute_config.qk_rope_head_dim);
        // V5: requests that fit the fixed slot count share one KV layout so
        // the previous turn's KV can be restored verbatim (session reuse).
        let kv_slots = kv_session_slots();
        let session_eligible = max_tokens <= kv_slots;
        let kv_tokens = if session_eligible { kv_slots } else { max_tokens };
        let kv_required = KVCache::required_f32(
            self.manifest.layers.len(),
            kv_tokens,
            compute_config.num_kv_heads,
            head_dim,
        )?;
        let mut kv_storage = vec![0.0f32; kv_required];
        let mut kv_cache = KVCache::from_scratch(
            &mut kv_storage,
            self.manifest.layers.len(),
            kv_tokens,
            compute_config.num_kv_heads,
            head_dim,
        )?;
        let session_meta = if session_eligible {
            Some((self.manifest.layers.len(), kv_tokens))
        } else {
            None
        };
        let mut tokens = request.token_ids.clone();
        let gemm = FusedInt4Gemm;
        let stop_token_ids = if !request.stop_token_ids.is_empty() {
            request.stop_token_ids.clone()
        } else if self.manifest.header.model_family == MODEL_FAMILY_DEEPSEEK_V4_FLASH {
            // DeepSeek's EOS is token 1; the GLM defaults never fire here,
            // so the server used to generate PAST the end of the answer
            // (wasted tokens, and clients that stop at EOS then abandoned
            // the stream mid-flight).
            vec![1]
        } else {
            self.config.stop_token_ids.clone()
        };

        // MTP draft (M2): with ZC_MTP=1 the module runs as a pure OBSERVER
        // in the decode loop - after each sampled token t+1 it drafts t+2
        // and the next iteration logs draft-vs-actual (accept rate preview
        // for M3). Main-path tokens are untouched by construction.
        let mtp_verify_requested = std::env::var("ZC_MTP_VERIFY")
            .map(|v| v == "1")
            .unwrap_or(false);
        let mtp_enabled = (std::env::var("ZC_MTP").map(|v| v == "1").unwrap_or(false)
            || mtp_verify_requested)
            && compute_config.attention_kind == AttentionKind::DeepSeekV4Mla
            && compute_config.hc_mult > 1;
        let mtp_module = if mtp_enabled {
            match self.ensure_mtp_module() {
                Ok(true) => self.mtp_module(),
                Ok(false) => {
                    eprintln!("mtp_draft disabled reason=no_mtp_tensors_or_index");
                    None
                }
                Err(err) => {
                    eprintln!("mtp_draft disabled error={err}");
                    None
                }
            }
        } else {
            None
        };
        // Dedicated single-layer KV cache for the MTP block. Draft positions
        // are LOCAL (0,1,2,... per generated token): rope scores depend only
        // on relative distance, and every cached row is real (no zero rows
        // from the un-drafted prompt history).
        let mut mtp_kv_storage = if mtp_module.is_some() {
            vec![
                0.0f32;
                KVCache::required_f32(1, max_tokens, compute_config.num_kv_heads, head_dim)?
            ]
        } else {
            Vec::new()
        };
        let mut mtp_kv_cache = if mtp_module.is_some() {
            Some(KVCache::from_scratch(
                &mut mtp_kv_storage,
                1,
                max_tokens,
                compute_config.num_kv_heads,
                head_dim,
            )?)
        } else {
            None
        };
        let mut mtp_hidden = if mtp_module.is_some() {
            vec![0.0f32; compute_config.hidden_size * hc_mult]
        } else {
            Vec::new()
        };
        let mut mtp_pos = 0usize;
        let mut mtp_failed = false;
        let mut mtp_last_draft: Option<u32> = None;
        let mut mtp_hits = 0u32;
        let mut mtp_checked = 0u32;
        let batch_prefill = std::env::var("ZC_BATCH_PREFILL").map(|v| v != "0").unwrap_or(true)
            && compute_config.attention_kind == AttentionKind::DeepSeekV4Mla
            && compute_config.hc_mult > 1
            && request.route_expert_ids.is_empty()
            && request.token_ids.len() > 1;
        if batch_prefill {
            // V5: skip the prefill of the prefix already covered by the
            // previous turn's KV (multi-turn chat resends the history).
            let restored = if session_eligible {
                self.try_restore_session(
                    &request.token_ids,
                    &mut kv_cache,
                    self.manifest.layers.len(),
                    kv_tokens,
                )
            } else {
                0
            };
            if restored == 0 {
                self.run_prefill_batch(
                    &mut scheduler,
                    &request.token_ids,
                    &mut hidden_states,
                    &mut kv_cache,
                    &mut scratch_buf,
                    &compute_config,
                    &gemm,
                )?;
            } else {
                let items: Vec<(u32, usize)> = request.token_ids[restored..]
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(offset, token_id)| (token_id, restored + offset))
                    .collect();
                let hc_dim = compute_config.hidden_size * hc_mult;
                let mut delta_hiddens = vec![0.0f32; items.len() * hc_dim];
                self.run_positions_batch(
                    &mut scheduler,
                    &items,
                    &mut delta_hiddens,
                    &mut kv_cache,
                    &mut scratch_buf,
                    &compute_config,
                    &gemm,
                )?;
                let last = items.len() - 1;
                hidden_states[..hc_dim]
                    .copy_from_slice(&delta_hiddens[last * hc_dim..(last + 1) * hc_dim]);
                eprintln!(
                    "prefill_batch positions={} mode=delta_after_session_reuse",
                    items.len()
                );
            }
        } else {
            for (prefill_index, token_id) in request.token_ids.iter().copied().enumerate() {
                self.fill_hidden_from_embedding_or_fallback(
                    &mut scheduler,
                    token_id,
                    &mut hidden_states,
                )?;
                self.run_model_for_one_token(
                    &mut scheduler,
                    prefill_index,
                    token_id,
                    &mut hidden_states,
                    &mut kv_cache,
                    &mut scratch_buf,
                    &compute_config,
                    &gemm,
                    &request.route_expert_ids,
                )?;
                crate::vlog!("prefill token_index={} token_id={}", prefill_index, token_id);
            }
        }

        // M3: exact-greedy speculative decode (draft + batched 2-position
        // verify pass). Only for pure greedy sampling; anything else falls
        // back to the sequential loop below (with the M2 observer if on).
        let mtp_verify = mtp_verify_requested
            && self.row_index.is_some()
            && request.sampling.temperature <= 0.0
            && request.sampling.top_k <= 1
            && (request.sampling.repetition_penalty - 1.0).abs() < f32::EPSILON
            && request.route_expert_ids.is_empty();
        if mtp_verify {
            if let (Some(module), Some(mtp_kv)) = (mtp_module.as_ref(), mtp_kv_cache.as_mut()) {
                eprintln!("mtp_verify enabled mode=exact_greedy_speculative");
                return self.run_decode_speculative(
                    writer,
                    &request,
                    &mut scheduler,
                    &mut tokens,
                    &mut hidden_states,
                    &mut kv_cache,
                    &mut scratch_buf,
                    &mut sampling_hidden,
                    &compute_config,
                    &gemm,
                    &stop_token_ids,
                    module,
                    mtp_kv,
                    &mut mtp_hidden,
                    session_meta,
                );
            }
            eprintln!("mtp_verify requested but module unavailable - sequential decode");
        }

        for index in 0..request.max_new_tokens {
            // Reference semantics: the FIRST new token is sampled from the
            // hidden of the last prefilled position (no extra forward pass).
            // Every following token embeds the freshly generated token and
            // runs it at its own position.
            if index > 0 {
                let token_pos = tokens.len() - 1;
                let current_token_id = tokens.last().copied().unwrap_or(0);
                self.fill_hidden_from_embedding_or_fallback(
                    &mut scheduler,
                    current_token_id,
                    &mut hidden_states,
                )?;
                self.run_model_for_one_token(
                    &mut scheduler,
                    token_pos,
                    current_token_id,
                    &mut hidden_states,
                    &mut kv_cache,
                    &mut scratch_buf,
                    &compute_config,
                    &gemm,
                    &request.route_expert_ids,
                )?;
            }

            self.prepare_sampling_hidden(&hidden_states, &mut sampling_hidden, &compute_config)?;
            let (next_token, next_logit, sample_source, candidates) = self
                .sample_next_token_from_lm_head(
                    &mut scheduler,
                    &sampling_hidden,
                    &mut scratch_buf,
                    &gemm,
                    &tokens,
                    &request.sampling,
                    stable_sample_seed(&request.request_id, index, request.sampling.seed),
                )?
                .map(|sample| {
                    (
                        sample.token_id,
                        sample.logit,
                        request.sampling.source(),
                        sample.candidates,
                    )
                })
                .unwrap_or_else(|| {
                    crate::vlog!("sampling source=hidden_state_placeholder");
                    (
                        sample_argmax_token(&hidden_states),
                        f32::NAN,
                        "hidden_state_placeholder",
                        Vec::new(),
                    )
                });
            tokens.push(next_token);
            write_event(
                writer,
                &GenerationEvent::Token {
                    request_id: request.request_id.clone(),
                    index,
                    token_id: next_token,
                    logit: next_logit,
                    source: sample_source,
                    candidates,
                },
            )?;

            // M2: compare the previous draft against the token the full
            // model actually produced (accept-rate preview, zero cost).
            if let Some(drafted) = mtp_last_draft.take() {
                mtp_checked += 1;
                if drafted == next_token {
                    mtp_hits += 1;
                }
                eprintln!(
                    "mtp_draft_check drafted={} actual={} hit={} hits={}/{}",
                    drafted,
                    next_token,
                    drafted == next_token,
                    mtp_hits,
                    mtp_checked
                );
            }

            if stop_token_ids.contains(&next_token) {
                if mtp_checked > 0 {
                    eprintln!(
                        "mtp_draft_summary hits={} checked={} rate={:.3}",
                        mtp_hits,
                        mtp_checked,
                        mtp_hits as f32 / mtp_checked as f32
                    );
                }
                if let Some((kv_layers, kv_slots)) = session_meta {
                    self.store_session(&tokens, &kv_cache, kv_layers, kv_slots);
                }
                write_event(
                    writer,
                    &GenerationEvent::Finished {
                        request_id: request.request_id,
                        generated_tokens: index + 1,
                        stop_reason: "stop_token",
                    },
                )?;
                writer.flush()?;
                return Ok(());
            }

            // M2: draft token t+2 from (hidden of the sampled position,
            // embed of t+1) BEFORE hidden_states is overwritten below.
            if !mtp_failed && index + 1 < request.max_new_tokens {
                if let (Some(module), Some(mtp_kv)) =
                    (mtp_module.as_ref(), mtp_kv_cache.as_mut())
                {
                    let draft_started = Instant::now();
                    match self.mtp_draft_next(
                        module,
                        &hidden_states,
                        next_token,
                        mtp_pos,
                        mtp_kv,
                        &mut mtp_hidden,
                        &mut scratch_buf,
                        &compute_config,
                        &gemm,
                    ) {
                        Ok(Some((draft_token, draft_logit))) => {
                            eprintln!(
                                "mtp_draft pos={} token_in={} draft={} logit={:.4} elapsed_ms={}",
                                mtp_pos,
                                next_token,
                                draft_token,
                                draft_logit,
                                draft_started.elapsed().as_millis()
                            );
                            mtp_last_draft = Some(draft_token);
                            mtp_pos += 1;
                        }
                        Ok(None) => {
                            eprintln!("mtp_draft skipped reason=missing_index_or_embed");
                        }
                        Err(err) => {
                            eprintln!(
                                "mtp_draft error={err} - drafting disabled for this request"
                            );
                            mtp_failed = true;
                        }
                    }
                }
            }

            self.fill_hidden_from_embedding_or_fallback(
                &mut scheduler,
                next_token,
                &mut hidden_states,
            )?;
        }

        if mtp_checked > 0 {
            eprintln!(
                "mtp_draft_summary hits={} checked={} rate={:.3}",
                mtp_hits,
                mtp_checked,
                mtp_hits as f32 / mtp_checked as f32
            );
        }
        if let Some((kv_layers, kv_slots)) = session_meta {
            self.store_session(&tokens, &kv_cache, kv_layers, kv_slots);
        }
        write_event(
            writer,
            &GenerationEvent::Finished {
                request_id: request.request_id,
                generated_tokens: request.max_new_tokens,
                stop_reason: "max_new_tokens",
            },
        )?;
        writer.flush()?;
        Ok(())
    }

    fn run_model_for_one_token(
        &self,
        scheduler: &mut MoEIoScheduler,
        token_pos: usize,
        current_token_id: u32,
        hidden_states: &mut [f32],
        kv_cache: &mut KVCache<'_>,
        scratch_buf: &mut [f32],
        compute_config: &ComputeConfig,
        gemm: &FusedInt4Gemm,
        route_expert_ids: &[u32],
    ) -> Result<(), GenerationError> {
        let layer_count = scheduler.manifest.layers.len() as u32;
        let mut compute_slot = 0usize;
        let local_layer_start = self.config.local_layer_start;
        let local_layer_end = self.config.local_layer_end.unwrap_or(u32::MAX);
        // Parking lot for prefetched blocks that complete while waiting for
        // the current layer (next-layer dense prefetch overlap).
        let mut parking: Vec<ReadyBlock> = Vec::new();
        for layer_index in 0..layer_count {
            let layer_desc = scheduler.layer(layer_index);
            let physical_layer_id = layer_desc.layer_id;
            if layer_desc.is_global_auxiliary() {
                crate::vlog!(
                    "skip layer={} source=global_auxiliary block_type=manifest",
                    physical_layer_id
                );
                continue;
            }
            if physical_layer_id < local_layer_start || physical_layer_id >= local_layer_end {
                crate::vlog!(
                    "skip layer={} source=local_layer_range start={} end={}",
                    physical_layer_id, local_layer_start, local_layer_end
                );
                continue;
            }
            let cached_dense = self.dense_cache.lock().unwrap().get(physical_layer_id);
            let mut ready: Vec<ReadyBlock>;
            let dense_arc: Option<std::sync::Arc<Vec<u8>>>;
            let dense_bytes: &[u8];
            if let Some(arc) = cached_dense {
                ready = Vec::new();
                dense_arc = Some(arc);
                dense_bytes = dense_arc.as_ref().unwrap().as_slice();
            } else {
                let parked = parking.iter().position(|block| {
                    block.ticket.layer_id == physical_layer_id
                        && matches!(block.ticket.kind, BlockKind::Dense)
                });
                if let Some(position) = parked {
                    ready = vec![parking.swap_remove(position)];
                } else {
                    scheduler.enqueue_dense_read(layer_index);
                    scheduler.submit_pending_until_full(usize::MAX)?;
                    ready = wait_for_layer_blocks_parked(
                        scheduler,
                        physical_layer_id,
                        1,
                        &mut parking,
                    )?;
                }
                let dense = ready
                    .iter()
                    .find(|block| matches!(block.ticket.kind, BlockKind::Dense))
                    .ok_or(GenerationError::MissingDenseBlock(physical_layer_id))?;
                let dense_ptr = block_to_ptr(scheduler, dense);
                dense_bytes = unsafe { std::slice::from_raw_parts(dense_ptr.ptr, dense_ptr.len) };
                dense_arc = None;
            }
            if is_non_compute_dense_block(dense_bytes)? {
                crate::vlog!(
                    "skip layer={} source=global_auxiliary dense_bytes={}",
                    physical_layer_id,
                    dense_bytes.len()
                );
                for block in &ready {
                    scheduler.release_ready_block(block)?;
                }
                continue;
            }
            if dense_arc.is_none() {
                self.dense_cache
                    .lock()
                    .unwrap()
                    .insert(physical_layer_id, dense_bytes);
            }
            // DeepSeek mHC without route hints: run the attention phase
            // first, route on the REAL gate input (post-attention working
            // stream, reference ordering), then read the selected experts
            // and finish the MoE phase.
            if compute_config.attention_kind == AttentionKind::DeepSeekV4Mla
                && compute_config.hc_mult > 1
                && route_expert_ids.is_empty()
            {
                // G1 speculative prefetch: route on the PRE-attention hidden
                // (cheap gate gemv) and start the expert reads BEFORE the
                // attention compute, so I/O overlaps compute. Hash layers
                // (tid2eid) predict exactly; for the others hits are reused,
                // misses are read as usual, spurious blocks are released.
                let spec_prefetch = std::env::var("ZC_SPEC_PREFETCH")
                    .map(|value| value != "0")
                    .unwrap_or(true);
                let mut spec_ids: Vec<u32> = Vec::new();
                if spec_prefetch && scheduler.available_buffers() > self.config.active_experts as usize
                {
                    let spec_hidden =
                        &hidden_states[..compute_config.hidden_size.min(hidden_states.len())];
                    if let Ok((spec_routes, _)) = routes_for_layer_with_dense_router(
                        scheduler,
                        layer_index,
                        physical_layer_id,
                        route_expert_ids,
                        self.config.active_experts,
                        dense_bytes,
                        spec_hidden,
                        scratch_buf,
                        gemm,
                        compute_config,
                        current_token_id,
                    ) {
                        if !spec_routes.is_empty() {
                            // G2: only prefetch experts that are not already
                            // in the RAM cache.
                            let to_fetch: Vec<ExpertRoute> = {
                                let mut ram = self.expert_ram.lock().unwrap();
                                spec_routes
                                    .iter()
                                    .filter(|route| {
                                        ram.get(physical_layer_id, route.expert_id).is_none()
                                    })
                                    .cloned()
                                    .collect()
                            };
                            if !to_fetch.is_empty() {
                                scheduler.enqueue_selected_experts(layer_index, &to_fetch)?;
                                scheduler.submit_pending_until_full(usize::MAX)?;
                            }
                            spec_ids = to_fetch.iter().map(|route| route.expert_id).collect();
                        }
                    }
                }
                let mut carry = unsafe {
                    compute_layer_deepseek_mhc_phase1(
                        dense_bytes,
                        compute_slot,
                        token_pos,
                        hidden_states,
                        kv_cache,
                        &mut ComputeScratch {
                            dequant_tile_f32: &mut *scratch_buf,
                        },
                        compute_config,
                    )?
                };
                let (mut routes, route_source) = routes_for_layer_with_dense_router(
                    scheduler,
                    layer_index,
                    physical_layer_id,
                    route_expert_ids,
                    self.config.active_experts,
                    dense_bytes,
                    &carry.gate_input,
                    scratch_buf,
                    gemm,
                    compute_config,
                    current_token_id,
                )?;
                normalize_route_weights(&mut routes, route_source, compute_config);
                crate::vlog!(
                    "router layer={} source={}_post_attention experts={:?} weights={:?}",
                    physical_layer_id,
                    route_source,
                    routes.iter().map(|route| route.expert_id).collect::<Vec<_>>(),
                    routes.iter().map(|route| route.score).collect::<Vec<_>>()
                );
                // G2: experts already in the RAM cache need no I/O at all.
                let ram_guards: Vec<(std::sync::Arc<Vec<u8>>, u32, f32)> = {
                    let mut ram = self.expert_ram.lock().unwrap();
                    routes
                        .iter()
                        .filter_map(|route| {
                            ram.get(physical_layer_id, route.expert_id)
                                .map(|arc| (arc, route.expert_id, route.score))
                        })
                        .collect()
                };
                // G1: the speculative experts are already in flight. First
                // collect them and RELEASE the spurious ones immediately
                // (frees I/O buffers), then enqueue only the misses.
                let missing: Vec<ExpertRoute> = routes
                    .iter()
                    .filter(|route| {
                        !spec_ids.contains(&route.expert_id)
                            && !ram_guards
                                .iter()
                                .any(|(_, expert_id, _)| *expert_id == route.expert_id)
                    })
                    .cloned()
                    .collect();
                if !ram_guards.is_empty() {
                    crate::vlog!(
                        "expert_ram layer={} hits={} of {}",
                        physical_layer_id,
                        ram_guards.len(),
                        routes.len()
                    );
                }
                if !spec_ids.is_empty() {
                    let spec_blocks = wait_for_layer_blocks_parked(
                        scheduler,
                        physical_layer_id,
                        spec_ids.len(),
                        &mut parking,
                    )?;
                    for block in spec_blocks {
                        let is_hit = matches!(block.ticket.kind, BlockKind::Expert)
                            && routes
                                .iter()
                                .any(|route| route.expert_id == block.ticket.expert_id);
                        if is_hit {
                            ready.push(block);
                        } else {
                            scheduler.release_ready_block(&block)?;
                        }
                    }
                    crate::vlog!(
                        "spec_prefetch layer={} spec={} hits={} misses={}",
                        physical_layer_id,
                        spec_ids.len(),
                        spec_ids.len().saturating_sub(missing.len()),
                        missing.len()
                    );
                }
                if !missing.is_empty() {
                    scheduler.enqueue_selected_experts(layer_index, &missing)?;
                }
                // Overlap: prefetch the next layer's dense block while this
                // layer's experts stream in (deterministic, not speculative).
                // Keep at least one buffer free for the expert stream.
                let next_index = layer_index + 1;
                if next_index < layer_count && scheduler.available_buffers() >= 2 {
                    let next_layer = scheduler.layer(next_index);
                    let next_physical = next_layer.layer_id;
                    if next_physical < local_layer_end
                        && !next_layer.is_global_auxiliary()
                        && self.dense_cache.lock().unwrap().get(next_physical).is_none()
                        && !parking
                            .iter()
                            .any(|block| block.ticket.layer_id == next_physical)
                    {
                        scheduler.enqueue_dense_read(next_index);
                    }
                }
                scheduler.submit_pending_until_full(usize::MAX)?;
                if !missing.is_empty() {
                    ready.extend(wait_for_layer_blocks_parked(
                        scheduler,
                        physical_layer_id,
                        missing.len(),
                        &mut parking,
                    )?);
                }
                let mut experts = ready
                    .iter()
                    .filter(|block| {
                        matches!(block.ticket.kind, BlockKind::Expert)
                            && routes
                                .iter()
                                .any(|route| route.expert_id == block.ticket.expert_id)
                    })
                    .map(|block| block_to_ptr_with_route(scheduler, block, &routes))
                    .collect::<Vec<_>>();
                // G2: append the RAM-cached experts (Arc guards keep the
                // bytes alive through phase 2).
                for (arc, expert_id, weight) in &ram_guards {
                    experts.push(IoBlockPtr {
                        kind: BlockKind::Expert,
                        layer_id: physical_layer_id,
                        expert_id: *expert_id,
                        route_weight: *weight,
                        ptr: arc.as_ptr(),
                        len: arc.len(),
                    });
                }
                let mut stats = LayerComputeStats {
                    dense_bytes: dense_bytes.len(),
                    expert_bytes: 0,
                    experts: experts.len(),
                    dequantized_values: 0,
                };
                unsafe {
                    compute_layer_deepseek_mhc_phase2(
                        dense_bytes,
                        compute_slot,
                        &experts,
                        &mut carry,
                        hidden_states,
                        &mut ComputeScratch {
                            dequant_tile_f32: &mut *scratch_buf,
                        },
                        compute_config,
                        gemm,
                        &mut stats,
                    )?;
                }
                crate::vlog!(
                    "compute layer={} compute_slot={} dense_bytes={} expert_bytes={} experts={} dequantized_values={}",
                    physical_layer_id,
                    compute_slot,
                    stats.dense_bytes,
                    stats.expert_bytes,
                    stats.experts,
                    stats.dequantized_values
                );
                // G2: remember the experts read from disk this token so hot
                // ones skip their NVMe read next time.
                {
                    let mut ram = self.expert_ram.lock().unwrap();
                    for block in &ready {
                        if matches!(block.ticket.kind, BlockKind::Expert)
                            && routes
                                .iter()
                                .any(|route| route.expert_id == block.ticket.expert_id)
                        {
                            let block_ptr = block_to_ptr(scheduler, block);
                            let bytes = unsafe {
                                std::slice::from_raw_parts(block_ptr.ptr, block_ptr.len)
                            };
                            ram.insert(physical_layer_id, block.ticket.expert_id, bytes);
                        }
                    }
                }
                for block in &ready {
                    scheduler.release_ready_block(block)?;
                }
                compute_slot += 1;
                continue;
            }

            // Router input: first hidden copy only. With mHC this is a
            // prefetch approximation (the reference gate consumes the
            // post-attention working stream); hash layers are exact since
            // they route by token id.
            let router_hidden = &hidden_states[..compute_config.hidden_size.min(hidden_states.len())];
            let (mut routes, route_source) = routes_for_layer_with_dense_router(
                scheduler,
                layer_index,
                physical_layer_id,
                route_expert_ids,
                self.config.active_experts,
                dense_bytes,
                router_hidden,
                scratch_buf,
                gemm,
                compute_config,
                current_token_id,
            )?;
            normalize_route_weights(&mut routes, route_source, compute_config);
            crate::vlog!(
                "router layer={} source={} experts={:?} weights={:?}",
                physical_layer_id,
                route_source,
                routes
                    .iter()
                    .map(|route| route.expert_id)
                    .collect::<Vec<_>>(),
                routes
                    .iter()
                    .map(|route| route.score)
                    .collect::<Vec<_>>()
            );
            scheduler.enqueue_selected_experts(layer_index, &routes)?;
            scheduler.submit_pending_until_full(usize::MAX)?;
            if !routes.is_empty() {
                ready.extend(wait_for_layer_blocks(
                    scheduler,
                    physical_layer_id,
                    routes.len(),
                )?);
            }
            let experts = ready
                .iter()
                .filter(|block| matches!(block.ticket.kind, BlockKind::Expert))
                .map(|block| block_to_ptr_with_route(scheduler, block, &routes))
                .collect::<Vec<_>>();
            let mut scratch = ComputeScratch {
                dequant_tile_f32: scratch_buf,
            };

            let stats = unsafe {
                compute_layer(
                    compute_slot,
                    token_pos,
                    dense_bytes.as_ptr(),
                    dense_bytes.len(),
                    &experts,
                    hidden_states,
                    kv_cache,
                    &mut scratch,
                    compute_config,
                    gemm,
                )?
            };
            crate::vlog!(
                "compute layer={} compute_slot={} dense_bytes={} expert_bytes={} experts={} dequantized_values={}",
                physical_layer_id,
                compute_slot,
                stats.dense_bytes,
                stats.expert_bytes,
                stats.experts,
                stats.dequantized_values
            );

            for block in &ready {
                scheduler.release_ready_block(block)?;
            }
            compute_slot += 1;
        }
        // Release any prefetched blocks that were never consumed (e.g. the
        // next layer turned out to be skipped or the loop ended).
        for block in &parking {
            scheduler.release_ready_block(block)?;
        }
        Ok(())
    }

    /// Replicate the first hidden copy across the remaining mHC copies
    /// (reference: `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`).
    fn replicate_hidden_for_hc(hidden_states: &mut [f32], hidden: usize) {
        if hidden_states.len() <= hidden {
            return;
        }
        let (first, rest) = hidden_states.split_at_mut(hidden);
        for copy in rest.chunks_mut(hidden) {
            let span = copy.len().min(hidden);
            copy[..span].copy_from_slice(&first[..span]);
        }
    }

    fn fill_hidden_from_embedding_or_fallback(
        &self,
        scheduler: &mut MoEIoScheduler,
        token_id: u32,
        hidden_states: &mut [f32],
    ) -> Result<(), GenerationError> {
        if self.try_fill_hidden_from_embedding(scheduler, token_id, hidden_states)? {
            crate::vlog!("embedding source=embed_tokens token_id={}", token_id);
        } else {
            crate::vlog!("embedding source=fill_token_hidden token_id={}", token_id);
            fill_token_hidden(hidden_states, token_id);
        }
        // mHC: the embedding fills the first copy; replicate across copies.
        let hidden = self.manifest.header.hidden_size as usize;
        if hidden > 0 && hidden_states.len() > hidden {
            Self::replicate_hidden_for_hc(hidden_states, hidden);
        }
        Ok(())
    }

    /// Batch prefill (E1): wrapper over `run_positions_batch` for the whole
    /// prompt at positions 0..n.
    #[allow(clippy::too_many_arguments)]
    fn run_prefill_batch(
        &self,
        scheduler: &mut MoEIoScheduler,
        token_ids: &[u32],
        hidden_states: &mut [f32],
        kv_cache: &mut KVCache<'_>,
        scratch_buf: &mut [f32],
        compute_config: &ComputeConfig,
        gemm: &FusedInt4Gemm,
    ) -> Result<(), GenerationError> {
        let npos = token_ids.len();
        let hc = compute_config.hc_mult.max(1);
        let hc_dim = compute_config.hidden_size * hc;
        let items: Vec<(u32, usize)> = token_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(position, token_id)| (token_id, position))
            .collect();
        let mut hiddens = vec![0.0f32; npos * hc_dim];
        self.run_positions_batch(
            scheduler,
            &items,
            &mut hiddens,
            kv_cache,
            scratch_buf,
            compute_config,
            gemm,
        )?;
        hidden_states[..hc_dim]
            .copy_from_slice(&hiddens[(npos - 1) * hc_dim..(npos - 1) * hc_dim + hc_dim]);
        eprintln!("prefill_batch positions={} mode=layer_major_union", npos);
        Ok(())
    }

    /// Layer-major batched forward of arbitrary (token, absolute position)
    /// items - the E1 machinery, generalized for the M3 speculative verify
    /// pass. Each layer's dense block and the UNION of the selected experts
    /// are read from NVMe once per layer instead of once per position. Math
    /// is identical to the per-token path (phase1 + post-attention routing
    /// per position, routed contributions accumulated expert-major with
    /// per-position normalized weights, shared expert + hc_post per
    /// position). Causality holds because items are processed in order
    /// within each layer. `hiddens` (npos * hc_dim) receives the final hc
    /// state of every item.
    #[allow(clippy::too_many_arguments)]
    fn run_positions_batch(
        &self,
        scheduler: &mut MoEIoScheduler,
        items: &[(u32, usize)],
        hiddens: &mut [f32],
        kv_cache: &mut KVCache<'_>,
        scratch_buf: &mut [f32],
        compute_config: &ComputeConfig,
        gemm: &FusedInt4Gemm,
    ) -> Result<(), GenerationError> {
        let npos = items.len();
        let hc = compute_config.hc_mult.max(1);
        let hidden = compute_config.hidden_size;
        let hc_dim = hidden * hc;
        // ZC_PROF=1: per-phase wall-clock breakdown of the whole pass.
        let prof = std::env::var("ZC_PROF").map(|v| v == "1").unwrap_or(false);
        let mut prof_dense = Duration::ZERO;
        let mut prof_phase1 = Duration::ZERO;
        let mut prof_route = Duration::ZERO;
        let mut prof_expert_io = Duration::ZERO;
        let mut prof_expert_compute = Duration::ZERO;
        let mut prof_finish = Duration::ZERO;
        let pass_started = Instant::now();
        for (slot, (token_id, _position)) in items.iter().enumerate() {
            self.fill_hidden_from_embedding_or_fallback(
                scheduler,
                *token_id,
                &mut hiddens[slot * hc_dim..(slot + 1) * hc_dim],
            )?;
        }

        let layer_count = scheduler.manifest.layers.len() as u32;
        let local_layer_start = self.config.local_layer_start;
        let local_layer_end = self.config.local_layer_end.unwrap_or(u32::MAX);
        let chunk_size = self.config.io_buffer_count.saturating_sub(2).max(1);
        let mut compute_slot = 0usize;
        for layer_index in 0..layer_count {
            let layer_desc = scheduler.layer(layer_index);
            let physical_layer_id = layer_desc.layer_id;
            if layer_desc.is_global_auxiliary() {
                crate::vlog!(
                    "skip layer={} source=global_auxiliary block_type=manifest",
                    physical_layer_id
                );
                continue;
            }
            if physical_layer_id < local_layer_start || physical_layer_id >= local_layer_end {
                crate::vlog!(
                    "skip layer={} source=local_layer_range start={} end={}",
                    physical_layer_id, local_layer_start, local_layer_end
                );
                continue;
            }

            let dense_started = Instant::now();
            let cached_dense = self.dense_cache.lock().unwrap().get(physical_layer_id);
            let mut dense_blocks: Vec<ReadyBlock> = Vec::new();
            let dense_arc: Option<std::sync::Arc<Vec<u8>>>;
            let dense_bytes: &[u8];
            if let Some(arc) = cached_dense {
                dense_arc = Some(arc);
                dense_bytes = dense_arc.as_ref().unwrap().as_slice();
            } else {
                scheduler.enqueue_dense_read(layer_index);
                scheduler.submit_pending_until_full(usize::MAX)?;
                dense_blocks = wait_for_layer_blocks(scheduler, physical_layer_id, 1)?;
                let dense = dense_blocks
                    .iter()
                    .find(|block| matches!(block.ticket.kind, BlockKind::Dense))
                    .ok_or(GenerationError::MissingDenseBlock(physical_layer_id))?;
                let dense_ptr = block_to_ptr(scheduler, dense);
                dense_bytes = unsafe { std::slice::from_raw_parts(dense_ptr.ptr, dense_ptr.len) };
                dense_arc = None;
            }
            if is_non_compute_dense_block(dense_bytes)? {
                crate::vlog!(
                    "skip layer={} source=global_auxiliary dense_bytes={}",
                    physical_layer_id,
                    dense_bytes.len()
                );
                for block in &dense_blocks {
                    scheduler.release_ready_block(block)?;
                }
                continue;
            }
            if dense_arc.is_none() {
                self.dense_cache
                    .lock()
                    .unwrap()
                    .insert(physical_layer_id, dense_bytes);
            }

            prof_dense += dense_started.elapsed();

            // Phase 1 + post-attention routing for every item, in order.
            // T1c: with the experts pack the ring is fully async, so each
            // position's expert reads are enqueued AS SOON as its routing is
            // known - the I/O overlaps the remaining positions' attention.
            // Opt-in (ZC_PROGRESSIVE=1): stalls under investigation - the
            // first expert wait starves when the pending queue and the
            // fixed-buffer budget interleave badly. The pack alone already
            // removes the per-shard open/close cost.
            let progressive_io = scheduler.io.has_pack()
                && std::env::var("ZC_PROGRESSIVE").map(|v| v == "1").unwrap_or(false);
            let mut enqueued_experts: std::collections::BTreeSet<u32> =
                std::collections::BTreeSet::new();
            let mut carries: Vec<DeepSeekMhcCarry> = Vec::with_capacity(npos);
            let mut per_pos_routes: Vec<Vec<ExpertRoute>> = Vec::with_capacity(npos);
            // T2: batched phase1 - every projection matrix is streamed once
            // for ALL positions instead of once per position (phase1 was
            // measured memory-bound on q/o weights, ~58% of the pass).
            let phase1_started = Instant::now();
            let batch_positions: Vec<usize> =
                items.iter().map(|&(_, position)| position).collect();
            let batch_carries = unsafe {
                compute_layer_deepseek_mhc_phase1_batch(
                    dense_bytes,
                    compute_slot,
                    &batch_positions,
                    &mut hiddens[..npos * hc_dim],
                    kv_cache,
                    &mut ComputeScratch {
                        dequant_tile_f32: &mut *scratch_buf,
                    },
                    compute_config,
                )?
            };
            prof_phase1 += phase1_started.elapsed();
            for (&(token_id, _position), carry) in items.iter().zip(batch_carries) {
                let route_started = Instant::now();
                let (mut routes, route_source) = routes_for_layer_with_dense_router(
                    scheduler,
                    layer_index,
                    physical_layer_id,
                    &[],
                    self.config.active_experts,
                    dense_bytes,
                    &carry.gate_input,
                    scratch_buf,
                    gemm,
                    compute_config,
                    token_id,
                )?;
                normalize_route_weights(&mut routes, route_source, compute_config);
                prof_route += route_started.elapsed();
                if progressive_io {
                    for route in &routes {
                        if enqueued_experts.insert(route.expert_id) {
                            scheduler.enqueue_expert_read(layer_index, route.expert_id)?;
                        }
                    }
                    scheduler.submit_pending_until_full(usize::MAX)?;
                }
                carries.push(carry);
                per_pos_routes.push(routes);
            }

            // Union of the selected experts with per-position normalized
            // weights (mirrors compute_active_experts normalization).
            let mut union: std::collections::BTreeMap<u32, Vec<(usize, f32)>> =
                std::collections::BTreeMap::new();
            for (position, routes) in per_pos_routes.iter().enumerate() {
                if routes.is_empty() {
                    continue;
                }
                let sum: f32 = routes.iter().map(|route| route.score).sum();
                let finite =
                    sum.is_finite() && sum > 0.0 && routes.iter().all(|r| r.score.is_finite());
                let uniform = 1.0 / routes.len() as f32;
                for route in routes {
                    let weight = if finite { route.score / sum } else { uniform };
                    union
                        .entry(route.expert_id)
                        .or_default()
                        .push((position, weight));
                }
            }

            let mut gate_flat = vec![0.0f32; npos * hidden];
            for (position, carry) in carries.iter().enumerate() {
                gate_flat[position * hidden..(position + 1) * hidden]
                    .copy_from_slice(&carry.gate_input[..hidden]);
            }
            let mut routed_acc = vec![0.0f32; npos * hidden];
            let mut routed_seen = vec![false; npos];
            let expert_ids: Vec<u32> = union.keys().copied().collect();
            let mut stats = LayerComputeStats::default();
            for chunk in expert_ids.chunks(chunk_size) {
                let chunk_routes: Vec<ExpertRoute> = chunk
                    .iter()
                    .map(|&expert_id| ExpertRoute {
                        expert_id,
                        score: 1.0,
                    })
                    .collect();
                let io_started = Instant::now();
                if !progressive_io {
                    scheduler.enqueue_selected_experts(layer_index, &chunk_routes)?;
                }
                scheduler.submit_pending_until_full(usize::MAX)?;
                let blocks =
                    wait_for_layer_blocks(scheduler, physical_layer_id, chunk_routes.len())?;
                prof_expert_io += io_started.elapsed();
                let compute_started = Instant::now();
                for block in &blocks {
                    if !matches!(block.ticket.kind, BlockKind::Expert) {
                        continue;
                    }
                    let Some(targets) = union.get(&block.ticket.expert_id) else {
                        continue;
                    };
                    for &(position, _) in targets {
                        routed_seen[position] = true;
                    }
                    let block_ptr = block_to_ptr(scheduler, block);
                    let expert_bytes =
                        unsafe { std::slice::from_raw_parts(block_ptr.ptr, block_ptr.len) };
                    unsafe {
                        accumulate_expert_for_positions(
                            expert_bytes,
                            &gate_flat,
                            hidden,
                            targets,
                            &mut routed_acc,
                            &mut ComputeScratch {
                                dequant_tile_f32: &mut *scratch_buf,
                            },
                            compute_config,
                            gemm,
                            &mut stats,
                        )?;
                    }
                }
                prof_expert_compute += compute_started.elapsed();
                let release_started = Instant::now();
                for block in &blocks {
                    scheduler.release_ready_block(block)?;
                }
                prof_expert_io += release_started.elapsed();
            }

            let finish_started = Instant::now();
            for position in 0..npos {
                let routed = if routed_seen[position] {
                    Some(&routed_acc[position * hidden..(position + 1) * hidden])
                } else {
                    None
                };
                unsafe {
                    finish_layer_deepseek_mhc_phase2_batch(
                        dense_bytes,
                        &mut carries[position],
                        routed.map(|slice| &slice[..]),
                        &mut hiddens[position * hc_dim..(position + 1) * hc_dim],
                        &mut ComputeScratch {
                            dequant_tile_f32: &mut *scratch_buf,
                        },
                        compute_config,
                        gemm,
                        &mut stats,
                    )?;
                }
            }
            prof_finish += finish_started.elapsed();
            crate::vlog!(
                "compute_batch layer={} positions={} experts_union={} expert_bytes={} dequantized_values={}",
                physical_layer_id,
                npos,
                expert_ids.len(),
                stats.expert_bytes,
                stats.dequantized_values
            );
            for block in &dense_blocks {
                scheduler.release_ready_block(block)?;
            }
            compute_slot += 1;
        }

        if prof {
            eprintln!(
                "prof_phase1_detail {}",
                crate::compute::phase1_prof::dump_and_reset()
            );
            let total = pass_started.elapsed();
            eprintln!(
                "prof_pass npos={} total_ms={} dense_ms={} phase1_ms={} route_ms={} expert_io_ms={} expert_compute_ms={} finish_ms={} other_ms={}",
                npos,
                total.as_millis(),
                prof_dense.as_millis(),
                prof_phase1.as_millis(),
                prof_route.as_millis(),
                prof_expert_io.as_millis(),
                prof_expert_compute.as_millis(),
                prof_finish.as_millis(),
                total
                    .saturating_sub(prof_dense)
                    .saturating_sub(prof_phase1)
                    .saturating_sub(prof_route)
                    .saturating_sub(prof_expert_io)
                    .saturating_sub(prof_expert_compute)
                    .saturating_sub(prof_finish)
                    .as_millis()
            );
        }
        Ok(())
    }

    fn try_fill_hidden_from_embedding(
        &self,
        scheduler: &mut MoEIoScheduler,
        token_id: u32,
        hidden_states: &mut [f32],
    ) -> Result<bool, GenerationError> {
        if let Some(row_index) = &self.row_index {
            if row_index.read_bf16_row("embed.weight", token_id as usize, hidden_states)? {
                crate::vlog!("embedding source=row_tensor_index token_id={}", token_id);
                return Ok(true);
            }
        }
        let layer_count = scheduler.manifest.layers.len() as u32;
        for layer_index in 0..layer_count {
            let physical_layer_id = scheduler.layer(layer_index).layer_id;
            scheduler.enqueue_dense_read(layer_index);
            scheduler.submit_pending_until_full(usize::MAX)?;
            let ready = wait_for_layer_blocks(scheduler, physical_layer_id, 1)?;
            let dense = ready
                .iter()
                .find(|block| matches!(block.ticket.kind, BlockKind::Dense))
                .ok_or(GenerationError::MissingDenseBlock(physical_layer_id))?;
            let dense_ptr = block_to_ptr(scheduler, dense);
            let dense_bytes = unsafe { std::slice::from_raw_parts(dense_ptr.ptr, dense_ptr.len) };
            let found = token_embedding_from_block(dense_bytes, token_id, hidden_states)?;
            for block in &ready {
                scheduler.release_ready_block(block)?;
            }
            if found {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Produce the single-stream hidden vector for the LM head. With mHC
    /// active this applies `hc_head` pooling (sigmoid gating over the hc
    /// copies) followed by the final weighted RMS norm (`norm.weight`),
    /// matching `ParallelHead.forward` in the reference implementation.
    fn prepare_sampling_hidden(
        &self,
        hidden_states: &[f32],
        sampling_hidden: &mut [f32],
        compute_config: &ComputeConfig,
    ) -> Result<(), GenerationError> {
        let hidden = compute_config.hidden_size;
        let hc = compute_config.hc_mult.max(1);
        if hc <= 1 || hidden_states.len() < hc * hidden {
            let span = sampling_hidden.len().min(hidden_states.len());
            sampling_hidden[..span].copy_from_slice(&hidden_states[..span]);
            return Ok(());
        }
        let mut pooled_applied = false;
        if let Some(row_index) = &self.row_index {
            let fn_bytes = hc * hc * hidden * 4;
            if let (Some(fn_rows), Some(scale_raw), Some(base_raw)) = (
                row_index.read_tensor_prefix("hc_head_fn", fn_bytes)?,
                row_index.read_tensor_prefix("hc_head_scale", 4)?,
                row_index.read_tensor_prefix("hc_head_base", hc * 4)?,
            ) {
                if fn_rows.len() == fn_bytes && scale_raw.len() == 4 && base_raw.len() == hc * 4 {
                    let scale =
                        f32::from_le_bytes([scale_raw[0], scale_raw[1], scale_raw[2], scale_raw[3]]);
                    let base = base_raw
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<_>>();
                    crate::compute::hc_head_pool(
                        &fn_rows,
                        scale,
                        &base,
                        hidden_states,
                        sampling_hidden,
                        hc,
                        hidden,
                        compute_config.hc_eps,
                    )?;
                    pooled_applied = true;
                }
            }
        }
        if !pooled_applied {
            // Fallback: first copy only (identity pooling).
            sampling_hidden[..hidden].copy_from_slice(&hidden_states[..hidden]);
        }
        // Final weighted RMS norm before the LM head.
        let mut norm_applied = false;
        if let Some(row_index) = &self.row_index {
            if let Some(norm_raw) = row_index.read_tensor_prefix("norm.weight", hidden * 2)? {
                if norm_raw.len() == hidden * 2 {
                    let mean_square = sampling_hidden[..hidden]
                        .iter()
                        .map(|value| value * value)
                        .sum::<f32>()
                        / hidden as f32;
                    let inv_rms = 1.0 / (mean_square + compute_config.hc_eps).sqrt();
                    for (value, chunk) in sampling_hidden[..hidden]
                        .iter_mut()
                        .zip(norm_raw.chunks_exact(2))
                    {
                        let weight_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        let weight = f32::from_bits((weight_bits as u32) << 16);
                        *value = *value * inv_rms * weight;
                    }
                    norm_applied = true;
                }
            }
        }
        if env::var("ZC_MHC_DEBUG").is_ok() {
            let pooled_l2 = sampling_hidden[..hidden]
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                .sqrt();
            crate::vlog!(
                "mhc_debug pooled_l2={:.6} sample={:?}",
                pooled_l2,
                &sampling_hidden[..8.min(hidden)]
            );
        }
        crate::vlog!(
            "math_fidelity component=hc_head_pool applied={} final_norm_applied={}",
            pooled_applied, norm_applied
        );
        Ok(())
    }

    fn sample_next_token_from_lm_head(
        &self,
        scheduler: &mut MoEIoScheduler,
        hidden_states: &[f32],
        scratch_buf: &mut [f32],
        gemm: &FusedInt4Gemm,
        previous_tokens: &[u32],
        sampling: &SamplingConfig,
        sample_seed: u64,
    ) -> Result<Option<SampledToken>, GenerationError> {
        if let Some(row_index) = &self.row_index {
            if let Some(candidates) =
                row_index.topk_bf16_lm_head(hidden_states, scratch_buf, sampling.top_k)?
            {
                let scanned_rows = row_index.lm_head_row_count().unwrap_or(candidates.len());
                let sample = sample_from_candidates(
                    candidates,
                    previous_tokens,
                    sampling,
                    sample_seed,
                );
                crate::vlog!(
                    "sampling source=row_tensor_index rows={} scratch_logits={} top_k={} top_p={} temperature={} repetition_penalty={}",
                    scanned_rows,
                    scratch_buf.len(),
                    sampling.top_k,
                    sampling.top_p,
                    sampling.temperature,
                    sampling.repetition_penalty
                );
                return Ok(Some(sample));
            }
        }
        let layer_count = scheduler.manifest.layers.len() as u32;
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(sampling.top_k);
        let mut best_layer: Option<u32> = None;
        let mut scanned_rows = 0usize;
        for layer_index in 0..layer_count {
            let layer_desc = scheduler.layer(layer_index);
            if scanned_rows > 0 && !layer_desc.is_global_auxiliary() {
                break;
            }
            let physical_layer_id = layer_desc.layer_id;
            scheduler.enqueue_dense_read(layer_index);
            scheduler.submit_pending_until_full(usize::MAX)?;
            let ready = wait_for_layer_blocks(scheduler, physical_layer_id, 1)?;
            let dense = ready
                .iter()
                .find(|block| matches!(block.ticket.kind, BlockKind::Dense))
                .ok_or(GenerationError::MissingDenseBlock(physical_layer_id))?;
            let dense_ptr = block_to_ptr(scheduler, dense);
            let dense_bytes = unsafe { std::slice::from_raw_parts(dense_ptr.ptr, dense_ptr.len) };
            let candidate = unsafe {
                lm_head_topk_score_chunked_from_block(
                    dense_bytes,
                    hidden_states,
                    scratch_buf,
                    gemm,
                    sampling.top_k,
                )?
            };
            let lm_head_rows = lm_head_row_count_from_block(dense_bytes)?;
            if let Some(rows) = lm_head_rows {
                scanned_rows += rows;
            }
            for block in &ready {
                scheduler.release_ready_block(block)?;
            }
            for (token, logit) in candidate {
                push_generation_topk(&mut best, sampling.top_k, token, logit);
                best_layer = Some(physical_layer_id);
            }
        }
        if !best.is_empty() {
            let sample = sample_from_candidates(
                best,
                previous_tokens,
                sampling,
                sample_seed,
            );
            crate::vlog!(
                "sampling source={} layer={} rows={} scratch_logits={} top_k={} top_p={} temperature={} repetition_penalty={}",
                sampling.source(),
                best_layer.unwrap_or(0),
                scanned_rows,
                scratch_buf.len(),
                sampling.top_k,
                sampling.top_p,
                sampling.temperature,
                sampling.repetition_penalty
            );
            return Ok(Some(sample));
        }
        Ok(None)
    }

    #[allow(dead_code)]
    pub async fn forward_hidden_to_next_node(
        &self,
        request_id: String,
        token_index: u32,
        next_layer: u32,
        hidden_states: &[f32],
    ) -> Result<Option<crate::server::cluster::ClusterMessage>, GenerationError> {
        let Some(addr) = self.config.cluster_next_node else {
            return Ok(None);
        };
        let response = crate::server::cluster::send_hidden_state(
            addr,
            request_id,
            token_index,
            next_layer,
            hidden_states,
        )
        .await
        .map_err(|err| GenerationError::Io(std::io::Error::new(std::io::ErrorKind::Other, err)))?;
        Ok(Some(response))
    }
}

fn write_speculative_finish<W: Write>(
    writer: &mut W,
    request_id: String,
    generated_tokens: u32,
    stop_reason: &'static str,
    hits: u32,
    checked: u32,
) -> Result<(), GenerationError> {
    if checked > 0 {
        eprintln!(
            "mtp_draft_summary hits={} checked={} rate={:.3} mode=verify",
            hits,
            checked,
            hits as f32 / checked as f32
        );
    }
    write_event(
        writer,
        &GenerationEvent::Finished {
            request_id,
            generated_tokens,
            stop_reason,
        },
    )?;
    writer.flush()?;
    Ok(())
}

fn push_generation_topk(best: &mut Vec<(u32, f32)>, top_k: usize, token: u32, logit: f32) {
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

fn sample_from_candidates(
    mut candidates: Vec<(u32, f32)>,
    previous_tokens: &[u32],
    sampling: &SamplingConfig,
    seed: u64,
) -> SampledToken {
    for (token, logit) in &mut candidates {
        if sampling.repetition_penalty != 1.0 && previous_tokens.contains(token) {
            if *logit >= 0.0 {
                *logit /= sampling.repetition_penalty;
            } else {
                *logit *= sampling.repetition_penalty;
            }
        }
    }
    candidates.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    if sampling.temperature <= 0.0 || sampling.top_k <= 1 {
        let (token_id, logit) = candidates[0];
        return SampledToken {
            token_id,
            logit,
            candidates: candidates
                .into_iter()
                .map(|(token_id, logit)| TokenCandidate { token_id, logit })
                .collect(),
        };
    }

    let temperature = sampling.temperature.max(1.0e-6);
    let max_logit = candidates[0].1;
    let mut weighted = candidates
        .into_iter()
        .map(|(token, logit)| {
            let weight = ((logit - max_logit) / temperature).exp();
            (token, logit, weight)
        })
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|(_, _, weight)| *weight).sum::<f32>();
    if total <= 0.0 || !total.is_finite() {
        return SampledToken {
            token_id: weighted[0].0,
            logit: weighted[0].1,
            candidates: weighted
                .into_iter()
                .map(|(token_id, logit, _)| TokenCandidate { token_id, logit })
                .collect(),
        };
    }
    for (_, _, weight) in &mut weighted {
        *weight /= total;
    }
    if sampling.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for (_, _, prob) in &weighted {
            cumulative += *prob;
            keep += 1;
            if cumulative >= sampling.top_p {
                break;
            }
        }
        weighted.truncate(keep.max(1));
        let renorm = weighted.iter().map(|(_, _, prob)| *prob).sum::<f32>();
        if renorm > 0.0 {
            for (_, _, prob) in &mut weighted {
                *prob /= renorm;
            }
        }
    }
    let mut threshold = sample_unit(seed);
    let candidate_view = weighted
        .iter()
        .map(|(token_id, logit, _)| TokenCandidate {
            token_id: *token_id,
            logit: *logit,
        })
        .collect::<Vec<_>>();
    for (token_id, logit, prob) in weighted {
        if threshold <= prob {
            return SampledToken {
                token_id,
                logit,
                candidates: candidate_view,
            };
        }
        threshold -= prob;
    }
    unreachable!("weighted candidates are non-empty")
}

fn stable_sample_seed(request_id: &str, index: u32, seed: u64) -> u64 {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15 ^ u64::from(index);
    for byte in request_id.as_bytes() {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 27;
    }
    state
}

fn sample_unit(seed: u64) -> f32 {
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    state ^= state >> 30;
    state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^= state >> 31;
    let value = (state >> 40) as u32;
    (value as f32) / ((1u32 << 24) as f32)
}

fn wait_for_layer_blocks(
    scheduler: &mut MoEIoScheduler,
    layer_id: u32,
    expected: usize,
) -> Result<Vec<ReadyBlock>, GenerationError> {
    let mut parking = Vec::new();
    let ready = wait_for_layer_blocks_parked(scheduler, layer_id, expected, &mut parking)?;
    // Legacy behavior for callers without a parking lot: other-layer
    // blocks are released immediately.
    for block in &parking {
        scheduler.release_ready_block(block)?;
    }
    Ok(ready)
}

/// Like `wait_for_layer_blocks`, but completed blocks that belong to other
/// layers (e.g. a prefetched next-layer dense block) are PARKED instead of
/// released, so the next iteration can consume them without re-reading.
fn wait_for_layer_blocks_parked(
    scheduler: &mut MoEIoScheduler,
    layer_id: u32,
    expected: usize,
    parking: &mut Vec<ReadyBlock>,
) -> Result<Vec<ReadyBlock>, GenerationError> {
    let mut ready = Vec::with_capacity(expected);
    // Drain matching blocks that were parked by a previous wait.
    let mut index = 0;
    while index < parking.len() {
        if parking[index].ticket.layer_id == layer_id {
            ready.push(parking.swap_remove(index));
        } else {
            index += 1;
        }
    }
    let mut spins = 0usize;
    let started = Instant::now();
    let timeout = scheduler_wait_timeout();
    while ready.len() < expected {
        scheduler.pump_completions()?;
        while let Some(block) = scheduler.pop_ready() {
            if block.ticket.layer_id == layer_id {
                ready.push(block);
            } else {
                parking.push(block);
            }
        }
        if ready.len() < expected {
            spins += 1;
            if started.elapsed() > timeout {
                return Err(GenerationError::SchedulerWaitBudgetExceeded);
            }
            if spins & 0x3ff == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }
    Ok(ready)
}

fn scheduler_wait_timeout() -> Duration {
    env::var("ZC_SCHEDULER_WAIT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_secs(300))
}

fn block_to_ptr(scheduler: &MoEIoScheduler, block: &ReadyBlock) -> IoBlockPtr {
    let buffer = &scheduler.io.io_buffers[block.ticket.fixed_buffer_index as usize];
    IoBlockPtr {
        kind: block.ticket.kind,
        layer_id: block.ticket.layer_id,
        expert_id: block.ticket.expert_id,
        route_weight: 1.0,
        ptr: buffer.ptr.as_ptr(),
        len: block.ticket.payload_bytes as usize,
    }
}

fn block_to_ptr_with_route(
    scheduler: &MoEIoScheduler,
    block: &ReadyBlock,
    routes: &[ExpertRoute],
) -> IoBlockPtr {
    let mut ptr = block_to_ptr(scheduler, block);
    if let Some(route) = routes
        .iter()
        .find(|route| route.expert_id == block.ticket.expert_id)
    {
        ptr.route_weight = route.score;
    } else {
        ptr.route_weight = f32::NAN;
    }
    ptr
}

fn normalize_route_weights(
    routes: &mut [ExpertRoute],
    route_source: &'static str,
    compute_config: &ComputeConfig,
) {
    // DeepSeek-V4 router weights are already sum-normalized and scaled by
    // route_scale inside route_experts_from_dense_block; softmax here would
    // destroy that parity.
    if route_source == "router" && compute_config.router_math == RouterMath::DeepSeekV4SqrtSoftplus
    {
        return;
    }
    normalize_route_weights_legacy(routes, route_source)
}

fn normalize_route_weights_legacy(routes: &mut [ExpertRoute], route_source: &'static str) {
    if routes.is_empty() {
        return;
    }
    if route_source != "router" {
        fill_uniform_route_weights(routes);
        return;
    }
    if routes.iter().any(|route| !route.score.is_finite()) {
        fill_uniform_route_weights(routes);
        return;
    }
    let max_score = routes
        .iter()
        .map(|route| route.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for route in routes.iter_mut() {
        route.score = (route.score - max_score).exp();
        sum += route.score;
    }
    if !sum.is_finite() || sum <= f32::EPSILON {
        fill_uniform_route_weights(routes);
        return;
    }
    for route in routes {
        route.score /= sum;
    }
}

fn fill_uniform_route_weights(routes: &mut [ExpertRoute]) {
    let weight = 1.0 / routes.len() as f32;
    for route in routes {
        route.score = weight;
    }
}

fn routes_for_layer(
    scheduler: &MoEIoScheduler,
    layer_id: u32,
    hinted: &[u32],
    active_experts: u32,
) -> Vec<ExpertRoute> {
    let layer = scheduler.layer(layer_id);
    if layer.experts.is_empty() {
        return Vec::new();
    }

    let mut expert_ids = if hinted.is_empty() {
        layer
            .experts
            .iter()
            .take(active_experts as usize)
            .map(|expert| expert.expert_id)
            .collect::<Vec<_>>()
    } else {
        hinted.to_vec()
    };
    expert_ids.retain(|expert_id| layer.experts.iter().any(|expert| expert.expert_id == *expert_id));
    expert_ids.truncate(active_experts as usize);
    expert_ids
        .into_iter()
        .enumerate()
        .map(|(index, expert_id)| ExpertRoute {
            expert_id,
            score: 1.0 / (index + 1) as f32,
        })
        .collect()
}

fn routes_for_layer_with_dense_router(
    scheduler: &MoEIoScheduler,
    manifest_layer_index: u32,
    physical_layer_id: u32,
    hinted: &[u32],
    active_experts: u32,
    dense: &[u8],
    hidden_states: &[f32],
    scratch_buf: &mut [f32],
    gemm: &FusedInt4Gemm,
    compute_config: &ComputeConfig,
    current_token_id: u32,
) -> Result<(Vec<ExpertRoute>, &'static str), GenerationError> {
    if !hinted.is_empty() {
        return Ok((
            routes_for_layer(scheduler, manifest_layer_index, hinted, active_experts),
            "hint",
        ));
    }
    let layer = scheduler.layer(manifest_layer_index);
    let available = layer
        .experts
        .iter()
        .map(|expert| expert.expert_id)
        .collect::<Vec<_>>();
    if let Some(probe_top_k) = router_probe_top_k() {
        if let Some(probe_routes) = router_top_experts_from_dense_block(
            dense,
            hidden_states,
            probe_top_k,
            scratch_buf,
            gemm,
        )? {
            let expert_ids = probe_routes
                .iter()
                .map(|route| route.expert_id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let scores = probe_routes
                .iter()
                .map(|route| format!("{:.6}", route.score))
                .collect::<Vec<_>>()
                .join(",");
            crate::vlog!(
                "router_probe layer={} manifest_layer={} top_all=[{}] scores=[{}] available={}",
                physical_layer_id,
                manifest_layer_index,
                expert_ids,
                scores,
                available.len()
            );
        }
    }
    match route_experts_from_dense_block(
        dense,
        hidden_states,
        active_experts as usize,
        &available,
        scratch_buf,
        gemm,
        RouteOptions::from_config(compute_config, physical_layer_id, current_token_id),
    )? {
        Some(routes) => Ok((routes, "router")),
        None => Ok((
            routes_for_layer(scheduler, manifest_layer_index, hinted, active_experts),
            "fallback",
        )),
    }
}

fn router_probe_top_k() -> Option<usize> {
    let value = env::var("ZC_ROUTER_PROBE_TOPK").ok()?;
    let parsed = value.parse::<usize>().ok()?;
    if parsed == 0 {
        None
    } else {
        Some(parsed.min(32))
    }
}

fn fill_token_hidden(state: &mut [f32], token_id: u32) {
    let seed = token_id as f32 / 1024.0;
    for (index, value) in state.iter_mut().enumerate() {
        let next = ((index as f32 + 1.0) * 0.000_976_562_5 + seed).sin();
        *value = next;
    }
}

fn sample_argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

fn write_event<W: Write>(writer: &mut W, event: &GenerationEvent) -> Result<(), GenerationError> {
    serde_json::to_writer(&mut *writer, event)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_route_weights_are_softmax_normalized() {
        let mut routes = vec![
            ExpertRoute {
                expert_id: 1,
                score: 2.0,
            },
            ExpertRoute {
                expert_id: 3,
                score: 1.0,
            },
        ];

        normalize_route_weights_legacy(&mut routes, "router");

        assert_close(routes[0].score, 0.7310586);
        assert_close(routes[1].score, 0.26894143);
    }

    #[test]
    fn deepseek_router_weights_skip_softmax_renormalization() {
        let mut routes = vec![
            ExpertRoute {
                expert_id: 1,
                score: 0.9,
            },
            ExpertRoute {
                expert_id: 3,
                score: 0.6,
            },
        ];
        let config = ComputeConfig::deepseek_v4_flash();

        normalize_route_weights(&mut routes, "router", &config);

        // Weights already normalized+scaled upstream: must pass through.
        assert_close(routes[0].score, 0.9);
        assert_close(routes[1].score, 0.6);
    }

    #[test]
    fn hinted_route_weights_fall_back_to_uniform() {
        let mut routes = vec![
            ExpertRoute {
                expert_id: 1,
                score: 10.0,
            },
            ExpertRoute {
                expert_id: 3,
                score: 1.0,
            },
        ];

        normalize_route_weights_legacy(&mut routes, "hint");

        assert_close(routes[0].score, 0.5);
        assert_close(routes[1].score, 0.5);
    }

    #[test]
    fn mtp_expert_id_parses_from_tensor_names() {
        assert_eq!(
            mtp_expert_id_from_name("mtp.0.ffn.experts.17.w1.weight"),
            Some(17)
        );
        assert_eq!(
            mtp_expert_id_from_name("mtp.0.ffn.experts.255.w2.scale"),
            Some(255)
        );
        assert_eq!(mtp_expert_id_from_name("mtp.0.ffn.experts.0.w3.weight"), Some(0));
        assert_eq!(mtp_expert_id_from_name("mtp.0.ffn.gate.weight"), None);
        assert_eq!(mtp_expert_id_from_name("mtp.0.attn.wq_a.weight"), None);
    }

    #[test]
    fn sampling_penalty_can_change_argmax() {
        let config = SamplingConfig::from_options(Some(0.0), Some(2), Some(1.0), Some(2.0), None);
        let selected = sample_from_candidates(vec![(7, 10.0), (8, 6.0)], &[7], &config, 0);

        assert_eq!(selected.token_id, 8);
        assert_close(selected.logit, 6.0);
        assert_eq!(selected.candidates[0].token_id, 8);
    }

    #[test]
    fn topk_temperature_sampling_is_seeded() {
        let config = SamplingConfig::from_options(Some(1.0), Some(3), Some(1.0), Some(1.0), Some(42));
        let first = sample_from_candidates(vec![(1, 3.0), (2, 2.5), (3, 2.0)], &[], &config, 42);
        let second = sample_from_candidates(vec![(1, 3.0), (2, 2.5), (3, 2.0)], &[], &config, 42);

        assert_eq!(first.token_id, second.token_id);
        assert_close(first.logit, second.logit);
    }

    #[test]
    fn session_match_len_covers_prefix_and_leaves_last_position() {
        let stored = vec![10, 11, 12, 13, 14];
        // Continuation: whole valid prefix reusable.
        assert_eq!(session_match_len(&stored, 4, &[10, 11, 12, 13, 14, 15, 16]), 4);
        // Prompt shorter than valid prefix (regenerate): leave one to prefill.
        assert_eq!(session_match_len(&stored, 4, &[10, 11, 12]), 2);
        // Divergence inside the prefix: full miss.
        assert_eq!(session_match_len(&stored, 4, &[10, 99, 12, 13, 14, 15]), 0);
        // Empty overlap: full miss.
        assert_eq!(session_match_len(&stored, 4, &[10]), 0);
    }

    #[test]
    fn same_conversation_extends_or_truncates_but_not_shared_system_prompt() {
        let convo_a = vec![1, 2, 3, 40, 41, 42];
        // Next turn resends history plus the new question.
        assert!(is_same_conversation(&convo_a, 5, &[1, 2, 3, 40, 41, 42, 50]));
        // Regenerate: shorter prompt, same conversation.
        assert!(is_same_conversation(&convo_a, 5, &[1, 2, 3, 40]));
        // Different conversation sharing the system prompt [1,2,3]: the
        // first question token diverges -> separate slot.
        assert!(!is_same_conversation(&convo_a, 5, &[1, 2, 3, 70, 71]));
    }

    #[test]
    fn store_slot_policy_replaces_own_slot_appends_then_evicts_lru() {
        let convo_a: Vec<u32> = vec![1, 2, 3, 40, 41, 42];
        let convo_b: Vec<u32> = vec![1, 2, 3, 70, 71];

        // Same conversation -> replace its slot, never a second one.
        let slots: Vec<(&[u32], usize, u64)> = vec![(&convo_a, 5, 7)];
        assert!(matches!(
            select_store_slot(&slots, &[1, 2, 3, 40, 41, 42, 50, 51], 2),
            StoreSlot::Replace(0)
        ));

        // New conversation with free capacity -> append.
        assert!(matches!(
            select_store_slot(&slots, &convo_b, 2),
            StoreSlot::Append
        ));

        // Full store, new conversation -> evict the least recently used.
        let slots: Vec<(&[u32], usize, u64)> = vec![(&convo_a, 5, 9), (&convo_b, 4, 3)];
        assert!(matches!(
            select_store_slot(&slots, &[1, 2, 3, 80, 81], 2),
            StoreSlot::Evict(1)
        ));
    }

    fn assert_close(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1.0e-5,
            "actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[allow(dead_code)]
pub fn conversion_command(model_dir: &Path, out: &Path) -> String {
    format!(
        "python tools/convert_safetensors.py --model-dir {} --out {} --quant int4 --pack-global-into-layer0",
        model_dir.display(),
        out.display()
    )
}

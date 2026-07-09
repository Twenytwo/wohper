use crate::model_format::{ModelFormatError, ModelManifest};
use crate::server::generation::{
    GenerationConfig, GenerationError, GenerationRequest, GenerationRuntime, SamplingConfig,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::SocketAddr;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const DEFAULT_SOCKET_PATH: &str = "/tmp/wohper-infer.sock";
const MAX_ENVELOPE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Error)]
pub enum ApiServerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("model format error: {0}")]
    Model(#[from] ModelFormatError),
    #[error("generation error: {0}")]
    Generation(#[from] GenerationError),
    #[error("prompt envelope is too large: {0} bytes")]
    EnvelopeTooLarge(u64),
}

#[derive(Clone, Debug)]
pub struct ApiServerConfig {
    pub socket_path: PathBuf,
    pub model_path: PathBuf,
    pub active_experts: u32,
    pub pipeline_depth: usize,
    pub io_buffer_count: usize,
    pub io_buffer_mb: usize,
    pub runtime_buffer_mb: usize,
    pub stop_token_id: u32,
    pub expert_cache_dir: PathBuf,
    pub expert_cache_gb: u64,
    pub expert_remote_endpoint: Option<String>,
    pub cluster_next_node: Option<SocketAddr>,
    pub local_layer_start: u32,
    pub local_layer_end: Option<u32>,
}

impl ApiServerConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: PathBuf::from(DEFAULT_SOCKET_PATH),
            model_path: model_path.into(),
            active_experts: 8,
            pipeline_depth: 4,
            io_buffer_count: 12,
            io_buffer_mb: 128,
            runtime_buffer_mb: 0,
            stop_token_id: crate::server::generation::GLM52_PRIMARY_EOS_TOKEN_ID,
            expert_cache_dir: PathBuf::from("cache/experts"),
            expert_cache_gb: 100,
            expert_remote_endpoint: None,
            cluster_next_node: None,
            local_layer_start: 0,
            local_layer_end: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PromptEnvelope {
    pub request_id: String,
    pub objective: String,
    #[serde(default)]
    pub compact_context: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub tools_allowed: Vec<String>,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub stop_token_id: Option<u32>,
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub route_hint: Option<RouteHint>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RouteHint {
    #[serde(default)]
    pub warm_layers: Vec<u32>,
    #[serde(default)]
    pub expert_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ApiResponse {
    pub request_id: String,
    pub status: &'static str,
    pub accepted: bool,
    pub objective_bytes: usize,
    pub compact_context_bytes: usize,
    pub planned_prefetch: PrefetchPlanSummary,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrefetchPlanSummary {
    pub model_layers: usize,
    pub active_experts: u32,
    pub pipeline_depth: usize,
    pub warm_layers: Vec<u32>,
    pub expert_ids: Vec<u32>,
}

pub struct SchedulerBridge {
    manifest: ModelManifest,
    active_experts: u32,
    pipeline_depth: usize,
}

impl SchedulerBridge {
    pub fn open(
        model_path: impl AsRef<Path>,
        active_experts: u32,
        pipeline_depth: usize,
    ) -> Result<Self, ApiServerError> {
        let manifest = ModelManifest::load(model_path)?;
        Ok(Self {
            manifest,
            active_experts,
            pipeline_depth,
        })
    }

    pub fn prepare_prefetch(&self, envelope: &PromptEnvelope) -> PrefetchPlanSummary {
        let layer_count = self.manifest.layers.len();
        let default_warm_layers = (0..self.pipeline_depth.min(layer_count))
            .map(|layer| layer as u32)
            .collect::<Vec<_>>();

        let warm_layers = envelope
            .route_hint
            .as_ref()
            .map(|hint| {
                hint.warm_layers
                    .iter()
                    .copied()
                    .filter(|layer| (*layer as usize) < layer_count)
                    .collect::<Vec<_>>()
            })
            .filter(|layers| !layers.is_empty())
            .unwrap_or(default_warm_layers);

        let expert_ids = envelope
            .route_hint
            .as_ref()
            .map(|hint| hint.expert_ids.clone())
            .filter(|experts| !experts.is_empty())
            .unwrap_or_else(|| (0..self.active_experts).collect());

        PrefetchPlanSummary {
            model_layers: layer_count,
            active_experts: self.active_experts,
            pipeline_depth: self.pipeline_depth,
            warm_layers,
            expert_ids,
        }
    }
}

pub struct ApiServer {
    config: ApiServerConfig,
    bridge: SchedulerBridge,
    /// Persistent generation runtime (E2): created lazily on the first
    /// generation request and reused for every following one, so the RAM
    /// caches (dense blocks, lm_head rows) stay warm across chat turns.
    runtime: std::sync::Mutex<Option<GenerationRuntime>>,
}

impl ApiServer {
    pub fn new(config: ApiServerConfig) -> Result<Self, ApiServerError> {
        let bridge = SchedulerBridge::open(
            &config.model_path,
            config.active_experts,
            config.pipeline_depth,
        )?;
        Ok(Self {
            config,
            bridge,
            runtime: std::sync::Mutex::new(None),
        })
    }

    pub fn serve(self) -> Result<(), ApiServerError> {
        remove_stale_socket(&self.config.socket_path)?;
        let listener = UnixListener::bind(&self.config.socket_path)?;
        for stream in listener.incoming() {
            match stream {
                // Per-request errors (client disconnects mid-stream, bad
                // envelopes, generation I/O) must NOT kill the persistent
                // server: log and keep serving. Agent clients abort and
                // time out routinely.
                Ok(stream) => {
                    if let Err(err) = self.handle_stream(stream) {
                        eprintln!("request failed (server keeps serving): {err}");
                    }
                }
                Err(err) => return Err(ApiServerError::Io(err)),
            }
        }
        Ok(())
    }

    fn handle_stream(&self, stream: UnixStream) -> Result<(), ApiServerError> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut payload = String::new();
        let bytes = reader.read_line(&mut payload)? as u64;
        if bytes > MAX_ENVELOPE_BYTES {
            return Err(ApiServerError::EnvelopeTooLarge(bytes));
        }

        let envelope: PromptEnvelope = serde_json::from_str(&payload)?;
        if !envelope.token_ids.is_empty() {
            return self.handle_generation(stream, envelope);
        }

        let planned_prefetch = self.bridge.prepare_prefetch(&envelope);
        let response = ApiResponse {
            request_id: envelope.request_id.clone(),
            status: "planned",
            accepted: true,
            objective_bytes: envelope.objective.len(),
            compact_context_bytes: envelope.compact_context.len(),
            planned_prefetch,
            message: "PromptEnvelope accepted; scheduler prefetch plan prepared".to_string(),
        };

        let mut writer = stream;
        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }

    fn handle_generation(
        &self,
        stream: UnixStream,
        envelope: PromptEnvelope,
    ) -> Result<(), ApiServerError> {
        let mut config = GenerationConfig::new(self.config.model_path.clone());
        config.active_experts = self.config.active_experts;
        config.pipeline_depth = self.config.pipeline_depth;
        config.io_buffer_count = self.config.io_buffer_count;
        config.io_buffer_bytes = self.config.io_buffer_mb * 1024 * 1024;
        config.runtime_buffer_bytes = self.config.runtime_buffer_mb * 1024 * 1024;
        config.stop_token_ids = vec![self.config.stop_token_id];
        config.expert_cache_dir = self.config.expert_cache_dir.clone();
        config.expert_cache_bytes = self.config.expert_cache_gb * 1024 * 1024 * 1024;
        config.expert_remote_endpoint = self.config.expert_remote_endpoint.clone();
        config.cluster_next_node = self.config.cluster_next_node;
        config.local_layer_start = self.config.local_layer_start;
        config.local_layer_end = self.config.local_layer_end;

        let request = GenerationRequest {
            request_id: envelope.request_id,
            token_ids: envelope.token_ids,
            max_new_tokens: envelope.max_new_tokens.unwrap_or(32).min(4096),
            sampling: SamplingConfig::from_options(
                envelope.temperature,
                envelope.top_k,
                envelope.top_p,
                envelope.repetition_penalty,
                envelope.seed,
            ),
            route_expert_ids: envelope
                .route_hint
                .as_ref()
                .map(|hint| hint.expert_ids.clone())
                .unwrap_or_default(),
            stop_token_ids: if envelope.stop_token_ids.is_empty() {
                envelope.stop_token_id.into_iter().collect()
            } else {
                envelope.stop_token_ids
            },
        };

        let mut guard = self.runtime.lock().unwrap();
        if guard.is_none() {
            *guard = Some(GenerationRuntime::open(config)?);
            eprintln!("generation runtime initialized (persistent: caches stay warm across requests)");
        }
        let runtime = guard.as_ref().expect("runtime just initialized");
        let mut writer = stream;
        runtime.generate_stream(request, &mut writer)?;
        Ok(())
    }
}

fn remove_stale_socket(path: &Path) -> Result<(), ApiServerError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(ApiServerError::Io(err)),
    }
}

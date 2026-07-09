use std::env;
use std::path::PathBuf;

use zc_infer_core::server::api::{ApiServer, ApiServerConfig, DEFAULT_SOCKET_PATH};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    socket: PathBuf,
    active_experts: u32,
    pipeline_depth: usize,
    io_buffer_count: usize,
    io_buffer_mb: usize,
    runtime_buffer_mb: usize,
    stop_token_id: u32,
    expert_cache_dir: PathBuf,
    expert_cache_gb: u64,
    expert_remote_endpoint: Option<String>,
    cluster_next_node: Option<std::net::SocketAddr>,
    local_layer_start: u32,
    local_layer_end: Option<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let mut config = ApiServerConfig::new(args.model);
    config.socket_path = args.socket;
    config.active_experts = args.active_experts;
    config.pipeline_depth = args.pipeline_depth;
    config.io_buffer_count = args.io_buffer_count;
    config.io_buffer_mb = args.io_buffer_mb;
    config.runtime_buffer_mb = args.runtime_buffer_mb;
    config.stop_token_id = args.stop_token_id;
    config.expert_cache_dir = args.expert_cache_dir;
    config.expert_cache_gb = args.expert_cache_gb;
    config.expert_remote_endpoint = args.expert_remote_endpoint;
    config.cluster_next_node = args.cluster_next_node;
    config.local_layer_start = args.local_layer_start;
    config.local_layer_end = args.local_layer_end;

    eprintln!(
        "wohper infer server listening on {}",
        config.socket_path.display()
    );
    ApiServer::new(config)?.serve()?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut socket = PathBuf::from(DEFAULT_SOCKET_PATH);
    let mut active_experts = 8;
    let mut pipeline_depth = 4;
    let mut io_buffer_count = 12;
    let mut io_buffer_mb = 128;
    let mut runtime_buffer_mb = 0;
    let mut stop_token_id = zc_infer_core::server::generation::GLM52_PRIMARY_EOS_TOKEN_ID;
    let mut expert_cache_dir = PathBuf::from("cache/experts");
    let mut expert_cache_gb = 100;
    let mut expert_remote_endpoint = None;
    let mut cluster_next_node = None;
    let mut local_layer_start = 0;
    let mut local_layer_end = None;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(iter.next().ok_or("--model needs a value")?)),
            "--socket" => socket = PathBuf::from(iter.next().ok_or("--socket needs a value")?),
            "--active-experts" => {
                active_experts = iter.next().ok_or("--active-experts needs a value")?.parse()?
            }
            "--pipeline-depth" => {
                pipeline_depth = iter.next().ok_or("--pipeline-depth needs a value")?.parse()?
            }
            "--io-buffer-count" => {
                io_buffer_count = iter
                    .next()
                    .ok_or("--io-buffer-count needs a value")?
                    .parse()?
            }
            "--io-buffer-mb" => {
                io_buffer_mb = iter.next().ok_or("--io-buffer-mb needs a value")?.parse()?
            }
            "--runtime-buffer-mb" => {
                runtime_buffer_mb = iter
                    .next()
                    .ok_or("--runtime-buffer-mb needs a value")?
                    .parse()?
            }
            "--stop-token-id" => {
                stop_token_id = iter.next().ok_or("--stop-token-id needs a value")?.parse()?
            }
            "--expert-cache-dir" => {
                expert_cache_dir = PathBuf::from(iter.next().ok_or("--expert-cache-dir needs a value")?)
            }
            "--expert-cache-gb" => {
                expert_cache_gb = iter.next().ok_or("--expert-cache-gb needs a value")?.parse()?
            }
            "--expert-remote-endpoint" => {
                expert_remote_endpoint = Some(iter.next().ok_or("--expert-remote-endpoint needs a value")?)
            }
            "--cluster-next-node" => {
                cluster_next_node = Some(iter.next().ok_or("--cluster-next-node needs a value")?.parse()?)
            }
            "--local-layer-start" => {
                local_layer_start = iter.next().ok_or("--local-layer-start needs a value")?.parse()?
            }
            "--local-layer-end" => {
                local_layer_end = Some(iter.next().ok_or("--local-layer-end needs a value")?.parse()?)
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        model: model.ok_or("--model is required")?,
        socket,
        active_experts,
        pipeline_depth,
        io_buffer_count,
        io_buffer_mb,
        runtime_buffer_mb,
        stop_token_id,
        expert_cache_dir,
        expert_cache_gb,
        expert_remote_endpoint,
        cluster_next_node,
        local_layer_start,
        local_layer_end,
    })
}

fn print_help() {
    println!(
        "zc_infer_server --model MODEL.bin [--socket /tmp/wohper-infer.sock] \
         [--active-experts N] [--pipeline-depth N] [--io-buffer-count N] \
         [--io-buffer-mb N] [--runtime-buffer-mb N] [--stop-token-id ID] \
         [--expert-cache-dir DIR] [--expert-cache-gb N] \
         [--expert-remote-endpoint URL] [--cluster-next-node HOST:PORT] \
         [--local-layer-start N] [--local-layer-end N]"
    );
}

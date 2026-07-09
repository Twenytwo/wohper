use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{Error as IoError, ErrorKind, Read};
use std::path::PathBuf;

use zc_infer_core::compute::{ExpertCacheConfig, ExpertLruCache};
use zc_infer_core::model_format::{ModelManifest, QUANT_BLOCK_MAGIC};
use zc_infer_core::ALIGN_2MB;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    endpoint: String,
    cache_dir: PathBuf,
    layer_id: u32,
    expert_id: u32,
    max_cache_bytes: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let manifest = ModelManifest::load(&args.model)?;
    let shard = manifest
        .expert_shard(args.layer_id, args.expert_id)
        .ok_or_else(|| {
            IoError::new(
                ErrorKind::NotFound,
                format!(
                    "missing shard ref for layer {} expert {}",
                    args.layer_id, args.expert_id
                ),
            )
        })?;

    println!(
        "manifest_ok layers={} layer={} expert={} shard_path={} shard_bytes={}",
        manifest.layers.len(),
        args.layer_id,
        args.expert_id,
        shard.path.display(),
        shard.disk_bytes
    );

    let mut cache = ExpertLruCache::new(ExpertCacheConfig {
        cache_dir: args.cache_dir.clone(),
        max_bytes: args.max_cache_bytes,
        remote_endpoint: Some(args.endpoint.clone()),
    })?;

    let fetched = cache.ensure_expert(args.layer_id, args.expert_id, None, Some(shard.disk_bytes))?;
    let metadata = std::fs::metadata(&fetched)?;
    let mut magic = [0u8; 8];
    File::open(&fetched)?.read_exact(&mut magic)?;

    println!(
        "cache_fetch_ok path={} bytes={} mod_2mb={} magic={}",
        fetched.display(),
        metadata.len(),
        metadata.len() % ALIGN_2MB,
        String::from_utf8_lossy(&magic)
    );

    if metadata.len() != shard.disk_bytes {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "fetched size mismatch: got {}, expected {}",
                metadata.len(),
                shard.disk_bytes
            ),
        )
        .into());
    }
    if metadata.len() % ALIGN_2MB != 0 {
        return Err(IoError::new(ErrorKind::InvalidData, "fetched file is not 2MB aligned").into());
    }
    if magic != QUANT_BLOCK_MAGIC {
        return Err(IoError::new(ErrorKind::InvalidData, "invalid ZCBLK01 magic").into());
    }

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut model = None;
    let mut endpoint = None;
    let mut cache_dir = None;
    let mut layer_id = 3u32;
    let mut expert_id = 0u32;
    let mut max_cache_bytes = 100 * 1024 * 1024 * 1024u64;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(next_value(&mut iter, "--model")?)),
            "--endpoint" => endpoint = Some(next_value(&mut iter, "--endpoint")?),
            "--cache-dir" => cache_dir = Some(PathBuf::from(next_value(&mut iter, "--cache-dir")?)),
            "--layer-id" => layer_id = next_value(&mut iter, "--layer-id")?.parse()?,
            "--expert-id" => expert_id = next_value(&mut iter, "--expert-id")?.parse()?,
            "--max-cache-bytes" => {
                max_cache_bytes = next_value(&mut iter, "--max-cache-bytes")?.parse()?
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(IoError::new(
                    ErrorKind::InvalidInput,
                    format!("unknown argument: {other}"),
                )
                .into())
            }
        }
    }

    Ok(Args {
        model: model.ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "--model is required"))?,
        endpoint: endpoint
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "--endpoint is required"))?,
        cache_dir: cache_dir
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "--cache-dir is required"))?,
        layer_id,
        expert_id,
        max_cache_bytes,
    })
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, IoError> {
    iter.next()
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, format!("{flag} needs a value")))
}

fn print_usage() {
    eprintln!(
        "zc_remote_fetch_smoke --model dense_core.bin --endpoint http://127.0.0.1:9101 \\
         --cache-dir cache/glm52-worker-l3e0 [--layer-id 3] [--expert-id 0]"
    );
}

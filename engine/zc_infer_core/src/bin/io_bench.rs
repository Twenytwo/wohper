use std::env;
use std::path::PathBuf;
use std::time::Instant;

use zc_infer_core::direct_io::DirectIoRuntime;
use zc_infer_core::model_format::ModelManifest;
use zc_infer_core::scheduler::{ExpertRoute, MoEIoScheduler};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    rounds: u32,
    active_experts: u32,
    compressed_buffer_mb: usize,
    runtime_buffer_mb: usize,
    io_buffer_count: usize,
    pipeline_depth: usize,
    print_plan_layers: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let manifest = ModelManifest::load(&args.model)?;

    let io = DirectIoRuntime::open(
        &args.model,
        args.compressed_buffer_mb * 1024 * 1024,
        args.runtime_buffer_mb * 1024 * 1024,
        args.io_buffer_count,
    )?;
    let mut scheduler = MoEIoScheduler::new(io, manifest);

    let started = Instant::now();
    let mut submitted_bytes = 0u64;
    let mut completed_bytes = 0u64;
    let mut completed_reads = 0u64;
    let mut scheduled_reads = 0u64;
    let mut max_pending = 0usize;
    let mut max_in_flight = 0u64;
    let max_request_window = args
        .pipeline_depth
        .max(1)
        .saturating_mul((args.active_experts as usize).saturating_add(1));

    let layer_count = scheduler.manifest.layers.len() as u64;
    let total_layer_plans = args.rounds as u64 * layer_count;
    let mut next_plan = 0u64;

    while next_plan < total_layer_plans || completed_reads < scheduled_reads {
        while next_plan < total_layer_plans
            && scheduler.pending_len() + (scheduler.in_flight() as usize) < max_request_window
        {
            let round = (next_plan / layer_count) as u32;
            let layer_id = (next_plan % layer_count) as u32;
            let selected = deterministic_routes(
                layer_id,
                round,
                scheduler.layer(layer_id).experts.len() as u32,
                args.active_experts,
            );
            if round == 0 && layer_id < args.print_plan_layers {
                println!(
                    "route_plan layer={layer_id} experts={}",
                    selected
                        .iter()
                        .map(|route| route.expert_id.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }

            submitted_bytes += scheduler.enqueue_layer_io_plan(layer_id, &selected)?;
            scheduled_reads += selected.len() as u64 + 1;
            next_plan += 1;
        }

        scheduler.submit_pending_until_full(usize::MAX)?;
        max_pending = max_pending.max(scheduler.pending_len());
        max_in_flight = max_in_flight.max(scheduler.in_flight());

        let before = completed_reads;
        collect_ready(&mut scheduler, &mut completed_reads, &mut completed_bytes)?;
        scheduler.submit_pending_until_full(usize::MAX)?;

        if completed_reads == before {
            scheduler.pump_completions()?;
            collect_ready(&mut scheduler, &mut completed_reads, &mut completed_bytes)?;
            std::hint::spin_loop();
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let submitted_gib = submitted_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let completed_gib = completed_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let rss_kb = read_rss_kb().unwrap_or(0);

    println!("model={}", args.model.display());
    println!("layers={}", scheduler.manifest.layers.len());
    println!("rounds={}", args.rounds);
    println!("active_experts={}", args.active_experts);
    println!("io_buffer_count={}", scheduler.io.buffer_count());
    println!("pipeline_depth={}", args.pipeline_depth);
    println!("submitted_bytes={submitted_bytes}");
    println!("completed_bytes={completed_bytes}");
    println!("completed_reads={completed_reads}");
    println!("scheduled_reads={scheduled_reads}");
    println!("max_pending_requests={max_pending}");
    println!("max_in_flight_reads={max_in_flight}");
    println!("available_buffers_end={}", scheduler.available_buffers());
    println!("elapsed_sec={elapsed:.6}");
    println!("submitted_gib={submitted_gib:.6}");
    println!("completed_gib={completed_gib:.6}");
    println!("completed_gib_per_sec={:.6}", completed_gib / elapsed.max(0.000001));
    println!("rss_kb={rss_kb}");

    Ok(())
}

fn deterministic_routes(
    layer_id: u32,
    round: u32,
    experts_per_layer: u32,
    active_experts: u32,
) -> Vec<ExpertRoute> {
    // Fixed deterministic router simulator.
    //
    // For the canonical benchmark shape (8 experts/layer, 2 active experts):
    //   round 0, layer 0 -> [1, 4]
    //   round 0, layer 1 -> [0, 7]
    //
    // This makes it obvious from logs whether conditional expert reads are
    // targeting only the expected expert blocks.
    const BASE_PATTERN: [[u32; 2]; 8] = [
        [1, 4],
        [0, 7],
        [2, 5],
        [3, 6],
        [4, 1],
        [7, 0],
        [5, 2],
        [6, 3],
    ];

    let mut routes: Vec<ExpertRoute> = Vec::with_capacity(active_experts as usize);
    for i in 0..active_experts {
        let base = if i < 2 {
            BASE_PATTERN[(layer_id as usize) % BASE_PATTERN.len()][i as usize]
        } else {
            layer_id.wrapping_mul(17).wrapping_add(i.wrapping_mul(5))
        };
        let expert_id = base.wrapping_add(round.wrapping_mul(3)) % experts_per_layer;
        if routes.iter().any(|route| route.expert_id == expert_id) {
            continue;
        }
        routes.push(ExpertRoute {
            expert_id,
            score: 1.0 / (i + 1) as f32,
        });
    }
    routes
}

fn collect_ready(
    scheduler: &mut MoEIoScheduler,
    completed_reads: &mut u64,
    completed_bytes: &mut u64,
) -> Result<(), Box<dyn std::error::Error>> {
    scheduler.pump_completions()?;
    while let Some(block) = scheduler.pop_ready() {
        *completed_reads += 1;
        if block.result > 0 {
            *completed_bytes += block.result as u64;
        }
        scheduler.release_ready_block(&block)?;
    }
    Ok(())
}

fn read_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut rounds = 1;
    let mut active_experts = 2;
    let mut compressed_buffer_mb = 256;
    let mut runtime_buffer_mb = 256;
    let mut io_buffer_count = 6;
    let mut pipeline_depth = 4;
    let mut print_plan_layers = 8;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(iter.next().ok_or("--model needs a value")?)),
            "--rounds" => rounds = iter.next().ok_or("--rounds needs a value")?.parse()?,
            "--active-experts" => {
                active_experts = iter.next().ok_or("--active-experts needs a value")?.parse()?
            }
            "--compressed-buffer-mb" => {
                compressed_buffer_mb = iter
                    .next()
                    .ok_or("--compressed-buffer-mb needs a value")?
                    .parse()?
            }
            "--runtime-buffer-mb" => {
                runtime_buffer_mb = iter
                    .next()
                    .ok_or("--runtime-buffer-mb needs a value")?
                    .parse()?
            }
            "--io-buffer-count" => {
                io_buffer_count = iter
                    .next()
                    .ok_or("--io-buffer-count needs a value")?
                    .parse()?
            }
            "--pipeline-depth" => {
                pipeline_depth = iter
                    .next()
                    .ok_or("--pipeline-depth needs a value")?
                    .parse()?
            }
            "--print-plan-layers" => {
                print_plan_layers = iter
                    .next()
                    .ok_or("--print-plan-layers needs a value")?
                    .parse()?
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
        rounds,
        active_experts,
        compressed_buffer_mb,
        runtime_buffer_mb,
        io_buffer_count,
        pipeline_depth,
        print_plan_layers,
    })
}

fn print_help() {
    println!(
        "io_bench --model MODEL.bin [--rounds N] [--active-experts N] \
         [--compressed-buffer-mb N] [--runtime-buffer-mb N] \
         [--io-buffer-count N] [--pipeline-depth N] [--print-plan-layers N]"
    );
}

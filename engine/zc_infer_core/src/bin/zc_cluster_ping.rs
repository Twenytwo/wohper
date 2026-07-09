use std::env;
use std::net::SocketAddr;

use zc_infer_core::server::cluster::send_hidden_state;

#[derive(Debug)]
struct Args {
    addr: SocketAddr,
    hidden_size: usize,
    next_layer: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let hidden_states = (0..args.hidden_size)
        .map(|index| (index as f32 * 0.001).sin())
        .collect::<Vec<_>>();
    let response = send_hidden_state(
        args.addr,
        "cluster-smoke".to_string(),
        0,
        args.next_layer,
        &hidden_states,
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut addr = None;
    let mut hidden_size = 4096usize;
    let mut next_layer = 26u32;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--addr" => addr = Some(iter.next().ok_or("--addr needs a value")?.parse()?),
            "--hidden-size" => {
                hidden_size = iter.next().ok_or("--hidden-size needs a value")?.parse()?
            }
            "--next-layer" => next_layer = iter.next().ok_or("--next-layer needs a value")?.parse()?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Args {
        addr: addr.ok_or("--addr is required")?,
        hidden_size,
        next_layer,
    })
}

fn print_help() {
    println!("zc_cluster_ping --addr HOST:PORT [--hidden-size N] [--next-layer N]");
}

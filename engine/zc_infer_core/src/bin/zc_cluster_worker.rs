use std::env;
use std::net::SocketAddr;

use zc_infer_core::server::cluster::{serve_cluster_node, ClusterMessage};

#[derive(Debug)]
struct Args {
    listen: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    eprintln!("wohper cluster worker listening on {}", args.listen);

    serve_cluster_node(args.listen, |message| async move {
        match message {
            ClusterMessage::HiddenState {
                request_id,
                token_index,
                next_layer,
                hidden_size,
                hidden_states,
            } => {
                eprintln!(
                    "hidden_state request_id={} token_index={} next_layer={} hidden_size={} payload_f32={}",
                    request_id,
                    token_index,
                    next_layer,
                    hidden_size,
                    hidden_states.len()
                );
                ClusterMessage::Ack {
                    request_id,
                    message: format!(
                        "worker accepted hidden state for layer {} ({} f32)",
                        next_layer,
                        hidden_states.len()
                    ),
                }
            }
            ClusterMessage::Ack {
                request_id,
                message,
            } => ClusterMessage::Ack {
                request_id,
                message: format!("worker received ack: {message}"),
            },
            ClusterMessage::Error {
                request_id,
                message,
            } => ClusterMessage::Error {
                request_id,
                message: format!("worker received error: {message}"),
            },
        }
    })
    .await?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut listen: SocketAddr = "0.0.0.0:9000".parse()?;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--listen" => listen = iter.next().ok_or("--listen needs a value")?.parse()?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Args { listen })
}

fn print_help() {
    println!("zc_cluster_worker [--listen 0.0.0.0:9000]");
}

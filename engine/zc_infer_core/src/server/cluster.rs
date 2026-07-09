use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ClusterError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("frame too large: {0} bytes")]
    FrameTooLarge(usize),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClusterMessage {
    HiddenState {
        request_id: String,
        token_index: u32,
        next_layer: u32,
        hidden_size: usize,
        hidden_states: Vec<f32>,
    },
    Ack {
        request_id: String,
        message: String,
    },
    Error {
        request_id: String,
        message: String,
    },
}

pub async fn send_hidden_state(
    addr: SocketAddr,
    request_id: String,
    token_index: u32,
    next_layer: u32,
    hidden_states: &[f32],
) -> Result<ClusterMessage, ClusterError> {
    let mut stream = TcpStream::connect(addr).await?;
    let message = ClusterMessage::HiddenState {
        request_id,
        token_index,
        next_layer,
        hidden_size: hidden_states.len(),
        hidden_states: hidden_states.to_vec(),
    };
    write_frame(&mut stream, &message).await?;
    read_frame(&mut stream).await
}

pub async fn serve_cluster_node<F, Fut>(
    addr: SocketAddr,
    mut handler: F,
) -> Result<(), ClusterError>
where
    F: FnMut(ClusterMessage) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ClusterMessage> + Send,
{
    let listener = TcpListener::bind(addr).await?;
    loop {
        let (mut stream, _) = listener.accept().await?;
        let message = read_frame(&mut stream).await?;
        let response = handler(message).await;
        write_frame(&mut stream, &response).await?;
    }
}

async fn write_frame(stream: &mut TcpStream, message: &ClusterMessage) -> Result<(), ClusterError> {
    let payload = serde_json::to_vec(message)?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(ClusterError::FrameTooLarge(payload.len()));
    }
    stream.write_u32_le(payload.len() as u32).await?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
    Ok(())
}

async fn read_frame(stream: &mut TcpStream) -> Result<ClusterMessage, ClusterError> {
    let len = stream.read_u32_le().await? as usize;
    if len > MAX_FRAME_BYTES {
        return Err(ClusterError::FrameTooLarge(len));
    }
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await?;
    Ok(serde_json::from_slice(&payload)?)
}

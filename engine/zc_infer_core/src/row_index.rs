use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use thiserror::Error;

const DEEPSEEK_BF16_DTYPE: u16 = 11;
const DEEPSEEK_BF16_QUANT: u16 = 2404;
const DEFAULT_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum RowIndexError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tensor index status is not ready: {0}")]
    NotReady(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("tensor is not row-addressable BF16: {0}")]
    NotRowBf16(String),
    #[error("row {row} is outside tensor {tensor} rows={rows}")]
    RowOutOfRange { tensor: String, row: usize, rows: usize },
    #[error("output too small: required={required}, actual={actual}")]
    OutputTooSmall { required: usize, actual: usize },
    #[error("invalid tensor index: {0}")]
    InvalidIndex(&'static str),
}

#[derive(Clone, Debug, Deserialize)]
pub struct RowTensorRecord {
    pub name: String,
    pub dtype_code: u16,
    pub quant_format: u16,
    #[allow(dead_code)]
    pub role_code: u16,
    #[allow(dead_code)]
    pub shape: Vec<usize>,
    pub absolute_data_offset: u64,
    pub data_bytes: u64,
    pub row_count: Option<usize>,
    pub row_width: Option<usize>,
    pub row_bytes: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RowTensorIndexFile {
    format: String,
    version: u32,
    status: String,
    core_file: String,
    tensors: Vec<RowTensorRecord>,
}

#[derive(Debug)]
pub struct RowTensorIndex {
    core_path: PathBuf,
    tensors: HashMap<String, RowTensorRecord>,
    /// Whole lm_head tensor kept in RAM when ZC_LMHEAD_CACHE=1 (~1GB for
    /// 129280x4096 bf16): saves the full-tensor NVMe read on every token.
    lm_head_cache: std::sync::OnceLock<Option<Vec<u8>>>,
}

impl RowTensorIndex {
    pub fn load_for_model(model_path: impl AsRef<Path>) -> Result<Option<Self>, RowIndexError> {
        let model_path = model_path.as_ref();
        let index_path = model_path.with_extension("tensor_index.json");
        if !index_path.exists() {
            return Ok(None);
        }
        Self::load(index_path).map(Some)
    }

    pub fn load(index_path: impl AsRef<Path>) -> Result<Self, RowIndexError> {
        let index_path = index_path.as_ref();
        let text = std::fs::read_to_string(index_path)?;
        let parsed: RowTensorIndexFile = serde_json::from_str(&text)?;
        // Legacy magic accepted: indexes built before the rename keep it.
        if (parsed.format != "wohper-row-tensor-index"
            && parsed.format != "zeroclaw-row-tensor-index")
            || parsed.version != 1
        {
            return Err(RowIndexError::InvalidIndex("unsupported format or version"));
        }
        if parsed.status != "ready" {
            return Err(RowIndexError::NotReady(parsed.status));
        }
        let core_path = resolve_core_path(index_path, &parsed.core_file);
        let mut tensors = HashMap::with_capacity(parsed.tensors.len());
        for tensor in parsed.tensors {
            tensors.insert(tensor.name.clone(), tensor);
        }
        Ok(Self {
            core_path,
            tensors,
            lm_head_cache: std::sync::OnceLock::new(),
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Raw bytes of a tensor read straight from the core file at its
    /// absolute offset (used by the MTP module to lift its tensors out of
    /// the giant global block, which the block scheduler cannot read).
    pub fn read_tensor_bytes(&self, name: &str) -> Result<Vec<u8>, RowIndexError> {
        let record = self
            .tensors
            .get(name)
            .ok_or_else(|| RowIndexError::MissingTensor(name.to_string()))?;
        let mut raw = vec![0u8; record.data_bytes as usize];
        let file = File::open(&self.core_path)?;
        read_exact_at(&file, &mut raw, record.absolute_data_offset)?;
        Ok(raw)
    }

    /// Names of all indexed tensors with a given prefix (sorted).
    pub fn tensor_names_with_prefix(&self, prefix: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .tensors
            .keys()
            .filter(|name| name.starts_with(prefix))
            .cloned()
            .collect();
        names.sort();
        names
    }

    pub fn tensor(&self, name: &str) -> Option<&RowTensorRecord> {
        self.tensors.get(name)
    }

    pub fn read_tensor_prefix(
        &self,
        tensor_name: &str,
        max_bytes: usize,
    ) -> Result<Option<Vec<u8>>, RowIndexError> {
        let Some(tensor) = self.tensors.get(tensor_name) else {
            return Ok(None);
        };
        let size = max_bytes.min(tensor.data_bytes as usize);
        let mut raw = vec![0u8; size];
        let file = File::open(&self.core_path)?;
        read_exact_at(&file, &mut raw, tensor.absolute_data_offset)?;
        Ok(Some(raw))
    }

    pub fn read_tensor_row_bytes(
        &self,
        tensor_name: &str,
        row: usize,
    ) -> Result<Option<Vec<u8>>, RowIndexError> {
        let Some(tensor) = self.tensors.get(tensor_name) else {
            return Ok(None);
        };
        let Some(row_bytes) = tensor.row_bytes else {
            return Err(RowIndexError::InvalidIndex("tensor is not row-addressable"));
        };
        let offset = row_offset(tensor, row, row_bytes)?;
        let mut raw = vec![0u8; row_bytes];
        let file = File::open(&self.core_path)?;
        read_exact_at(&file, &mut raw, offset)?;
        Ok(Some(raw))
    }

    pub fn lm_head_row_count(&self) -> Option<usize> {
        self.tensors
            .get("head.weight")
            .or_else(|| self.tensors.get("lm_head.weight"))
            .and_then(|tensor| tensor.row_count)
    }

    pub fn read_bf16_row(
        &self,
        tensor_name: &str,
        row: usize,
        out: &mut [f32],
    ) -> Result<bool, RowIndexError> {
        let Some(tensor) = self.tensors.get(tensor_name) else {
            return Ok(false);
        };
        validate_bf16_row_tensor(tensor)?;
        let row_count = tensor.row_count.unwrap_or(0);
        if row >= row_count {
            return Ok(false);
        }
        let row_width = tensor.row_width.unwrap_or(0);
        if out.len() < row_width {
            return Err(RowIndexError::OutputTooSmall {
                required: row_width,
                actual: out.len(),
            });
        }
        let row_bytes = tensor.row_bytes.unwrap_or(0);
        let offset = row_offset(tensor, row, row_bytes)?;
        let mut raw = vec![0u8; row_bytes];
        let file = File::open(&self.core_path)?;
        read_exact_at(&file, &mut raw, offset)?;
        decode_bf16_into(&raw, &mut out[..row_width])?;
        Ok(true)
    }

    pub fn topk_bf16_lm_head(
        &self,
        hidden: &[f32],
        scratch: &mut [f32],
        top_k: usize,
    ) -> Result<Option<Vec<(u32, f32)>>, RowIndexError> {
        let Some(tensor) = self
            .tensors
            .get("head.weight")
            .or_else(|| self.tensors.get("lm_head.weight"))
        else {
            return Ok(None);
        };
        validate_bf16_row_tensor(tensor)?;
        let row_count = tensor.row_count.unwrap_or(0);
        let row_width = tensor.row_width.unwrap_or(0);
        let row_bytes = tensor.row_bytes.unwrap_or(0);
        if hidden.len() < row_width {
            return Err(RowIndexError::OutputTooSmall {
                required: row_width,
                actual: hidden.len(),
            });
        }
        if scratch.is_empty() {
            return Err(RowIndexError::OutputTooSmall {
                required: 1,
                actual: 0,
            });
        }

        let cache_enabled = std::env::var("ZC_LMHEAD_CACHE")
            .map(|value| value.trim() == "1")
            .unwrap_or(false);
        let cached: Option<&Vec<u8>> = if cache_enabled {
            self.lm_head_cache
                .get_or_init(|| {
                    let total = row_count.checked_mul(row_bytes)?;
                    let mut buffer = vec![0u8; total];
                    let file = File::open(&self.core_path).ok()?;
                    let offset = row_offset(tensor, 0, row_bytes).ok()?;
                    read_exact_at(&file, &mut buffer, offset).ok()?;
                    eprintln!("lmhead_cache loaded rows={} bytes={}", row_count, total);
                    Some(buffer)
                })
                .as_ref()
        } else {
            None
        };

        let rows_per_chunk = scratch
            .len()
            .min((DEFAULT_CHUNK_BYTES / row_bytes.max(1)).max(1))
            .max(1);
        let mut raw = vec![0u8; if cached.is_some() { 0 } else { rows_per_chunk * row_bytes }];
        let file = if cached.is_some() {
            None
        } else {
            Some(File::open(&self.core_path)?)
        };
        let mut best = Vec::with_capacity(top_k.max(1).min(128));
        let mut row_start = 0usize;
        while row_start < row_count {
            let rows = rows_per_chunk.min(row_count - row_start);
            let bytes = rows * row_bytes;
            let raw: &[u8] = if let Some(buffer) = cached {
                &buffer[row_start * row_bytes..row_start * row_bytes + bytes]
            } else {
                let offset = row_offset(tensor, row_start, row_bytes)?;
                let file = file.as_ref().expect("file open when cache disabled");
                read_exact_at(file, &mut raw[..bytes], offset)?;
                &raw[..bytes]
            };
            crate::deepseek_v4::parallel_rows_f32::<RowIndexError>(
                &mut scratch[..rows],
                16,
                &|chunk_offset, chunk| {
                    for (local, slot) in chunk.iter_mut().enumerate() {
                        let start = (chunk_offset + local) * row_bytes;
                        *slot =
                            dot_bf16_row(&raw[start..start + row_bytes], &hidden[..row_width])?;
                    }
                    Ok(())
                },
            )?;
            for (local_row, &score) in scratch[..rows].iter().enumerate() {
                push_topk(&mut best, top_k.max(1), (row_start + local_row) as u32, score);
            }
            row_start += rows;
        }
        best.sort_by(|(_, left), (_, right)| right.total_cmp(left));
        Ok(Some(best))
    }
}

fn resolve_core_path(index_path: &Path, core_file: &str) -> PathBuf {
    let core = PathBuf::from(core_file);
    if core.is_absolute() {
        core
    } else {
        index_path.parent().unwrap_or_else(|| Path::new(".")).join(core)
    }
}

fn validate_bf16_row_tensor(tensor: &RowTensorRecord) -> Result<(), RowIndexError> {
    if tensor.dtype_code != DEEPSEEK_BF16_DTYPE
        || tensor.quant_format != DEEPSEEK_BF16_QUANT
        || tensor.row_count.is_none()
        || tensor.row_width.is_none()
        || tensor.row_bytes.is_none()
    {
        return Err(RowIndexError::NotRowBf16(tensor.name.clone()));
    }
    let row_count = tensor.row_count.unwrap();
    let row_bytes = tensor.row_bytes.unwrap();
    if row_count == 0 || row_bytes == 0 || tensor.data_bytes < row_bytes as u64 {
        return Err(RowIndexError::InvalidIndex("invalid row tensor dimensions"));
    }
    Ok(())
}

fn row_offset(tensor: &RowTensorRecord, row: usize, row_bytes: usize) -> Result<u64, RowIndexError> {
    let row_count = tensor.row_count.unwrap_or(0);
    if row >= row_count {
        return Err(RowIndexError::RowOutOfRange {
            tensor: tensor.name.clone(),
            row,
            rows: row_count,
        });
    }
    let relative = row
        .checked_mul(row_bytes)
        .ok_or(RowIndexError::InvalidIndex("row offset overflow"))?;
    tensor
        .absolute_data_offset
        .checked_add(relative as u64)
        .ok_or(RowIndexError::InvalidIndex("absolute row offset overflow"))
}

fn decode_bf16_into(raw: &[u8], out: &mut [f32]) -> Result<(), RowIndexError> {
    if raw.len() < out.len() * 2 {
        return Err(RowIndexError::InvalidIndex("BF16 row too small"));
    }
    for (index, slot) in out.iter_mut().enumerate() {
        let offset = index * 2;
        let raw = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
        *slot = f32::from_bits((raw as u32) << 16);
    }
    Ok(())
}

fn dot_bf16_row(raw: &[u8], input: &[f32]) -> Result<f32, RowIndexError> {
    // Delegates to the compute kernel (AVX2 dispatch + multi-accumulator).
    crate::compute::dot_bf16_row(raw, input)
        .map_err(|_| RowIndexError::InvalidIndex("BF16 row too small"))
}

fn push_topk(best: &mut Vec<(u32, f32)>, top_k: usize, token: u32, logit: f32) {
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

#[cfg(unix)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let read = file.read_at(buf, offset)?;
        if read == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short row-index read"));
        }
        offset += read as u64;
        let (_, rest) = buf.split_at_mut(read);
        buf = rest;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_bf16_row_values() {
        let mut raw = Vec::new();
        for value in [1.0f32, -2.0, 0.5] {
            raw.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        let mut out = [0.0f32; 3];
        decode_bf16_into(&raw, &mut out).unwrap();
        assert_eq!(out, [1.0, -2.0, 0.5]);
    }

    #[test]
    fn topk_keeps_largest_scores() {
        let mut best = Vec::new();
        push_topk(&mut best, 2, 1, 0.1);
        push_topk(&mut best, 2, 2, 2.0);
        push_topk(&mut best, 2, 3, 1.0);
        best.sort_by(|(_, left), (_, right)| right.total_cmp(left));
        assert_eq!(best, vec![(2, 2.0), (3, 1.0)]);
    }
}

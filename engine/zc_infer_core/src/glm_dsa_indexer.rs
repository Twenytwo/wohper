//! GLM-DSA indexer math core.
//!
//! This module is intentionally independent from MODEL.bin parsing. It provides
//! the bounded top-k selection primitive that the runtime can wire to real
//! indexer tensors after parity fixtures exist.

pub const GLM52_INDEX_TOPK: usize = 2048;
pub const GLM52_INDEX_N_HEADS: usize = 32;
pub const GLM52_INDEX_HEAD_DIM: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct IndexerConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            n_heads: GLM52_INDEX_N_HEADS,
            head_dim: GLM52_INDEX_HEAD_DIM,
            top_k: GLM52_INDEX_TOPK,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexerError {
    Shape(&'static str),
    EmptyContext,
}

impl std::fmt::Display for IndexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(message) => write!(f, "GLM-DSA indexer shape error: {message}"),
            Self::EmptyContext => write!(f, "GLM-DSA indexer context is empty"),
        }
    }
}

impl std::error::Error for IndexerError {}

/// Select causal key positions using GLM-DSA indexer scoring.
///
/// Shapes:
/// - `query_heads`: `[n_heads * head_dim]`
/// - `key_cache`: `[context_len * head_dim]`
/// - `head_weights`: `[n_heads]`
///
/// The score for each key position is:
///
/// `sum_h(head_weight[h] * relu(dot(q[h], k[pos]) / sqrt(head_dim)))`.
pub fn select_topk_positions(
    query_heads: &[f32],
    key_cache: &[f32],
    head_weights: &[f32],
    context_len: usize,
    causal_len: usize,
    config: IndexerConfig,
) -> Result<Vec<usize>, IndexerError> {
    if context_len == 0 || causal_len == 0 {
        return Err(IndexerError::EmptyContext);
    }
    if config.n_heads == 0 || config.head_dim == 0 || config.top_k == 0 {
        return Err(IndexerError::Shape("config values must be positive"));
    }
    if query_heads.len() < config.n_heads * config.head_dim {
        return Err(IndexerError::Shape("query_heads too small"));
    }
    if key_cache.len() < context_len * config.head_dim {
        return Err(IndexerError::Shape("key_cache too small"));
    }
    if head_weights.len() < config.n_heads {
        return Err(IndexerError::Shape("head_weights too small"));
    }

    let visible = context_len.min(causal_len);
    let keep = visible.min(config.top_k);
    if keep == visible {
        return Ok((0..visible).collect());
    }

    let scale = 1.0f32 / (config.head_dim as f32).sqrt();
    let mut best = Vec::<(usize, f32)>::with_capacity(keep);
    for pos in 0..visible {
        let key = &key_cache[pos * config.head_dim..(pos + 1) * config.head_dim];
        let mut score = 0.0f32;
        for head in 0..config.n_heads {
            let query =
                &query_heads[head * config.head_dim..(head + 1) * config.head_dim];
            let dot = query
                .iter()
                .zip(key.iter())
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale;
            score += head_weights[head] * dot.max(0.0);
        }
        push_topk(&mut best, keep, pos, score);
    }
    best.sort_by(|(left_pos, left_score), (right_pos, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_pos.cmp(right_pos))
    });
    Ok(best.into_iter().map(|(pos, _)| pos).collect())
}

pub fn reuse_shared_indexer_positions(previous_full_layer: &[usize]) -> Vec<usize> {
    previous_full_layer.to_vec()
}

fn push_topk(best: &mut Vec<(usize, f32)>, keep: usize, pos: usize, score: f32) {
    if best.len() < keep {
        best.push((pos, score));
        return;
    }
    if let Some((worst_index, _)) = best.iter().enumerate().min_by(
        |(_, (left_pos, left_score)), (_, (right_pos, right_score))| {
            left_score
                .total_cmp(right_score)
                .then_with(|| right_pos.cmp(left_pos))
        },
    ) {
        let (_, worst_score) = best[worst_index];
        if score > worst_score {
            best[worst_index] = (pos, score);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_context_returns_full_causal_prefix() {
        let config = IndexerConfig {
            n_heads: 1,
            head_dim: 2,
            top_k: 4,
        };
        let selected = select_topk_positions(
            &[1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            &[1.0],
            3,
            3,
            config,
        )
        .unwrap();

        assert_eq!(selected, vec![0, 1, 2]);
    }

    #[test]
    fn long_context_selects_highest_relu_scores() {
        let config = IndexerConfig {
            n_heads: 1,
            head_dim: 2,
            top_k: 2,
        };
        let selected = select_topk_positions(
            &[1.0, 0.0],
            &[
                1.0, 0.0,  // pos0 score 1/sqrt2
                -5.0, 0.0, // pos1 relu zero
                3.0, 0.0,  // pos2 score 3/sqrt2
                2.0, 0.0,  // pos3 score 2/sqrt2
            ],
            &[1.0],
            4,
            4,
            config,
        )
        .unwrap();

        assert_eq!(selected, vec![2, 3]);
    }

    #[test]
    fn causal_len_hides_future_positions() {
        let config = IndexerConfig {
            n_heads: 1,
            head_dim: 1,
            top_k: 1,
        };
        let selected = select_topk_positions(
            &[1.0],
            &[1.0, 10.0, 100.0],
            &[1.0],
            3,
            2,
            config,
        )
        .unwrap();

        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn shared_indexer_reuses_previous_full_layer_positions() {
        let selected = reuse_shared_indexer_positions(&[7, 3, 1]);
        assert_eq!(selected, vec![7, 3, 1]);
    }
}

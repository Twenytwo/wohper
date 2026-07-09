pub mod compute;
pub mod deepseek_v4;
pub mod direct_io;
pub mod glm_dsa_indexer;
pub mod model_format;
pub mod row_index;
pub mod server;
pub mod scheduler;

pub const ALIGN_2MB: u64 = 2 * 1024 * 1024;

/// Per-layer/per-token hot-path logging gate (ZC_VERBOSE=1). The decode
/// loop emits hundreds of trace lines per pass; behind Docker's log driver
/// that is measurable overhead, so tracing is opt-in.
pub fn verbose_logging() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ZC_VERBOSE")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

/// `eprintln!` gated behind `ZC_VERBOSE=1` - for hot-path trace lines only
/// (startup, errors and summary lines keep plain `eprintln!`).
#[macro_export]
macro_rules! vlog {
    ($($arg:tt)*) => {
        if $crate::verbose_logging() {
            eprintln!($($arg)*);
        }
    };
}

#[inline]
pub const fn is_aligned_2mb(value: u64) -> bool {
    value & (ALIGN_2MB - 1) == 0
}

#[inline]
pub const fn align_up_2mb(value: u64) -> u64 {
    (value + ALIGN_2MB - 1) & !(ALIGN_2MB - 1)
}

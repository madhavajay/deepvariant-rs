//! Local realignment subsystem (port-in-progress).
//!
//! Mirrors `deepvariant/realigner/`. Currently provides:
//!
//!   - `ssw` — Smith-Waterman pairwise alignment (classical, not the
//!     SIMD/striped variant the upstream uses; same numerical result).
//!
//! Not yet ported:
//!   - `debruijn_graph` — haplotype assembly from candidate reads
//!   - `window_selector` — picks regions for re-assembly
//!   - `fast_pass_aligner` — orchestrates SSW alignment of reads to
//!     assembled haplotypes
//!
//! Without these, indel candidate calling lags upstream by a few percent
//! on heavily mutated regions.

pub mod debruijn;
pub mod fast_pass;
pub mod orchestrator;
pub mod ssw;
pub mod window_selector;

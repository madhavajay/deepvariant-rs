//! Genomics compute. Modules added per port milestone.
//!
//! Most modules are pure-compute (channels, layout, math, phasing,
//! features, …) and build for wasm32-unknown-unknown. The two that
//! pull in TFRecord I/O via `dv-io` (which transitively brings
//! flate2/bzip2/xz) are gated behind the `io` Cargo feature, on by
//! default.

pub mod allelecounter;
pub mod alt_aligned_pileup;
pub mod direct_phasing;
pub mod fuzzy_support;
pub mod gvcf;
pub mod make_examples;
pub mod math;
pub mod merge_phased_reads;
pub mod methylation_aware_phasing;
pub mod nucleus;
pub mod phasing_apply;
pub mod pileup_image;
#[cfg(feature = "io")]
pub mod postprocess;
pub mod realigner;
pub mod small_model;
pub mod utils;
pub mod variant_calling;
pub mod vcf;

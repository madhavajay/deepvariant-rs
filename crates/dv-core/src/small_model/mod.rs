//! Small-model fast-path feature extraction.
//!
//! Port of `deepvariant/small_model/make_small_model_examples.py`.
//! Produces a fixed-length feature vector that the small WGS model
//! (`models/small_wgs.onnx`, input shape `[batch, 70]`) consumes:
//!
//!   * 12 base features (read counts / depths / VAFs / MAPQ / BQ / strand)
//!   * 7 variant features (snp/ins/del flags, lengths, multiallelic flags)
//!   * `vaf_window_size` (default 51) per-position VAF context features,
//!     ordered from offset `-half` through `+half` of the variant start.
//!
//! Feature order matches upstream's Python dict-insertion order, which is
//! the order Keras saw at training time. Don't reorder.
//!
//! All feature *values* are integers in upstream (Python `int`); we keep
//! them integers internally and only cast to `f32` at the very end so a
//! mistake in arithmetic surfaces as a clear value rather than a silent
//! float drift.

use std::collections::HashMap;

use dv_proto::nucleus_v1::Variant;

pub mod features;

/// One read's per-allele attributes — the only data the small-model
/// features need from the read.
#[derive(Debug, Clone, Copy)]
pub struct ReadAttrs {
    pub mapping_quality: u8,
    pub avg_base_quality: u8,
    pub is_reverse_strand: bool,
}

/// Default VAF context window size (matches upstream WGS config).
pub const VAF_CONTEXT_WINDOW_SIZE: usize = 51;

/// Default GQ thresholds for accepting a small-model call (Phred). Below
/// these the candidate is bumped to the big model.
pub const SNP_GQ_THRESHOLD: f64 = 25.0;
pub const INDEL_GQ_THRESHOLD: f64 = 30.0;

/// Tells whether this candidate should go through the small model at all.
pub fn is_eligible(variant: &Variant) -> bool {
    // Upstream defaults: accept_snps=True, accept_indels=True,
    // accept_multiallelics=True. So everything is eligible.
    let _ = variant;
    true
}

/// Compute the 70-feature vector (with default window size) for one
/// candidate × alt-allele-indices combo.
pub fn compute(
    variant: &Variant,
    alt_allele_indices: &[i32],
    ref_reads: &[ReadAttrs],
    alt_reads: &[ReadAttrs],
    total_depth: i32,
    vaf_at_position: &HashMap<i64, i32>,
) -> Vec<f32> {
    features::compute_with_window(
        variant,
        alt_allele_indices,
        ref_reads,
        alt_reads,
        total_depth,
        vaf_at_position,
        VAF_CONTEXT_WINDOW_SIZE,
    )
}

/// Convert a softmax probability vector (`[ref, het, hom_alt]`) into a
/// bounded-Phred GQ score, matching upstream's
/// `genomics_math.ptrue_to_bounded_phred(max_p, max_confidence=1-1e-7)`.
///
/// Returns the GQ for the most-likely class.
pub fn bounded_phred(probs: &[f32]) -> f64 {
    let max_p = probs
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let bounded = (max_p as f64).min(1.0 - 1e-7).max(1e-7);
    -10.0 * (1.0 - bounded).log10()
}

/// Decide if `probs` clears the configured GQ threshold for this variant
/// type. SNPs get a lower bar than indels (per upstream defaults).
pub fn passes_threshold(variant: &Variant, alt_allele_indices: &[i32], probs: &[f32]) -> bool {
    let is_snp_call = features::is_snp(variant, alt_allele_indices);
    let threshold = if is_snp_call {
        SNP_GQ_THRESHOLD
    } else {
        INDEL_GQ_THRESHOLD
    };
    bounded_phred(probs) >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_phred_high_confidence_caps() {
        // p=1.0 should map to ≥70 (= -10*log10(1e-7))
        let v = bounded_phred(&[1.0, 0.0, 0.0]);
        assert!(v >= 65.0, "got {v}");
        // p=0.5 → -10*log10(0.5) ≈ 3.01
        let v = bounded_phred(&[0.0, 0.5, 0.5]);
        assert!((2.5..=3.5).contains(&v), "got {v}");
    }
}

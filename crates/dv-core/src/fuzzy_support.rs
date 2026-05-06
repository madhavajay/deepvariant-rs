//! Read-supports-variant *fuzzy* support classifier.
//!
//! Port of upstream
//! `deepvariant/channels/read_supports_variant_fuzzy_channel.cc::ReadSupportsAlt`.
//! For long-read pipelines (PacBio HiFi, ONT) the indel error model is
//! sloppy enough that a read often reports an alt that's 1–2 bp
//! different from the candidate's *primary* alt but corresponds to
//! the same haplotype. This classifier converts a per-(candidate,
//! read) pair into a "support code" that
//! `pileup_image::channels::read_supports_variant_fuzzy_read` then
//! maps to a pixel value.
//!
//! Support codes (all defined in `pileup_image::channels`):
//!   * 0  = no support
//!   * 1  = exact alt support
//!   * 2  = other-alt support (read supports a different alt than
//!          the one being painted)
//!   * 8  = fuzzy match within 3 bp of the painted alt's size, same
//!          haplotype phase
//!   * 9  = fuzzy match within 2 bp
//!   * 10 = fuzzy match within 1 bp
//!
//! The encoder is in `pileup_image::channels`; this module is the
//! classifier that lives upstream of it.
//!
//! Inputs are designed to be plain Rust types so callers don't need
//! to assemble a full `DeepVariantCall` proto:
//!   * `variant_alts`         — the candidate's complete alt list (in
//!                              the order they appear on the Variant
//!                              proto).
//!   * `pileup_alt_indices`   — the indices into `variant_alts` of the
//!                              alts being painted in this pileup
//!                              (1 entry for biallelic, 1–2 for
//!                              het-alt rendering).
//!   * `allele_support`       — map from alt-allele bases → list of
//!                              read keys (`fragment_name/read_number`)
//!                              that support that alt.
//!   * `alt_allele_phases`    — per-candidate-alt phase value from the
//!                              `ALT_PS` info field. Index `i` is the
//!                              phase for `variant_alts[i]`. 0 = unknown
//!                              / both haplotypes.
//!   * `read_key`             — the read's `fragment_name/read_number`.
//!   * `read_hp`              — the read's `HP` aux-tag value (0 / 1 / 2).

use std::collections::HashMap;

/// Classify a single (candidate, read) pair into a fuzzy support code.
///
/// Mirrors upstream's `ReadSupportsAlt` plus its inner `CalculateReadSupport`.
/// The lookup is O(N · M) where N = number of alts and M = average
/// reads-per-alt; for typical short-read coverage that's <100 ops.
#[allow(clippy::too_many_arguments)]
pub fn classify_read_support(
    variant_alts: &[String],
    pileup_alt_indices: &[i32],
    allele_support: &HashMap<String, Vec<String>>,
    alt_allele_phases: &[i32],
    read_key: &str,
    read_hp: i32,
) -> i32 {
    let pileup_alts: Vec<&String> = pileup_alt_indices
        .iter()
        .filter_map(|&i| variant_alts.get(i as usize))
        .collect();

    for alt in variant_alts {
        let supporters = match allele_support.get(alt) {
            Some(s) => s,
            None => continue,
        };
        let read_supports_this_alt =
            supporters.iter().any(|name| name == read_key);
        if !read_supports_this_alt {
            continue;
        }
        let alt_in_pileup = pileup_alts.iter().any(|p| *p == alt);
        if alt_in_pileup {
            return 1; // exact
        }
        // The read supports a non-pileup alt — try to fuzzy-match it
        // against each of the pileup alts.
        for image_alt in &pileup_alts {
            let image_alt_global_index = variant_alts
                .iter()
                .position(|a| a == *image_alt);
            let image_global = match image_alt_global_index {
                Some(i) => i,
                None => continue,
            };
            let image_phase = alt_allele_phases.get(image_global).copied().unwrap_or(0);
            // Phase compatibility: phase 0 on either side is "any
            // haplotype"; otherwise both must agree.
            let phase_ok = image_phase == 0 || read_hp == 0 || image_phase == read_hp;
            if !phase_ok {
                continue;
            }
            let size_diff =
                ((image_alt.len() as i64) - (alt.len() as i64)).unsigned_abs() as usize;
            // Upstream uses 1-bp and 2-bp fuzzy codes; 3-bp is defined
            // but never returned by the algorithm. Match upstream.
            match size_diff {
                1 => return 10,
                2 => return 9,
                _ => {}
            }
        }
        // Read supports an alt that we couldn't fuzzy-match — flag
        // it as supporting *some* alt at this position.
        return 2;
    }
    0
}

/// Helper: look up the `ALT_PS` info-field value list and project it
/// onto a per-alt phase vector. Mirrors upstream's `CalculateAlelePhases`.
///
/// `info_values_for_alt_ps` is the raw list of integer values from
/// `Variant.info[<alt_ps_key>]`. Upstream stores ref-phase at index 0
/// and per-alt phases at 1..=N, so we slice off the first element.
/// Missing entries default to 0.
pub fn calculate_allele_phases(
    info_values_for_alt_ps: &[i32],
    num_alt_alleles: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; num_alt_alleles];
    for i in 0..num_alt_alleles {
        if i + 1 < info_values_for_alt_ps.len() {
            out[i] = info_values_for_alt_ps[i + 1];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support_map(alts: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
        alts.iter()
            .map(|(a, reads)| {
                (
                    (*a).to_string(),
                    reads.iter().map(|s| (*s).to_string()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn exact_match_returns_1() {
        let variant_alts = vec!["C".to_string()];
        let support =
            support_map(&[("C", &["FRAG1/1", "FRAG2/1"])]);
        let code = classify_read_support(
            &variant_alts,
            &[0],
            &support,
            &[0],
            "FRAG1/1",
            0,
        );
        assert_eq!(code, 1);
    }

    #[test]
    fn no_supporting_read_returns_0() {
        let variant_alts = vec!["C".to_string()];
        let support = support_map(&[("C", &["FRAG2/1"])]);
        let code = classify_read_support(
            &variant_alts,
            &[0],
            &support,
            &[0],
            "FRAG_OTHER/1",
            0,
        );
        assert_eq!(code, 0);
    }

    #[test]
    fn fuzzy_match_1bp_with_compatible_phase_returns_10() {
        // Pileup paints alt index 0 = "AAAA" (4 chars).
        // Candidate also has alt index 1 = "AAA" (3 chars; 1bp shorter).
        // Read supports alt 1 (the non-pileup one), HP matches phase.
        let variant_alts = vec!["AAAA".to_string(), "AAA".to_string()];
        let support = support_map(&[("AAA", &["READ1/1"])]);
        // alt_allele_phases: phase for AAAA = 1, phase for AAA = 1
        let phases = vec![1, 1];
        let code = classify_read_support(
            &variant_alts,
            &[0],
            &support,
            &phases,
            "READ1/1",
            1, // matching HP
        );
        assert_eq!(code, 10);
    }

    #[test]
    fn fuzzy_match_2bp_returns_9() {
        let variant_alts = vec!["AAAAA".to_string(), "AAA".to_string()];
        let support = support_map(&[("AAA", &["READ1/1"])]);
        let phases = vec![1, 1];
        let code = classify_read_support(
            &variant_alts,
            &[0], // paint AAAAA
            &support,
            &phases,
            "READ1/1",
            1,
        );
        assert_eq!(code, 9);
    }

    #[test]
    fn fuzzy_match_3bp_falls_through_to_2() {
        // 3 bp size diff doesn't match the upstream's 1/2 bp window,
        // so the classifier returns 2 ("other alt").
        let variant_alts = vec!["AAAAAA".to_string(), "AAA".to_string()];
        let support = support_map(&[("AAA", &["READ1/1"])]);
        let phases = vec![1, 1];
        let code = classify_read_support(
            &variant_alts,
            &[0], // paint AAAAAA (6 chars; 3 bp diff from AAA)
            &support,
            &phases,
            "READ1/1",
            1,
        );
        assert_eq!(code, 2);
    }

    #[test]
    fn fuzzy_match_phase_mismatch_returns_2() {
        // Read has HP=2; image alt has phase=1; non-zero on both
        // sides means phases must agree. They don't, so no fuzzy code.
        let variant_alts = vec!["AAAA".to_string(), "AAA".to_string()];
        let support = support_map(&[("AAA", &["READ1/1"])]);
        let phases = vec![1, 1]; // image_alt (AAAA) → phase 1
        let code = classify_read_support(
            &variant_alts,
            &[0],
            &support,
            &phases,
            "READ1/1",
            2, // HP=2 doesn't match phase=1
        );
        assert_eq!(code, 2); // other-alt support, not fuzzy
    }

    #[test]
    fn fuzzy_match_phase_zero_unblocks_match() {
        // image phase = 0 ("any") → phase check always passes.
        let variant_alts = vec!["AAAA".to_string(), "AAA".to_string()];
        let support = support_map(&[("AAA", &["READ1/1"])]);
        let phases = vec![0, 1];
        let code = classify_read_support(
            &variant_alts,
            &[0],
            &support,
            &phases,
            "READ1/1",
            2, // HP=2 — but image_phase=0 means we don't care
        );
        assert_eq!(code, 10);
    }

    #[test]
    fn read_hp_zero_unblocks_match() {
        // Read HP=0 ("unknown") → phase check passes.
        let variant_alts = vec!["AAAA".to_string(), "AAA".to_string()];
        let support = support_map(&[("AAA", &["READ1/1"])]);
        let phases = vec![1, 1];
        let code = classify_read_support(
            &variant_alts,
            &[0],
            &support,
            &phases,
            "READ1/1",
            0, // unknown HP — pass anyway
        );
        assert_eq!(code, 10);
    }

    #[test]
    fn other_alt_support_returns_2() {
        // Read supports an alt that isn't in the pileup AND can't be
        // fuzzy-matched (size diff is 0 since both alts are SNVs).
        let variant_alts = vec!["C".to_string(), "G".to_string()];
        let support = support_map(&[("G", &["READ1/1"])]);
        let phases = vec![1, 1];
        let code = classify_read_support(
            &variant_alts,
            &[0], // paint C
            &support,
            &phases,
            "READ1/1",
            1,
        );
        assert_eq!(code, 2);
    }

    #[test]
    fn calculate_allele_phases_drops_ref_index() {
        // info[ALT_PS] = [ref_phase, alt0_phase, alt1_phase]
        let info = vec![0, 1, 2];
        let phases = calculate_allele_phases(&info, 2);
        assert_eq!(phases, vec![1, 2]);
    }

    #[test]
    fn calculate_allele_phases_pads_with_zero() {
        let info = vec![0, 1]; // missing alt1 phase
        let phases = calculate_allele_phases(&info, 2);
        assert_eq!(phases, vec![1, 0]);
    }

    #[test]
    fn calculate_allele_phases_empty_info_all_zero() {
        let phases = calculate_allele_phases(&[], 3);
        assert_eq!(phases, vec![0, 0, 0]);
    }
}

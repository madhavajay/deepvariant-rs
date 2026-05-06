//! Apply direct-phasing output to variant records.
//!
//! `direct_phasing::DirectPhasing::phased_variants()` yields per-position
//! `PhasedVariant { position, phase_1_bases, phase_2_bases,
//! is_first_in_block }` entries. Postprocess needs to:
//!
//!   * Group those into phase-set blocks (PS = leftmost position in
//!     the block, since `is_first_in_block` marks block starts).
//!   * Switch each matching variant's genotype representation from
//!     `0/1` to `0|1` (phased), with the order reflecting which alt
//!     belongs to phase 1 vs phase 2.
//!   * Set the FORMAT `PS` field on the record.
//!
//! This module is a pure-logic helper: it takes a `Vec<PhasedVariant>`
//! and a slice of `Variant` records, returns a map keyed by genomic
//! position with the per-variant updates the caller should apply.

use std::collections::HashMap;

use dv_proto::nucleus_v1::{value, ListValue, Value, Variant, VariantCall};

use crate::direct_phasing::PhasedVariant;

/// One variant's phasing update.
#[derive(Debug, Clone, PartialEq)]
pub struct PhasingUpdate {
    /// Genotype tuple in phase order (e.g. `[0, 1]` for `0|1`,
    /// `[1, 0]` for `1|0`). Always length 2 for phased variants.
    pub genotype: [i32; 2],
    /// Phase set ID — the genomic start (0-based) of the first
    /// variant in this block.
    pub phase_set: i64,
}

/// Build a per-position map of phasing updates from
/// `phased_variants`. Block boundaries (`is_first_in_block=true`)
/// reset the running PS to the current variant's position; subsequent
/// variants inherit that PS.
///
/// The order of `phase_1_bases` / `phase_2_bases` in the input
/// determines the phased genotype. We match each `PhasedVariant`
/// against the corresponding `Variant` record by start position; the
/// alt-allele indices are looked up from the variant's
/// `alternate_bases`. If the phased alt isn't in the variant's alt
/// list (e.g. the phasing reported a different allele), the variant
/// is skipped — phasing only applies when alleles agree.
pub fn build_phasing_updates(
    phased_variants: &[PhasedVariant],
    variants: &[Variant],
) -> HashMap<i64, PhasingUpdate> {
    let mut by_position: HashMap<i64, &Variant> = HashMap::new();
    for v in variants {
        by_position.insert(v.start, v);
    }

    let mut updates: HashMap<i64, PhasingUpdate> = HashMap::new();
    let mut current_ps: Option<i64> = None;

    for pv in phased_variants {
        if pv.is_first_in_block || current_ps.is_none() {
            current_ps = Some(pv.position);
        }
        let ps = current_ps.expect("ps initialized");
        let v = match by_position.get(&pv.position) {
            Some(v) => v,
            None => continue,
        };

        // Resolve each haplotype's bases to an allele index. 0 = REF.
        let resolve = |bases: &str| -> Option<i32> {
            if bases == v.reference_bases {
                return Some(0);
            }
            v.alternate_bases
                .iter()
                .position(|a| a == bases)
                .map(|i| (i + 1) as i32)
        };
        let g1 = match resolve(&pv.phase_1_bases) {
            Some(g) => g,
            None => continue,
        };
        let g2 = match resolve(&pv.phase_2_bases) {
            Some(g) => g,
            None => continue,
        };

        updates.insert(
            pv.position,
            PhasingUpdate {
                genotype: [g1, g2],
                phase_set: ps,
            },
        );
    }

    updates
}

/// Apply a `PhasingUpdate` in place to a `VariantCall`. Sets
/// `is_phased=true`, replaces `genotype`, and sets `PS` info field.
/// No-op on unphased variants — the caller is expected to look up the
/// update from the map and only call this when there's something to
/// apply.
pub fn apply_phasing_update(call: &mut VariantCall, update: &PhasingUpdate) {
    call.is_phased = true;
    call.genotype = vec![update.genotype[0], update.genotype[1]];
    call.info.insert(
        "PS".to_string(),
        ListValue {
            values: vec![Value {
                kind: Some(value::Kind::IntValue(update.phase_set as i32)),
            }],
        },
    );
}

/// Convenience helper: given a slice of `(Variant, &PhasingUpdate)`
/// pairs, produce a vector of phased variants. Used by callers that
/// want a one-shot transformation. Returns clones so the input is not
/// modified.
pub fn apply_to_variants(
    phased_variants: &[PhasedVariant],
    variants: &[Variant],
) -> Vec<Variant> {
    let updates = build_phasing_updates(phased_variants, variants);
    variants
        .iter()
        .map(|v| {
            let mut v = v.clone();
            if let Some(update) = updates.get(&v.start) {
                if v.calls.is_empty() {
                    v.calls.push(VariantCall::default());
                }
                apply_phasing_update(&mut v.calls[0], update);
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pv(position: i64, p1: &str, p2: &str, first: bool) -> PhasedVariant {
        PhasedVariant {
            position,
            phase_1_bases: p1.into(),
            phase_2_bases: p2.into(),
            is_first_in_block: first,
        }
    }

    fn var(start: i64, refb: &str, alts: &[&str]) -> Variant {
        Variant {
            reference_name: "chr1".into(),
            start,
            end: start + refb.len() as i64,
            reference_bases: refb.into(),
            alternate_bases: alts.iter().map(|s| (*s).to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn single_block_three_variants_share_ps() {
        // Three SNVs in one block, first variant marks the block.
        let phased = vec![
            pv(100, "A", "C", true),
            pv(110, "G", "T", false),
            pv(120, "T", "G", false),
        ];
        let variants = vec![
            var(100, "A", &["C"]),
            var(110, "G", &["T"]),
            var(120, "T", &["G"]),
        ];
        let updates = build_phasing_updates(&phased, &variants);
        assert_eq!(updates.len(), 3);
        assert_eq!(updates[&100].phase_set, 100);
        assert_eq!(updates[&110].phase_set, 100);
        assert_eq!(updates[&120].phase_set, 100);
        // 100: phase 1 = REF (A), phase 2 = alt-1 (C)
        assert_eq!(updates[&100].genotype, [0, 1]);
        // 120: phase 1 = REF (T), phase 2 = alt-1 (G)
        assert_eq!(updates[&120].genotype, [0, 1]);
    }

    #[test]
    fn two_blocks_have_distinct_ps() {
        let phased = vec![
            pv(100, "A", "C", true),
            pv(110, "G", "T", false),
            pv(200, "T", "G", true), // new block
            pv(210, "C", "A", false),
        ];
        let variants = vec![
            var(100, "A", &["C"]),
            var(110, "G", &["T"]),
            var(200, "T", &["G"]),
            var(210, "C", &["A"]),
        ];
        let updates = build_phasing_updates(&phased, &variants);
        assert_eq!(updates[&100].phase_set, 100);
        assert_eq!(updates[&110].phase_set, 100);
        assert_eq!(updates[&200].phase_set, 200);
        assert_eq!(updates[&210].phase_set, 200);
    }

    #[test]
    fn swapped_phase_is_recorded_as_one_zero() {
        // Phased output has alt on phase 1, ref on phase 2 →
        // genotype = [1, 0].
        let phased = vec![pv(100, "C", "A", true)];
        let variants = vec![var(100, "A", &["C"])];
        let updates = build_phasing_updates(&phased, &variants);
        assert_eq!(updates[&100].genotype, [1, 0]);
        assert_eq!(updates[&100].phase_set, 100);
    }

    #[test]
    fn unmatched_position_is_dropped() {
        // PhasedVariant exists but no Variant at that position.
        let phased = vec![pv(100, "A", "C", true)];
        let variants: Vec<Variant> = vec![];
        let updates = build_phasing_updates(&phased, &variants);
        assert!(updates.is_empty());
    }

    #[test]
    fn non_matching_alt_is_dropped() {
        // The phased alt "G" isn't in the variant's alt list ["C"].
        let phased = vec![pv(100, "A", "G", true)];
        let variants = vec![var(100, "A", &["C"])];
        let updates = build_phasing_updates(&phased, &variants);
        assert!(updates.is_empty());
    }

    #[test]
    fn apply_phasing_update_sets_call_fields() {
        let mut call = VariantCall::default();
        let update = PhasingUpdate {
            genotype: [1, 0],
            phase_set: 12345,
        };
        apply_phasing_update(&mut call, &update);
        assert!(call.is_phased);
        assert_eq!(call.genotype, vec![1, 0]);
        let ps = call.info.get("PS").expect("PS field");
        let kind = ps.values[0].kind.as_ref().unwrap();
        match kind {
            value::Kind::IntValue(n) => assert_eq!(*n, 12345),
            _ => panic!("PS should be IntValue"),
        }
    }

    #[test]
    fn apply_to_variants_round_trip() {
        let phased = vec![pv(100, "A", "C", true), pv(110, "T", "G", false)];
        let variants = vec![var(100, "A", &["C"]), var(110, "T", &["G"])];
        let phased_vars = apply_to_variants(&phased, &variants);
        assert_eq!(phased_vars.len(), 2);
        for v in &phased_vars {
            let call = &v.calls[0];
            assert!(call.is_phased);
            assert!(call.info.contains_key("PS"));
        }
    }

    #[test]
    fn first_in_block_implicit_when_no_running_ps() {
        // is_first_in_block=false on the very first PhasedVariant
        // shouldn't crash; we treat the absence of a running PS as a
        // new block.
        let phased = vec![pv(100, "A", "C", false)];
        let variants = vec![var(100, "A", &["C"])];
        let updates = build_phasing_updates(&phased, &variants);
        assert_eq!(updates[&100].phase_set, 100);
    }
}

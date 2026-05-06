//! gVCF generation: interleave variant calls with non-variant blocks.
//!
//! Mirrors `third_party/nucleus/io/merge_variants.cc`.

#[cfg(feature = "io")]
use std::path::Path;

#[cfg(feature = "io")]
use anyhow::{Context, Result};
#[cfg(feature = "io")]
use prost::Message;

use dv_proto::nucleus_v1::{value, ListValue, Value, Variant};

use crate::math;

/// Symbolic gVCF "any other allele" marker.
pub const GVCF_ALT_ALLELE: &str = "<*>";

/// log10 likelihood assigned to genotypes containing `<*>` for variants
/// promoted into gVCF (`_GVCF_ALT_ALLELE_GL` in upstream).
pub const GVCF_ALT_ALLELE_GL: f64 = -99.0;

/// Subtract max GL so the highest is 0 — matches `ZeroScaleGl`.
pub fn zero_shift_gl(variant: &mut Variant) {
    if let Some(call) = variant.calls.first_mut() {
        if call.genotype_likelihood.is_empty() {
            return;
        }
        let max = call
            .genotype_likelihood
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        for g in &mut call.genotype_likelihood {
            *g -= max;
        }
    }
}

/// Append `<*>` alt + extend GL/AD/VAF in-place. Idempotent — does nothing
/// if `<*>` is already present.
pub fn transform_to_gvcf(variant: &mut Variant) {
    if variant
        .alternate_bases
        .iter()
        .any(|a| a == GVCF_ALT_ALLELE)
    {
        return;
    }
    variant.alternate_bases.push(GVCF_ALT_ALLELE.to_string());

    let n_alts_after = variant.alternate_bases.len();
    if let Some(call) = variant.calls.first_mut() {
        // Match upstream's `alternate_bases.size() + 1` post-push: for diploid
        // adding `<*>` introduces `n_alts_after + 1` new genotype entries
        // (= n_alts_before + 2). E.g. 1 alt + <*> → 3 new GL entries; for
        // 2 alts + <*> → 4 new entries.
        for _ in 0..(n_alts_after + 1) {
            call.genotype_likelihood.push(GVCF_ALT_ALLELE_GL);
        }
        // Append a 0 to AD (R-indexed: REF + alts).
        if let Some(ad) = call.info.get_mut("AD") {
            ad.values.push(Value {
                kind: Some(value::Kind::IntValue(0)),
            });
        }
        // Append a 0 to VAF (A-indexed: alts only).
        if let Some(vaf) = call.info.get_mut("VAF") {
            vaf.values.push(Value {
                kind: Some(value::Kind::NumberValue(0.0)),
            });
        }
    }

    // Recompute PL from the fresh GL list (zero-shift then truncate).
    if let Some(call) = variant.calls.first_mut() {
        if !call.genotype_likelihood.is_empty() {
            let max = call
                .genotype_likelihood
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let pls: Vec<i32> = call
                .genotype_likelihood
                .iter()
                .map(|gl| (-10.0 * (gl - max)) as i32)
                .collect();
            call.info.insert(
                "PL".to_string(),
                ListValue {
                    values: pls
                        .iter()
                        .map(|&n| Value {
                            kind: Some(value::Kind::IntValue(n)),
                        })
                        .collect(),
                },
            );
        }
    }
}

/// Read non-variant gVCF blocks from a TFRecord shard. Behind the `io`
/// feature so wasm builds (which can't link bgzf/flate2 without help)
/// don't try to compile it.
#[cfg(feature = "io")]
pub fn load_nonvariants<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<Variant>> {
    let mut out = Vec::new();
    for p in paths {
        let mut r = dv_io::tfrecord::open_reader(p.as_ref())
            .with_context(|| format!("open gvcf shard {}", p.as_ref().display()))?;
        while let Some(rec) = r.read_record()? {
            out.push(Variant::decode(&*rec).context("decode Variant")?);
        }
    }
    // Sort by (contig, start, end) to be safe.
    out.sort_by(|a, b| {
        (a.reference_name.as_str(), a.start, a.end).cmp(&(
            b.reference_name.as_str(),
            b.start,
            b.end,
        ))
    });
    Ok(out)
}

/// Merge a sorted variant stream with a sorted non-variant stream into the
/// gVCF ordering. Variants always go through `zero_shift_gl` then
/// `transform_to_gvcf` before emission. Subsumed non-variants are dropped.
/// Truncation of overlapping non-variants is supported when `ref_lookup` is
/// provided (returns the REF base at a given 0-based position) — for
/// chr20 quickstart all blocks are pre-split, so passing `None` is fine.
pub fn merge_streams(
    variants: Vec<Variant>,
    nonvariants: Vec<Variant>,
    contig_index: impl Fn(&str) -> Option<u32>,
    mut ref_lookup: Option<&mut dyn FnMut(&str, i64) -> Option<String>>,
) -> Vec<Variant> {
    let mut out = Vec::new();
    let mut vi = 0usize;
    let mut ni = 0usize;
    let mut nv_carry: Option<Variant> = None;

    let take_v = |i: &mut usize, vs: &Vec<Variant>| -> Option<Variant> {
        if *i < vs.len() {
            let v = vs[*i].clone();
            *i += 1;
            Some(v)
        } else {
            None
        }
    };

    let mut cur_variant = take_v(&mut vi, &variants);
    let mut cur_nv = nv_carry.take().or_else(|| take_v(&mut ni, &nonvariants));

    while cur_variant.is_some() || cur_nv.is_some() {
        let v_idx = cur_variant.as_ref().and_then(|v| contig_index(&v.reference_name));
        let n_idx = cur_nv.as_ref().and_then(|n| contig_index(&n.reference_name));

        match (cur_variant.take(), cur_nv.take()) {
            (Some(mut v), None) => {
                zero_shift_gl(&mut v);
                transform_to_gvcf(&mut v);
                out.push(v);
                cur_variant = take_v(&mut vi, &variants);
            }
            (None, Some(n)) => {
                out.push(n);
                cur_nv = take_v(&mut ni, &nonvariants);
            }
            (Some(mut v), Some(mut n)) => {
                let v_contig = v_idx.unwrap_or(u32::MAX);
                let n_contig = n_idx.unwrap_or(u32::MAX);
                let variant_first = v_contig < n_contig
                    || (v_contig == n_contig && v.end <= n.start);
                let nonvariant_first = n_contig < v_contig
                    || (n_contig == v_contig && n.end <= v.start);
                if variant_first {
                    zero_shift_gl(&mut v);
                    transform_to_gvcf(&mut v);
                    out.push(v);
                    cur_variant = take_v(&mut vi, &variants);
                    cur_nv = Some(n);
                } else if nonvariant_first {
                    out.push(n);
                    cur_nv = take_v(&mut ni, &nonvariants);
                    cur_variant = Some(v);
                } else {
                    // overlap
                    if n.start < v.start {
                        // Emit left-truncated nonvariant [n.start, v.start)
                        let mut left = n.clone();
                        left.end = v.start;
                        out.push(left);
                    }
                    if n.end > v.end {
                        // Carry right-truncated nonvariant [v.end, n.end)
                        n.start = v.end;
                        if let Some(lookup) = ref_lookup.as_deref_mut() {
                            if let Some(b) = lookup(&n.reference_name, n.start) {
                                n.reference_bases = b;
                            }
                        }
                        cur_nv = Some(n);
                    } else {
                        // subsumed
                        cur_nv = take_v(&mut ni, &nonvariants);
                    }
                    cur_variant = Some(v);
                }
            }
            (None, None) => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dv_proto::nucleus_v1::VariantCall;

    fn make_variant(start: i64, end: i64) -> Variant {
        Variant {
            reference_name: "chr20".into(),
            start,
            end,
            reference_bases: "A".into(),
            alternate_bases: vec!["T".into()],
            calls: vec![VariantCall {
                genotype: vec![0, 1],
                genotype_likelihood: vec![-3.0, 0.0, -10.0],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_nonvariant(start: i64, end: i64) -> Variant {
        Variant {
            reference_name: "chr20".into(),
            start,
            end,
            reference_bases: "T".into(),
            alternate_bases: vec![GVCF_ALT_ALLELE.into()],
            calls: vec![VariantCall {
                genotype: vec![0, 0],
                genotype_likelihood: vec![0.0, -13.0, -135.0],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn ci(name: &str) -> Option<u32> {
        if name == "chr20" {
            Some(0)
        } else {
            None
        }
    }

    #[test]
    fn zero_shift_centers_max() {
        let mut v = make_variant(100, 101);
        zero_shift_gl(&mut v);
        assert_eq!(
            v.calls[0].genotype_likelihood,
            vec![-3.0, 0.0, -10.0]
        );
    }

    #[test]
    fn transform_appends_star() {
        let mut v = make_variant(100, 101);
        transform_to_gvcf(&mut v);
        assert_eq!(v.alternate_bases, vec!["T", GVCF_ALT_ALLELE]);
        // Original GL had 3 entries (3 alleles incl REF). After adding `<*>`
        // we have 4 alleles → 6 diploid genotypes → 3 new entries appended.
        assert_eq!(v.calls[0].genotype_likelihood.len(), 6);
    }

    #[test]
    fn merge_sequential() {
        let nv1 = make_nonvariant(10, 20); // before
        let v = make_variant(20, 21);
        let nv2 = make_nonvariant(21, 30); // after
        let merged = merge_streams(vec![v.clone()], vec![nv1.clone(), nv2.clone()], ci, None);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].start, nv1.start);
        assert_eq!(merged[1].start, v.start);
        // variant became gvcf
        assert!(merged[1].alternate_bases.contains(&GVCF_ALT_ALLELE.to_string()));
        assert_eq!(merged[2].start, nv2.start);
    }

    #[test]
    fn merge_subsumed_dropped() {
        let v = make_variant(20, 30);
        let nv = make_nonvariant(22, 25); // fully inside variant
        let merged = merge_streams(vec![v], vec![nv], ci, None);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_left_truncate() {
        let v = make_variant(20, 30);
        let nv = make_nonvariant(15, 25); // overlaps left
        let merged = merge_streams(vec![v], vec![nv], ci, None);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start, 15);
        assert_eq!(merged[0].end, 20); // truncated
    }
}

#[allow(dead_code)]
fn _silence_math_unused() -> f64 {
    math::MAX_CONFIDENCE
}

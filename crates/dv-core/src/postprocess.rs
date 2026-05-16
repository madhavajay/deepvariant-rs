//! Postprocess CallVariantsOutput into final Variant calls (M2).
//!
//! Pipeline:
//!   read CVOs → sort by genomic coordinate → group by variant range
//!   → merge predictions (single + multi-allelic) → genotype call + QUAL
//!   → set GT/GQ/PL/MID + filter → emit Variant proto

use std::cmp::Ordering;
use std::collections::HashMap;

#[cfg(feature = "io")]
use anyhow::{Context, Result};
#[cfg(feature = "io")]
use prost::Message;

use dv_proto::dv::CallVariantsOutput;
use dv_proto::nucleus_v1::{value, ListValue, Value, Variant, VariantCall};

use crate::math;

/// Default flags from `postprocess_variants.py`. Mirror upstream's defaults.
#[derive(Debug, Clone, Copy)]
pub struct PostprocessOptions {
    pub qual_filter: f64,
    pub multi_allelic_qual_filter: f64,
    pub cnn_homref_call_min_gq: f64,
    pub group_variants: bool,
    pub qual_precision: u32,
}

impl Default for PostprocessOptions {
    fn default() -> Self {
        Self {
            qual_filter: 1.0,
            multi_allelic_qual_filter: 1.0,
            cnn_homref_call_min_gq: 20.0,
            group_variants: true,
            qual_precision: 7,
        }
    }
}

/// Filter labels (from `dv_vcf_constants.py`).
pub mod filter {
    pub const PASS: &str = "PASS";
    pub const REF: &str = "RefCall";
    pub const LOW_QUAL: &str = "LowQual";
    pub const NO_CALL: &str = "NoCall";
}

/// Read CVO records from one or more TFRecord shards into a single Vec.
#[cfg(feature = "io")]
pub fn load_cvos<P: AsRef<std::path::Path>>(paths: &[P]) -> Result<Vec<CallVariantsOutput>> {
    let mut out = Vec::new();
    for p in paths {
        let mut r = dv_io::tfrecord::open_reader(p.as_ref())
            .with_context(|| format!("open CVO shard {}", p.as_ref().display()))?;
        while let Some(rec) = r.read_record()? {
            out.push(CallVariantsOutput::decode(&*rec).context("decode CVO")?);
        }
    }
    Ok(out)
}

/// Sort CVOs by (contig name, start, end, ref_bases, sorted alts, sorted alt_allele_indices).
/// Mirrors upstream's `process_single_sites_tfrecords` C++ sort behavior.
pub fn sort_cvos(cvos: &mut [CallVariantsOutput]) {
    cvos.sort_by(|a, b| compare_cvos(a, b));
}

fn variant_key(v: &Variant) -> (&str, i64, i64, &str, Vec<&str>) {
    let mut alts: Vec<&str> = v.alternate_bases.iter().map(|s| s.as_str()).collect();
    alts.sort();
    (
        v.reference_name.as_str(),
        v.start,
        v.end,
        v.reference_bases.as_str(),
        alts,
    )
}

fn compare_cvos(a: &CallVariantsOutput, b: &CallVariantsOutput) -> Ordering {
    let av = a.variant.as_ref().expect("CVO.variant present");
    let bv = b.variant.as_ref().expect("CVO.variant present");
    variant_key(av).cmp(&variant_key(bv)).then_with(|| {
        let mut ai: Vec<i32> = a
            .alt_allele_indices
            .as_ref()
            .map(|x| x.indices.clone())
            .unwrap_or_default();
        let mut bi: Vec<i32> = b
            .alt_allele_indices
            .as_ref()
            .map(|x| x.indices.clone())
            .unwrap_or_default();
        ai.sort();
        bi.sort();
        ai.cmp(&bi)
    })
}

/// Group adjacent CVOs at the same variant range AND with the same canonical
/// alt set (after sort). Returns groups in input order; each inner Vec is one
/// variant locus' CVOs.
///
/// CVOs at the same `(ref_name, start, end)` but with **different** alt sets
/// originate from different candidates (e.g. one from the allele counter and
/// one from a realigner-assembled haplotype) and must NOT be grouped:
/// `merge_predictions_multi` treats `cvos[0].variant.alternate_bases` as the
/// canonical alt list shared by every CVO in the group, so mixing alt sets
/// would cause `get_alt_alleles_to_remove` to insert / preserve the wrong
/// alleles and `prune_alleles` to silently empty the alt list (n_alleles=1
/// → assertion in `add_call_to_variant`).
pub fn group_cvos(sorted: Vec<CallVariantsOutput>) -> Vec<Vec<CallVariantsOutput>> {
    let mut groups: Vec<Vec<CallVariantsOutput>> = Vec::new();
    let mut cur_key: Option<(String, i64, i64, Vec<String>)> = None;
    for cvo in sorted {
        let v = cvo.variant.as_ref().expect("variant");
        let mut alts: Vec<String> = v.alternate_bases.clone();
        alts.sort();
        let k = (v.reference_name.clone(), v.start, v.end, alts);
        match &cur_key {
            Some(prev) if prev == &k => groups.last_mut().unwrap().push(cvo),
            _ => {
                cur_key = Some(k);
                groups.push(vec![cvo]);
            }
        }
    }
    groups
}

/// For diploid (P=2) and N alts, return (index, [h2, h1]) such that
/// genotype index ↔ unordered pair of allele indices in VCF spec order.
/// Matches `most_likely_genotype` for ploidy=2.
pub fn most_likely_genotype_diploid(predictions: &[f64], n_alleles: usize) -> (usize, [i32; 2]) {
    assert!(n_alleles >= 2, "n_alleles must be >= 2");
    let argmax = argmax_f64(predictions);
    let mut index = 0usize;
    for h1 in 0..=n_alleles {
        for h2 in 0..=h1 {
            if index == argmax {
                return (index, [h2 as i32, h1 as i32]);
            }
            index += 1;
        }
    }
    panic!("no genotype index for predictions; argmax={argmax}, n_alleles={n_alleles}")
}

fn argmax_f64(xs: &[f64]) -> usize {
    let mut idx = 0usize;
    let mut best = xs[0];
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx
}

/// Returns `(gq, qual)` matching upstream `compute_quals`.
/// `gq` = phred(predictions[index]) rounded;
/// `qual` = phred(min(sum(predictions[1..]), 1.0)) rounded to qual_precision decimals.
pub fn compute_quals(predictions: &[f64], index: usize, opts: &PostprocessOptions) -> (i32, f64) {
    let gq = math::ptrue_to_bounded_phred(predictions[index]).round() as i32;
    let pvar: f64 = predictions[1..].iter().sum::<f64>().min(1.0);
    let qual = math::ptrue_to_bounded_phred(pvar);
    let scale = 10f64.powi(opts.qual_precision as i32);
    let qual_rounded = (qual * scale).round() / scale;
    (gq, qual_rounded)
}

/// PL per genotype as upstream computes it: zero-shift the log10 likelihoods
/// so the max is 0, then apply `phred = -10 * log10p` and **truncate**
/// (C++ implicit double→int) — not round.
/// See `vcf_conversion.cc:1188-1240`.
pub fn compute_pl(predictions: &[f64]) -> Vec<i32> {
    let log10s: Vec<f64> = predictions
        .iter()
        .map(|&p| math::perror_to_bounded_log10_perror(p))
        .collect();
    let max = log10s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    log10s
        .iter()
        .map(|x| (-10.0 * (x - max)) as i32)
        .collect()
}

/// Mirror Python `set_format(call, key, value)` — set
/// `call.info[key] = ListValue([Value(string_value=value)])`.
pub fn set_call_string_info(call: &mut VariantCall, key: &str, value: &str) {
    call.info.insert(
        key.to_string(),
        ListValue {
            values: vec![Value {
                kind: Some(value::Kind::StringValue(value.to_string())),
            }],
        },
    );
}

pub fn get_call_string_info<'a>(call: &'a VariantCall, key: &str) -> Option<&'a str> {
    call.info.get(key).and_then(|lv| {
        lv.values.first().and_then(|v| match &v.kind {
            Some(value::Kind::StringValue(s)) => Some(s.as_str()),
            _ => None,
        })
    })
}

/// Sum of AD values for a call, used by `uncall_gt_if_no_ad`.
pub fn ad_sum(call: &VariantCall) -> i64 {
    call.info
        .get("AD")
        .map(|lv| {
            lv.values
                .iter()
                .filter_map(|v| match &v.kind {
                    Some(value::Kind::IntValue(n)) => Some(*n as i64),
                    _ => None,
                })
                .sum()
        })
        .unwrap_or(0)
}

/// Single-allelic merge: trivial pass-through of the only CVO's probabilities.
/// (Multi-allelic resolution lives in `merge_predictions_multi`.)
pub fn merge_predictions_single(cvos: &[CallVariantsOutput]) -> (Variant, Vec<f64>) {
    assert_eq!(cvos.len(), 1, "single-allelic merge expects exactly 1 CVO");
    let cvo = &cvos[0];
    let variant = cvo.variant.clone().expect("CVO.variant present");
    let preds: Vec<f64> = cvo.genotype_probabilities.clone();
    (variant, preds)
}

/// Compute the set of alt alleles to drop based on per-alt single-CVO quality.
/// Mirrors `get_alt_alleles_to_remove(qual_filter)`.
pub fn get_alt_alleles_to_remove(
    cvos: &[CallVariantsOutput],
    qual_filter: f64,
    opts: &PostprocessOptions,
) -> std::collections::HashSet<String> {
    let mut to_remove = std::collections::HashSet::new();
    if qual_filter <= 0.0 || cvos.is_empty() {
        return to_remove;
    }
    let canonical = cvos[0].variant.as_ref().expect("variant");
    let mut max_qual: Option<f64> = None;
    let mut max_qual_allele: Option<String> = None;
    for cvo in cvos {
        let aai = cvo.alt_allele_indices.as_ref().map(|x| x.indices.as_slice()).unwrap_or(&[]);
        if aai.len() != 1 {
            continue;
        }
        let (_, qual) = compute_quals(&cvo.genotype_probabilities, 0, opts);
        let alt_index = aai[0] as usize;
        if let Some(allele) = canonical.alternate_bases.get(alt_index).cloned() {
            if max_qual.is_none() || max_qual.unwrap() < qual {
                max_qual = Some(qual);
                max_qual_allele = Some(allele.clone());
            }
            if qual < qual_filter {
                to_remove.insert(allele);
            }
        }
    }
    // If everything would be removed, keep the highest-qual one.
    if to_remove.len() == canonical.alternate_bases.len() {
        if let Some(a) = max_qual_allele {
            to_remove.remove(&a);
        }
    }
    to_remove
}

/// Re-index allele-indexed format fields after pruning. `keep_index_in_orig`
/// is the predicate that says whether the i-th value of the original list
/// should be retained.
fn reindex_list_value(
    lv: &mut ListValue,
    keep_index_in_orig: impl Fn(usize) -> bool,
) {
    let kept: Vec<Value> = lv
        .values
        .iter()
        .enumerate()
        .filter(|(i, _)| keep_index_in_orig(*i))
        .map(|(_, v)| v.clone())
        .collect();
    lv.values = kept;
}

const ALT_INDEXED_FIELDS: &[(&str, bool)] = &[
    ("AD", true),  // ref-prefixed
    ("VAF", false),
    ("MF", true),
    ("MD", true),
    ("NAD", true),
    ("NAF", false),
];

/// Returns a Variant clone with the named alt alleles removed and AD/VAF/...
/// reindexed accordingly.
pub fn prune_alleles(
    variant: &Variant,
    alts_to_remove: &std::collections::HashSet<String>,
) -> Variant {
    if alts_to_remove.is_empty() {
        return variant.clone();
    }
    let mut new_v = variant.clone();
    let original_alts = variant.alternate_bases.clone();
    // Build a per-original-index keep predicate.
    let keep_alt_idx: Vec<bool> = original_alts
        .iter()
        .map(|a| !alts_to_remove.contains(a))
        .collect();

    // Rewrite alt list.
    new_v.alternate_bases = original_alts
        .iter()
        .enumerate()
        .filter_map(|(i, a)| if keep_alt_idx[i] { Some(a.clone()) } else { None })
        .collect();

    for call in &mut new_v.calls {
        for (field, ref_is_zero) in ALT_INDEXED_FIELDS {
            if let Some(lv) = call.info.get_mut(*field) {
                if *ref_is_zero {
                    reindex_list_value(lv, |i| i == 0 || keep_alt_idx.get(i - 1).copied().unwrap_or(false));
                } else {
                    reindex_list_value(lv, |i| keep_alt_idx.get(i).copied().unwrap_or(false));
                }
            }
        }
    }
    new_v
}

/// Multi-allelic merge using upstream's "product" mode (default), with
/// alt-pruning by single-CVO qual filter.
pub fn merge_predictions_multi(
    cvos: &[CallVariantsOutput],
    multi_allelic_qual_filter: f64,
    opts: &PostprocessOptions,
) -> (Variant, Vec<f64>) {
    assert!(!cvos.is_empty());
    let original_variant = cvos[0].variant.clone().expect("variant");
    let original_alts: Vec<String> = original_variant.alternate_bases.clone();
    let to_remove = get_alt_alleles_to_remove(cvos, multi_allelic_qual_filter, opts);
    let canonical = prune_alleles(&original_variant, &to_remove);

    // Collect per-CVO probs+alt-subset, skipping CVOs whose alts include any pruned allele.
    let example_info: Vec<(Vec<f64>, std::collections::HashSet<String>)> = cvos
        .iter()
        .filter_map(|cvo| {
            let aai = cvo.alt_allele_indices.as_ref()?.indices.as_slice();
            let alts_subset: std::collections::HashSet<String> = aai
                .iter()
                .filter_map(|&i| original_alts.get(i as usize).cloned())
                .collect();
            if alts_subset.iter().any(|a| to_remove.contains(a)) {
                return None;
            }
            Some((cvo.genotype_probabilities.clone(), alts_subset))
        })
        .collect();

    // Genotype ordering for the PRUNED allele set.
    let alleles: Vec<String> = std::iter::once(canonical.reference_bases.clone())
        .chain(canonical.alternate_bases.iter().cloned())
        .collect();
    let mut ordering: Vec<(String, String)> = Vec::new();
    for h1 in 0..alleles.len() {
        for h2 in 0..=h1 {
            ordering.push((alleles[h2].clone(), alleles[h1].clone()));
        }
    }

    // Product-fuse per genotype.
    let mut predictions = Vec::with_capacity(ordering.len());
    for (a1, a2) in &ordering {
        let mut prob_per_example = Vec::with_capacity(example_info.len());
        for (probs, alts_subset) in &example_info {
            let overlap = alts_subset.contains(a1) as usize + alts_subset.contains(a2) as usize;
            prob_per_example.push(probs.get(overlap).copied().unwrap_or(0.0));
        }
        predictions.push(prob_per_example.iter().product::<f64>());
    }

    // Normalize (matches `normalize_predictions`).
    let sum: f64 = predictions.iter().sum();
    if sum > 0.0 {
        for p in &mut predictions {
            *p /= sum;
        }
    } else {
        let n = predictions.len() as f64;
        for p in &mut predictions {
            *p = 1.0 / n;
        }
    }

    (canonical, predictions)
}

/// `uncall_homref_gt_if_lowqual` — set GT to ./. if it's a low-GQ hom-ref.
fn uncall_homref_if_lowqual(variant: &mut Variant, min_gq: f64) {
    if variant.calls.is_empty() {
        return;
    }
    let call = &variant.calls[0];
    if call.genotype != [0, 0] {
        return;
    }
    let gq = get_call_int_info(call, "GQ").unwrap_or(i32::MAX);
    if (gq as f64) < min_gq {
        let call = &mut variant.calls[0];
        call.genotype = vec![-1, -1];
    }
}

fn get_call_int_info(call: &VariantCall, key: &str) -> Option<i32> {
    call.info.get(key).and_then(|lv| {
        lv.values.first().and_then(|v| match &v.kind {
            Some(value::Kind::IntValue(n)) => Some(*n),
            _ => None,
        })
    })
}

/// `compute_filter_fields` — derive PASS/RefCall/LowQual/NoCall from the variant.
fn compute_filter(variant: &Variant, qual_filter: f64) -> &'static str {
    if let Some(call) = variant.calls.first() {
        if call.genotype.iter().all(|&g| g == -1) {
            return filter::NO_CALL;
        }
        if call.genotype.iter().all(|&g| g == 0) {
            return filter::REF;
        }
    }
    if variant.quality < qual_filter {
        return filter::LOW_QUAL;
    }
    filter::PASS
}

/// Apply prediction merge → genotype call → set GT/GQ/PL → filter → return final variant.
pub fn add_call_to_variant(
    mut variant: Variant,
    predictions: &[f64],
    sample_name: &str,
    opts: &PostprocessOptions,
) -> Variant {
    let n_alleles = variant.alternate_bases.len() + 1;
    let (idx, genotype) = most_likely_genotype_diploid(predictions, n_alleles);
    let (gq, qual) = compute_quals(predictions, idx, opts);
    let pls = compute_pl(predictions);

    if variant.calls.is_empty() {
        variant.calls.push(VariantCall::default());
    }
    let call = &mut variant.calls[0];
    call.call_set_name = sample_name.to_string();
    call.genotype = genotype.to_vec();

    // GQ as int
    call.info.insert(
        "GQ".to_string(),
        ListValue {
            values: vec![Value {
                kind: Some(value::Kind::IntValue(gq)),
            }],
        },
    );
    // Genotype likelihoods
    call.genotype_likelihood = predictions
        .iter()
        .map(|&p| math::perror_to_bounded_log10_perror(p))
        .collect();
    // PL as ListValue ints (used by VCF writer)
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

    variant.quality = qual;

    // uncall_gt_if_no_ad
    if ad_sum(&variant.calls[0]) == 0 {
        let c = &mut variant.calls[0];
        c.genotype = vec![-1, -1];
        c.genotype_likelihood = vec![0.0, 0.0];
        c.info.insert(
            "GQ".to_string(),
            ListValue {
                values: vec![Value {
                    kind: Some(value::Kind::IntValue(0i32)),
                }],
            },
        );
    }

    let f = compute_filter(&variant, opts.qual_filter);
    variant.filter = vec![f.to_string()];

    uncall_homref_if_lowqual(&mut variant, opts.cnn_homref_call_min_gq);
    variant
}

/// Resolve CVOs by model: when both small-model and large-model CVOs exist for
/// the same locus, choose based on small-model GQ vs threshold (matches
/// `resolve_call_variant_outputs_by_model` for default behavior of preferring
/// large-model when the small-model GQ is below threshold).
///
/// For the default flow used by chr20 quickstart we just keep all CVOs and let
/// merge handle them. Real upstream behavior depends on the
/// `--resolve_call_variants_outputs_by_model` flag (default false).
pub fn process_cvos_into_variants(
    cvos: Vec<CallVariantsOutput>,
    sample_name: &str,
    opts: &PostprocessOptions,
) -> Vec<Variant> {
    let mut sorted = cvos;
    sort_cvos(&mut sorted);
    let groups = group_cvos(sorted);
    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        // Sort within-group by sorted alt_allele_indices so multi-allelic
        // merge sees a deterministic order (matches `_sort_grouped_variants`).
        let mut group = group;
        group.sort_by(|a, b| {
            let mut ai = a
                .alt_allele_indices
                .as_ref()
                .map(|x| x.indices.clone())
                .unwrap_or_default();
            let mut bi = b
                .alt_allele_indices
                .as_ref()
                .map(|x| x.indices.clone())
                .unwrap_or_default();
            ai.sort();
            bi.sort();
            ai.cmp(&bi)
        });
        let (variant, predictions) = if group.len() == 1 {
            merge_predictions_single(&group)
        } else {
            merge_predictions_multi(&group, opts.multi_allelic_qual_filter, opts)
        };
        out.push(add_call_to_variant(
            variant,
            &predictions,
            sample_name,
            opts,
        ));
    }
    out
}

#[allow(dead_code)]
fn _hashmap_unused_warning_silencer() -> HashMap<String, String> {
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn most_likely_genotype_homref() {
        let (idx, gt) = most_likely_genotype_diploid(&[0.99, 0.005, 0.005], 2);
        assert_eq!(idx, 0);
        assert_eq!(gt, [0, 0]);
    }

    #[test]
    fn most_likely_genotype_het() {
        let (idx, gt) = most_likely_genotype_diploid(&[0.005, 0.99, 0.005], 2);
        assert_eq!(idx, 1);
        assert_eq!(gt, [0, 1]);
    }

    #[test]
    fn most_likely_genotype_homalt() {
        let (idx, gt) = most_likely_genotype_diploid(&[0.005, 0.005, 0.99], 2);
        assert_eq!(idx, 2);
        assert_eq!(gt, [1, 1]);
    }

    #[test]
    fn compute_quals_homalt() {
        let opts = PostprocessOptions::default();
        let (gq, qual) = compute_quals(&[0.001, 0.001, 0.998], 2, &opts);
        // GQ ≈ phred(0.998) ≈ 27; QUAL ≈ phred(0.002) ≈ 27
        assert!(gq >= 25 && gq <= 30, "gq={gq}");
        assert!(qual >= 25.0 && qual <= 30.0, "qual={qual}");
    }

    #[test]
    fn compute_pl_homref_dominant() {
        let pls = compute_pl(&[0.999, 0.0009, 0.0001]);
        assert_eq!(pls.len(), 3);
        // PL[0] should be smallest (most likely); others larger
        assert!(pls[0] < pls[1], "pls={pls:?}");
        assert!(pls[0] < pls[2], "pls={pls:?}");
    }

    #[test]
    fn genotype_index_for_3_alleles() {
        // n_alleles=3 (REF + 2 alts) → 6 genotypes: 0/0,0/1,1/1,0/2,1/2,2/2
        let (idx, gt) = most_likely_genotype_diploid(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3);
        assert_eq!(idx, 5);
        assert_eq!(gt, [2, 2]);
    }

    /// Two distinct candidate variants at the same `(ref_name, start, end)`
    /// but with **different** alt sets — e.g. one from the allele counter
    /// (`alts=["A","G"]`) and one from a realigner-assembled deletion
    /// (`alts=["G"]`) — must be processed as separate VCF records, not
    /// merged. Pre-fix, `group_cvos` keyed only on the range and
    /// `merge_predictions_multi` would silently empty the alt list, then
    /// `add_call_to_variant` panicked with `n_alleles must be >= 2`.
    /// Reproducer: full chr20 HG003 hit one such locus
    /// (chr20:35167420-35167422 ref="GN").
    #[test]
    fn group_cvos_splits_distinct_alt_sets_at_same_range() {
        use dv_proto::dv::call_variants_output::AltAlleleIndices;

        fn cvo(alts: &[&str], aai: &[i32], probs: &[f64]) -> CallVariantsOutput {
            CallVariantsOutput {
                variant: Some(Variant {
                    reference_name: "chr20".into(),
                    start: 35_167_420,
                    end: 35_167_422,
                    reference_bases: "GN".into(),
                    alternate_bases: alts.iter().map(|s| s.to_string()).collect(),
                    ..Default::default()
                }),
                alt_allele_indices: Some(AltAlleleIndices { indices: aai.to_vec() }),
                genotype_probabilities: probs.to_vec(),
                debug_info: None,
            }
        }
        // Realigner-discovered deletion (1 alt) at the same range as a
        // candidate-caller multi-allelic variant (2 alts). High-confidence
        // hom-alt probs so the multi-allelic qual filter keeps both alts.
        let cvos = vec![
            cvo(&["A", "G"], &[0], &[0.001, 0.001, 0.998]), // multi-allelic, alt 0
            cvo(&["A", "G"], &[1], &[0.001, 0.001, 0.998]), // multi-allelic, alt 1
            cvo(&["G"],      &[0], &[0.001, 0.001, 0.998]), // realigner deletion
        ];
        let opts = PostprocessOptions::default();
        // Pre-fix this would panic in add_call_to_variant
        // ("n_alleles must be >= 2").
        let variants = process_cvos_into_variants(cvos, "S1", &opts);
        // Two records: the multi-allelic ["A","G"] candidate and the
        // ["G"] deletion candidate, processed independently.
        assert_eq!(variants.len(), 2, "got {} variants", variants.len());
        let mut alt_sets: Vec<Vec<String>> =
            variants.iter().map(|v| v.alternate_bases.clone()).collect();
        alt_sets.sort();
        assert_eq!(
            alt_sets,
            vec![
                vec!["A".to_string(), "G".to_string()],
                vec!["G".to_string()],
            ],
            "expected the 2-alt and 1-alt candidates as separate records"
        );
        // No 0-alt record slipped through.
        for v in &variants {
            assert!(!v.alternate_bases.is_empty(), "0-alt record");
        }
    }
}

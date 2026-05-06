//! Methylation-aware phasing — port of
//! `deepvariant/methylation_aware_phasing.cc` (~483 LOC).
//!
//! Long-read sequencing (PacBio HiFi, ONT) carries methylation
//! base-modification calls that can disambiguate read phasing at
//! REF-only sites where direct phasing has nothing to lock onto.
//! Algorithm:
//!   1. Use direct-phasing's already-phased reads as the "anchor"
//!      haplotypes 1 and 2.
//!   2. Find methylated REF sites where hap1 vs hap2 read methylation
//!      distributions differ (Wilcoxon rank-sum p < 0.05) and pass
//!      coverage / mean-difference / stddev filters.
//!   3. For each unphased read, vote on hap1 vs hap2 by comparing its
//!      per-site methylation level to the haplotype-mean methylation;
//!      majority wins (≥3 votes required).
//!   4. Iterate until convergence or `max_iter`.
//!
//! We work with a small data type rather than the full
//! `DeepVariantCall_ReadSupport` proto so the same module is usable
//! from contexts that don't pass full proto messages around. The
//! call-site converter is straightforward (read_name +
//! methylation_level are both directly available on the proto).

use std::collections::HashMap;

/// One read's methylation evidence at one site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethylRead {
    pub read_name: String,
    /// Methylation level encoded 0..=255 (0 = unset/no methylation).
    pub methylation_level: u8,
}

/// One methylated REF site with the per-read evidence and an output
/// p-value (set by `identify_informative_sites_and_update_p_values`).
#[derive(Debug, Clone, PartialEq)]
pub struct MethylCall {
    pub position: i64,
    pub ref_reads: Vec<MethylRead>,
    pub methylation_p_value: f64,
}

const P_THRESHOLD: f64 = 0.05;
const RANK_SUM_VARIANCE_DENOMINATOR: f64 = 12.0;

/// Convert a 0..=255 methylation byte to a probability in [0.0, 1.0].
/// Returns -1.0 when the byte is 0 (treated as "no data" by upstream).
pub fn get_methylation_level_at_site(read: &MethylRead) -> f64 {
    if read.methylation_level == 0 {
        -1.0
    } else {
        read.methylation_level as f64 / 255.0
    }
}

/// Standard normal CDF via Abramowitz–Stegun 7.1.26 (max error 1.5e-7).
fn normal_cdf(x: f64) -> f64 {
    fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let y = 1.0
            - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-ax * ax).exp();
        sign * y
    }
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Two-sided Wilcoxon rank-sum (Mann–Whitney U) test on the two
/// methylation samples, with average ranks for ties. Returns -1.0 if
/// either sample is empty, otherwise an asymptotic p-value via the
/// normal approximation. Mirrors upstream exactly.
pub fn wilcoxon_rank_sum_test(hap1_methyl: &[f64], hap2_methyl: &[f64]) -> f64 {
    let n1 = hap1_methyl.len();
    let n2 = hap2_methyl.len();
    if n1 == 0 || n2 == 0 {
        return -1.0;
    }

    #[derive(Clone, Copy)]
    struct RV {
        value: f64,
        group: u8, // 0 = hap1, 1 = hap2
    }
    let mut combined: Vec<RV> = Vec::with_capacity(n1 + n2);
    for &v in hap1_methyl {
        combined.push(RV { value: v, group: 0 });
    }
    for &v in hap2_methyl {
        combined.push(RV { value: v, group: 1 });
    }
    combined.sort_by(|a, b| a.value.partial_cmp(&b.value).expect("NaN in methyl"));

    // Average ranks for ties (1-based ranks).
    let mut ranks = vec![0.0f64; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        while j + 1 < combined.len() && combined[j + 1].value == combined[i].value {
            j += 1;
        }
        let avg_rank = (i + j + 2) as f64 / 2.0;
        for k in i..=j {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }

    let mut rank_sum_1 = 0.0f64;
    for k in 0..combined.len() {
        if combined[k].group == 0 {
            rank_sum_1 += ranks[k];
        }
    }

    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let u1 = rank_sum_1 - n1f * (n1f + 1.0) / 2.0;
    let u2 = n1f * n2f - u1;
    let u = u1.min(u2);

    let mean_u = n1f * n2f / 2.0;
    let std_u = (n1f * n2f * (n1f + n2f + 1.0) / RANK_SUM_VARIANCE_DENOMINATOR).sqrt();
    let z = (u - mean_u) / std_u;
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Return indices of `reads` whose phase entry equals `target_phase`.
pub fn extract_reads_by_phase(
    reads: &[MethylRead],
    phases: &[i32],
    target_phase: i32,
) -> Vec<usize> {
    debug_assert_eq!(reads.len(), phases.len());
    reads
        .iter()
        .zip(phases.iter())
        .enumerate()
        .filter_map(|(i, (_, p))| if *p == target_phase { Some(i) } else { None })
        .collect()
}

/// Internal helper — same as `extract_reads_by_phase` but returns the
/// reads themselves (cloned). Useful for tests that want the
/// upstream-style "vector of pointers" view.
pub fn extract_reads_by_phase_owned(
    reads: &[MethylRead],
    phases: &[i32],
    target_phase: i32,
) -> Vec<MethylRead> {
    extract_reads_by_phase(reads, phases, target_phase)
        .into_iter()
        .map(|i| reads[i].clone())
        .collect()
}

/// Compose `read_name/read_number` for cross-referencing reads against
/// methylation site evidence. Mirrors upstream `ReadKeyForMethylationAwarePhasing`.
pub fn read_key_for_methylation_aware_phasing(fragment_name: &str, read_number: i32) -> String {
    format!("{}/{}", fragment_name, read_number)
}

/// Run the differential-methylation filter + Wilcoxon test on each
/// methylated REF call. Updates `call.methylation_p_value` in place.
/// Returns clones of the calls that pass the p<0.05 threshold.
///
/// Filters (matching upstream order):
///   * ≥2 hap1 reads AND ≥2 hap2 reads with valid methylation
///   * ≥6 total reads
///   * |mean(hap1) - mean(hap2)| ≥ 0.25
///   * stddev ≤ 0.2 on both haplotypes
///   * 0 ≤ p-value < 0.05
pub fn identify_informative_sites_and_update_p_values(
    hap1_names: &[String],
    hap2_names: &[String],
    calls: &mut [MethylCall],
) -> Vec<MethylCall> {
    use std::collections::HashSet;
    let hap1_set: HashSet<&str> = hap1_names.iter().map(|s| s.as_str()).collect();
    let hap2_set: HashSet<&str> = hap2_names.iter().map(|s| s.as_str()).collect();

    let mut informative = Vec::new();
    for call in calls.iter_mut() {
        let mut hap1_methyl: Vec<f64> = Vec::new();
        let mut hap2_methyl: Vec<f64> = Vec::new();
        for support in &call.ref_reads {
            let m = get_methylation_level_at_site(support);
            if m < 0.0 {
                continue;
            }
            if hap1_set.contains(support.read_name.as_str()) {
                hap1_methyl.push(m);
            } else if hap2_set.contains(support.read_name.as_str()) {
                hap2_methyl.push(m);
            }
        }

        if hap1_methyl.len() < 2 || hap2_methyl.len() < 2 {
            continue;
        }
        if hap1_methyl.len() + hap2_methyl.len() < 6 {
            continue;
        }
        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
        let h1m = mean(&hap1_methyl);
        let h2m = mean(&hap2_methyl);
        if (h1m - h2m).abs() < 0.25 {
            continue;
        }
        let stddev = |xs: &[f64], m: f64| -> f64 {
            let s: f64 = xs.iter().map(|v| (v - m).powi(2)).sum();
            (s / xs.len() as f64).sqrt()
        };
        if stddev(&hap1_methyl, h1m) > 0.2 || stddev(&hap2_methyl, h2m) > 0.2 {
            continue;
        }
        let p_value = wilcoxon_rank_sum_test(&hap1_methyl, &hap2_methyl);
        call.methylation_p_value = p_value;
        if p_value >= 0.0 && p_value < P_THRESHOLD {
            informative.push(call.clone());
        }
    }
    informative
}

/// Vote a single unphased read into hap1, hap2, or 0 (uncertain) by
/// comparing its per-site methylation level to the per-site average
/// methylation of hap1 / hap2 reads. Closer mean wins; ≥3 votes and
/// strict majority required.
pub fn haplotype_vote_with_methylation(
    unphased_read_name: &str,
    informative_calls: &[MethylCall],
    hap1_names: &[String],
    hap2_names: &[String],
) -> i32 {
    use std::collections::HashSet;
    let hap1_set: HashSet<&str> = hap1_names.iter().map(|s| s.as_str()).collect();
    let hap2_set: HashSet<&str> = hap2_names.iter().map(|s| s.as_str()).collect();

    let mut hap1_votes = 0i32;
    let mut hap2_votes = 0i32;

    for call in informative_calls {
        // Methylation level for the unphased read at this site.
        let read_methyl = call
            .ref_reads
            .iter()
            .find(|r| r.read_name == unphased_read_name)
            .map(get_methylation_level_at_site)
            .unwrap_or(-1.0);
        if read_methyl < 0.0 {
            continue;
        }

        let mut hap1_map: HashMap<String, f64> = HashMap::new();
        let mut hap2_map: HashMap<String, f64> = HashMap::new();
        for support in &call.ref_reads {
            let m = get_methylation_level_at_site(support);
            if m < 0.0 {
                continue;
            }
            if hap1_set.contains(support.read_name.as_str()) {
                hap1_map.insert(support.read_name.clone(), m);
            } else if hap2_set.contains(support.read_name.as_str()) {
                hap2_map.insert(support.read_name.clone(), m);
            }
        }
        if hap1_map.is_empty() || hap2_map.is_empty() {
            continue;
        }
        let mean = |m: &HashMap<String, f64>| -> f64 {
            m.values().sum::<f64>() / m.len() as f64
        };
        let h1m = mean(&hap1_map);
        let h2m = mean(&hap2_map);
        if (read_methyl - h1m).abs() < (read_methyl - h2m).abs() {
            hap1_votes += 1;
        } else {
            hap2_votes += 1;
        }
    }

    if hap1_votes >= 3 && hap1_votes > hap2_votes {
        1
    } else if hap2_votes >= 3 && hap2_votes > hap1_votes {
        2
    } else {
        0
    }
}

/// Iterative methylation-aware phasing. Returns the updated phase
/// vector (same order as `reads_to_phase`) plus the final p-values for
/// each methylated REF site.
///
/// `reads_to_phase` carries (read_name) per read in the same order as
/// `initial_read_phases`. The methylation evidence comes from
/// `methylated_ref_sites`; reads not present at any site get treated
/// as having no methylation data.
pub fn perform_methylation_aware_phasing(
    reads_to_phase: &[String],
    initial_read_phases: &[i32],
    methylated_ref_sites: &mut [MethylCall],
    max_iter: usize,
) -> (Vec<i32>, Vec<f64>) {
    debug_assert_eq!(reads_to_phase.len(), initial_read_phases.len());

    // Build a name → methylation level map by aggregating over all sites.
    // (Upstream uses the *last* observation per read; we match that.)
    let mut read_support_map: HashMap<String, MethylRead> = HashMap::new();
    for call in methylated_ref_sites.iter() {
        for r in &call.ref_reads {
            read_support_map.insert(r.read_name.clone(), r.clone());
        }
    }
    let full_supports: Vec<MethylRead> = reads_to_phase
        .iter()
        .map(|key| {
            read_support_map.get(key).cloned().unwrap_or(MethylRead {
                read_name: key.clone(),
                methylation_level: 0,
            })
        })
        .collect();

    let mut current_phases = initial_read_phases.to_vec();

    for _iter in 0..max_iter {
        let hap1_names: Vec<String> = extract_reads_by_phase(&full_supports, &current_phases, 1)
            .into_iter()
            .map(|i| full_supports[i].read_name.clone())
            .collect();
        let hap2_names: Vec<String> = extract_reads_by_phase(&full_supports, &current_phases, 2)
            .into_iter()
            .map(|i| full_supports[i].read_name.clone())
            .collect();
        let num_phased = hap1_names.len() + hap2_names.len();
        let num_unphased = full_supports.len() - num_phased;
        if num_unphased == 0 {
            break;
        }

        let informative = identify_informative_sites_and_update_p_values(
            &hap1_names,
            &hap2_names,
            methylated_ref_sites,
        );

        let mut newly_phased = 0usize;
        for i in 0..full_supports.len() {
            if current_phases[i] != 0 {
                continue;
            }
            let vote = haplotype_vote_with_methylation(
                &full_supports[i].read_name,
                &informative,
                &hap1_names,
                &hap2_names,
            );
            if vote > 0 {
                current_phases[i] = vote;
                newly_phased += 1;
            }
        }
        if newly_phased == 0 {
            break;
        }
    }

    let p_values: Vec<f64> = methylated_ref_sites
        .iter()
        .map(|c| c.methylation_p_value)
        .collect();
    (current_phases, p_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(name: &str, level: u8) -> MethylRead {
        MethylRead {
            read_name: name.into(),
            methylation_level: level,
        }
    }

    /// Mirrors upstream `DistinctDistributionsIsDifferentiallyMethylated`.
    #[test]
    fn wilcoxon_distinct_distributions_significant() {
        let h1 = vec![0.10, 0.15, 0.20, 0.12, 0.18];
        let h2 = vec![0.75, 0.80, 0.85, 0.78, 0.82];
        let p = wilcoxon_rank_sum_test(&h1, &h2);
        assert!(p < 0.05, "expected p<0.05, got {}", p);
    }

    /// Mirrors upstream `IdenticalDistributionsIsDifferentiallyMethylated`.
    #[test]
    fn wilcoxon_identical_distributions_not_significant() {
        let h1 = vec![0.35, 0.40, 0.45, 0.50, 0.42];
        let h2 = vec![0.35, 0.40, 0.45, 0.50, 0.42];
        let p = wilcoxon_rank_sum_test(&h1, &h2);
        assert!(p > 0.05, "expected p>0.05, got {}", p);
    }

    /// Mirrors upstream `EmptyHaplotypesIsNotDifferentiallyMethylated`.
    #[test]
    fn wilcoxon_empty_returns_minus_one() {
        let empty: Vec<f64> = vec![];
        assert_eq!(wilcoxon_rank_sum_test(&empty, &empty), -1.0);
        assert_eq!(wilcoxon_rank_sum_test(&empty, &[0.2, 0.4, 0.6]), -1.0);
    }

    /// Mirrors upstream `WilcoxonRankSumSortOrderMatters`.
    #[test]
    fn wilcoxon_clear_separation_significant() {
        let h1 = vec![0.9, 0.85, 0.88, 0.95, 0.92];
        let h2 = vec![0.1, 0.12, 0.15, 0.05, 0.09];
        let p = wilcoxon_rank_sum_test(&h1, &h2);
        assert!(p < 0.01, "expected p<0.01, got {}", p);
    }

    /// Mirrors upstream `WilcoxonRankSumGroupAssignmentMatters`.
    #[test]
    fn wilcoxon_group_assignment_3v3_significant() {
        let h1 = vec![0.1, 0.2, 0.3];
        let h2 = vec![0.8, 0.9, 1.0];
        let p = wilcoxon_rank_sum_test(&h1, &h2);
        assert!(p < 0.05, "expected p<0.05, got {}", p);
    }

    /// Mirrors upstream `GetMethylationLevelAtSiteReturnsNormalized`.
    #[test]
    fn get_methylation_level_normalized() {
        let r = read("x", 128);
        let v = get_methylation_level_at_site(&r);
        assert!((v - 128.0 / 255.0).abs() < 1e-6);
    }

    /// Mirrors upstream `GetMethylationLevelAtSiteReturnsMinusOne`.
    #[test]
    fn get_methylation_level_unset_minus_one() {
        let r = read("x", 0);
        assert_eq!(get_methylation_level_at_site(&r), -1.0);
    }

    /// Mirrors upstream `ReadKeyMatchesExpectedFormat`.
    #[test]
    fn read_key_format() {
        assert_eq!(
            read_key_for_methylation_aware_phasing("frag", 1),
            "frag/1"
        );
    }

    /// Mirrors upstream `ExtractReadsByPhaseReturnsCorrectSubset`.
    #[test]
    fn extract_reads_by_phase_returns_correct_subset() {
        let reads = vec![read("a", 0), read("b", 0), read("c", 0)];
        let phases = vec![0, 2, 2];
        let idxs = extract_reads_by_phase(&reads, &phases, 2);
        assert_eq!(idxs, vec![1, 2]);
        let owned = extract_reads_by_phase_owned(&reads, &phases, 2);
        assert_eq!(
            owned.iter().map(|r| r.read_name.as_str()).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
    }

    fn make_methyl_call(
        pos: i64,
        n_hap1: usize,
        n_hap2: usize,
        hap1_methyl: u8,
        hap2_methyl: u8,
    ) -> MethylCall {
        let mut reads = Vec::new();
        for i in 0..n_hap1 {
            reads.push(read(&format!("hap1_{}", i), hap1_methyl));
        }
        for i in 0..n_hap2 {
            reads.push(read(&format!("hap2_{}", i), hap2_methyl));
        }
        MethylCall {
            position: pos,
            ref_reads: reads,
            methylation_p_value: 0.0,
        }
    }

    /// Mirrors upstream `IdentifyInformativeSitesFiltersCorrectly`.
    #[test]
    fn identify_informative_sites_filters_correctly() {
        // Build hap1/hap2 names (we use the same naming convention as
        // the upstream test fixture).
        let mut hap1_names: Vec<String> = Vec::new();
        let mut hap2_names: Vec<String> = Vec::new();

        let informative_call = make_methyl_call(100, 3, 3, 25, 230);
        let low_coverage = make_methyl_call(200, 1, 1, 25, 230);
        let low_total = make_methyl_call(300, 2, 2, 25, 230);
        let low_mean_diff = make_methyl_call(200, 3, 3, 125, 130);
        let mut high_stddev = make_methyl_call(400, 3, 3, 10, 200);
        // The upstream test mutates each call's read_infos directly to
        // populate hap1/hap2_reads. We mirror the side-effect by adding
        // the read_names at the end so all five sites' reads land in
        // both hap1_names / hap2_names sets together.
        for call in [
            &informative_call,
            &low_coverage,
            &low_total,
            &low_mean_diff,
            &high_stddev,
        ] {
            for r in &call.ref_reads {
                if r.read_name.starts_with("hap1") {
                    hap1_names.push(r.read_name.clone());
                } else {
                    hap2_names.push(r.read_name.clone());
                }
            }
        }
        // Add the extra "hap1_3" read with methyl=250 (to spike stddev).
        high_stddev.ref_reads.push(read("hap1_3", 250));
        hap1_names.push("hap1_3".into());

        let mut calls = vec![
            informative_call,
            low_coverage,
            low_total,
            low_mean_diff,
            high_stddev,
        ];
        let informative =
            identify_informative_sites_and_update_p_values(&hap1_names, &hap2_names, &mut calls);
        assert_eq!(informative.len(), 1);
        assert_eq!(informative[0].position, 100);
        // upstream expects 0.049534 ± 1e-6 — our normal-CDF approx is
        // ~1e-7 accurate so we should agree to ~1e-5.
        assert!(
            (informative[0].methylation_p_value - 0.049534).abs() < 1e-4,
            "got p={}",
            informative[0].methylation_p_value
        );
    }

    /// Mirrors upstream `HaplotypeVoteWithMethylationVotesCorrectly`.
    #[test]
    fn haplotype_vote_with_methylation_votes_correctly() {
        let mut informative_calls = Vec::new();
        for i in 0..3 {
            let call = MethylCall {
                position: 1000 + i,
                ref_reads: vec![
                    read("hap1", 25),     // ~0.1
                    read("hap2", 230),    // ~0.9
                    read("unphased", 240),// closer to hap2
                ],
                methylation_p_value: 0.0,
            };
            informative_calls.push(call);
        }
        let hap1_names = vec!["hap1".to_string()];
        let hap2_names = vec!["hap2".to_string()];
        let vote = haplotype_vote_with_methylation(
            "unphased",
            &informative_calls,
            &hap1_names,
            &hap2_names,
        );
        assert_eq!(vote, 2);
    }
}

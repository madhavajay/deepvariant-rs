//! Port of the most-used pieces of `third_party/nucleus/util/variant_utils.py`.
//!
//! Upstream is ~1000 LOC; this is the subset relevant to candidate
//! generation, postprocess, and the test suite.

use dv_proto::nucleus_v1::Variant;

/// Half-open `[start, end)` range tuple.
pub fn variant_range(v: &Variant) -> (&str, i64, i64) {
    (v.reference_name.as_str(), v.start, v.end)
}

/// Variant has at least one alt of length != ref length.
pub fn is_indel(v: &Variant) -> bool {
    let r = v.reference_bases.len();
    v.alternate_bases.iter().any(|a| a.len() != r)
}

pub fn is_snp(v: &Variant) -> bool {
    v.reference_bases.len() == 1
        && !v.alternate_bases.is_empty()
        && v.alternate_bases.iter().all(|a| a.len() == 1)
}

pub fn is_biallelic(v: &Variant) -> bool {
    v.alternate_bases.len() == 1
}

pub fn is_multiallelic(v: &Variant) -> bool {
    v.alternate_bases.len() > 1
}

pub fn is_insertion(reference: &str, alt: &str) -> bool {
    alt.len() > reference.len()
}

pub fn is_deletion(reference: &str, alt: &str) -> bool {
    alt.len() < reference.len()
}

pub fn has_insertion(v: &Variant) -> bool {
    v.alternate_bases.iter().any(|a| is_insertion(&v.reference_bases, a))
}

pub fn has_deletion(v: &Variant) -> bool {
    v.alternate_bases.iter().any(|a| is_deletion(&v.reference_bases, a))
}

/// Two variants overlap iff they share a contig and their ranges intersect.
pub fn variants_overlap(a: &Variant, b: &Variant) -> bool {
    a.reference_name == b.reference_name && a.start < b.end && b.start < a.end
}

/// Strip a common prefix and a non-mandatory common suffix to get the
/// minimal-length representation.
/// Mirrors `simplify_alleles` for ref + a single alt.
pub fn simplify_alleles(reference: &str, alt: &str) -> (String, String) {
    let r_bytes = reference.as_bytes();
    let a_bytes = alt.as_bytes();
    // Strip common suffix while both are > 1 char.
    let mut r_end = r_bytes.len();
    let mut a_end = a_bytes.len();
    while r_end > 1 && a_end > 1 && r_bytes[r_end - 1] == a_bytes[a_end - 1] {
        r_end -= 1;
        a_end -= 1;
    }
    // Strip common prefix while both are > 1 char (must keep an anchor base).
    let mut r_start = 0usize;
    let mut a_start = 0usize;
    while r_start + 1 < r_end && a_start + 1 < a_end && r_bytes[r_start] == a_bytes[a_start] {
        r_start += 1;
        a_start += 1;
    }
    (
        std::str::from_utf8(&r_bytes[r_start..r_end]).unwrap().to_string(),
        std::str::from_utf8(&a_bytes[a_start..a_end]).unwrap().to_string(),
    )
}

/// Number of diploid genotypes for `n_alts` alts: `(n+1)*(n+2)/2 - 1`?
/// No — for diploid: `(n+1) choose 2 + (n+1) = (n+1)(n+2)/2` genotypes
/// where n = n_alts (REF counted as allele 0).
pub fn num_diploid_genotypes(n_alts: usize) -> usize {
    let n = n_alts + 1; // total alleles
    n * (n + 1) / 2
}

/// Return the canonical genotype ordering as `(idx, h2_str, h1_str)` tuples
/// per VCF spec. For ploidy=2, alleles indexed 0..=n_alts (0 = REF).
pub fn genotype_ordering_in_likelihoods(v: &Variant) -> Vec<(usize, &str, &str)> {
    let alleles: Vec<&str> = std::iter::once(v.reference_bases.as_str())
        .chain(v.alternate_bases.iter().map(|s| s.as_str()))
        .collect();
    let mut out = Vec::with_capacity(num_diploid_genotypes(v.alternate_bases.len()));
    let mut idx = 0usize;
    for h1 in 0..alleles.len() {
        for h2 in 0..=h1 {
            out.push((idx, alleles[h2], alleles[h1]));
            idx += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(reference: &str, alts: &[&str]) -> Variant {
        Variant {
            reference_name: "chr1".into(),
            start: 100,
            end: 100 + reference.len() as i64,
            reference_bases: reference.into(),
            alternate_bases: alts.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn is_snp_examples() {
        assert!(is_snp(&v("A", &["T"])));
        assert!(is_snp(&v("A", &["T", "C"])));
        assert!(!is_snp(&v("AT", &["A"])));
        assert!(!is_snp(&v("A", &["AT"])));
    }

    #[test]
    fn is_indel_examples() {
        assert!(is_indel(&v("AT", &["A"])));
        assert!(is_indel(&v("A", &["AT"])));
        assert!(!is_indel(&v("A", &["T"])));
    }

    #[test]
    fn biallelic_vs_multiallelic() {
        assert!(is_biallelic(&v("A", &["T"])));
        assert!(!is_biallelic(&v("A", &["T", "C"])));
        assert!(is_multiallelic(&v("A", &["T", "C"])));
    }

    #[test]
    fn insertion_deletion_predicates() {
        assert!(is_insertion("A", "AT"));
        assert!(is_deletion("AT", "A"));
        assert!(!is_insertion("A", "T"));
        assert!(has_insertion(&v("A", &["AT"])));
        assert!(has_deletion(&v("AT", &["A"])));
        assert!(!has_insertion(&v("A", &["T"])));
    }

    #[test]
    fn overlap_examples() {
        let a = v("A", &["T"]);
        let mut b = v("C", &["G"]);
        b.start = 100;
        b.end = 101;
        assert!(variants_overlap(&a, &b));
        b.start = 200;
        b.end = 210;
        assert!(!variants_overlap(&a, &b));
    }

    #[test]
    fn simplify_strips_common_suffix() {
        assert_eq!(simplify_alleles("ATG", "AG"), ("AT".into(), "A".into()));
        assert_eq!(simplify_alleles("AT", "ACT"), ("A".into(), "AC".into()));
        assert_eq!(simplify_alleles("A", "T"), ("A".into(), "T".into())); // SNV unchanged
    }

    #[test]
    fn num_diploid_genotypes_examples() {
        assert_eq!(num_diploid_genotypes(0), 1); // REF only — 0/0
        assert_eq!(num_diploid_genotypes(1), 3); // 0/0, 0/1, 1/1
        assert_eq!(num_diploid_genotypes(2), 6); // 0/0,0/1,1/1,0/2,1/2,2/2
        assert_eq!(num_diploid_genotypes(3), 10);
    }

    #[test]
    fn genotype_ordering_for_2_alts() {
        let var = v("A", &["T", "C"]);
        let ord = genotype_ordering_in_likelihoods(&var);
        // VCF spec: 0/0, 0/1, 1/1, 0/2, 1/2, 2/2
        let want = [("A", "A"), ("A", "T"), ("T", "T"), ("A", "C"), ("T", "C"), ("C", "C")];
        for (i, (h2, h1)) in want.iter().enumerate() {
            assert_eq!((ord[i].1, ord[i].2), (*h2, *h1), "i={i}");
        }
    }
}

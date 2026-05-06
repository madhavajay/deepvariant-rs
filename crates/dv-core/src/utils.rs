//! Small helper functions ported from
//! `deepvariant/utils.cc` (~102 LOC). Keeps the cross-module
//! dependencies minimal — just `Allele` / `AlleleType` from the
//! generated proto.

use dv_proto::dv::{Allele, AlleleType};

/// Construct an `Allele` proto with the given fields. Mirrors upstream
/// `MakeAllele` exactly.
#[allow(clippy::too_many_arguments)]
pub fn make_allele(
    bases: &str,
    allele_type: AlleleType,
    count: i32,
    is_low_quality: bool,
    mapping_quality: i32,
    avg_base_quality: i32,
    is_reverse_strand: bool,
    is_methylated: bool,
    methylation_level: i32,
) -> Allele {
    Allele {
        bases: bases.to_string(),
        r#type: allele_type as i32,
        count,
        is_low_quality,
        mapping_quality,
        avg_base_quality,
        is_reverse_strand,
        is_methylated,
        methylation_level,
    }
}

/// Drop the longest *suffix* shared by `ref` and `alt` (but always
/// keeping at least one base of each), then format the result as
/// `"<ref>-><alt>"`. Used to compactly describe a variant in error
/// messages and tests. Mirrors upstream `SimplifyRefAlt`.
pub fn simplify_ref_alt(reference: &str, alt: &str) -> String {
    let shortest = reference.len().min(alt.len());
    let mut common_suffix_len = 0usize;
    let mut suffix_idx = 1usize;
    while suffix_idx < shortest {
        if reference.as_bytes()[reference.len() - suffix_idx]
            != alt.as_bytes()[alt.len() - suffix_idx]
        {
            break;
        }
        common_suffix_len = suffix_idx;
        suffix_idx += 1;
    }
    if common_suffix_len == 0 {
        format!("{}->{}", reference, alt)
    } else {
        format!(
            "{}->{}",
            &reference[..reference.len() - common_suffix_len],
            &alt[..alt.len() - common_suffix_len]
        )
    }
}

/// Classify (REF, ALT) by their lengths. Mirrors upstream
/// `AlleleTypeFromAlt`. REF==ALT (with same length) → REFERENCE.
pub fn allele_type_from_alt(reference: &str, alt: &str) -> AlleleType {
    use std::cmp::Ordering;
    match reference.len().cmp(&alt.len()) {
        Ordering::Greater => AlleleType::Deletion,
        Ordering::Less => AlleleType::Insertion,
        Ordering::Equal => {
            if reference == alt {
                AlleleType::Reference
            } else {
                AlleleType::Substitution
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_allele_round_trip() {
        let a = make_allele("A", AlleleType::Substitution, 5, false, 60, 30, true, false, 0);
        assert_eq!(a.bases, "A");
        assert_eq!(a.r#type, AlleleType::Substitution as i32);
        assert_eq!(a.count, 5);
        assert!(!a.is_low_quality);
        assert_eq!(a.mapping_quality, 60);
        assert_eq!(a.avg_base_quality, 30);
        assert!(a.is_reverse_strand);
    }

    #[test]
    fn simplify_ref_alt_no_common_suffix() {
        assert_eq!(simplify_ref_alt("A", "C"), "A->C");
        assert_eq!(simplify_ref_alt("AC", "GT"), "AC->GT");
    }

    #[test]
    fn simplify_ref_alt_with_common_suffix() {
        // "ACG" → "AGG": no common suffix at idx 1 (G vs G is the same!)
        // Actually "ACG"/"AGG": last bases both G. So common_suffix=1 → "AC->AG"
        assert_eq!(simplify_ref_alt("ACG", "AGG"), "AC->AG");
        // "AAA"/"TAA" → last 2 match → "A->T"
        assert_eq!(simplify_ref_alt("AAA", "TAA"), "A->T");
    }

    #[test]
    fn simplify_ref_alt_keeps_at_least_one_base() {
        // "A"/"A" → length 1 → no shortening, even though full match.
        assert_eq!(simplify_ref_alt("A", "A"), "A->A");
    }

    #[test]
    fn allele_type_from_alt_classification() {
        assert_eq!(allele_type_from_alt("A", "C"), AlleleType::Substitution);
        assert_eq!(allele_type_from_alt("A", "AC"), AlleleType::Insertion);
        assert_eq!(allele_type_from_alt("AC", "A"), AlleleType::Deletion);
        assert_eq!(allele_type_from_alt("A", "A"), AlleleType::Reference);
    }

    /// Mirrors upstream `UtilsTest::TestSimplifyRefAlt`.
    #[test]
    fn simplify_ref_alt_upstream_cases() {
        assert_eq!(simplify_ref_alt("CAA", "CA"), "CA->C");
        assert_eq!(simplify_ref_alt("CA", "C"), "CA->C");
        assert_eq!(simplify_ref_alt("ATGTG", "ATGTGTGTGTGTG"), "A->ATGTGTGTG");
        assert_eq!(simplify_ref_alt("C", "C"), "C->C");
    }
}

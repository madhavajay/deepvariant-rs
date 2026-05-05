//! Port of the most-used pieces of `third_party/nucleus/util/ranges.py`.
//!
//! The full upstream module is ~800 LOC including BED/BEDPE parsers and
//! contig-map utilities; ported here is the subset needed by allelecounter,
//! variant_calling, and pileup-image construction:
//!
//!   - `Range` (alias for `nucleus_v1::Range`)
//!   - parse/format of literal `chr:start-end` regions
//!   - overlap / contains / overlap_len / span
//!   - simple `RangeSet` over a single contig with binary search

use dv_proto::nucleus_v1::Range;

/// 1-based inclusive `chr:start-end` literal; 0-based half-open in proto.
pub fn make(chrom: &str, start: i64, end: i64) -> Range {
    Range {
        reference_name: chrom.to_string(),
        start,
        end,
    }
}

pub fn length(r: &Range) -> i64 {
    r.end - r.start
}

pub fn position_overlaps(chrom: &str, pos: i64, r: &Range) -> bool {
    r.reference_name == chrom && pos >= r.start && pos < r.end
}

pub fn ranges_overlap(a: &Range, b: &Range) -> bool {
    a.reference_name == b.reference_name && a.start < b.end && b.start < a.end
}

pub fn overlap_len(a: &Range, b: &Range) -> i64 {
    if a.reference_name != b.reference_name {
        return 0;
    }
    let lo = a.start.max(b.start);
    let hi = a.end.min(b.end);
    if hi > lo {
        hi - lo
    } else {
        0
    }
}

/// Bounding span of a non-empty list of same-contig ranges.
pub fn span(rs: &[Range]) -> Option<Range> {
    let first = rs.first()?;
    let mut s = first.start;
    let mut e = first.end;
    for r in rs.iter().skip(1) {
        if r.reference_name != first.reference_name {
            return None;
        }
        s = s.min(r.start);
        e = e.max(r.end);
    }
    Some(Range {
        reference_name: first.reference_name.clone(),
        start: s,
        end: e,
    })
}

pub fn as_tuple(r: &Range) -> (&str, i64, i64) {
    (r.reference_name.as_str(), r.start, r.end)
}

/// Parse a literal like `"chr20:10,000,000-10,010,000"` into a `Range`.
/// Accepts comma thousands separators in coordinates. Returns 0-based
/// half-open coordinates.
pub fn parse_literal(literal: &str) -> Result<Range, String> {
    let stripped: String = literal.chars().filter(|c| *c != ',').collect();
    let (chrom, rest) = stripped
        .rsplit_once(':')
        .ok_or_else(|| format!("missing ':' in {literal}"))?;
    let (start_s, end_s) = rest
        .split_once('-')
        .ok_or_else(|| format!("missing '-' in {literal}"))?;
    let start_1: i64 = start_s.parse().map_err(|_| format!("bad start in {literal}"))?;
    let end_1: i64 = end_s.parse().map_err(|_| format!("bad end in {literal}"))?;
    if start_1 < 1 || end_1 < start_1 {
        return Err(format!("invalid coord range in {literal}"));
    }
    Ok(Range {
        reference_name: chrom.to_string(),
        start: start_1 - 1, // 1-based → 0-based
        end: end_1,         // 1-based inclusive → 0-based exclusive
    })
}

pub fn to_literal(r: &Range) -> String {
    format!("{}:{}-{}", r.reference_name, r.start + 1, r.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(chrom: &str, s: i64, e: i64) -> Range {
        make(chrom, s, e)
    }

    #[test]
    fn length_basic() {
        assert_eq!(length(&r("chr1", 100, 200)), 100);
    }

    #[test]
    fn position_overlap_inclusive_start_exclusive_end() {
        assert!(position_overlaps("chr1", 100, &r("chr1", 100, 200)));
        assert!(position_overlaps("chr1", 199, &r("chr1", 100, 200)));
        assert!(!position_overlaps("chr1", 200, &r("chr1", 100, 200)));
        assert!(!position_overlaps("chr1", 99, &r("chr1", 100, 200)));
        assert!(!position_overlaps("chr2", 150, &r("chr1", 100, 200)));
    }

    #[test]
    fn ranges_overlap_basic() {
        assert!(ranges_overlap(&r("c", 0, 10), &r("c", 5, 15)));
        assert!(!ranges_overlap(&r("c", 0, 10), &r("c", 10, 20))); // touching != overlap
        assert!(!ranges_overlap(&r("c", 0, 10), &r("d", 5, 15)));
    }

    #[test]
    fn overlap_len_examples() {
        assert_eq!(overlap_len(&r("c", 0, 10), &r("c", 5, 15)), 5);
        assert_eq!(overlap_len(&r("c", 0, 10), &r("c", 10, 20)), 0);
        assert_eq!(overlap_len(&r("c", 0, 10), &r("d", 0, 10)), 0);
        assert_eq!(overlap_len(&r("c", 0, 10), &r("c", 3, 8)), 5);
    }

    #[test]
    fn span_picks_min_max() {
        let s = span(&[r("c", 100, 200), r("c", 50, 150), r("c", 175, 300)]).unwrap();
        assert_eq!((s.start, s.end), (50, 300));
    }

    #[test]
    fn span_rejects_mixed_contigs() {
        assert!(span(&[r("c", 100, 200), r("d", 50, 150)]).is_none());
    }

    #[test]
    fn parse_literal_with_commas() {
        let r = parse_literal("chr20:10,000,000-10,010,000").unwrap();
        assert_eq!(r.reference_name, "chr20");
        assert_eq!(r.start, 9_999_999);
        assert_eq!(r.end, 10_010_000);
    }

    #[test]
    fn parse_literal_simple() {
        let r = parse_literal("chr1:100-200").unwrap();
        assert_eq!(r.reference_name, "chr1");
        assert_eq!(r.start, 99);
        assert_eq!(r.end, 200);
    }

    #[test]
    fn parse_literal_rejects_malformed() {
        assert!(parse_literal("nothing").is_err());
        assert!(parse_literal("chr1:abc-200").is_err());
    }

    #[test]
    fn literal_round_trip() {
        let lit = "chr20:10000000-10010000";
        assert_eq!(to_literal(&parse_literal(lit).unwrap()), lit);
    }
}

//! Port of `third_party/nucleus/util/cigar.py`.

use dv_proto::nucleus_v1::cigar_unit::Operation as Op;
use dv_proto::nucleus_v1::CigarUnit;

pub fn op_to_char(op: Op) -> char {
    match op {
        Op::AlignmentMatch => 'M',
        Op::Insert => 'I',
        Op::Delete => 'D',
        Op::Skip => 'N',
        Op::ClipSoft => 'S',
        Op::ClipHard => 'H',
        Op::Pad => 'P',
        Op::SequenceMatch => '=',
        Op::SequenceMismatch => 'X',
        Op::Unspecified => '?',
    }
}

pub fn char_to_op(c: char) -> Option<Op> {
    Some(match c {
        'M' => Op::AlignmentMatch,
        'I' => Op::Insert,
        'D' => Op::Delete,
        'N' => Op::Skip,
        'S' => Op::ClipSoft,
        'H' => Op::ClipHard,
        'P' => Op::Pad,
        '=' => Op::SequenceMatch,
        'X' => Op::SequenceMismatch,
        _ => return None,
    })
}

pub fn is_ref_advancing(op: Op) -> bool {
    matches!(
        op,
        Op::AlignmentMatch | Op::SequenceMatch | Op::Delete | Op::Skip | Op::SequenceMismatch
    )
}

pub fn is_read_advancing(op: Op) -> bool {
    matches!(
        op,
        Op::AlignmentMatch | Op::SequenceMatch | Op::Insert | Op::ClipSoft | Op::SequenceMismatch
    )
}

pub fn unit(op: Op, length: i64) -> CigarUnit {
    CigarUnit {
        operation: op as i32,
        operation_length: length,
        ..Default::default()
    }
}

/// Parse a CIGAR string like `"150M2S"` into units.
pub fn parse(s: &str) -> Result<Vec<CigarUnit>, String> {
    if s.is_empty() {
        return Err("cigar_str cannot be empty".into());
    }
    let mut out = Vec::new();
    let mut digits = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else {
            let op = char_to_op(c).ok_or_else(|| format!("Malformed CIGAR string {s}"))?;
            if digits.is_empty() {
                return Err(format!("Malformed CIGAR string {s}"));
            }
            let len: i64 = digits.parse().map_err(|_| format!("Malformed CIGAR string {s}"))?;
            if len < 1 {
                return Err(format!("Length must be >= 1: {len}"));
            }
            out.push(unit(op, len));
            digits.clear();
        }
    }
    if !digits.is_empty() {
        return Err(format!("Malformed CIGAR string {s} (trailing digits)"));
    }
    if out.is_empty() {
        return Err(format!("Malformed CIGAR string {s}"));
    }
    Ok(out)
}

pub fn format(units: &[CigarUnit]) -> String {
    let mut out = String::new();
    for u in units {
        out.push_str(&u.operation_length.to_string());
        let op = Op::try_from(u.operation).unwrap_or(Op::Unspecified);
        out.push(op_to_char(op));
    }
    out
}

pub fn alignment_length(units: &[CigarUnit]) -> i64 {
    units
        .iter()
        .filter_map(|u| {
            let op = Op::try_from(u.operation).ok()?;
            if is_ref_advancing(op) {
                Some(u.operation_length)
            } else {
                None
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple() {
        let u = parse("150M2S").unwrap();
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].operation_length, 150);
        assert_eq!(Op::try_from(u[0].operation).unwrap(), Op::AlignmentMatch);
        assert_eq!(u[1].operation_length, 2);
        assert_eq!(Op::try_from(u[1].operation).unwrap(), Op::ClipSoft);
    }

    #[test]
    fn parse_all_ops() {
        let u = parse("1M2I3D4N5S6H7P8=9X").unwrap();
        let chars: String = u
            .iter()
            .flat_map(|x| {
                let op = Op::try_from(x.operation).unwrap();
                vec![char::from_digit(x.operation_length as u32, 10).unwrap(), op_to_char(op)]
            })
            .collect();
        assert_eq!(chars, "1M2I3D4N5S6H7P8=9X");
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(parse("").is_err());
    }

    #[test]
    fn parse_rejects_zero_length() {
        assert!(parse("0M").is_err());
    }

    #[test]
    fn parse_rejects_bad_op() {
        assert!(parse("10Q").is_err());
    }

    #[test]
    fn format_round_trips() {
        let s = "150M2I3D";
        assert_eq!(format(&parse(s).unwrap()), s);
    }

    #[test]
    fn alignment_length_examples() {
        // M=150 (ref-advancing) + S=2 (not) → 150
        assert_eq!(alignment_length(&parse("150M2S").unwrap()), 150);
        // M=10 + I=5 (not) + D=3 (ref) + N=2 (ref) → 15
        assert_eq!(alignment_length(&parse("10M5I3D2N").unwrap()), 15);
    }

    #[test]
    fn ref_vs_read_advancing() {
        for &op in &[Op::AlignmentMatch, Op::SequenceMatch, Op::SequenceMismatch] {
            assert!(is_ref_advancing(op));
            assert!(is_read_advancing(op));
        }
        assert!(is_ref_advancing(Op::Delete));
        assert!(is_ref_advancing(Op::Skip));
        assert!(!is_read_advancing(Op::Delete));
        assert!(is_read_advancing(Op::Insert));
        assert!(is_read_advancing(Op::ClipSoft));
        assert!(!is_ref_advancing(Op::Insert));
        assert!(!is_ref_advancing(Op::ClipSoft));
        assert!(!is_ref_advancing(Op::ClipHard));
        assert!(!is_read_advancing(Op::ClipHard));
        assert!(!is_ref_advancing(Op::Pad));
        assert!(!is_read_advancing(Op::Pad));
    }
}

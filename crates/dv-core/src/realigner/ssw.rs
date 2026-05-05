//! Pairwise local alignment via classical Smith-Waterman with affine gap
//! penalty. The upstream port uses libssw's striped SIMD variant for
//! speed; this is the equivalent classical implementation, producing the
//! same alignments but slower.
//!
//! References:
//!   Smith TF, Waterman MS. Identification of common molecular
//!   subsequences. J Mol Biol. 1981;147(1):195-7.
//!   Gotoh O. An improved algorithm for matching biological sequences.
//!   J Mol Biol. 1982;162(3):705-8.

#[derive(Debug, Clone, Copy)]
pub struct ScoreParams {
    pub match_score: i32,
    pub mismatch_penalty: i32, // positive value
    pub gap_open_penalty: i32, // positive
    pub gap_extend_penalty: i32, // positive
}

impl Default for ScoreParams {
    fn default() -> Self {
        // Defaults align with deepvariant/realigner/ssw.h.
        Self {
            match_score: 4,
            mismatch_penalty: 6,
            gap_open_penalty: 8,
            gap_extend_penalty: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alignment {
    /// 0-based start index in the reference (target) sequence.
    pub ref_begin: usize,
    /// 0-based end index (exclusive) in the reference.
    pub ref_end: usize,
    /// 0-based start in query (read).
    pub query_begin: usize,
    /// 0-based end (exclusive) in query.
    pub query_end: usize,
    /// Best alignment score.
    pub score: i32,
    /// CIGAR-like trace as `(op, length)` ops covering the local region.
    /// Ops: 'M' = match/mismatch, 'I' = ins to ref, 'D' = del from ref.
    pub cigar: Vec<(char, usize)>,
}

#[derive(Clone, Copy)]
enum Trace {
    None,
    Match,
    Up,    // gap in query (deletion from ref)
    Left,  // gap in ref (insertion into ref)
}

/// Run Smith-Waterman with affine gap penalties. Returns `None` if the
/// best alignment score is 0 (no positive-score alignment found).
pub fn align(query: &[u8], reference: &[u8], p: ScoreParams) -> Option<Alignment> {
    let n = query.len();
    let m = reference.len();
    if n == 0 || m == 0 {
        return None;
    }

    // H = best score ending at (i, j); E = best score with last op gap-in-query;
    // F = best score with last op gap-in-ref. Affine penalties: opening costs
    // gap_open, each subsequent base costs gap_extend.
    let neg_inf = i32::MIN / 2;
    let mut h = vec![vec![0i32; m + 1]; n + 1];
    let mut e = vec![vec![neg_inf; m + 1]; n + 1];
    let mut f = vec![vec![neg_inf; m + 1]; n + 1];
    let mut tb = vec![vec![Trace::None; m + 1]; n + 1];

    let mut best = 0i32;
    let mut best_pos = (0usize, 0usize);

    for i in 1..=n {
        for j in 1..=m {
            // E[i][j]: best score with last op being 'D' (gap in query, advance ref)
            let e_open = h[i][j - 1] - (p.gap_open_penalty + p.gap_extend_penalty);
            let e_ext = e[i][j - 1] - p.gap_extend_penalty;
            e[i][j] = e_open.max(e_ext);

            // F[i][j]: best score with last op being 'I' (gap in ref, advance query)
            let f_open = h[i - 1][j] - (p.gap_open_penalty + p.gap_extend_penalty);
            let f_ext = f[i - 1][j] - p.gap_extend_penalty;
            f[i][j] = f_open.max(f_ext);

            // M = score(query[i-1] vs reference[j-1]) + H[i-1][j-1]
            let s = if query[i - 1].eq_ignore_ascii_case(&reference[j - 1]) {
                p.match_score
            } else {
                -p.mismatch_penalty
            };
            let m_score = h[i - 1][j - 1] + s;

            let mut best_h = 0;
            let mut tr = Trace::None;
            if m_score > best_h {
                best_h = m_score;
                tr = Trace::Match;
            }
            if e[i][j] > best_h {
                best_h = e[i][j];
                tr = Trace::Left;
            }
            if f[i][j] > best_h {
                best_h = f[i][j];
                tr = Trace::Up;
            }
            h[i][j] = best_h;
            tb[i][j] = tr;

            if best_h > best {
                best = best_h;
                best_pos = (i, j);
            }
        }
    }

    if best == 0 {
        return None;
    }

    // Trace back from best_pos until we hit a 0 score.
    let (mut i, mut j) = best_pos;
    let ref_end = j;
    let query_end = i;
    let mut cigar_rev: Vec<(char, usize)> = Vec::new();
    let push_op = |c: &mut Vec<(char, usize)>, op: char| match c.last_mut() {
        Some((last_op, len)) if *last_op == op => *len += 1,
        _ => c.push((op, 1)),
    };
    while i > 0 && j > 0 && h[i][j] > 0 {
        match tb[i][j] {
            Trace::Match => {
                push_op(&mut cigar_rev, 'M');
                i -= 1;
                j -= 1;
            }
            Trace::Left => {
                push_op(&mut cigar_rev, 'D');
                j -= 1;
            }
            Trace::Up => {
                push_op(&mut cigar_rev, 'I');
                i -= 1;
            }
            Trace::None => break,
        }
    }
    let ref_begin = j;
    let query_begin = i;
    cigar_rev.reverse();

    Some(Alignment {
        ref_begin,
        ref_end,
        query_begin,
        query_end,
        score: best,
        cigar: cigar_rev,
    })
}

/// Render a CIGAR-like trace as a string (e.g. "5M2I3M").
pub fn cigar_string(cig: &[(char, usize)]) -> String {
    let mut s = String::new();
    for (op, len) in cig {
        s.push_str(&len.to_string());
        s.push(*op);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn align_default(q: &[u8], r: &[u8]) -> Alignment {
        align(q, r, ScoreParams::default()).expect("alignment")
    }

    #[test]
    fn perfect_match() {
        let q = b"ACGTACGT";
        let r = b"ACGTACGT";
        let a = align_default(q, r);
        assert_eq!(a.cigar, vec![('M', 8)]);
        assert_eq!(a.score, 8 * 4);
        assert_eq!(a.ref_begin, 0);
        assert_eq!(a.ref_end, 8);
    }

    #[test]
    fn substring_match() {
        let q = b"GGGACGTACGTAAA";
        let r = b"ACGTACGT";
        let a = align_default(q, r);
        // Expect 8M aligned in middle of the query; ref [0, 8).
        assert_eq!(a.cigar, vec![('M', 8)]);
        assert_eq!(a.score, 8 * 4);
        assert_eq!(a.ref_begin, 0);
        assert_eq!(a.ref_end, 8);
        assert_eq!(a.query_begin, 3);
        assert_eq!(a.query_end, 11);
    }

    #[test]
    fn single_mismatch() {
        let q = b"ACGTACGT";
        let r = b"ACGTTCGT"; // mismatch at pos 4
        let a = align_default(q, r);
        // 4M on each side, with mismatch tolerated; total length 8 still matches as 8M.
        assert_eq!(cigar_string(&a.cigar), "8M");
        // Score = 7 matches * 4 + 1 mismatch * -6 = 22
        assert_eq!(a.score, 7 * 4 - 6);
    }

    #[test]
    fn insertion_in_query() {
        // Long flanking matches force the gap-paid path to win locally.
        // Query has TTTT inserted relative to reference.
        let q = b"ACGTACGTAAGGCCAATTTTCCAAGGCC";
        let r = b"ACGTACGTAAGGCCAACCAAGGCC";
        let a = align_default(q, r);
        let total_i: usize = a.cigar.iter().filter(|(op, _)| *op == 'I').map(|(_, l)| l).sum();
        assert!(
            total_i > 0,
            "expected insertion in cigar; got {}",
            cigar_string(&a.cigar)
        );
    }

    #[test]
    fn deletion_in_query() {
        // Reference has TTTT relative to query (a deletion in the read).
        let q = b"ACGTACGTAAGGCCAACCAAGGCC";
        let r = b"ACGTACGTAAGGCCAATTTTCCAAGGCC";
        let a = align_default(q, r);
        let total_d: usize = a.cigar.iter().filter(|(op, _)| *op == 'D').map(|(_, l)| l).sum();
        assert!(
            total_d > 0,
            "expected deletion in cigar; got {}",
            cigar_string(&a.cigar)
        );
    }

    #[test]
    fn returns_none_when_score_zero() {
        // Mismatch every base -> negative scores -> best is 0.
        let q = b"AAAA";
        let r = b"GGGG";
        let result = align(q, r, ScoreParams::default());
        assert!(result.is_none());
    }

    #[test]
    fn cigar_string_renders() {
        assert_eq!(cigar_string(&[('M', 5), ('I', 2), ('M', 3)]), "5M2I3M");
    }
}

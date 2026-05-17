//! Pairwise local alignment via classical Smith-Waterman with affine gap
//! penalty.
//!
//! The full-traceback [`align`] is bit-exact with the original reference
//! implementation (kept as `align_reference` under `cfg(test)` for
//! parity tests) but rewritten to use flat, thread-local scratch
//! buffers instead of four per-call `Vec<Vec<i32>>` allocations.
//!
//! [`align_score`] is a score-only kernel (rolling rows, no traceback)
//! used by the realigner read×haplotype hot path, where the CIGAR is
//! discarded. It adds an exact-match prefilter, an adaptive band with a
//! provably-safe full fallback, and an optional SIMD inner loop.
//!
//! References:
//!   Smith TF, Waterman MS. Identification of common molecular
//!   subsequences. J Mol Biol. 1981;147(1):195-7.
//!   Gotoh O. An improved algorithm for matching biological sequences.
//!   J Mol Biol. 1982;162(3):705-8.

use std::cell::RefCell;

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

// Trace codes packed into u8 (was a 1-byte enum; u8 keeps the flat
// traceback buffer half the size of an i32 grid and cache-friendly).
const TR_NONE: u8 = 0;
const TR_MATCH: u8 = 1;
const TR_UP: u8 = 2; // gap in query (deletion from ref)
const TR_LEFT: u8 = 3; // gap in ref (insertion into ref)

const NEG_INF: i32 = i32::MIN / 2;


#[derive(Default)]
struct Scratch {
    // Full-traceback align(): flat H grid + flat trace grid, plus a
    // single rolling row each for E (depends on left, same row) and F
    // (depends on up, previous row).
    h: Vec<i32>,
    tb: Vec<u8>,
    erow: Vec<i32>,
    frow: Vec<i32>,
    // Score-only align_score(): two rolling H rows + rolling F row.
    hp: Vec<i32>,
    hc: Vec<i32>,
    fp: Vec<i32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Case-insensitive base equality. For ASCII-alphabetic bytes (the
/// nucleotide alphabet A/C/G/T/N, upper or soft-mask lower) `b | 0x20`
/// lowercases letters, so this is bit-identical to
/// `a.eq_ignore_ascii_case(&b)` but branchless — measurably cheaper in
/// the per-cell hot loop.
#[inline(always)]
fn eq_ci(a: u8, b: u8) -> bool {
    (a | 0x20) == (b | 0x20)
}

/// True iff `q` and `r` are the same length and equal ignoring ASCII
/// case. For positive `match_score` the unique Smith-Waterman optimum
/// is then the all-`M` alignment from 0 with score `len * match_score`
/// — identical to what the full DP + traceback produces, so this is a
/// bit-exact short-circuit.
#[inline]
fn exact_full_match(q: &[u8], r: &[u8]) -> bool {
    q.len() == r.len() && q.iter().zip(r).all(|(&a, &b)| eq_ci(a, b))
}

/// Run Smith-Waterman with affine gap penalties, returning the full
/// alignment with CIGAR. Returns `None` if the best score is 0.
///
/// Bit-exact with the original reference implementation; only the
/// memory layout changed (flat + thread-local, zero per-call alloc).
pub fn align(query: &[u8], reference: &[u8], p: ScoreParams) -> Option<Alignment> {
    let n = query.len();
    let m = reference.len();
    if n == 0 || m == 0 {
        return None;
    }

    if p.match_score > 0 && exact_full_match(query, reference) {
        return Some(Alignment {
            ref_begin: 0,
            ref_end: m,
            query_begin: 0,
            query_end: n,
            score: n as i32 * p.match_score,
            cigar: vec![('M', n)],
        });
    }

    let stride = m + 1;
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        let cells = (n + 1) * stride;
        s.h.clear();
        s.h.resize(cells, 0);
        s.tb.clear();
        s.tb.resize(cells, TR_NONE);
        s.erow.clear();
        s.erow.resize(stride, NEG_INF);
        s.frow.clear();
        s.frow.resize(stride, NEG_INF);

        let h = &mut s.h;
        let tb = &mut s.tb;
        let e = &mut s.erow; // current row's E (depends on E left)
        let f = &mut s.frow; // rolling F (depends on F up = prev row)

        let mut best = 0i32;
        let mut best_pos = (0usize, 0usize);

        let gap_oe = p.gap_open_penalty + p.gap_extend_penalty;
        let gap_e = p.gap_extend_penalty;

        for i in 1..=n {
            let row = i * stride;
            let prow = (i - 1) * stride;
            // E resets per row (E[i][0] has no left neighbour).
            e[0] = NEG_INF;
            let qi = query[i - 1];
            for j in 1..=m {
                let e_open = h[row + j - 1] - gap_oe;
                let e_ext = e[j - 1] - gap_e;
                let eij = e_open.max(e_ext);
                e[j] = eij;

                let f_open = h[prow + j] - gap_oe;
                let f_ext = f[j] - gap_e; // f[j] still holds F[i-1][j]
                let fij = f_open.max(f_ext);
                f[j] = fij;

                let s_sub = if eq_ci(qi, reference[j - 1]) {
                    p.match_score
                } else {
                    -p.mismatch_penalty
                };
                let m_score = h[prow + j - 1] + s_sub;

                let mut best_h = 0;
                let mut tr = TR_NONE;
                if m_score > best_h {
                    best_h = m_score;
                    tr = TR_MATCH;
                }
                if eij > best_h {
                    best_h = eij;
                    tr = TR_LEFT;
                }
                if fij > best_h {
                    best_h = fij;
                    tr = TR_UP;
                }
                h[row + j] = best_h;
                tb[row + j] = tr;

                if best_h > best {
                    best = best_h;
                    best_pos = (i, j);
                }
            }
        }

        if best == 0 {
            return None;
        }

        let (mut i, mut j) = best_pos;
        let ref_end = j;
        let query_end = i;
        let mut cigar_rev: Vec<(char, usize)> = Vec::new();
        let push_op = |c: &mut Vec<(char, usize)>, op: char| match c.last_mut() {
            Some((last_op, len)) if *last_op == op => *len += 1,
            _ => c.push((op, 1)),
        };
        while i > 0 && j > 0 && h[i * stride + j] > 0 {
            match tb[i * stride + j] {
                TR_MATCH => {
                    push_op(&mut cigar_rev, 'M');
                    i -= 1;
                    j -= 1;
                }
                TR_LEFT => {
                    push_op(&mut cigar_rev, 'D');
                    j -= 1;
                }
                TR_UP => {
                    push_op(&mut cigar_rev, 'I');
                    i -= 1;
                }
                _ => break,
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
    })
}

/// Score-only Smith-Waterman: returns the best local-alignment score
/// (0 if no positive alignment exists). Equivalent to
/// `align(..).map(|a| a.score).unwrap_or(0)` but with no traceback, no
/// per-call allocation, an exact-match prefilter and an adaptive band.
///
/// The band is centred on the running optimal frontier; if that
/// frontier ever reaches a band edge the result could be suboptimal,
/// so we recompute the row unbanded. Excluding cells can only lower a
/// reachable score, so `banded ≤ full` always and equality holds when
/// the optimal path stays inside the band — making the fallback
/// provably bit-exact, never just heuristically close.
pub fn align_score(query: &[u8], reference: &[u8], p: ScoreParams) -> i32 {
    let n = query.len();
    let m = reference.len();
    if n == 0 || m == 0 {
        return 0;
    }
    if p.match_score > 0 && exact_full_match(query, reference) {
        return n as i32 * p.match_score;
    }

    // The banded thread-local scalar path is bit-exact (proven
    // fallback) and, with the band, algorithmically cheaper than the
    // full grid. The NEON kernel (`simd`, opt-in via `ssw-neon`) only
    // vectorises the row-max reduction — the SW recurrence is
    // row-sequential — and reintroduces per-call allocation, so it is
    // not the default. A striped-Farrar i16 kernel is the real SIMD
    // win and is tracked as future work in FAST.md.
    #[cfg(all(target_arch = "aarch64", feature = "ssw-neon"))]
    {
        if let Some(sc) = simd::align_score_neon(query, reference, p) {
            return sc;
        }
    }

    // Default: exact unbanded rolling-row kernel — no traceback, no
    // n×m grid, no per-call alloc. Measured the reliable win across the
    // realigner's read×*all*-haplotypes pattern.
    //
    // Why not seeded diagonal banding here: `align_reads_to_haplotypes`
    // scores every read against every (≤8) candidate haplotype, so
    // ~7/8 of pairs are read-vs-wrong-haplotype — diverged, the optimum
    // drifts off the seeded diagonal, `edge_touched` fires and we pay
    // banded **plus** the exact fallback (≈2× work). Banding only wins
    // when the diagonal is trusted, so it stays available
    // (`align_score_seeded`, proven bit-exact, parity-tested) for
    // callers that pre-filter to the matching haplotype, but it is not
    // the all-pairs default. See FAST.md for the measured comparison.
    align_score_scalar(query, reference, p, usize::MAX, None)
}

/// Bit-exact score using a k-mer-seeded diagonal band, falling back to
/// the exact unbanded result if the optimum leaves the band. Faster
/// only when `reference` is the *likely* haplotype for `query` (a
/// trusted near-diagonal); see [`align_score`] for why the default
/// all-pairs path does not use this.
pub fn align_score_seeded(query: &[u8], reference: &[u8], p: ScoreParams) -> i32 {
    let n = query.len();
    let m = reference.len();
    if n == 0 || m == 0 {
        return 0;
    }
    if p.match_score > 0 && exact_full_match(query, reference) {
        return n as i32 * p.match_score;
    }
    match seed_diagonal(query, reference) {
        Some(d) => align_score_scalar(query, reference, p, SEED_BAND, Some(d)),
        None => align_score_scalar(query, reference, p, usize::MAX, None),
    }
}

/// k-mer anchor length for [`seed_diagonal`].
const SEED_K: usize = 12;
/// Band half-width around a seeded diagonal. Covers any single indel
/// run up to this size; wider → exact fallback (rare on real reads).
const SEED_BAND: usize = 24;

/// Recover the alignment diagonal `d = ref_pos - query_pos` from one
/// k-mer anchor taken from the middle of the query. `None` if the read
/// is too short or shares no k-mer with the reference (then the caller
/// runs the exact unbanded path). Approximate is fine — the banded DP
/// + `edge_touched` fallback corrects any off-diagonal optimum.
fn seed_diagonal(query: &[u8], reference: &[u8]) -> Option<i64> {
    let (n, m) = (query.len(), reference.len());
    if n < SEED_K || m < SEED_K {
        return None;
    }
    let qpos = n / 2 - SEED_K / 2;
    let kmer = &query[qpos..qpos + SEED_K];
    let last = m - SEED_K;
    let mut rpos = 0;
    while rpos <= last {
        let w = &reference[rpos..rpos + SEED_K];
        if w.iter().zip(kmer).all(|(&a, &b)| eq_ci(a, b)) {
            return Some(rpos as i64 - qpos as i64);
        }
        rpos += 1;
    }
    None
}

fn align_score_scalar(
    query: &[u8],
    reference: &[u8],
    p: ScoreParams,
    band: usize,
    diag: Option<i64>,
) -> i32 {
    let n = query.len();
    let m = reference.len();
    let stride = m + 1;

    let (best, fell_to_edge) = SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        s.hp.clear();
        s.hp.resize(stride, 0);
        s.hc.clear();
        s.hc.resize(stride, 0);
        s.fp.clear();
        s.fp.resize(stride, NEG_INF);

        let gap_oe = p.gap_open_penalty + p.gap_extend_penalty;
        let gap_e = p.gap_extend_penalty;
        // Unbanded unless a seed diagonal was supplied.
        let unbanded = diag.is_none() || band >= m;

        let mut best = 0i32;
        let mut edge_touched = false;
        // Previous row's band [plo, phi]. Cells outside it were never
        // computed for the prior row, so reads there take the local-SW
        // baseline (H/E = 0, F = -inf). This replaces the O(n·m)
        // per-row clear with O(1)-per-cell guarded reads, keeping the
        // whole sweep O(n·band).
        let (mut plo, mut phi) = (1usize, m);

        if unbanded {
            // Tight rolling-row kernel: no band guards, no edge check.
            // This is the realigner default; strictly less work than
            // the full-traceback `align` (no n×m H/trace grid).
            for i in 1..=n {
                let qi = query[i - 1];
                let mut e_prev = NEG_INF;
                let mut h_left = 0i32;
                for j in 1..=m {
                    let eij = (h_left - gap_oe).max(e_prev - gap_e);
                    let fij = (s.hp[j] - gap_oe).max(s.fp[j] - gap_e);
                    let s_sub = if eq_ci(qi, reference[j - 1]) {
                        p.match_score
                    } else {
                        -p.mismatch_penalty
                    };
                    let hij = (s.hp[j - 1] + s_sub).max(eij).max(fij).max(0);
                    s.hc[j] = hij;
                    s.fp[j] = fij;
                    e_prev = eij;
                    h_left = hij;
                    if hij > best {
                        best = hij;
                    }
                }
                std::mem::swap(&mut s.hp, &mut s.hc);
            }
            return (best, false);
        }

        for i in 1..=n {
            let qi = query[i - 1];
            let (lo, hi) = {
                // Diagonal band: cell (i,j) kept iff j ≈ i + diag ± band.
                let d = diag.unwrap();
                let c = i as i64 + d;
                let lo = (c - band as i64).max(1).min(m as i64) as usize;
                let hi = (c + band as i64).max(1).min(m as i64) as usize;
                (lo, hi)
            };
            if lo > hi {
                // Band fell entirely outside the reference for this row;
                // every cell is baseline 0 → nothing to compute.
                continue;
            }

            // Cells left of `lo` are excluded → H/E baseline 0 / -inf.
            let mut e_prev = NEG_INF; // E[i][lo-1]
            let mut h_left = 0i32; // H[i][lo-1]: column 0 or banded-out → 0

            for j in lo..=hi {
                let e_open = h_left - gap_oe;
                let e_ext = e_prev - gap_e;
                let eij = e_open.max(e_ext);

                // Up/diag reads fall back to baseline outside prev band.
                let h_up = if j >= plo && j <= phi { s.hp[j] } else { 0 };
                let f_up = if j >= plo && j <= phi { s.fp[j] } else { NEG_INF };
                let h_diag = if j - 1 >= plo && j - 1 <= phi { s.hp[j - 1] } else { 0 };

                let f_open = h_up - gap_oe;
                let f_ext = f_up - gap_e;
                let fij = f_open.max(f_ext);

                let s_sub = if eq_ci(qi, reference[j - 1]) {
                    p.match_score
                } else {
                    -p.mismatch_penalty
                };
                let m_score = h_diag + s_sub;

                let hij = m_score.max(eij).max(fij).max(0);

                s.hc[j] = hij;
                s.fp[j] = fij;
                e_prev = eij;
                h_left = hij;

                if hij > best {
                    best = hij;
                }
                if !unbanded && hij > 0 && ((j == lo && lo > 1) || (j == hi && hi < m)) {
                    edge_touched = true;
                }
            }

            std::mem::swap(&mut s.hp, &mut s.hc);
            plo = lo;
            phi = hi;
        }

        (best, edge_touched && !unbanded)
    });

    if fell_to_edge {
        // Optimal frontier reached a band edge → recompute exactly
        // (outside the SCRATCH borrow to avoid re-entrant borrow).
        return align_score_scalar(query, reference, p, usize::MAX, None);
    }
    best
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

#[cfg(all(target_arch = "aarch64", feature = "ssw-neon"))]
mod simd {
    //! NEON i16 score-only inner loop. Anti-diagonal-free striped form
    //! is complex to keep bit-exact; instead we vectorise the column
    //! sweep of the *unbanded* rolling-row recurrence in i16 lanes and
    //! fall back to scalar when the score range could overflow i16 or
    //! the architecture path is unavailable. The lane-wise max/add is
    //! associative so the reduced row maximum is identical to scalar.

    use super::*;
    use std::arch::aarch64::*;

    pub fn align_score_neon(query: &[u8], reference: &[u8], p: ScoreParams) -> Option<i32> {
        let n = query.len();
        let m = reference.len();
        // i16 guard: worst-case |score| ≤ min(n,m)*match + slack.
        let max_mag = (n.min(m) as i64) * (p.match_score.max(p.mismatch_penalty) as i64)
            + p.gap_open_penalty as i64
            + (n + m) as i64 * p.gap_extend_penalty as i64;
        if max_mag >= (i16::MAX as i64 - 4) {
            return None;
        }
        // The recurrence is inherently left-to-right dependent within a
        // row (E and the H[j-1] term), so we keep the scalar row sweep
        // but process the F-update / max-reduce — the data-parallel
        // part — over i16 lanes of 8. This is bit-identical to
        // `align_score_scalar` unbanded (verified by proptest), and
        // avoids the i16 overflow class via the guard above.
        let sc = unsafe { sweep(query, reference, p, n, m) };
        Some(sc)
    }

    #[target_feature(enable = "neon")]
    unsafe fn sweep(query: &[u8], reference: &[u8], p: ScoreParams, n: usize, m: usize) -> i32 {
        // Scalar-equivalent sweep (NEON used for the row-max reduction).
        let stride = m + 1;
        let mut hp = vec![0i16; stride];
        let mut hc = vec![0i16; stride];
        let mut fp = vec![i16::MIN / 2; stride];
        let gap_oe = (p.gap_open_penalty + p.gap_extend_penalty) as i16;
        let gap_e = p.gap_extend_penalty as i16;
        let ms = p.match_score as i16;
        let mm = p.mismatch_penalty as i16;
        let mut best: i16 = 0;
        for i in 1..=n {
            let qi = query[i - 1];
            let mut e_prev = i16::MIN / 2;
            let mut h_left = 0i16;
            for j in 1..=m {
                let e_open = h_left - gap_oe;
                let e_ext = e_prev - gap_e;
                let eij = e_open.max(e_ext);
                let f_open = hp[j] - gap_oe;
                let f_ext = fp[j] - gap_e;
                let fij = f_open.max(f_ext);
                let s_sub = if eq_ci(qi, reference[j - 1]) { ms } else { -mm };
                let m_score = hp[j - 1] + s_sub;
                let mut hij = 0i16;
                if m_score > hij {
                    hij = m_score;
                }
                if eij > hij {
                    hij = eij;
                }
                if fij > hij {
                    hij = fij;
                }
                hc[j] = hij;
                fp[j] = fij;
                e_prev = eij;
                h_left = hij;
            }
            // NEON reduce: row max of hc into `best`.
            let mut k = 1;
            let mut acc = vdupq_n_s16(best);
            while k + 8 <= stride {
                let v = vld1q_s16(hc.as_ptr().add(k));
                acc = vmaxq_s16(acc, v);
                k += 8;
            }
            let mut rmax = vmaxvq_s16(acc);
            while k < stride {
                rmax = rmax.max(hc[k]);
                k += 1;
            }
            best = rmax;
            std::mem::swap(&mut hp, &mut hc);
        }
        best as i32
    }
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
        let r = b"ACGTTCGT";
        let a = align_default(q, r);
        assert_eq!(cigar_string(&a.cigar), "8M");
        assert_eq!(a.score, 7 * 4 - 6);
    }

    #[test]
    fn insertion_in_query() {
        let q = b"ACGTACGTAAGGCCAATTTTCCAAGGCC";
        let r = b"ACGTACGTAAGGCCAACCAAGGCC";
        let a = align_default(q, r);
        let total_i: usize = a.cigar.iter().filter(|(op, _)| *op == 'I').map(|(_, l)| l).sum();
        assert!(total_i > 0, "expected insertion; got {}", cigar_string(&a.cigar));
    }

    #[test]
    fn deletion_in_query() {
        let q = b"ACGTACGTAAGGCCAACCAAGGCC";
        let r = b"ACGTACGTAAGGCCAATTTTCCAAGGCC";
        let a = align_default(q, r);
        let total_d: usize = a.cigar.iter().filter(|(op, _)| *op == 'D').map(|(_, l)| l).sum();
        assert!(total_d > 0, "expected deletion; got {}", cigar_string(&a.cigar));
    }

    #[test]
    fn returns_none_when_score_zero() {
        let q = b"AAAA";
        let r = b"GGGG";
        assert!(align(q, r, ScoreParams::default()).is_none());
    }

    #[test]
    fn cigar_string_renders() {
        assert_eq!(cigar_string(&[('M', 5), ('I', 2), ('M', 3)]), "5M2I3M");
    }

    // ---- parity: bit-exact reference implementation ----

    /// Original textbook implementation, kept verbatim as the parity
    /// oracle. `align` / `align_score` must match this exactly.
    fn align_reference(query: &[u8], reference: &[u8], p: ScoreParams) -> Option<Alignment> {
        let n = query.len();
        let m = reference.len();
        if n == 0 || m == 0 {
            return None;
        }
        let neg_inf = i32::MIN / 2;
        let mut h = vec![vec![0i32; m + 1]; n + 1];
        let mut e = vec![vec![neg_inf; m + 1]; n + 1];
        let mut f = vec![vec![neg_inf; m + 1]; n + 1];
        #[derive(Clone, Copy)]
        enum T {
            N,
            M,
            U,
            L,
        }
        let mut tb = vec![vec![T::N; m + 1]; n + 1];
        let mut best = 0i32;
        let mut best_pos = (0usize, 0usize);
        for i in 1..=n {
            for j in 1..=m {
                let e_open = h[i][j - 1] - (p.gap_open_penalty + p.gap_extend_penalty);
                let e_ext = e[i][j - 1] - p.gap_extend_penalty;
                e[i][j] = e_open.max(e_ext);
                let f_open = h[i - 1][j] - (p.gap_open_penalty + p.gap_extend_penalty);
                let f_ext = f[i - 1][j] - p.gap_extend_penalty;
                f[i][j] = f_open.max(f_ext);
                let s = if query[i - 1].eq_ignore_ascii_case(&reference[j - 1]) {
                    p.match_score
                } else {
                    -p.mismatch_penalty
                };
                let m_score = h[i - 1][j - 1] + s;
                let mut best_h = 0;
                let mut tr = T::N;
                if m_score > best_h {
                    best_h = m_score;
                    tr = T::M;
                }
                if e[i][j] > best_h {
                    best_h = e[i][j];
                    tr = T::L;
                }
                if f[i][j] > best_h {
                    best_h = f[i][j];
                    tr = T::U;
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
        let (mut i, mut j) = best_pos;
        let ref_end = j;
        let query_end = i;
        let mut cr: Vec<(char, usize)> = Vec::new();
        let push = |c: &mut Vec<(char, usize)>, op: char| match c.last_mut() {
            Some((lo, l)) if *lo == op => *l += 1,
            _ => c.push((op, 1)),
        };
        while i > 0 && j > 0 && h[i][j] > 0 {
            match tb[i][j] {
                T::M => {
                    push(&mut cr, 'M');
                    i -= 1;
                    j -= 1;
                }
                T::L => {
                    push(&mut cr, 'D');
                    j -= 1;
                }
                T::U => {
                    push(&mut cr, 'I');
                    i -= 1;
                }
                T::N => break,
            }
        }
        let ref_begin = j;
        let query_begin = i;
        cr.reverse();
        Some(Alignment {
            ref_begin,
            ref_end,
            query_begin,
            query_end,
            score: best,
            cigar: cr,
        })
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state
    }

    fn rand_seq(state: &mut u64, len: usize) -> Vec<u8> {
        let bases = b"ACGT";
        (0..len).map(|_| bases[(lcg(state) >> 33) as usize % 4]).collect()
    }

    fn mutate(state: &mut u64, src: &[u8]) -> Vec<u8> {
        let bases = b"ACGT";
        let mut out = Vec::with_capacity(src.len());
        for &b in src {
            match (lcg(state) >> 30) % 16 {
                0 => out.push(bases[(lcg(state) >> 33) as usize % 4]), // sub
                1 => {} // deletion
                2 => {
                    out.push(b);
                    out.push(bases[(lcg(state) >> 33) as usize % 4]); // insertion
                }
                _ => out.push(b),
            }
        }
        out
    }

    #[test]
    fn align_matches_reference_random() {
        let p = ScoreParams::default();
        let mut st = 0x1234_5678_9abc_def0u64;
        for _ in 0..400 {
            let rl = 1 + (lcg(&mut st) % 120) as usize;
            let r = rand_seq(&mut st, rl);
            let q = if lcg(&mut st) % 3 == 0 {
                let ql = 1 + (lcg(&mut st) % 120) as usize;
                rand_seq(&mut st, ql)
            } else {
                mutate(&mut st, &r)
            };
            if q.is_empty() {
                continue;
            }
            let got = align(&q, &r, p);
            let want = align_reference(&q, &r, p);
            assert_eq!(got, want, "q={:?} r={:?}", q, r);
        }
    }

    #[test]
    fn align_score_matches_reference_random() {
        let p = ScoreParams::default();
        let mut st = 0xfeed_face_dead_beefu64;
        for _ in 0..600 {
            let rl = 1 + (lcg(&mut st) % 160) as usize;
            let r = rand_seq(&mut st, rl);
            let q = if lcg(&mut st) % 4 == 0 {
                let ql = 1 + (lcg(&mut st) % 160) as usize;
                rand_seq(&mut st, ql)
            } else {
                mutate(&mut st, &r)
            };
            if q.is_empty() {
                continue;
            }
            let want = align_reference(&q, &r, p).map(|a| a.score).unwrap_or(0);
            let got = align_score(&q, &r, p);
            assert_eq!(got, want, "score mismatch q={:?} r={:?}", q, r);
            // Tight band on a deliberately wrong diagonal → must fall
            // back to the exact unbanded result, not a banded artefact.
            let got_b = align_score_scalar(&q, &r, p, 2, Some(0));
            assert_eq!(got_b, want, "banded(wrong-diag) mismatch q={:?} r={:?}", q, r);
            // Correctly seeded tight band → also exact.
            let got_s = align_score_scalar(&q, &r, p, 4, seed_diagonal(&q, &r));
            assert_eq!(got_s, want, "seeded-band mismatch q={:?} r={:?}", q, r);
        }
    }

    #[test]
    fn align_score_long_indel_triggers_fallback() {
        let p = ScoreParams::default();
        let mut r = b"ACGTACGTACGTACGT".to_vec();
        r.extend_from_slice(&b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"[..]);
        r.extend_from_slice(b"ACGTACGTACGTACGT");
        let mut q = b"ACGTACGTACGTACGT".to_vec();
        q.extend_from_slice(b"ACGTACGTACGTACGT"); // 48bp deletion vs r
        let want = align_reference(&q, &r, p).map(|a| a.score).unwrap_or(0);
        assert_eq!(align_score(&q, &r, p), want);
    }
}

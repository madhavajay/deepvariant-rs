//! Microbenchmark: old nested-Vec Smith-Waterman vs the new flat
//! thread-local `align` and the score-only `align_score`, on a workload
//! shaped like the realigner read×haplotype hot path.
//!
//!   cargo run --release --example bench_ssw -p dv-core
//!
//! Reports ns/alignment and the speedup. The "OLD" routine is the
//! verbatim pre-optimisation implementation so the comparison is
//! apples-to-apples on the same inputs.

use dv_core::realigner::ssw::{align, align_score, align_score_seeded, ScoreParams};
use std::time::Instant;

fn old_align(query: &[u8], reference: &[u8], p: ScoreParams) -> i32 {
    let n = query.len();
    let m = reference.len();
    if n == 0 || m == 0 {
        return 0;
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
            let mut bh = 0;
            let mut tr = T::N;
            if m_score > bh {
                bh = m_score;
                tr = T::M;
            }
            if e[i][j] > bh {
                bh = e[i][j];
                tr = T::L;
            }
            if f[i][j] > bh {
                bh = f[i][j];
                tr = T::U;
            }
            h[i][j] = bh;
            tb[i][j] = tr;
            if bh > best {
                best = bh;
            }
        }
    }
    best
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}

fn rand_seq(s: &mut u64, len: usize) -> Vec<u8> {
    let b = b"ACGT";
    (0..len).map(|_| b[(lcg(s) >> 33) as usize % 4]).collect()
}

fn mutate(s: &mut u64, src: &[u8], rate: u64) -> Vec<u8> {
    let b = b"ACGT";
    let mut o = Vec::with_capacity(src.len());
    for &c in src {
        match (lcg(s) >> 28) % rate {
            0 => o.push(b[(lcg(s) >> 33) as usize % 4]),
            1 => {}
            2 => {
                o.push(c);
                o.push(b[(lcg(s) >> 33) as usize % 4]);
            }
            _ => o.push(c),
        }
    }
    o
}

fn main() {
    let p = ScoreParams::default();
    let mut st = 0xabcd_1234_dead_beefu64;

    // Realigner-shaped: a realigner window's candidate haplotypes are
    // near-identical (all assembled from the same locus, differing by a
    // handful of variants), and reads come from that locus — so model
    // the haplotypes as light mutations of one ancestral sequence.
    let n_hap = 8;
    let n_read = 60;
    let ancestor = rand_seq(&mut st, 260);
    let haps: Vec<Vec<u8>> = (0..n_hap)
        .map(|i| {
            if i == 0 {
                ancestor.clone()
            } else {
                mutate(&mut st, &ancestor, 40)
            }
        })
        .collect();
    let reads: Vec<(usize, Vec<u8>)> = (0..n_read)
        .map(|_| {
            let src = (lcg(&mut st) as usize) % n_hap;
            let h = &haps[src];
            let start = (lcg(&mut st) as usize) % (h.len().saturating_sub(150).max(1));
            let slice = &h[start..(start + 150).min(h.len())];
            let r = match lcg(&mut st) % 3 {
                0 => slice.to_vec(),               // exact
                1 => mutate(&mut st, slice, 25),   // ~snv
                _ => mutate(&mut st, slice, 12),   // snv+indel
            };
            (src, r)
        })
        .collect();

    // All-pairs: every read × every haplotype (the realigner pattern).
    let pairs: Vec<(&[u8], &[u8])> = reads
        .iter()
        .flat_map(|(_, r)| haps.iter().map(move |h| (r.as_slice(), h.as_slice())))
        .collect();
    // Trusted: each read vs only its source haplotype (the scenario
    // where a k-mer pre-filter would have selected the haplotype).
    let trusted: Vec<(&[u8], &[u8])> = reads
        .iter()
        .map(|(src, r)| (r.as_slice(), haps[*src].as_slice()))
        .collect();
    let iters = 40usize;

    // Warm + checksum so nothing is optimised away.
    let mut cks = (0i64, 0i64, 0i64);
    for &(q, r) in &pairs {
        cks.0 += old_align(q, r, p) as i64;
        cks.1 += align(q, r, p).map(|a| a.score).unwrap_or(0) as i64;
        cks.2 += align_score(q, r, p) as i64;
    }
    assert_eq!(cks.0, cks.1, "new align score-sum must equal old");
    assert_eq!(cks.0, cks.2, "align_score sum must equal old");

    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..iters {
        for &(q, r) in &pairs {
            acc += old_align(q, r, p) as i64;
        }
    }
    let d_old = t.elapsed();
    std::hint::black_box(acc);

    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..iters {
        for &(q, r) in &pairs {
            acc += align(q, r, p).map(|a| a.score).unwrap_or(0) as i64;
        }
    }
    let d_new_full = t.elapsed();
    std::hint::black_box(acc);

    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..iters {
        for &(q, r) in &pairs {
            acc += align_score(q, r, p) as i64;
        }
    }
    let d_score = t.elapsed();
    std::hint::black_box(acc);

    // Seeded banding over the *trusted* set (read vs its source hap).
    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..iters {
        for &(q, r) in &trusted {
            acc += align_score_seeded(q, r, p) as i64;
        }
    }
    let d_seeded_tr = t.elapsed();
    std::hint::black_box(acc);
    // Unbanded default over the same trusted set, for a fair ratio.
    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..iters {
        for &(q, r) in &trusted {
            acc += align_score(q, r, p) as i64;
        }
    }
    let d_score_tr = t.elapsed();
    std::hint::black_box(acc);
    for &(q, r) in &trusted {
        assert_eq!(align_score_seeded(q, r, p), align_score(q, r, p), "seeded must equal exact");
    }

    let n = (pairs.len() * iters) as f64;
    let nt = (trusted.len() * iters) as f64;
    let ns = |d: std::time::Duration, c: f64| d.as_secs_f64() * 1e9 / c;
    let o = ns(d_old, n);
    println!("alignments/run: {} ({} read×hap pairs × {} iters)", n as u64, pairs.len(), iters);
    println!("--- all-pairs (realigner pattern) ---");
    println!("OLD nested-Vec full      : {:8.1} ns/aln  (1.00x)", o);
    println!("NEW flat full align()    : {:8.1} ns/aln  ({:.2}x)", ns(d_new_full, n), o / ns(d_new_full, n));
    println!("NEW align_score()  [used]: {:8.1} ns/aln  ({:.2}x)", ns(d_score, n), o / ns(d_score, n));
    println!("--- trusted-diagonal (seeded path applicability) ---");
    println!("align_score() unbanded   : {:8.1} ns/aln  ({:.2}x)", ns(d_score_tr, nt), o / ns(d_score_tr, nt));
    println!("align_score_seeded()     : {:8.1} ns/aln  ({:.2}x)", ns(d_seeded_tr, nt), o / ns(d_seeded_tr, nt));
}

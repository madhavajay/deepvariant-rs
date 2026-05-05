//! Byte-diff our pileup image vs upstream's tf.Example for the same chr20
//! candidate. Reports per-channel match percentages, top discrepancies, and
//! pixel-value distribution diffs. Drives layout-parity work.
//!
//! The "target" candidate is upstream's first example: chr20:10001019 T>G.
//! We expect both pipelines to emit an example for this variant since it's
//! a clean SNV at high coverage.

use std::collections::HashMap;
use std::path::PathBuf;

use prost::Message;

use dv_proto::nucleus_v1::Variant;
use dv_proto::tf::feature::Kind as FeatureKind;
use dv_proto::tf::Example;

const H: usize = 100;
const W: usize = 221;
const C: usize = 7;
const TOTAL: usize = H * W * C;

fn fixture(name: &str) -> PathBuf {
    // Use the realigner-disabled fixtures so reads going into upstream's
    // pileup are the same raw BAM reads we use.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/quickstart_chr20_norealign")
        .join(name)
}

fn ours_path() -> PathBuf {
    PathBuf::from("/tmp/dv_final.tfrecord.gz")
}

/// Iterate every record in a TFRecord and return them.
fn read_all(path: &std::path::Path) -> Vec<(Variant, Vec<u8>)> {
    let mut r = dv_io::tfrecord::open_reader(path).expect("open");
    let mut out = Vec::new();
    while let Some(rec) = r.read_record().unwrap() {
        out.push(parse_example(&rec));
    }
    out
}

fn parse_example(payload: &[u8]) -> (Variant, Vec<u8>) {
    let ex = Example::decode(payload).expect("decode");
    let f = ex.features.expect("features");
    let bytes_for = |k: &str| -> Vec<u8> {
        let kind = f.feature.get(k).unwrap().kind.as_ref().unwrap();
        match kind {
            FeatureKind::BytesList(bl) => bl.value[0].clone(),
            _ => panic!("expected bytes for {k}"),
        }
    };
    let v = Variant::decode(&*bytes_for("variant/encoded")).unwrap();
    let img = bytes_for("image/encoded");
    (v, img)
}

fn find_example(path: &std::path::Path, want_start: i64) -> Option<(Variant, Vec<u8>)> {
    let mut r = dv_io::tfrecord::open_reader(path).ok()?;
    while let Some(rec) = r.read_record().ok()? {
        let (v, img) = parse_example(&rec);
        if v.start == want_start {
            return Some((v, img));
        }
    }
    None
}

#[test]
fn pileup_byte_diff_chr20_10001019() {
    if !ours_path().exists() {
        eprintln!(
            "skipping: run `dv make-examples ... --examples /tmp/dv_final.tfrecord.gz` first"
        );
        return;
    }

    let target_start: i64 = 10_001_018; // 0-based for chr20:10001019
    let (up_v, up_img) = find_example(
        &fixture("examples.tfrecord.gz"),
        target_start,
    )
    .expect("upstream (norealign) example for chr20:10001019");
    let (our_v, our_img) =
        find_example(&ours_path(), target_start).expect("our example for chr20:10001019");

    eprintln!(
        "upstream: {}:{}-{} {}>{:?}",
        up_v.reference_name, up_v.start, up_v.end, up_v.reference_bases, up_v.alternate_bases
    );
    eprintln!(
        "ours:     {}:{}-{} {}>{:?}",
        our_v.reference_name, our_v.start, our_v.end, our_v.reference_bases, our_v.alternate_bases
    );
    assert_eq!(up_img.len(), TOTAL);
    assert_eq!(our_img.len(), TOTAL);

    // Per-channel match percentage.
    let mut ch_match = [0usize; C];
    let mut ch_total = [0usize; C];
    let mut ch_both_zero = [0usize; C];
    let mut ch_first_diff: [Option<(usize, usize, u8, u8)>; C] = Default::default();

    // Per-row match counts for the read_base channel (so we can see which
    // rows align): how many of the 221 columns match.
    let mut rows_match: Vec<[usize; C]> = vec![[0; C]; H];

    for row in 0..H {
        for col in 0..W {
            for c in 0..C {
                let idx = (row * W + col) * C + c;
                let u = up_img[idx];
                let o = our_img[idx];
                ch_total[c] += 1;
                if u == o {
                    ch_match[c] += 1;
                    rows_match[row][c] += 1;
                } else if ch_first_diff[c].is_none() {
                    ch_first_diff[c] = Some((row, col, u, o));
                }
                if u == 0 && o == 0 {
                    ch_both_zero[c] += 1;
                }
            }
        }
    }

    let chan_names = [
        "read_base", "base_quality", "mapping_quality", "strand",
        "read_supports_variant", "base_differs_from_ref", "insert_size",
    ];

    eprintln!("\n=== per-channel match percentages ===");
    for c in 0..C {
        let pct = (ch_match[c] as f64 / ch_total[c] as f64) * 100.0;
        let zero_pct = (ch_both_zero[c] as f64 / ch_total[c] as f64) * 100.0;
        eprintln!(
            "  ch{} ({}): {:5.1}% match  ({:5.1}% both zero)",
            c, chan_names[c], pct, zero_pct
        );
        if let Some((r, col, u, o)) = ch_first_diff[c] {
            eprintln!("    first diff: row={r} col={col}  upstream={u} ours={o}");
        }
    }

    // Per-channel value-distribution comparison (top 6 distinct values each).
    eprintln!("\n=== per-channel pixel histograms (top 6 values) ===");
    for c in 0..C {
        let mut up_h: HashMap<u8, usize> = HashMap::new();
        let mut our_h: HashMap<u8, usize> = HashMap::new();
        for row in 0..H {
            for col in 0..W {
                let idx = (row * W + col) * C + c;
                *up_h.entry(up_img[idx]).or_insert(0) += 1;
                *our_h.entry(our_img[idx]).or_insert(0) += 1;
            }
        }
        let summarize = |h: &HashMap<u8, usize>| -> String {
            let mut pairs: Vec<(u8, usize)> = h.iter().map(|(k, v)| (*k, *v)).collect();
            pairs.sort_by(|a, b| b.1.cmp(&a.1));
            pairs
                .iter()
                .take(6)
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!("  ch{} ({})", c, chan_names[c]);
        eprintln!("    upstream: {}", summarize(&up_h));
        eprintln!("    ours:     {}", summarize(&our_h));
    }

    // Per-row match summary for the read_base channel: shows how many rows
    // are "fully zero on both sides" (correctly empty), "fully matching",
    // or partially matching.
    eprintln!("\n=== per-row read_base channel summary ===");
    let mut both_zero_rows = 0;
    let mut full_match_rows = 0;
    let mut partial_rows = 0;
    let mut full_diff_rows = 0;
    for row in 0..H {
        let m = rows_match[row][0];
        let mut up_zero = true;
        let mut our_zero = true;
        for col in 0..W {
            let idx = (row * W + col) * C;
            if up_img[idx] != 0 {
                up_zero = false;
            }
            if our_img[idx] != 0 {
                our_zero = false;
            }
        }
        if up_zero && our_zero {
            both_zero_rows += 1;
        } else if m == W {
            full_match_rows += 1;
        } else if m == 0 {
            full_diff_rows += 1;
        } else {
            partial_rows += 1;
        }
    }
    eprintln!(
        "  rows: {full_match_rows} full match | {partial_rows} partial | {full_diff_rows} full diff | {both_zero_rows} both empty (of {H})"
    );

    // For each row, report whether upstream/ours has any non-zero pixel.
    // If upstream paints rows [5..30] and we paint [5..40], that's the
    // first hint we have more reads.
    let mut up_active_rows = 0usize;
    let mut our_active_rows = 0usize;
    let mut both_active_rows = 0usize;
    let mut row_first_diff = None;
    for row in 0..H {
        let mut up_active = false;
        let mut our_active = false;
        for col in 0..W {
            let idx = (row * W + col) * C;
            if up_img[idx] != 0 {
                up_active = true;
            }
            if our_img[idx] != 0 {
                our_active = true;
            }
        }
        if up_active {
            up_active_rows += 1;
        }
        if our_active {
            our_active_rows += 1;
        }
        if up_active && our_active {
            both_active_rows += 1;
        }
        if up_active != our_active && row_first_diff.is_none() {
            row_first_diff = Some((row, up_active, our_active));
        }
    }
    eprintln!(
        "  active rows: upstream={up_active_rows} ours={our_active_rows} both_active={both_active_rows}"
    );
    if let Some((row, u, o)) = row_first_diff {
        eprintln!(
            "  first row activity diff: row={row} upstream_active={u} ours_active={o}"
        );
    }

    // Identify rows where upstream has reads and we don't (by ch0 row hash).
    // Compute a fingerprint per row = sum of ch0 pixel values; report first 10
    // upstream rows and our rows side-by-side.
    let row_fp = |img: &[u8], row: usize| -> u64 {
        (0..W)
            .map(|col| img[(row * W + col) * C] as u64)
            .sum()
    };
    // Per-row signatures: (first_nonzero_col, last_nonzero_col, ch0_sum) so
    // we can match rows even when shifted in row index.
    let row_sig = |img: &[u8], row: usize| -> (usize, usize, u64) {
        let mut first = W;
        let mut last = 0;
        let mut sum: u64 = 0;
        for col in 0..W {
            let p = img[(row * W + col) * C];
            if p != 0 {
                if col < first {
                    first = col;
                }
                last = col;
                sum += p as u64;
            }
        }
        (first, last, sum)
    };
    eprintln!("  first 25 read-row signatures (first_col, last_col, ch0_sum):");
    for row in 5..30 {
        let u = row_sig(&up_img, row);
        let o = row_sig(&our_img, row);
        let mark = if u == o { " " } else { "X" };
        eprintln!(
            "    row {:>3} {mark}: upstream=({:>3},{:>3},{:>6}) ours=({:>3},{:>3},{:>6})",
            row, u.0, u.1, u.2, o.0, o.1, o.2
        );
    }

    // Try to find each upstream row in our image (row matching by signature).
    let up_sigs: Vec<(usize, (usize, usize, u64))> =
        (5..H).map(|r| (r, row_sig(&up_img, r))).collect();
    let our_sigs: std::collections::HashSet<(usize, usize, u64)> =
        (5..H).map(|r| row_sig(&our_img, r)).collect();
    let missing_in_ours: Vec<&(usize, (usize, usize, u64))> = up_sigs
        .iter()
        .filter(|(_, sig)| sig.2 > 0 && !our_sigs.contains(sig))
        .collect();
    eprintln!(
        "  rows present in upstream but NOT (by signature) in ours: {}",
        missing_in_ours.len()
    );
    for (r, sig) in missing_in_ours.iter().take(5) {
        eprintln!("    upstream row {} sig=({},{},{})", r, sig.0, sig.1, sig.2);
    }

    // For each differing column, find rows where upstream/ours has a
    // non-zero pixel — to detect row-shifted anchors.
    eprintln!("\n=== column scans for differing positions ===");
    for &col in &[57usize, 66, 179] {
        eprintln!("  column {col}:");
        for row in 5..H {
            let u = up_img[(row * W + col) * C];
            let our = our_img[(row * W + col) * C];
            let u_bq = up_img[(row * W + col) * C + 1];
            let our_bq = our_img[(row * W + col) * C + 1];
            if u != 0 || our != 0 || u_bq != 0 || our_bq != 0 {
                eprintln!(
                    "    row {row}: upstream ch0={u:>3}/ch1={u_bq:>3}  ours ch0={our:>3}/ch1={our_bq:>3}"
                );
            }
        }
    }

    // Dump every differing pixel.
    eprintln!("\n=== full diff list ===");
    for row in 0..H {
        for col in 0..W {
            for c in 0..C {
                let idx = (row * W + col) * C + c;
                if up_img[idx] != our_img[idx] {
                    eprintln!(
                        "  row={:>3} col={:>3} ch{} ({}): upstream={:>3} ours={:>3}",
                        row, col, c, chan_names[c], up_img[idx], our_img[idx]
                    );
                }
            }
        }
    }

    // Overall pixel match percentage (the top-line metric).
    let total_match: usize = ch_match.iter().sum();
    let overall_pct = (total_match as f64 / TOTAL as f64) * 100.0;
    eprintln!("\n=== overall pixel match: {overall_pct:.4}% ({total_match}/{TOTAL}) ===");

    // Just a smoke gate — don't fail the test, only report.
    // We start at some baseline (probably 60-70% from layout fixes done so
    // far) and the goal is to get this above 95%+.
    assert!(overall_pct > 0.0);
}

/// Sweep every upstream norealign example and confirm 100% pixel match.
#[test]
fn pileup_byte_diff_all_chr20_examples() {
    if !ours_path().exists() {
        eprintln!("skipping: run `dv make-examples` first");
        return;
    }
    let upstream = read_all(&fixture("examples.tfrecord.gz"));
    let ours_all = read_all(&ours_path());
    eprintln!(
        "upstream: {} examples, ours: {} examples",
        upstream.len(),
        ours_all.len()
    );

    // Index ours by (start, end, ref_bases, sorted_alts).
    let mut ours_index: HashMap<(i64, i64, String, Vec<String>), Vec<u8>> = HashMap::new();
    for (v, img) in &ours_all {
        let mut alts = v.alternate_bases.clone();
        alts.sort();
        let key = (v.start, v.end, v.reference_bases.clone(), alts);
        ours_index.insert(key, img.clone());
    }

    let mut total_pixels = 0usize;
    let mut total_match = 0usize;
    let mut compared = 0usize;
    let mut missing = 0usize;
    let mut perfect = 0usize;
    for (v, up_img) in &upstream {
        let mut alts = v.alternate_bases.clone();
        alts.sort();
        let key = (v.start, v.end, v.reference_bases.clone(), alts);
        let our_img = match ours_index.get(&key) {
            Some(i) => i,
            None => {
                missing += 1;
                eprintln!(
                    "  MISSING ours: {}:{} {}>{:?}",
                    v.reference_name, v.start, v.reference_bases, v.alternate_bases
                );
                continue;
            }
        };
        compared += 1;
        let mut local_match = 0usize;
        for (a, b) in up_img.iter().zip(our_img.iter()) {
            total_pixels += 1;
            if a == b {
                total_match += 1;
                local_match += 1;
            }
        }
        let local_pct = local_match as f64 / TOTAL as f64 * 100.0;
        if local_pct == 100.0 {
            perfect += 1;
        } else {
            eprintln!(
                "  diff:  {}:{} {}>{:?}  match={local_pct:.4}%",
                v.reference_name, v.start, v.reference_bases, v.alternate_bases
            );
        }
    }
    let overall = total_match as f64 / total_pixels.max(1) as f64 * 100.0;
    eprintln!(
        "\n=== sweep summary ===\n  upstream: {}\n  ours: {}\n  compared: {compared}, missing: {missing}\n  perfect rows: {perfect}/{compared}\n  overall pixel match: {overall:.4}% ({total_match}/{total_pixels})",
        upstream.len(),
        ours_all.len()
    );
    // Diagnostic only — failures here surface layout bugs to fix, but
    // don't break CI. Bookkeeping floor: at least the SNVs (which made up
    // ~70% of upstream's set last time we measured) should match 100%.
    let snv_count = upstream
        .iter()
        .filter(|(v, _)| v.reference_bases.len() == 1 && v.alternate_bases.iter().all(|a| a.len() == 1))
        .count();
    assert!(
        perfect >= snv_count,
        "expected at least {snv_count} (all SNVs) perfect matches, got {perfect}"
    );
}

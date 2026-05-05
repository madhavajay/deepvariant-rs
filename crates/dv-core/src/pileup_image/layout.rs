//! Pileup image rasterization.
//!
//! Mirrors `deepvariant/pileup_image_native.cc` +
//! `pileup_channel_lib.cc::CalculateBaseLevelData`. Key behaviors:
//!
//!   - Reference band of `reference_band_height` rows at top, each
//!     row is a copy of the per-column ref-base channel encoding.
//!   - For each read, walk CIGAR and paint columns:
//!       * `M`/`=`/`X`: paint each base at column `ref_i - image_start`
//!       * `INSERT`/`SOFT_CLIP`: paint a single anchor pixel at column
//!         `ref_i - 1 - image_start` using the indel-anchor base char
//!       * `DELETE`/`SKIP`: paint a single anchor pixel at column
//!         `ref_i - image_start` using the indel-anchor base char,
//!         then advance ref_i by op_len
//!       * `HARD_CLIP`/`PAD`: ignored
//!   - If a read has a low-quality base at the candidate variant
//!     position, the read is dropped (returns `None` from encode_read).
//!   - Reads are downsampled with a seeded mt19937_64 shuffle when
//!     coverage exceeds `max_reads = height - reference_band_height`.
//!   - Reads are stable-sorted by `(hap_index, allele_support_group,
//!     alignment_position)` before being placed in image rows.
//!   - Empty rows (when coverage < max_reads) are left as zero-filled.

use crate::pileup_image::channels::{self, ChannelKind};
use crate::pileup_image::options::PileupOptions;

/// Per-read context needed to paint a pileup row.
#[derive(Debug, Clone)]
pub struct PileupRead<'a> {
    pub ref_start: i64,
    pub cigar: &'a [(char, i64)],
    pub seq: &'a [u8],
    pub base_quality: &'a [u8],
    pub mapping_quality: u8,
    pub is_reverse_strand: bool,
    pub fragment_length: i32,
    pub supports_variant: bool,
    pub hp_tag: u8,
    /// Used as the final sort tiebreaker (matches upstream's
    /// `read.fragment_name()`).
    pub fragment_name: &'a str,
    /// Final sort tiebreaker after fragment_name (matches upstream's
    /// `read.read_number()` — typically 0 or 1 for paired reads).
    pub read_number: i32,
}

/// 0-based ref position of the candidate variant; used to drop reads
/// with low-quality bases at the call site (matches upstream's behavior).
#[derive(Debug, Clone, Copy)]
pub struct VariantContext {
    pub variant_pos: i64,
    pub min_base_quality_at_call: u8,
}

#[derive(Debug, Clone, Copy)]
pub enum CigarOp {
    AlignmentMatch,
    Insert,
    SoftClip,
    Delete,
    Skip,
    HardClip,
    Pad,
    SequenceMatch,
    SequenceMismatch,
}

impl CigarOp {
    fn from_char(c: char) -> Option<Self> {
        Some(match c {
            'M' => CigarOp::AlignmentMatch,
            'I' => CigarOp::Insert,
            'S' => CigarOp::SoftClip,
            'D' => CigarOp::Delete,
            'N' => CigarOp::Skip,
            'H' => CigarOp::HardClip,
            'P' => CigarOp::Pad,
            '=' => CigarOp::SequenceMatch,
            'X' => CigarOp::SequenceMismatch,
            _ => return None,
        })
    }
}

/// Result of painting one read row. `None` if the read should be dropped.
fn encode_read_row(
    img: &mut [u8],
    row: usize,
    width: usize,
    _c: usize,
    image_start: i64,
    ref_bases: &[u8],
    read: &PileupRead<'_>,
    channel_kinds: &[ChannelKind],
    opts: &PileupOptions,
    ctx: Option<VariantContext>,
) -> bool {
    let mut ref_i = read.ref_start;
    let mut read_i = 0usize;

    // Helper: paint one column for this read.
    let paint_at = |img: &mut [u8],
                       ref_pos: i64,
                       read_pos: usize,
                       read_base: u8|
     -> Result<(), ()> {
        let col_signed = ref_pos - image_start;
        if col_signed < 0 || (col_signed as usize) >= width {
            return Ok(());
        }
        let col = col_signed as usize;
        let bq = read.base_quality.get(read_pos).copied().unwrap_or(0) as i32;
        // Drop the read entirely if it has a low-qual base at the call site.
        if let Some(c) = ctx {
            if ref_pos == c.variant_pos && (bq as u8) < c.min_base_quality_at_call {
                return Err(());
            }
        }
        let ref_base = ref_bases[col];
        let base_idx = (row * width + col) * c_channels(channel_kinds);
        for (ci, kind) in channel_kinds.iter().enumerate() {
            img[base_idx + ci] = read_pixel(*kind, read_base, ref_base, bq, read, opts);
        }
        Ok(())
    };

    let anchor_char = b'*'; // upstream's `indel_anchoring_base_char`
    for &(op_char, len) in read.cigar {
        let op = match CigarOp::from_char(op_char) {
            Some(o) => o,
            None => continue, // unrecognized op: skip
        };
        let len_us = len as usize;
        match op {
            CigarOp::AlignmentMatch | CigarOp::SequenceMatch | CigarOp::SequenceMismatch => {
                for i in 0..len_us {
                    let rp = ref_i + i as i64;
                    let read_b = read.seq[read_i + i].to_ascii_uppercase();
                    if paint_at(img, rp, read_i + i, read_b).is_err() {
                        return false;
                    }
                }
                ref_i += len;
                read_i += len_us;
            }
            CigarOp::Insert => {
                if ref_i > 0 {
                    let _ = paint_at(img, ref_i - 1, read_i, anchor_char);
                }
                read_i += len_us;
            }
            CigarOp::SoftClip => {
                // Upstream's `CalculateChannels` lambda leaves `read_base = 0`
                // for SOFT_CLIP, which causes the paint check to skip. So
                // soft clips advance read_i but paint nothing.
                read_i += len_us;
            }
            CigarOp::Delete | CigarOp::Skip => {
                // Upstream's `CalculateChannels` decrements ref_i by 1 inside
                // the action lambda for DELETE ops, so the anchor is painted
                // at the column *before* the deletion start, not at it.
                if read_i > 0 && ref_i > 0 {
                    let _ = paint_at(img, ref_i - 1, read_i - 1, anchor_char);
                }
                ref_i += len;
            }
            CigarOp::HardClip | CigarOp::Pad => {}
        }
    }
    true
}

fn c_channels(kinds: &[ChannelKind]) -> usize {
    kinds.len()
}

fn read_pixel(
    kind: ChannelKind,
    read_base: u8,
    ref_base: u8,
    base_q: i32,
    read: &PileupRead<'_>,
    opts: &PileupOptions,
) -> u8 {
    match kind {
        ChannelKind::ReadBase => channels::read_base_read(read_base, opts),
        ChannelKind::BaseQuality => channels::base_quality_read(base_q, opts),
        ChannelKind::MappingQuality => {
            channels::mapping_quality_read(read.mapping_quality as i32, opts)
        }
        ChannelKind::Strand => channels::strand_read(read.is_reverse_strand, opts),
        ChannelKind::ReadSupportsVariant => {
            channels::read_supports_variant_read(read.supports_variant, opts)
        }
        ChannelKind::BaseDiffersFromRef => {
            channels::base_differs_from_ref_read(read_base, ref_base, opts)
        }
        ChannelKind::InsertSize => channels::insert_size_read(read.fragment_length),
    }
}

fn ref_pixel(kind: ChannelKind, ref_base: u8, opts: &PileupOptions) -> u8 {
    match kind {
        ChannelKind::ReadBase => channels::read_base_ref(ref_base, opts),
        ChannelKind::BaseQuality => channels::base_quality_ref(opts),
        ChannelKind::MappingQuality => channels::mapping_quality_ref(opts),
        ChannelKind::Strand => channels::strand_ref(opts),
        ChannelKind::ReadSupportsVariant => channels::read_supports_variant_ref(opts),
        ChannelKind::BaseDiffersFromRef => channels::base_differs_from_ref_ref(opts),
        ChannelKind::InsertSize => channels::insert_size_ref(),
    }
}

/// Sort key for stable ordering of reads in the pileup.
/// Matches upstream's `SortImageRows` in `pileup_image_native.cc:75`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PileupSortKey {
    hap: i32,
    allele_group: i32,
    position: i64,
    fragment_name: String,
    read_number: i32,
}

/// Render the pileup image for one candidate.
///
/// Read selection: shuffle deterministically with a `mt19937_64(seed)`
/// then take the first `height - reference_band_height`. Sort by
/// `(hap_tag, allele_support_group, ref_start)`.
///
/// `image_start` is the 0-based ref position of column 0 of the image.
/// `ref_bases` must have length exactly `width`.
pub fn render(
    image_start: i64,
    width: usize,
    height: usize,
    reference_band_height: usize,
    ref_bases: &[u8],
    reads: &[PileupRead<'_>],
    channel_kinds: &[ChannelKind],
    opts: &PileupOptions,
    ctx: Option<VariantContext>,
    random_seed: u64,
) -> Vec<u8> {
    debug_assert_eq!(ref_bases.len(), width);
    debug_assert!(reference_band_height < height);
    let c = channel_kinds.len();
    let mut img = vec![0u8; height * width * c];

    // Reference rows.
    for row in 0..reference_band_height {
        for col in 0..width {
            let r = ref_bases[col];
            let base_idx = (row * width + col) * c;
            for (ci, kind) in channel_kinds.iter().enumerate() {
                img[base_idx + ci] = ref_pixel(*kind, r, opts);
            }
        }
    }

    // Read rows: select up to (height - ref_band) reads with a deterministic shuffle.
    let max_reads = height - reference_band_height;
    let selected: Vec<usize> = if reads.len() <= max_reads {
        (0..reads.len()).collect()
    } else {
        // Deterministic mt19937_64 shuffle of indices.
        let mut indices: Vec<usize> = (0..reads.len()).collect();
        deterministic_shuffle(&mut indices, random_seed);
        indices.truncate(max_reads);
        indices
    };

    // Encode each selected read into a temporary row buffer and collect the
    // sort keys; drop reads whose call-site base is too low quality.
    struct RowBuf {
        key: PileupSortKey,
        pixels: Vec<u8>,
    }
    let mut row_bufs: Vec<RowBuf> = Vec::with_capacity(selected.len());
    let mut tmp_row = vec![0u8; width * c];
    for &idx in &selected {
        let read = &reads[idx];
        // We render into the start of `img` at row=0 then copy out — but to
        // avoid clobbering the ref rows we use a separate `tmp_row` of width
        // pixels and then copy.
        for px in &mut tmp_row {
            *px = 0;
        }
        // We use a fake `img` that's just `tmp_row` to drive encode_read_row.
        let ok = encode_read_row(
            &mut tmp_row,
            0,
            width,
            c,
            image_start,
            ref_bases,
            read,
            channel_kinds,
            opts,
            ctx,
        );
        if !ok {
            continue;
        }
        row_bufs.push(RowBuf {
            key: PileupSortKey {
                hap: read.hp_tag as i32,
                // Upstream's `sort_by_alt_allele_support` defaults to false,
                // so by default all reads share the same allele_group and
                // sort is purely by (hap, position, name, read#).
                allele_group: 0,
                position: read.ref_start,
                fragment_name: read.fragment_name.to_string(),
                read_number: read.read_number,
            },
            pixels: tmp_row.clone(),
        });
    }

    // Stable sort by sort key.
    row_bufs.sort_by(|a, b| a.key.cmp(&b.key));

    for (i, rb) in row_bufs.iter().enumerate().take(max_reads) {
        let row = reference_band_height + i;
        let dst_start = row * width * c;
        img[dst_start..dst_start + width * c].copy_from_slice(&rb.pixels);
    }

    img
}

/// Deterministic Fisher-Yates shuffle using the same algorithm shape as
/// `std::shuffle` over `std::mt19937_64`. Picks indices via modulo —
/// matches libstdc++/libc++'s common implementation strategy. Exact pixel
/// parity with upstream's specific shuffle byte-stream may require
/// further tuning.
fn deterministic_shuffle(indices: &mut [usize], seed: u64) {
    use rand_mt::Mt64;
    let mut rng = Mt64::new(seed);
    let n = indices.len();
    if n < 2 {
        return;
    }
    // Fisher-Yates: for i from n-1 down to 1, swap indices[i] with indices[j]
    // where j is uniform in [0, i].
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read<'a>(
        ref_start: i64,
        cigar: &'a [(char, i64)],
        seq: &'a [u8],
        bq: &'a [u8],
        supports: bool,
    ) -> PileupRead<'a> {
        PileupRead {
            ref_start,
            cigar,
            seq,
            base_quality: bq,
            mapping_quality: 60,
            is_reverse_strand: false,
            fragment_length: 300,
            supports_variant: supports,
            hp_tag: 0,
            fragment_name: "test",
            read_number: 0,
        }
    }

    #[test]
    fn renders_dimensions() {
        let opts = PileupOptions::default();
        let kinds = [
            ChannelKind::ReadBase,
            ChannelKind::BaseQuality,
            ChannelKind::MappingQuality,
            ChannelKind::Strand,
            ChannelKind::ReadSupportsVariant,
            ChannelKind::BaseDiffersFromRef,
            ChannelKind::InsertSize,
        ];
        let bq = [40u8; 10];
        let reads = vec![read(100, &[('M', 10)], b"AAAAACAAAA", &bq, true)];
        let img = render(100, 10, 100, 5, b"AAAAAAAAAA", &reads, &kinds, &opts, None, 42);
        assert_eq!(img.len(), 100 * 10 * 7);
    }

    #[test]
    fn ref_rows_have_ref_pixel() {
        let opts = PileupOptions::default();
        let kinds = [ChannelKind::ReadBase];
        let img = render(100, 4, 8, 4, b"ACGT", &[], &kinds, &opts, None, 0);
        for row in 0..4 {
            for col in 0..4 {
                let want = channels::base_color(b"ACGT"[col], &opts);
                assert_eq!(img[row * 4 + col], want, "row {row} col {col}");
            }
        }
    }

    #[test]
    fn read_row_paints_match_ops() {
        let opts = PileupOptions::default();
        let kinds = [ChannelKind::ReadBase];
        let bq = [40u8; 4];
        let reads = vec![read(0, &[('M', 4)], b"ACGT", &bq, false)];
        let img = render(0, 4, 6, 0, b"AAAA", &reads, &kinds, &opts, None, 0);
        for (col, &b) in b"ACGT".iter().enumerate() {
            let got = img[col];
            assert_eq!(got, channels::base_color(b, &opts));
        }
    }

    #[test]
    fn deletion_renders_one_anchor_pixel() {
        // 2M3D2M: read covers cols 0,1 (AA), the delete anchor goes at the
        // column *before* the deletion start (col 1, overwriting the prior
        // M paint with anchor=0 in read_base channel), then cols 5,6 (GG).
        // Cols 2,3,4 stay zero (within the deletion span).
        let opts = PileupOptions::default();
        let kinds = [ChannelKind::ReadBase];
        let bq = [40u8; 4];
        let reads = vec![read(0, &[('M', 2), ('D', 3), ('M', 2)], b"AAGG", &bq, false)];
        let img = render(0, 7, 6, 0, b"AAAAAAA", &reads, &kinds, &opts, None, 0);
        // Col 0: A pixel
        assert_eq!(img[0], channels::base_color(b'A', &opts));
        // Col 1: anchor (overwrites the M paint because anchor is at ref_i-1)
        assert_eq!(img[1], 0);
        // Cols 2,3,4: untouched (deletion span, zero)
        assert_eq!(img[2], 0);
        assert_eq!(img[3], 0);
        assert_eq!(img[4], 0);
        // Cols 5,6: G pixel
        assert_eq!(img[5], channels::base_color(b'G', &opts));
        assert_eq!(img[6], channels::base_color(b'G', &opts));
    }

    #[test]
    fn insertion_paints_anchor_at_prior_column() {
        // 2M3I2M: cols 0,1 (AA), then anchor at col 1 (ref_i-1=1) for the
        // insertion, then cols 2,3 (GG). Note: anchor at col 1 OVERWRITES the
        // M paint at col 1 — upstream lets this happen too because the for
        // loop fills columns in CIGAR order.
        let opts = PileupOptions::default();
        let kinds = [ChannelKind::ReadBase];
        let bq = [40u8; 7];
        let reads = vec![read(0, &[('M', 2), ('I', 3), ('M', 2)], b"AACCCGG", &bq, false)];
        let img = render(0, 4, 6, 0, b"AAAA", &reads, &kinds, &opts, None, 0);
        // Col 0: A pixel
        assert_eq!(img[0], channels::base_color(b'A', &opts));
        // Col 1: anchor (overwritten by I op at ref_i-1=1)
        assert_eq!(img[1], 0);
        // Cols 2,3: G pixel
        assert_eq!(img[2], channels::base_color(b'G', &opts));
        assert_eq!(img[3], channels::base_color(b'G', &opts));
    }

    #[test]
    fn low_qual_at_call_site_drops_read() {
        let opts = PileupOptions::default();
        let kinds = [ChannelKind::ReadBase];
        let bq = vec![40u8, 40, 5, 40]; // pos 2 is low quality
        let reads = vec![read(0, &[('M', 4)], b"ACGT", &bq, false)];
        let ctx = Some(VariantContext {
            variant_pos: 2,
            min_base_quality_at_call: 10,
        });
        let img = render(0, 4, 6, 0, b"AAAA", &reads, &kinds, &opts, ctx, 0);
        // Read should be dropped → row 0 all zero.
        for col in 0..4 {
            assert_eq!(img[col], 0, "col {col}");
        }
    }

    #[test]
    fn deterministic_seed_is_stable() {
        let opts = PileupOptions::default();
        let kinds = [ChannelKind::ReadBase];
        let bq = [40u8; 10];
        let reads: Vec<PileupRead<'_>> = (0..50)
            .map(|i| read(i, &[('M', 10)], b"ACGTACGTAC", &bq, false))
            .collect();
        let img1 = render(0, 50, 20, 5, &vec![b'A'; 50], &reads, &kinds, &opts, None, 12345);
        let img2 = render(0, 50, 20, 5, &vec![b'A'; 50], &reads, &kinds, &opts, None, 12345);
        assert_eq!(img1, img2, "same seed → same image");
    }
}

//! Per-channel pixel encoders for pileup images.
//!
//! Each channel produces a single byte per column (0..=254) for both
//! the "read row" and the "ref row". Each function is a port of one
//! file under `deepvariant/channels/*_channel.cc`.
//!
//! Channels currently ported (covers WGS default channel set
//! `[1, 2, 3, 4, 5, 6, 19]`, except #19 insert_size):
//!   - 1: read_base
//!   - 2: base_quality
//!   - 3: mapping_quality
//!   - 4: strand
//!   - 5: read_supports_variant
//!   - 6: base_differs_from_ref

use crate::pileup_image::options::PileupOptions;

pub const MAX_PIXEL_VALUE: f32 = 254.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    ReadBase,
    BaseQuality,
    MappingQuality,
    Strand,
    ReadSupportsVariant,
    BaseDiffersFromRef,
    InsertSize,
}

impl ChannelKind {
    /// Map a `DeepVariantChannelEnum` integer to a known channel.
    /// Returns `None` for channels we haven't ported yet.
    pub fn from_proto_index(idx: i32) -> Option<Self> {
        Some(match idx {
            1 => Self::ReadBase,
            2 => Self::BaseQuality,
            3 => Self::MappingQuality,
            4 => Self::Strand,
            5 => Self::ReadSupportsVariant,
            6 => Self::BaseDiffersFromRef,
            19 => Self::InsertSize,
            _ => return None,
        })
    }
}

/// Channel 19: insert_size. Read row = scaled `|fragment_length|`,
/// capped at `MAX_FRAGMENT_LENGTH = 1000`. Ref row = MAX (254).
pub const MAX_FRAGMENT_LENGTH: i32 = 1000;

pub fn insert_size_read(fragment_length: i32) -> u8 {
    let len = fragment_length.unsigned_abs() as i32;
    let len = len.min(MAX_FRAGMENT_LENGTH);
    (MAX_PIXEL_VALUE * (len as f32 / MAX_FRAGMENT_LENGTH as f32)) as u8
}
pub fn insert_size_ref() -> u8 {
    MAX_PIXEL_VALUE as u8
}

/// Channel 12: avg_base_quality. Mean of `aligned_quality` capped at 93.
pub const MAX_AVG_BASE_QUALITY: i32 = 93;

pub fn avg_base_quality_read(aligned_quality: &[u8]) -> u8 {
    if aligned_quality.is_empty() {
        return 0;
    }
    let sum: u32 = aligned_quality.iter().map(|&q| q as u32).sum();
    let avg = (sum / aligned_quality.len() as u32) as i32;
    scale_color(avg, MAX_AVG_BASE_QUALITY)
}
pub fn avg_base_quality_ref() -> u8 {
    MAX_PIXEL_VALUE as u8
}

/// Channel 15: gc_content. GC fraction of `aligned_sequence` × 100,
/// capped at 100.
pub const MAX_GC_CONTENT: i32 = 100;

pub fn gc_content_read(aligned_sequence: &[u8]) -> u8 {
    if aligned_sequence.is_empty() {
        return 0;
    }
    let gc = aligned_sequence
        .iter()
        .filter(|b| matches!(**b, b'G' | b'C' | b'g' | b'c'))
        .count();
    let pct = ((gc as f32 / aligned_sequence.len() as f32) * 100.0) as i32;
    scale_color(pct, MAX_GC_CONTENT)
}
pub fn gc_content_ref(ref_bases: &[u8]) -> u8 {
    gc_content_read(ref_bases)
}

/// Channel 13: identity. `match_count / read_len * 100` scaled to 0..254.
/// `match_count` sums lengths of `M` and `=` ops only (mismatches/indels
/// don't count).
pub fn identity_read<I>(cigar: I, read_len: usize) -> u8
where
    I: IntoIterator<Item = (char, i64)>,
{
    if read_len == 0 {
        return 0;
    }
    let mut matches = 0i64;
    for (op, len) in cigar {
        if op == 'M' || op == '=' {
            matches += len;
        }
    }
    let pct = (matches as f32 / read_len as f32 * 100.0) as i32;
    scale_color(pct, 100)
}
pub fn identity_ref() -> u8 {
    MAX_PIXEL_VALUE as u8
}

/// Channel 18: blank. Always 0.
pub fn blank_read() -> u8 {
    0
}
pub fn blank_ref() -> u8 {
    0
}

/// Channel 7: haplotype_tag. `HP` SAM tag value (0/1/2) → distinct
/// pixel values. None of these are calibrated against a model spec —
/// upstream uses fixed colors per HP value.
pub fn haplotype_tag_read(hp_tag: u8, opts: &PileupOptions) -> u8 {
    match hp_tag {
        1 => MAX_PIXEL_VALUE as u8,
        2 => (MAX_PIXEL_VALUE * 0.5) as u8,
        _ => opts.positive_strand_color as u8, // No HP tag → neutral
    }
}
pub fn haplotype_tag_ref(opts: &PileupOptions) -> u8 {
    opts.positive_strand_color as u8
}

/// Channel 8: allele_frequency. Maps `f ∈ (min_nonzero_af, 1.0]` to a
/// log-scale pixel: `((log10(min_nz) - log10(f)) / log10(min_nz)) * 254`.
/// Values at or below the floor go to 0.
pub fn allele_frequency_read(allele_freq: f32, min_nonzero_af: f32) -> u8 {
    if allele_freq <= min_nonzero_af || min_nonzero_af <= 0.0 {
        return 0;
    }
    let log10_af = allele_freq.log10();
    let log10_min = min_nonzero_af.log10();
    let v = ((log10_min - log10_af) / log10_min) * MAX_PIXEL_VALUE;
    v.clamp(0.0, MAX_PIXEL_VALUE) as u8
}
pub fn allele_frequency_ref(min_nonzero_af: f32) -> u8 {
    allele_frequency_read(0.0, min_nonzero_af)
}

/// Channel 22: mean_coverage. Encoded as a per-row constant rather than a
/// per-pixel value upstream — read-row pixel = scaled coverage, ref-row
/// pixel = 254. We model it the same way: `read` returns the scaled value,
/// `ref` returns saturated.
pub fn mean_coverage_read(coverage: i32, max_coverage: i32) -> u8 {
    scale_color(coverage, max_coverage.max(1))
}
pub fn mean_coverage_ref() -> u8 {
    MAX_PIXEL_VALUE as u8
}

/// Channel 11: read_mapping_percent. `(read_aligned_length / read_total_length) * 100`,
/// scaled to 0–254.
pub fn read_mapping_percent_read(aligned_len: i32, total_len: i32) -> u8 {
    if total_len <= 0 {
        return 0;
    }
    let pct = ((aligned_len as f32 / total_len as f32) * 100.0) as i32;
    scale_color(pct, 100)
}
pub fn read_mapping_percent_ref() -> u8 {
    MAX_PIXEL_VALUE as u8
}

/// Channel 17: homopolymer_weighted. For each position, the length of the
/// homopolymer run it belongs to. E.g. `ATCGGGAA` → `[1, 1, 1, 3, 3, 3, 2, 2]`.
/// Cap at 30, scale to 0..254.
pub const MAX_HOMOPOLYMER_WEIGHTED: i32 = 30;

pub fn homopolymer_weights(seq: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; seq.len()];
    if seq.is_empty() {
        return out;
    }
    let mut weight = 1i32;
    for i in 1..seq.len() {
        if seq[i] == seq[i - 1] {
            weight += 1;
        } else {
            for j in 0..weight {
                out[i - 1 - j as usize] = weight.min(255) as u8;
            }
            weight = 1;
        }
    }
    for j in 0..weight {
        out[seq.len() - 1 - j as usize] = weight.min(255) as u8;
    }
    out
}

pub fn homopolymer_weighted_read_at(seq: &[u8], read_pos: usize) -> u8 {
    let weights = homopolymer_weights(seq);
    let v = weights.get(read_pos).copied().unwrap_or(0) as i32;
    scale_color(v, MAX_HOMOPOLYMER_WEIGHTED)
}
pub fn homopolymer_weighted_ref_at(ref_bases: &[u8], col: usize) -> u8 {
    homopolymer_weighted_read_at(ref_bases, col)
}

/// Channel 16: is_homopolymer. 254 if position is part of a homopolymer
/// run of length >= 2, else 0.
pub fn is_homopolymer_read_at(seq: &[u8], read_pos: usize) -> u8 {
    let weights = homopolymer_weights(seq);
    if weights.get(read_pos).copied().unwrap_or(0) >= 2 {
        MAX_PIXEL_VALUE as u8
    } else {
        0
    }
}
pub fn is_homopolymer_ref_at(ref_bases: &[u8], col: usize) -> u8 {
    is_homopolymer_read_at(ref_bases, col)
}

/// Channel 14: gap_compressed_identity. Treats consecutive
/// insertion/deletion bases as a single mismatch (= "gap-compressed").
/// `matches / (matches + mismatches + indel_runs) * 100`, scaled.
pub fn gap_compressed_identity_read<I>(cigar: I) -> u8
where
    I: IntoIterator<Item = (char, i64)>,
{
    let mut matches = 0i64;
    let mut mismatches = 0i64;
    let mut indel_runs = 0i64;
    for (op, len) in cigar {
        match op {
            'M' | '=' => matches += len,
            'X' => mismatches += len,
            'I' | 'D' => indel_runs += 1,
            _ => {}
        }
    }
    let denom = matches + mismatches + indel_runs;
    if denom == 0 {
        return 0;
    }
    let pct = ((matches as f32 / denom as f32) * 100.0) as i32;
    scale_color(pct, 100)
}
pub fn gap_compressed_identity_ref() -> u8 {
    MAX_PIXEL_VALUE as u8
}

#[inline]
fn scale_color(value: i32, max_val: i32) -> u8 {
    let v = value.min(max_val).max(0) as f32;
    (MAX_PIXEL_VALUE * (v / max_val as f32)) as u8
}

/// Map a base character (uppercase) to its color.
pub fn base_color(base: u8, opts: &PileupOptions) -> u8 {
    match base {
        b'A' => (opts.base_color_offset_a_and_g + opts.base_color_stride * 3) as u8,
        b'G' => (opts.base_color_offset_a_and_g + opts.base_color_stride * 2) as u8,
        b'T' => (opts.base_color_offset_t_and_c + opts.base_color_stride) as u8,
        b'C' => opts.base_color_offset_t_and_c as u8,
        _ => 0,
    }
}

#[inline]
fn alpha_to_pixel(alpha: f32) -> u8 {
    (MAX_PIXEL_VALUE * alpha) as u8
}

/// Channel 1: read_base. Read row = color of the read's base.
/// Ref row = color of the ref base.
pub fn read_base_read(read_base: u8, opts: &PileupOptions) -> u8 {
    base_color(read_base, opts)
}
pub fn read_base_ref(ref_base: u8, opts: &PileupOptions) -> u8 {
    base_color(ref_base, opts)
}

/// Channel 2: base_quality. Read row = scaled base quality.
/// Ref row = scaled `reference_base_quality`.
pub fn base_quality_read(base_quality: i32, opts: &PileupOptions) -> u8 {
    scale_color(base_quality, opts.base_quality_cap)
}
pub fn base_quality_ref(opts: &PileupOptions) -> u8 {
    scale_color(opts.reference_base_quality, opts.base_quality_cap)
}

/// Channel 3: mapping_quality. Read row = scaled MAPQ.
/// Ref row = scaled `reference_base_quality` against `base_quality_cap`
/// (matches upstream's quirk: ref row uses base_quality_cap, not mapq_cap).
pub fn mapping_quality_read(mapping_quality: i32, opts: &PileupOptions) -> u8 {
    scale_color(mapping_quality, opts.mapping_quality_cap)
}
pub fn mapping_quality_ref(opts: &PileupOptions) -> u8 {
    scale_color(opts.reference_base_quality, opts.base_quality_cap)
}

/// Channel 4: strand. positive_strand_color or negative_strand_color
/// depending on the read's reverse-strand flag. Ref row = positive.
pub fn strand_read(is_reverse_strand: bool, opts: &PileupOptions) -> u8 {
    (if is_reverse_strand {
        opts.negative_strand_color
    } else {
        opts.positive_strand_color
    }) as u8
}
pub fn strand_ref(opts: &PileupOptions) -> u8 {
    opts.positive_strand_color as u8
}

/// Channel 5: read_supports_variant. `supports` flag derived externally.
pub fn read_supports_variant_read(supports: bool, opts: &PileupOptions) -> u8 {
    alpha_to_pixel(if supports {
        opts.allele_supporting_read_alpha
    } else {
        opts.allele_unsupporting_read_alpha
    })
}
pub fn read_supports_variant_ref(opts: &PileupOptions) -> u8 {
    alpha_to_pixel(opts.allele_unsupporting_read_alpha)
}

/// Channel 6: base_differs_from_ref. Matching → reference_matching_alpha,
/// mismatching → reference_mismatching_alpha. Ref row = matching.
pub fn base_differs_from_ref_read(read_base: u8, ref_base: u8, opts: &PileupOptions) -> u8 {
    let alpha = if read_base.eq_ignore_ascii_case(&ref_base) {
        opts.reference_matching_read_alpha
    } else {
        opts.reference_mismatching_read_alpha
    };
    alpha_to_pixel(alpha)
}
pub fn base_differs_from_ref_ref(opts: &PileupOptions) -> u8 {
    alpha_to_pixel(opts.reference_matching_read_alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pileup_image::options::PileupOptions;

    fn opts() -> PileupOptions {
        PileupOptions::default()
    }

    #[test]
    fn base_colors_match_upstream_defaults() {
        let o = opts();
        assert_eq!(base_color(b'A', &o), 250);
        assert_eq!(base_color(b'G', &o), 180);
        assert_eq!(base_color(b'T', &o), 100);
        assert_eq!(base_color(b'C', &o), 30);
        assert_eq!(base_color(b'N', &o), 0);
    }

    #[test]
    fn base_quality_scales_to_pixel() {
        let o = opts();
        assert_eq!(base_quality_read(40, &o), 254);
        assert_eq!(base_quality_read(0, &o), 0);
        assert_eq!(base_quality_read(20, &o), 127);
        // Cap exceeded
        assert_eq!(base_quality_read(99, &o), 254);
    }

    #[test]
    fn mapping_quality_scales_to_pixel() {
        let o = opts();
        assert_eq!(mapping_quality_read(60, &o), 254);
        assert_eq!(mapping_quality_read(30, &o), 127);
    }

    #[test]
    fn strand_colors() {
        let o = opts();
        assert_eq!(strand_read(false, &o), 70);
        assert_eq!(strand_read(true, &o), 240);
        assert_eq!(strand_ref(&o), 70);
    }

    #[test]
    fn supports_variant_alphas() {
        let o = opts();
        assert_eq!(read_supports_variant_read(true, &o), 254); // alpha=1.0
        assert_eq!(read_supports_variant_read(false, &o), 152); // alpha=0.6 → 152.4
        assert_eq!(read_supports_variant_ref(&o), 152);
    }

    #[test]
    fn base_differs_from_ref_alphas() {
        let o = opts();
        // Match: alpha=0.2 → 50.8 → 50
        assert_eq!(base_differs_from_ref_read(b'A', b'A', &o), 50);
        // Mismatch: alpha=1.0 → 254
        assert_eq!(base_differs_from_ref_read(b'C', b'A', &o), 254);
        // Case-insensitive
        assert_eq!(base_differs_from_ref_read(b'a', b'A', &o), 50);
        assert_eq!(base_differs_from_ref_ref(&o), 50);
    }

    #[test]
    fn insert_size_normalizes_to_pixel() {
        // Cap at 1000.
        assert_eq!(insert_size_read(0), 0);
        assert_eq!(insert_size_read(500), 127);
        assert_eq!(insert_size_read(1000), 254);
        assert_eq!(insert_size_read(2000), 254); // capped
        // Negative values count their absolute.
        assert_eq!(insert_size_read(-500), 127);
        assert_eq!(insert_size_ref(), 254);
    }

    #[test]
    fn channel_kind_proto_indices() {
        assert_eq!(ChannelKind::from_proto_index(1), Some(ChannelKind::ReadBase));
        assert_eq!(ChannelKind::from_proto_index(6), Some(ChannelKind::BaseDiffersFromRef));
        assert_eq!(ChannelKind::from_proto_index(19), Some(ChannelKind::InsertSize));
        assert_eq!(ChannelKind::from_proto_index(99), None);
    }

    #[test]
    fn avg_base_quality_means_correctly() {
        let q = [40u8, 30, 30, 30]; // mean = 32
        assert_eq!(avg_base_quality_read(&q), scale_color(32, MAX_AVG_BASE_QUALITY));
        assert_eq!(avg_base_quality_read(&[]), 0);
        assert_eq!(avg_base_quality_ref(), 254);
    }

    #[test]
    fn gc_content_examples() {
        assert_eq!(gc_content_read(b"AAAA"), 0); // 0%
        assert_eq!(gc_content_read(b"GCGC"), 254); // 100% → cap
        assert_eq!(gc_content_read(b"AAGC"), 127); // 50%
        assert_eq!(gc_content_read(b""), 0);
        // Lowercase
        assert_eq!(gc_content_read(b"gcGC"), 254);
    }

    #[test]
    fn identity_examples() {
        // 5M = full match → 100%
        let cigar = vec![('M', 5)];
        assert_eq!(identity_read(cigar, 5), 254);
        // 3M2I = 60% match
        let cigar = vec![('M', 3), ('I', 2)];
        assert_eq!(identity_read(cigar, 5), scale_color(60, 100));
        // = ops also count
        let cigar = vec![('=', 4)];
        assert_eq!(identity_read(cigar, 4), 254);
        assert_eq!(identity_read(vec![], 0), 0);
    }

    #[test]
    fn blank_is_zero() {
        assert_eq!(blank_read(), 0);
        assert_eq!(blank_ref(), 0);
    }

    #[test]
    fn haplotype_tag_examples() {
        let o = opts();
        assert_eq!(haplotype_tag_read(1, &o), 254);
        assert_eq!(haplotype_tag_read(2, &o), 127);
        // No HP → neutral (positive_strand_color = 70)
        assert_eq!(haplotype_tag_read(0, &o), 70);
        assert_eq!(haplotype_tag_ref(&o), 70);
    }

    #[test]
    fn allele_frequency_log_scaling() {
        let min = 0.001;
        // Below floor → 0
        assert_eq!(allele_frequency_read(0.0, min), 0);
        assert_eq!(allele_frequency_read(0.001, min), 0);
        // At max (af=1.0): log10(1)=0 → (log10(min)-0)/log10(min)=1 → 254
        assert_eq!(allele_frequency_read(1.0, min), 254);
        // Mid: af=0.1, log10=-1, log10(min)=-3, (-3-(-1))/-3 = 2/3 → 169
        let v = allele_frequency_read(0.1, min);
        assert!((160..=180).contains(&v), "got {v}");
        assert_eq!(allele_frequency_ref(min), 0);
    }

    #[test]
    fn mean_coverage_examples() {
        assert_eq!(mean_coverage_read(0, 100), 0);
        assert_eq!(mean_coverage_read(50, 100), 127);
        assert_eq!(mean_coverage_read(100, 100), 254);
        assert_eq!(mean_coverage_read(200, 100), 254); // clamped
        assert_eq!(mean_coverage_ref(), 254);
    }

    #[test]
    fn read_mapping_percent_examples() {
        assert_eq!(read_mapping_percent_read(50, 100), 127);
        assert_eq!(read_mapping_percent_read(100, 100), 254);
        assert_eq!(read_mapping_percent_read(0, 100), 0);
        assert_eq!(read_mapping_percent_read(50, 0), 0);
        assert_eq!(read_mapping_percent_ref(), 254);
    }

    #[test]
    fn gap_compressed_identity_examples() {
        // 5M = 100% match
        assert_eq!(gap_compressed_identity_read(vec![('M', 5)]), 254);
        // 5M2I → 5 matches + 1 indel run = 5/6 ≈ 83 → ≈211
        let v = gap_compressed_identity_read(vec![('M', 5), ('I', 2)]);
        assert!((200..=220).contains(&v), "got {v}");
        // 4M5X = 4/(4+5) = 44 → ≈111
        let v = gap_compressed_identity_read(vec![('M', 4), ('X', 5)]);
        assert!((100..=120).contains(&v), "got {v}");
        assert_eq!(gap_compressed_identity_ref(), 254);
    }

    #[test]
    fn homopolymer_weights_examples() {
        // ATCGGGAA -> 1,1,1,3,3,3,2,2
        assert_eq!(homopolymer_weights(b"ATCGGGAA"), vec![1, 1, 1, 3, 3, 3, 2, 2]);
        assert_eq!(homopolymer_weights(b"AAAA"), vec![4, 4, 4, 4]);
        assert!(homopolymer_weights(b"").is_empty());
        assert_eq!(homopolymer_weights(b"A"), vec![1]);
        assert_eq!(homopolymer_weights(b"ATAT"), vec![1, 1, 1, 1]);
    }

    #[test]
    fn homopolymer_weighted_pixel() {
        let seq = b"ATCGGGAA";
        // run len 1 → 254 * 1/30 ≈ 8
        assert_eq!(homopolymer_weighted_read_at(seq, 0), 8);
        // run len 3 → 254 * 3/30 ≈ 25
        assert_eq!(homopolymer_weighted_read_at(seq, 3), 25);
        // run len 2 → 254 * 2/30 ≈ 16
        assert_eq!(homopolymer_weighted_read_at(seq, 6), 16);
        assert_eq!(homopolymer_weighted_read_at(seq, 100), 0);
    }

    #[test]
    fn is_homopolymer_pixel() {
        let seq = b"ATCGGGAA";
        assert_eq!(is_homopolymer_read_at(seq, 0), 0);
        assert_eq!(is_homopolymer_read_at(seq, 1), 0);
        assert_eq!(is_homopolymer_read_at(seq, 3), 254);
        assert_eq!(is_homopolymer_read_at(seq, 6), 254);
    }
}

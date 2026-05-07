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
    /// Channel #9. Computed by re-rendering the pileup with reads
    /// realigned to alt-haplotype #1 and taking the
    /// `BaseDiffersFromRef` channel of that re-render. Populated by
    /// `pileup_image::layout::render_alt_aligned_channel` rather than
    /// the per-pixel encoders.
    DiffChannelsAlternateAllele1,
    /// Channel #10. Same as `DiffChannelsAlternateAllele1` but for the
    /// second alt allele in a multi-allelic candidate; for a biallelic
    /// candidate this duplicates allele-1.
    DiffChannelsAlternateAllele2,
    /// Channel #20. Re-rendered `ReadBase` against alt-haplotype #1.
    BaseChannelsAlternateAllele1,
    /// Channel #21. Re-rendered `ReadBase` against alt-haplotype #2.
    BaseChannelsAlternateAllele2,
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
            9 => Self::DiffChannelsAlternateAllele1,
            10 => Self::DiffChannelsAlternateAllele2,
            19 => Self::InsertSize,
            20 => Self::BaseChannelsAlternateAllele1,
            21 => Self::BaseChannelsAlternateAllele2,
            _ => return None,
        })
    }

    /// Whether this channel is filled by the alt-aligned re-render
    /// path (`render_alt_aligned_channel`) rather than the per-pixel
    /// encoders in the main `render` call.
    pub fn is_alt_aligned(self) -> bool {
        matches!(
            self,
            Self::DiffChannelsAlternateAllele1
                | Self::DiffChannelsAlternateAllele2
                | Self::BaseChannelsAlternateAllele1
                | Self::BaseChannelsAlternateAllele2
        )
    }

    /// For an alt-aligned channel, which alt allele index it tracks
    /// (0 for `_1`, 1 for `_2`). Panics for non-alt-aligned channels;
    /// callers should gate on `is_alt_aligned`.
    pub fn alt_index(self) -> usize {
        match self {
            Self::DiffChannelsAlternateAllele1 | Self::BaseChannelsAlternateAllele1 => 0,
            Self::DiffChannelsAlternateAllele2 | Self::BaseChannelsAlternateAllele2 => 1,
            _ => panic!("alt_index called on non-alt-aligned channel"),
        }
    }

    /// For an alt-aligned channel, the per-pixel encoder used during
    /// the alt-haplotype re-render. Diff channels reuse the
    /// `BaseDiffersFromRef` encoder; base channels reuse `ReadBase`.
    pub fn alt_aligned_underlying(self) -> Option<Self> {
        Some(match self {
            Self::DiffChannelsAlternateAllele1 | Self::DiffChannelsAlternateAllele2 => {
                Self::BaseDiffersFromRef
            }
            Self::BaseChannelsAlternateAllele1 | Self::BaseChannelsAlternateAllele2 => {
                Self::ReadBase
            }
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
/// run of length >= 3, else 0. Threshold matches upstream's
/// `IsHomopolymerChannel::IsHomopolymer` (looks at 3-base windows).
pub fn is_homopolymer_read_at(seq: &[u8], read_pos: usize) -> u8 {
    let weights = homopolymer_weights(seq);
    if weights.get(read_pos).copied().unwrap_or(0) >= 3 {
        MAX_PIXEL_VALUE as u8
    } else {
        0
    }
}
pub fn is_homopolymer_ref_at(ref_bases: &[u8], col: usize) -> u8 {
    is_homopolymer_read_at(ref_bases, col)
}

/// Channel 23 (`base_methylation`) and Channel 24 (`base_6mA`).
///
/// Both share the same encoder: a per-read modification probability vector
/// `mod_probs` (raw byte values 0..=255 from the BAM `MM/ML` tags) is
/// linearly rescaled into pixel space against `255` (matching upstream's
/// `ScaleColorVector(.., 255)`). Ref row is always 0.
///
/// Caller is responsible for selecting the right vector: `5mC` for #23,
/// `6mA` for #24.
pub fn base_methylation_read_at(mod_probs: Option<&[u8]>, read_index: usize) -> u8 {
    let probs = match mod_probs {
        Some(p) if !p.is_empty() => p,
        _ => return 0,
    };
    let v = probs.get(read_index).copied().unwrap_or(0) as i32;
    // Upstream scales against max_val=255, not 254.
    let v = v.min(255) as f32;
    (MAX_PIXEL_VALUE * (v / 255.0)) as u8
}
pub fn base_methylation_ref() -> u8 {
    0
}
pub fn base_6ma_read_at(mod_probs: Option<&[u8]>, read_index: usize) -> u8 {
    base_methylation_read_at(mod_probs, read_index)
}
pub fn base_6ma_ref() -> u8 {
    0
}

/// Channel 26: supplementary_alignment. Read-row = scaled
/// `allele_supporting_read_alpha` if the read carries the SAM
/// supplementary flag, else `allele_unsupporting_read_alpha`. Ref row =
/// raw `allele_unsupporting_read_alpha` (yes — upstream really does drop
/// the alpha→pixel scaling on the ref row, see
/// supplementary_alignment_channel.cc:63).
pub fn supplementary_alignment_read(is_supplementary: bool, opts: &PileupOptions) -> u8 {
    let alpha = if is_supplementary {
        opts.allele_supporting_read_alpha
    } else {
        opts.allele_unsupporting_read_alpha
    };
    alpha_to_pixel(alpha)
}
pub fn supplementary_alignment_ref(opts: &PileupOptions) -> u8 {
    opts.allele_unsupporting_read_alpha as u8
}

/// Channel 27: allele_sample_probability. `sqrt(supporting / total) * 254`.
/// Square root weights low probabilities more. Ref row = 0.
pub fn allele_sample_probability_read(supporting: i32, total: i32) -> u8 {
    if total <= 0 {
        return 0;
    }
    let v = supporting.clamp(0, total) as f32;
    let prob = (v / total as f32).sqrt();
    (MAX_PIXEL_VALUE * prob) as u8
}
pub fn allele_sample_probability_ref() -> u8 {
    0
}

/// Channel 25: read_supports_variant_fuzzy. The caller computes a "support
/// code" using upstream's logic (see `read_supports_variant_fuzzy_channel.cc`):
///   * 0 → no support → unsupporting alpha
///   * 1 → exact alt support → supporting alpha
///   * 2 → other-alt support → other-allele-supporting alpha
///   * 8/9/10 → fuzzy match within 3/2/1bp → 0.70/0.80/0.90
///
/// This module just maps the code to a pixel; the heavy phasing-aware
/// classification belongs in the candidate-iteration pass that has access
/// to the `DeepVariantCall` and `HP/PS` tags.
pub const FUZZY_SUPPORT_NONE: i32 = 0;
pub const FUZZY_SUPPORT_EXACT: i32 = 1;
pub const FUZZY_SUPPORT_OTHER_ALT: i32 = 2;
pub const FUZZY_SUPPORT_3BP: i32 = 8;
pub const FUZZY_SUPPORT_2BP: i32 = 9;
pub const FUZZY_SUPPORT_1BP: i32 = 10;

const FUZZY_ALPHA_1BP: f32 = 0.90;
const FUZZY_ALPHA_2BP: f32 = 0.80;
const FUZZY_ALPHA_3BP: f32 = 0.70;

pub fn read_supports_variant_fuzzy_read(support_code: i32, opts: &PileupOptions) -> u8 {
    let alpha = match support_code {
        FUZZY_SUPPORT_NONE => opts.allele_unsupporting_read_alpha,
        FUZZY_SUPPORT_EXACT => opts.allele_supporting_read_alpha,
        FUZZY_SUPPORT_1BP => FUZZY_ALPHA_1BP,
        FUZZY_SUPPORT_2BP => FUZZY_ALPHA_2BP,
        FUZZY_SUPPORT_3BP => FUZZY_ALPHA_3BP,
        _ => opts.other_allele_supporting_read_alpha,
    };
    alpha_to_pixel(alpha)
}
pub fn read_supports_variant_fuzzy_ref(opts: &PileupOptions) -> u8 {
    alpha_to_pixel(opts.allele_unsupporting_read_alpha)
}

/// Channels 28 / 29: homopolymer_insertion_quality / homopolymer_deletion_quality.
///
/// Per Ultima Genomics convention, each read base has a `tp` tag value
/// indicating the direction (sign) and magnitude of the most likely indel
/// error encoded by `QUAL[i]`. This encoder collapses each homopolymer run
/// into a single Phred quality reflecting the sum of error probabilities
/// in the requested direction (insertion = positive tp, deletion =
/// negative tp). The resulting per-base quality is mapped to a pixel via
/// `BaseQualityColor` (cap 93).
///
/// If `tp` is absent or the wrong length, all positions default to the
/// max quality color (94 ≈ `254 * 93/93`).
pub const MAX_Q_SCORE: i32 = 93;

#[inline]
fn base_quality_color(qual: i32) -> u8 {
    scale_color(qual, MAX_Q_SCORE)
}

pub fn homopolymer_indel_quality(seq: &[u8], qualities: &[u8], tps: &[i8], is_deletion: bool) -> Vec<u8> {
    let n = seq.len();
    let default = base_quality_color(MAX_Q_SCORE);
    let mut out = vec![default; n];
    if n == 0 || qualities.len() != n || tps.len() != n {
        return out;
    }
    let weights = homopolymer_weights(seq);
    let mut i = 0usize;
    while i < n {
        let hmer_len = weights[i] as usize;
        if hmer_len == 0 {
            i += 1;
            continue;
        }
        let end = (i + hmer_len).min(n);
        let mut error_prob = 0f64;
        for j in i..end {
            let tp = tps[j];
            if tp == 0 {
                continue;
            }
            let is_del_err = tp < 0;
            if is_del_err == is_deletion {
                let q = qualities[j] as f64;
                error_prob += 10f64.powf(q / -10.0);
            }
        }
        let q = if error_prob == 0.0 {
            MAX_Q_SCORE
        } else {
            let v = (-10.0 * error_prob.log10()) as i32;
            v.min(MAX_Q_SCORE)
        };
        let pixel = base_quality_color(q);
        for j in i..end {
            out[j] = pixel;
        }
        i = end;
    }
    out
}

pub fn homopolymer_insertion_quality_read_at(
    seq: &[u8],
    qualities: &[u8],
    tps: &[i8],
    read_pos: usize,
) -> u8 {
    let v = homopolymer_indel_quality(seq, qualities, tps, false);
    v.get(read_pos).copied().unwrap_or(0)
}
pub fn homopolymer_insertion_quality_ref() -> u8 {
    0
}

pub fn homopolymer_deletion_quality_read_at(
    seq: &[u8],
    qualities: &[u8],
    tps: &[i8],
    read_pos: usize,
) -> u8 {
    let v = homopolymer_indel_quality(seq, qualities, tps, true);
    v.get(read_pos).copied().unwrap_or(0)
}
pub fn homopolymer_deletion_quality_ref() -> u8 {
    0
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

    /// Mirrors upstream `IdentityTest::PacBioStyleCigar`:
    /// `"5=", "1X", "4="` → identity = 90% → scale_color(90, 100).
    #[test]
    fn identity_pacbio_style_cigar() {
        let cigar = vec![('=', 5), ('X', 1), ('=', 4)];
        assert_eq!(identity_read(cigar, 10), scale_color(90, 100));
    }

    /// Mirrors upstream `GapCompressedIdentityTest::DeletionCase`:
    /// `"3M", "4D", "3M"` → matches=6, indel runs=1 → 6/7 ≈ 85.7%.
    #[test]
    fn gap_compressed_identity_deletion_case() {
        let cigar = vec![('M', 3), ('D', 4), ('M', 3)];
        let v = gap_compressed_identity_read(cigar);
        // 6 / (6 + 0 + 1) = 6/7 ≈ 85.71%; scaled: 254 * 6/7 ≈ 217.7 → 217.
        assert!((215..=220).contains(&v), "got {v}");
    }

    /// Mirrors upstream `GapCompressedIdentityTest::PacBioStyleCigar`:
    /// `"3=", "2X", "2I", "3="` → matches=6, mismatches=2, indel runs=1
    /// → 6/9 ≈ 67%.
    #[test]
    fn gap_compressed_identity_pacbio_style_cigar() {
        let cigar = vec![('=', 3), ('X', 2), ('I', 2), ('=', 3)];
        let v = gap_compressed_identity_read(cigar);
        let expected = scale_color(66, 100); // (6*100)/9 = 66.66 → trunc 66
        // upstream's int division yields 66, not 67.
        assert!(
            (v as i32 - expected as i32).abs() <= 2,
            "got {v} expected near {expected}"
        );
    }

    /// Mirrors upstream `AvgBaseQualityTest::BasicCase`. Read of 10
    /// bases with quality 1..=10 → mean = 55/10 = 5 → scaled.
    #[test]
    fn avg_base_quality_upstream_basic_case() {
        let q: Vec<u8> = (1..=10u8).collect();
        let v = avg_base_quality_read(&q);
        // sum=55, avg=5 (integer division), scaled to 0..254 against
        // MAX_AVG_BASE_QUALITY=93: 254 * 5 / 93 = 13.65 → 13.
        let expected = scale_color(5, MAX_AVG_BASE_QUALITY);
        assert_eq!(v, expected);
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

    /// Mirrors upstream `HomoPolymerWeightedTest::BasicCase`.
    #[test]
    fn homopolymer_weights_upstream_basic_case() {
        // "GATTGGGCCCCAAAAA" → runs of 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5
        let expected: Vec<u8> = vec![1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5];
        assert_eq!(homopolymer_weights(b"GATTGGGCCCCAAAAA"), expected);
    }

    /// Mirrors upstream `HomoPolymerWeightedTest::WeightedHomoPolymerMax`.
    /// Run length cap is 255 (u8::MAX); we use a 30bp run to confirm
    /// no overflow at our smaller `MAX_HOMOPOLYMER_WEIGHTED=30` cap.
    #[test]
    fn homopolymer_weights_long_run() {
        let s: Vec<u8> = (0..30).map(|_| b'A').collect();
        let weights = homopolymer_weights(&s);
        assert_eq!(weights.len(), 30);
        assert!(weights.iter().all(|&w| w == 30));
        // Pixel scale: 30/30 → 254.
        assert_eq!(homopolymer_weighted_read_at(&s, 15), 254);
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
        // ATCGGGAA: only the GGG run (offsets 3,4,5) qualifies.
        // Run length 2 ("AA" at the end) does NOT qualify per upstream
        // (threshold is 3-or-more).
        let seq = b"ATCGGGAA";
        assert_eq!(is_homopolymer_read_at(seq, 0), 0);
        assert_eq!(is_homopolymer_read_at(seq, 1), 0);
        assert_eq!(is_homopolymer_read_at(seq, 3), 254);
        assert_eq!(is_homopolymer_read_at(seq, 4), 254);
        assert_eq!(is_homopolymer_read_at(seq, 5), 254);
        assert_eq!(is_homopolymer_read_at(seq, 6), 0); // "AA" run len 2
        assert_eq!(is_homopolymer_read_at(seq, 7), 0);
    }

    /// Mirrors upstream `IsHomoPolymerTest::IsHomopolymerBeginning`.
    #[test]
    fn is_homopolymer_beginning() {
        let seq = b"GGGATAATA";
        let expected = [254, 254, 254, 0, 0, 0, 0, 0, 0];
        for (i, &want) in expected.iter().enumerate() {
            assert_eq!(is_homopolymer_read_at(seq, i), want, "offset {i}");
        }
    }

    /// Mirrors upstream `IsHomoPolymerTest::IsHomopolymerMiddle`.
    #[test]
    fn is_homopolymer_middle() {
        let seq = b"ATTGGGTTA";
        let expected = [0, 0, 0, 254, 254, 254, 0, 0, 0];
        for (i, &want) in expected.iter().enumerate() {
            assert_eq!(is_homopolymer_read_at(seq, i), want, "offset {i}");
        }
    }

    /// Mirrors upstream `IsHomoPolymerTest::IsHomopolymerEnd`.
    #[test]
    fn is_homopolymer_end() {
        let seq = b"ATAATAGGG";
        let expected = [0, 0, 0, 0, 0, 0, 254, 254, 254];
        for (i, &want) in expected.iter().enumerate() {
            assert_eq!(is_homopolymer_read_at(seq, i), want, "offset {i}");
        }
    }

    /// Mirrors upstream `IsHomoPolymerTest::IsHomopolymerAll`.
    #[test]
    fn is_homopolymer_all_one_run() {
        let seq = b"AAAAAAAAA";
        for i in 0..seq.len() {
            assert_eq!(is_homopolymer_read_at(seq, i), 254);
        }
    }

    #[test]
    fn base_methylation_scales_against_255() {
        // 0 → 0, 255 → 254, 128 → ~127
        let v = vec![0u8, 128, 255];
        assert_eq!(base_methylation_read_at(Some(&v), 0), 0);
        assert_eq!(base_methylation_read_at(Some(&v), 1), 127);
        assert_eq!(base_methylation_read_at(Some(&v), 2), 254);
        // Out of range → 0
        assert_eq!(base_methylation_read_at(Some(&v), 99), 0);
        // Missing tag → 0
        assert_eq!(base_methylation_read_at(None, 0), 0);
        assert_eq!(base_methylation_read_at(Some(&[]), 0), 0);
        assert_eq!(base_methylation_ref(), 0);
        // 6mA shares the encoder.
        assert_eq!(base_6ma_read_at(Some(&v), 1), 127);
        assert_eq!(base_6ma_ref(), 0);
    }

    #[test]
    fn supplementary_alignment_alphas() {
        let o = opts();
        // is_supplementary=true → supporting alpha=1.0 → 254
        assert_eq!(supplementary_alignment_read(true, &o), 254);
        // false → unsupporting alpha=0.6 → 152
        assert_eq!(supplementary_alignment_read(false, &o), 152);
        // ref row drops alpha→pixel scaling (matches upstream quirk):
        // raw alpha cast to u8 → 0 (since 0.6 truncates)
        assert_eq!(supplementary_alignment_ref(&o), 0);
    }

    #[test]
    fn allele_sample_probability_sqrt_scaling() {
        // total=0 → 0
        assert_eq!(allele_sample_probability_read(5, 0), 0);
        // 0/10 → 0
        assert_eq!(allele_sample_probability_read(0, 10), 0);
        // 10/10 → sqrt(1)*254 = 254
        assert_eq!(allele_sample_probability_read(10, 10), 254);
        // 5/10 → sqrt(0.5)*254 ≈ 179
        let v = allele_sample_probability_read(5, 10);
        assert!((175..=185).contains(&v), "got {v}");
        // 1/100 → sqrt(0.01)*254 = 25.4 → 25
        let v = allele_sample_probability_read(1, 100);
        assert!((20..=30).contains(&v), "got {v}");
        assert_eq!(allele_sample_probability_ref(), 0);
    }

    #[test]
    fn read_supports_variant_fuzzy_codes() {
        let o = opts();
        // Exact = supporting (alpha 1.0) → 254
        assert_eq!(read_supports_variant_fuzzy_read(FUZZY_SUPPORT_EXACT, &o), 254);
        // Other = other-alt (alpha 0.6) → 152
        assert_eq!(read_supports_variant_fuzzy_read(FUZZY_SUPPORT_OTHER_ALT, &o), 152);
        // 1bp = 0.90 → 228
        let v = read_supports_variant_fuzzy_read(FUZZY_SUPPORT_1BP, &o);
        assert!((225..=230).contains(&v), "got {v}");
        // 2bp = 0.80 → 203
        let v = read_supports_variant_fuzzy_read(FUZZY_SUPPORT_2BP, &o);
        assert!((200..=205).contains(&v), "got {v}");
        // 3bp = 0.70 → 177
        let v = read_supports_variant_fuzzy_read(FUZZY_SUPPORT_3BP, &o);
        assert!((175..=180).contains(&v), "got {v}");
        // None = unsupporting (0.6) → 152
        assert_eq!(read_supports_variant_fuzzy_read(FUZZY_SUPPORT_NONE, &o), 152);
        // Unknown → defaults to other-alt
        assert_eq!(read_supports_variant_fuzzy_read(99, &o), 152);
        assert_eq!(read_supports_variant_fuzzy_ref(&o), 152);
    }

    #[test]
    fn homopolymer_indel_quality_no_tp_falls_back_to_max_q() {
        // No tp tag → vector returns max-Q color for every position.
        let seq = b"AAAAA";
        let q = vec![30u8; 5];
        let tps = vec![];
        let max = base_quality_color(MAX_Q_SCORE);
        let v = homopolymer_indel_quality(seq, &q, &tps, true);
        assert!(v.iter().all(|&x| x == max));
    }

    #[test]
    fn homopolymer_indel_quality_runs_collapse() {
        // AAAAA homopolymer of length 5. tp = [0, 1, 0, 0, 0] (one insertion err).
        // Quality=30 at the marked position.  error_prob = 10^-3.
        // hmer_directed_quality = -10*log10(10^-3) = 30. Pixel = scale_color(30, 93) = 81.
        let seq = b"AAAAA";
        let q = vec![30u8; 5];
        let tps = vec![0i8, 1, 0, 0, 0];
        let v = homopolymer_indel_quality(seq, &q, &tps, false); // is_deletion=false
        // All five positions in the run get the same pixel value.
        assert_eq!(v.len(), 5);
        let expected = base_quality_color(30);
        assert!(v.iter().all(|&x| x == expected), "got {v:?}");

        // For deletion direction, none of these tps qualify (tp>0 is insertion),
        // so error_prob=0 → max-Q pixel.
        let v = homopolymer_indel_quality(seq, &q, &tps, true);
        let max = base_quality_color(MAX_Q_SCORE);
        assert!(v.iter().all(|&x| x == max));

        // Per-position helpers should hit the same pixel.
        assert_eq!(
            homopolymer_insertion_quality_read_at(seq, &q, &tps, 2),
            expected
        );
        assert_eq!(
            homopolymer_deletion_quality_read_at(seq, &q, &tps, 2),
            max
        );
        assert_eq!(homopolymer_insertion_quality_ref(), 0);
        assert_eq!(homopolymer_deletion_quality_ref(), 0);
    }
}

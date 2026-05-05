//! Pileup image generation (port-in-progress).
//!
//! Mirrors `deepvariant/pileup_image_native.cc` + the
//! `deepvariant/channels/*_channel.cc` set of pixel encoders.
//!
//! Currently ported:
//!   - `options::PileupOptions` (with WGS-default values)
//!   - `channels::*` for channels 1–6 (read_base, base_quality,
//!     mapping_quality, strand, read_supports_variant,
//!     base_differs_from_ref).
//!
//! Not yet ported:
//!   - The pileup image *layout* (read sorting, deletion/insertion
//!     anchoring, soft-clip handling).
//!   - Channels 7–29 (haplotype_tag, allele_frequency, insert_size, etc.)
//!   - alt-aligned pileup variants.

pub mod channels;
pub mod layout;
pub mod options;

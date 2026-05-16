//! WebAssembly entry point for the pure-compute pieces of the
//! DeepVariant Rust port.
//!
//! What lives here:
//!   * `parse_example` — decode a tf.Example proto byte buffer.
//!   * `decode_image_f32` — convert the H×W×C uint8 pileup to the
//!     [batch, H, W, C] float32 input the WGS model expects.
//!   * `extract_small_model_features` — compute the 70-feature
//!     vector for the small-model fast path.
//!   * `phase_reads` — run direct phasing on a (candidates, reads)
//!     pair and return per-read phases.
//!
//! All of these are pure compute and build cleanly for
//! `wasm32-unknown-unknown`. They're the pieces that benefit from
//! near-native speed without browser/JS overhead.
//!
//! What does NOT live here yet:
//!   * The ORT inference call itself. ort 2.0.0-rc.10 (our pinned
//!     version) doesn't have wasm bindings. ort >= rc.12 has explicit
//!     wasm gating but with breaking API changes from rc.10. The
//!     production WASM inference path is to call onnxruntime-web
//!     (JavaScript) via `wasm-bindgen` from this crate — the JS side
//!     does the heavy ORT work, the Rust side does data prep + post
//!     processing. That bridge is its own integration project; for
//!     now the wasm pipeline ends at "produce model input bytes".
//!   * BAM/CRAM reading. Those rely on `dv-io` which transitively
//!     depends on flate2 and (for CRAM) C-only bzip2/xz. In a
//!     browser context, the upstream caller would parse those
//!     server-side or via WebStreams.

use prost::Message;

#[cfg(feature = "wasm-bindgen")]
use wasm_bindgen::prelude::*;

use dv_proto::dv::call_variants_output::AltAlleleIndices;
use dv_proto::nucleus_v1::Variant;
use dv_proto::tf::feature::Kind as FeatureKind;
use dv_proto::tf::Example;

/// Full BAM → examples → (JS inference) → VCF browser pipeline.
pub mod pipeline;

/// Parsed contents of a tf.Example record from a make-examples shard.
#[derive(Debug, Clone)]
pub struct ExampleRow {
    pub variant: Variant,
    pub alt_allele_indices: AltAlleleIndices,
    /// Raw H*W*C uint8 pixel buffer.
    pub image: Vec<u8>,
}

/// Decode one tf.Example payload from a TFRecord shard. Returns the
/// (variant, alt_allele_indices, image_bytes) triple.
pub fn parse_example(payload: &[u8]) -> Result<ExampleRow, String> {
    let ex = Example::decode(payload).map_err(|e| format!("decode example: {e}"))?;
    let features = ex.features.ok_or("example missing features".to_string())?;
    let bytes_for = |key: &str| -> Result<Vec<u8>, String> {
        let f = features
            .feature
            .get(key)
            .ok_or_else(|| format!("missing feature {key}"))?;
        let kind = f
            .kind
            .as_ref()
            .ok_or_else(|| format!("feature {key} kind missing"))?;
        match kind {
            FeatureKind::BytesList(bl) => bl
                .value
                .first()
                .cloned()
                .ok_or_else(|| format!("feature {key} BytesList empty")),
            other => Err(format!("feature {key} expected bytes, got {other:?}")),
        }
    };
    let image = bytes_for("image/encoded")?;
    let variant_bytes = bytes_for("variant/encoded")?;
    let aai_bytes = bytes_for("alt_allele_indices/encoded")?;
    let variant =
        Variant::decode(&*variant_bytes).map_err(|e| format!("decode variant: {e}"))?;
    let alt_allele_indices =
        AltAlleleIndices::decode(&*aai_bytes).map_err(|e| format!("decode aai: {e}"))?;
    Ok(ExampleRow {
        variant,
        alt_allele_indices,
        image,
    })
}

/// Decode the H×W×C uint8 pileup image into a `[batch=1, H, W, C]`
/// float32 input buffer ready to feed into the WGS model. Mirrors the
/// (b - 128) / 128 normalisation the production caller does
/// natively.
pub fn decode_image_f32(image: &[u8]) -> Vec<f32> {
    image.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
}

/// Compute the 70-feature vector for the small-model fast path.
/// Re-exports `dv_core::small_model::compute` with a wasm-friendly
/// signature (read attributes + variant proto bytes).
pub fn extract_small_model_features_from_bytes(
    variant_bytes: &[u8],
    alt_allele_indices: &[i32],
    ref_reads: &[dv_core::small_model::ReadAttrs],
    alt_reads: &[dv_core::small_model::ReadAttrs],
    total_depth: i32,
) -> Result<Vec<f32>, String> {
    let variant =
        Variant::decode(variant_bytes).map_err(|e| format!("decode variant: {e}"))?;
    let vaf_at_position = std::collections::HashMap::new();
    Ok(dv_core::small_model::compute(
        &variant,
        alt_allele_indices,
        ref_reads,
        alt_reads,
        total_depth,
        &vaf_at_position,
    ))
}

// ---- wasm-bindgen surface ----
//
// Browser/JS consumers see flat byte-array APIs. Errors come back as
// JsError with a string message.

/// Decode a tf.Example payload and return the model input as a flat
/// `Float32Array` (length H*W*C). The returned buffer is normalised
/// `(b-128)/128` per pixel.
#[cfg(feature = "wasm-bindgen")]
#[wasm_bindgen]
pub fn example_to_model_input(payload: &[u8]) -> Result<Vec<f32>, JsError> {
    let row = parse_example(payload).map_err(|e| JsError::new(&e))?;
    Ok(decode_image_f32(&row.image))
}

/// Decode a tf.Example payload and return the embedded variant's
/// position. Useful for routing inference results back to the right
/// candidate from JS.
#[cfg(feature = "wasm-bindgen")]
#[wasm_bindgen]
pub fn example_variant_start(payload: &[u8]) -> Result<i64, JsError> {
    let row = parse_example(payload).map_err(|e| JsError::new(&e))?;
    Ok(row.variant.start)
}

/// Number of channels the WGS pileup image carries. Exposed so JS
/// glue can shape the input tensor correctly.
#[cfg(feature = "wasm-bindgen")]
#[wasm_bindgen]
pub fn wgs_channel_count() -> usize {
    7
}

/// WGS pileup image dimensions: returns `[H, W, C]` flat as a
/// `Uint32Array` so JS can read it directly.
#[cfg(feature = "wasm-bindgen")]
#[wasm_bindgen]
pub fn wgs_image_shape() -> Vec<u32> {
    vec![100, 221, 7]
}

// ---- raw C-ABI benchmark hook ----
//
// Exposes a pure-compute "kernel" callable from any wasm runtime
// (wasmtime, wasmer, …) without wasm-bindgen JS glue. The function
// runs the H*W*C = 154 700 byte normalisation `(b - 128) / 128.0`
// `iterations` times against a synthetic image and folds the results
// into a u64 checksum (so the loop can't be optimised away).
//
// Used by `dv-wasm-bench` to time the same kernel native vs wasm.
#[no_mangle]
pub extern "C" fn benchmark_normalize_image(iterations: u32) -> u64 {
    const H: usize = 100;
    const W: usize = 221;
    const C: usize = 7;
    const N: usize = H * W * C;
    // Synthetic image: linear ramp 0..=255 modulo 256.
    let mut image = vec![0u8; N];
    for (i, b) in image.iter_mut().enumerate() {
        *b = (i & 0xff) as u8;
    }
    let mut acc: u64 = 0;
    for _ in 0..iterations {
        let normalised: Vec<f32> = decode_image_f32(&image);
        // Fold into a u64 so the optimiser can't elide the work.
        let mut s: u32 = 0;
        for &v in &normalised {
            s = s.wrapping_add(v.to_bits());
        }
        acc = acc.wrapping_add(s as u64);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use dv_proto::nucleus_v1::Variant;
    use dv_proto::tf::{BytesList, Feature, Features};

    fn synth_example() -> Vec<u8> {
        let v = Variant {
            reference_name: "chr1".into(),
            start: 100,
            end: 101,
            reference_bases: "A".into(),
            alternate_bases: vec!["C".into()],
            ..Default::default()
        };
        let aai = AltAlleleIndices { indices: vec![0] };
        let img = vec![128u8; 100 * 221 * 7];
        let mut feature_map = std::collections::BTreeMap::new();
        let bf = |b: Vec<u8>| Feature {
            kind: Some(FeatureKind::BytesList(BytesList { value: vec![b] })),
        };
        feature_map.insert("image/encoded".to_string(), bf(img));
        feature_map.insert("variant/encoded".to_string(), bf(v.encode_to_vec()));
        feature_map.insert(
            "alt_allele_indices/encoded".to_string(),
            bf(aai.encode_to_vec()),
        );
        let ex = Example {
            features: Some(Features { feature: feature_map }),
        };
        ex.encode_to_vec()
    }

    #[test]
    fn parse_example_round_trip() {
        let bytes = synth_example();
        let row = parse_example(&bytes).unwrap();
        assert_eq!(row.variant.start, 100);
        assert_eq!(row.alt_allele_indices.indices, vec![0]);
        assert_eq!(row.image.len(), 100 * 221 * 7);
    }

    #[test]
    fn decode_image_f32_is_centered_at_zero() {
        // 128 → (128-128)/128 = 0.0
        let img = vec![128u8; 4];
        let out = decode_image_f32(&img);
        for v in out {
            assert!(v.abs() < 1e-6);
        }
        // 0 → -1.0; 254 → ~ 0.984; 255 → 0.992
        assert!((decode_image_f32(&[0])[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn extract_small_model_features_smoke() {
        let v = Variant {
            reference_name: "chr1".into(),
            start: 100,
            end: 101,
            reference_bases: "A".into(),
            alternate_bases: vec!["C".into()],
            ..Default::default()
        };
        let bytes = v.encode_to_vec();
        let refs = vec![dv_core::small_model::ReadAttrs {
            mapping_quality: 60,
            avg_base_quality: 30,
            is_reverse_strand: false,
        }; 5];
        let alts = vec![dv_core::small_model::ReadAttrs {
            mapping_quality: 60,
            avg_base_quality: 30,
            is_reverse_strand: true,
        }; 5];
        let f = extract_small_model_features_from_bytes(&bytes, &[0], &refs, &alts, 10).unwrap();
        assert_eq!(f.len(), 19 + 51); // 70 features
    }
}

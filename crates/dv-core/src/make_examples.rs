//! Build tf.Example records for candidate variants.
//!
//! Port-in-progress of `deepvariant/make_examples_native.cc`. The
//! upstream version handles many cases (alt-aligned pileups,
//! multi-sample, methylation, somatic, etc.); this is the SNV-focused
//! foundation. Each example contains:
//!
//!   - `image/encoded`: raw bytes of the H×W×C uint8 pileup image
//!   - `variant/encoded`: serialized Variant proto
//!   - `alt_allele_indices/encoded`: serialized
//!     `CallVariantsOutput.AltAlleleIndices` proto

use std::collections::BTreeMap;

use prost::Message;

use dv_proto::dv::call_variants_output::AltAlleleIndices;
use dv_proto::nucleus_v1::Variant;
use dv_proto::tf::feature::Kind as FeatureKind;
use dv_proto::tf::{BytesList, Example, Feature, Features};

/// Serialize one (variant, alt_allele_indices, pileup_image) triple as a
/// `tf.Example` byte buffer ready to be written to a TFRecord shard.
pub fn build_example(variant: &Variant, alt_indices: &[i32], image: &[u8]) -> Vec<u8> {
    let aai = AltAlleleIndices {
        indices: alt_indices.to_vec(),
    };
    let mut feature_map: BTreeMap<String, Feature> = BTreeMap::new();
    feature_map.insert(
        "image/encoded".to_string(),
        bytes_feature(image.to_vec()),
    );
    feature_map.insert(
        "variant/encoded".to_string(),
        bytes_feature(variant.encode_to_vec()),
    );
    feature_map.insert(
        "alt_allele_indices/encoded".to_string(),
        bytes_feature(aai.encode_to_vec()),
    );
    let example = Example {
        features: Some(Features { feature: feature_map }),
    };
    example.encode_to_vec()
}

fn bytes_feature(bytes: Vec<u8>) -> Feature {
    Feature {
        kind: Some(FeatureKind::BytesList(BytesList { value: vec![bytes] })),
    }
}

/// Decode a tf.Example payload back into its three components, for testing.
pub fn parse_example(payload: &[u8]) -> Result<(Variant, AltAlleleIndices, Vec<u8>), String> {
    let ex = Example::decode(payload).map_err(|e| format!("decode example: {e}"))?;
    let features = ex
        .features
        .ok_or_else(|| "example missing features".to_string())?;
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
    let variant = Variant::decode(&*variant_bytes).map_err(|e| format!("decode variant: {e}"))?;
    let aai = AltAlleleIndices::decode(&*aai_bytes).map_err(|e| format!("decode aai: {e}"))?;
    Ok((variant, aai, image))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_variant() -> Variant {
        Variant {
            reference_name: "chr20".into(),
            start: 10_000_116,
            end: 10_000_117,
            reference_bases: "C".into(),
            alternate_bases: vec!["T".into()],
            ..Default::default()
        }
    }

    #[test]
    fn round_trip_example() {
        let v = synth_variant();
        let img = vec![42u8; 100 * 221 * 7];
        let bytes = build_example(&v, &[0], &img);
        let (got_v, got_aai, got_img) = parse_example(&bytes).unwrap();
        assert_eq!(got_v, v);
        assert_eq!(got_aai.indices, vec![0]);
        assert_eq!(got_img, img);
    }

    #[test]
    fn round_trip_multi_alt() {
        let mut v = synth_variant();
        v.alternate_bases = vec!["T".into(), "G".into()];
        let img = vec![7u8; 100 * 221 * 7];
        let bytes = build_example(&v, &[0, 1], &img);
        let (got_v, got_aai, _) = parse_example(&bytes).unwrap();
        assert_eq!(got_v.alternate_bases, vec!["T", "G"]);
        assert_eq!(got_aai.indices, vec![0, 1]);
    }
}

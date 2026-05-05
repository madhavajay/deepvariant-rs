use std::io::Result;

fn main() -> Result<()> {
    let proto_root = "proto";
    let protos = &[
        "deepvariant/protos/deepvariant.proto",
        "deepvariant/protos/realigner.proto",
        "deepvariant/protos/resources.proto",
        "third_party/nucleus/protos/cigar.proto",
        "third_party/nucleus/protos/example.proto",
        "third_party/nucleus/protos/feature.proto",
        "third_party/nucleus/protos/position.proto",
        "third_party/nucleus/protos/range.proto",
        "third_party/nucleus/protos/reads.proto",
        "third_party/nucleus/protos/reference.proto",
        "third_party/nucleus/protos/struct.proto",
        "third_party/nucleus/protos/variants.proto",
    ];

    for p in protos {
        println!("cargo:rerun-if-changed={proto_root}/{p}");
    }

    let mut cfg = prost_build::Config::new();
    cfg.include_file("_includes.rs");
    cfg.compile_protos(protos, &[proto_root])?;
    Ok(())
}

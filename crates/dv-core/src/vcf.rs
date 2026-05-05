//! VCF writer — emits text directly (matches upstream byte-equally where
//! deterministic). No noodles VCF header types; we only use noodles for
//! BGZF compression at the I/O layer.
//!
//! Header schema mirrors `dv_vcf_constants.deepvariant_header()`.

use std::io::Write;

use anyhow::{Context, Result};

use dv_proto::nucleus_v1::{value, ContigInfo, Variant, VariantCall};

const DEEPVARIANT_VERSION: &str = "1.10.0";

const HEADER_PRE: &str = "##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description=\"All filters passed\">
##FILTER=<ID=RefCall,Description=\"Genotyping model thinks this site is reference.\">
##FILTER=<ID=LowQual,Description=\"Confidence in this variant being real is below calling threshold.\">
##FILTER=<ID=NoCall,Description=\"Site has depth=0 resulting in no call.\">
##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position (for use with symbolic alleles)\">
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Conditional genotype quality\">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read depth\">
##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description=\"Minimum DP observed within the GVCF block.\">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Read depth for each allele\">
##FORMAT=<ID=VAF,Number=A,Type=Float,Description=\"Variant allele fractions.\">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods rounded to the closest integer\">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description=\"Phase set\">
##FORMAT=<ID=MF,Number=R,Type=Float,Description=\"Methylation fraction for each of the reference and alternate allele\">
##FORMAT=<ID=MD,Number=R,Type=Integer,Description=\"Methylation depth for each of the reference and alternate allele\">
##FORMAT=<ID=MT,Number=1,Type=String,Description=\"Methylation type: 0/0=Unmethylated, 0/1=Heterozygous, 1/1=Methylated\">
##FORMAT=<ID=MI,Number=1,Type=Float,Description=\"Allele-specific methylation score: p-value for Wilcoxon Rank-Sum test based on the observed difference in methylation between haplotypes.\">
##FORMAT=<ID=MED_DP,Number=1,Type=Integer,Description=\"Median DP observed within the GVCF block rounded to the nearest integer.\">
##FORMAT=<ID=MID,Number=1,Type=String,Description=\"Identifies which model called this variant.\">
";

/// Write the upstream-style VCF header to `w`.
/// Order: fixed prelude → `##DeepVariant_version` → `##contig` lines → column header.
pub fn write_header<W: Write>(
    w: &mut W,
    contigs: &[ContigInfo],
    sample_names: &[&str],
) -> Result<()> {
    w.write_all(HEADER_PRE.as_bytes())?;
    writeln!(w, "##DeepVariant_version={DEEPVARIANT_VERSION}")?;
    for c in contigs {
        if c.n_bases > 0 {
            writeln!(w, "##contig=<ID={},length={}>", c.name, c.n_bases)?;
        } else {
            writeln!(w, "##contig=<ID={}>", c.name)?;
        }
    }
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")?;
    if !sample_names.is_empty() {
        write!(w, "\tFORMAT")?;
        for s in sample_names {
            write!(w, "\t{s}")?;
        }
    }
    writeln!(w)?;
    Ok(())
}

/// Emit one variant record line.
pub fn write_variant_line<W: Write>(
    w: &mut W,
    variant: &Variant,
    format_keys: &[&str],
) -> Result<()> {
    write_record_line(w, variant, format_keys, /*emit_end_info*/ false)
}

/// Emit a record line for gVCF: nonvariant blocks (alts == ["<*>"]) get
/// `INFO=END=<end>` and FORMAT `GT:GQ:MIN_DP:PL`; variant lines get the
/// full FORMAT with `<*>` appended to alts.
pub fn write_gvcf_line<W: Write>(w: &mut W, variant: &Variant) -> Result<()> {
    let is_nonvariant = variant.alternate_bases.len() == 1
        && variant.alternate_bases[0] == crate::gvcf::GVCF_ALT_ALLELE;
    let format_keys: &[&str] = if is_nonvariant {
        &["GT", "GQ", "MIN_DP", "PL"]
    } else {
        &["GT", "GQ", "DP", "AD", "VAF", "MID", "PL"]
    };
    write_record_line(w, variant, format_keys, /*emit_end_info*/ is_nonvariant)
}

fn write_record_line<W: Write>(
    w: &mut W,
    variant: &Variant,
    format_keys: &[&str],
    emit_end_info: bool,
) -> Result<()> {
    let chrom = &variant.reference_name;
    let pos = variant.start + 1; // VCF is 1-based
    let id = if variant.names.is_empty() {
        ".".into()
    } else {
        variant.names.join(";")
    };
    let reference = &variant.reference_bases;
    let alts = if variant.alternate_bases.is_empty() {
        ".".into()
    } else {
        variant.alternate_bases.join(",")
    };
    let qual = format_qual(variant.quality);
    let filter = if variant.filter.is_empty() {
        ".".into()
    } else {
        variant.filter.join(";")
    };
    let info = if emit_end_info {
        format!("END={}", variant.end)
    } else {
        ".".into()
    };

    write!(
        w,
        "{chrom}\t{pos}\t{id}\t{reference}\t{alts}\t{qual}\t{filter}\t{info}"
    )
    .context("write fixed cols")?;

    if !variant.calls.is_empty() {
        write!(w, "\t{}", format_keys.join(":"))?;
        for call in &variant.calls {
            let mut parts = Vec::with_capacity(format_keys.len());
            for k in format_keys {
                parts.push(format_call_field(call, k));
            }
            write!(w, "\t{}", parts.join(":"))?;
        }
    }
    writeln!(w)?;
    Ok(())
}

fn format_qual(q: f64) -> String {
    if q == 0.0 {
        return "0".into();
    }
    let rounded = (q * 10.0).round() / 10.0;
    if rounded.fract() == 0.0 {
        format!("{}", rounded as i64)
    } else {
        format!("{:.1}", rounded)
    }
}

fn format_call_field(call: &VariantCall, key: &str) -> String {
    match key {
        "GT" => format_gt(call),
        "GQ" => int_field(call, "GQ").unwrap_or_else(|| ".".into()),
        "DP" => int_field(call, "DP").unwrap_or_else(|| ".".into()),
        "MIN_DP" => int_field(call, "MIN_DP").unwrap_or_else(|| ".".into()),
        "MED_DP" => int_field(call, "MED_DP").unwrap_or_else(|| ".".into()),
        "AD" => list_int_field(call, "AD"),
        "VAF" => list_float_field(call, "VAF"),
        // PL: prefer info[PL] (already-computed by postprocess) but fall
        // back to deriving from GL via zero-shift + phred-truncate.
        "PL" => {
            if call.info.contains_key("PL") {
                list_int_field(call, "PL")
            } else if !call.genotype_likelihood.is_empty() {
                let max = call
                    .genotype_likelihood
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                call.genotype_likelihood
                    .iter()
                    .map(|gl| ((-10.0 * (gl - max)) as i32).to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            } else {
                ".".into()
            }
        }
        "PS" => int_field(call, "PS").unwrap_or_else(|| ".".into()),
        "MF" => list_float_field(call, "MF"),
        "MD" => list_int_field(call, "MD"),
        "MT" => string_field(call, "MT").unwrap_or_else(|| ".".into()),
        "MI" => string_field(call, "MI").unwrap_or_else(|| ".".into()),
        "MID" => string_field(call, "MID").unwrap_or_else(|| ".".into()),
        _ => ".".into(),
    }
}

fn format_gt(call: &VariantCall) -> String {
    if call.genotype.is_empty() {
        return ".".into();
    }
    let sep = if call.is_phased { "|" } else { "/" };
    call.genotype
        .iter()
        .map(|g| if *g < 0 { ".".into() } else { g.to_string() })
        .collect::<Vec<_>>()
        .join(sep)
}

fn int_field(call: &VariantCall, key: &str) -> Option<String> {
    call.info.get(key).and_then(|lv| {
        lv.values.first().and_then(|v| match &v.kind {
            Some(value::Kind::IntValue(n)) => Some(n.to_string()),
            _ => None,
        })
    })
}

fn list_int_field(call: &VariantCall, key: &str) -> String {
    let Some(lv) = call.info.get(key) else {
        return ".".into();
    };
    let parts: Vec<String> = lv
        .values
        .iter()
        .map(|v| match &v.kind {
            Some(value::Kind::IntValue(n)) => n.to_string(),
            _ => ".".into(),
        })
        .collect();
    if parts.is_empty() {
        ".".into()
    } else {
        parts.join(",")
    }
}

fn list_float_field(call: &VariantCall, key: &str) -> String {
    let Some(lv) = call.info.get(key) else {
        return ".".into();
    };
    let parts: Vec<String> = lv
        .values
        .iter()
        .map(|v| match &v.kind {
            Some(value::Kind::NumberValue(n)) => format_float_short(*n),
            Some(value::Kind::IntValue(n)) => format!("{}", *n as f64),
            _ => ".".into(),
        })
        .collect();
    if parts.is_empty() {
        ".".into()
    } else {
        parts.join(",")
    }
}

fn string_field(call: &VariantCall, key: &str) -> Option<String> {
    call.info.get(key).and_then(|lv| {
        lv.values.first().and_then(|v| match &v.kind {
            Some(value::Kind::StringValue(s)) => Some(s.clone()),
            _ => None,
        })
    })
}

fn format_float_short(x: f64) -> String {
    let s = format!("{:.6}", x);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".into()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_qual_examples() {
        assert_eq!(format_qual(36.8), "36.8");
        assert_eq!(format_qual(36.0), "36");
        assert_eq!(format_qual(0.0), "0");
    }

    #[test]
    fn format_float_short_examples() {
        assert_eq!(format_float_short(0.545455), "0.545455");
        assert_eq!(format_float_short(0.5), "0.5");
        assert_eq!(format_float_short(1.0), "1");
    }
}

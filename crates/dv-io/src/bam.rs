//! BAM reading via `noodles-bam`.

use std::path::Path;

use anyhow::{Context, Result};

pub use noodles::bam;
pub use noodles::sam::Header;

pub type BamReader =
    noodles::bam::io::Reader<noodles::bgzf::Reader<Box<dyn std::io::Read>>>;

/// Open a BAM (no index required) and return its parsed header.
/// The bam::Reader internally bgzf-decodes; we just hand it a Read.
pub fn open(path: impl AsRef<Path>) -> Result<(Header, BamReader)> {
    let path = path.as_ref();
    let f = std::fs::File::open(path).with_context(|| format!("open BAM {}", path.display()))?;
    let buf = std::io::BufReader::new(f);
    let inner: Box<dyn std::io::Read> = Box::new(buf);
    let mut reader = noodles::bam::io::Reader::new(inner);
    let header = reader.read_header().context("read BAM header")?;
    Ok((header, reader))
}

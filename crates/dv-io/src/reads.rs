//! Format-agnostic alignment reader. Dispatches by file extension:
//!
//!   * `*.bam`  → `noodles-bam` reader (no reference required)
//!   * `*.cram` → `noodles-cram` reader (reference required for
//!                 sequence-decompression)
//!
//! Both formats produce records that implement
//! `sam::alignment::Record`, so the callback API below hands the
//! caller a `&dyn Record` and lets them use it identically.
//!
//! Why callback rather than iterator: the two underlying iterators
//! have very different lifetime/ownership shapes (BAM yields owned
//! `bam::Record`s; CRAM yields owned `RecordBuf`s, but indexed reading
//! requires holding a borrow into the reader). Wrapping both in a
//! single `Iterator<Item = Box<dyn Record + '_>>` is doable but ugly.
//! `for_each_record` keeps the API simple at zero allocation cost per
//! record.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use noodles::sam::alignment::Record;
use noodles::sam::Header;

/// Alignment input format detected from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentFormat {
    Bam,
    Cram,
}

impl AlignmentFormat {
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase());
        match ext.as_deref() {
            Some("bam") => Ok(Self::Bam),
            Some("cram") => Ok(Self::Cram),
            other => Err(anyhow!(
                "unrecognised alignment file extension {:?}; expected .bam or .cram",
                other
            )),
        }
    }
}

type BamRdr = noodles::bam::io::Reader<noodles::bgzf::Reader<Box<dyn std::io::Read>>>;
type CramRdr = noodles::cram::io::Reader<std::fs::File>;

/// Format-dispatched alignment reader.
pub enum AlignmentReader {
    Bam(BamRdr),
    Cram(CramRdr),
}

/// Open an alignment file (BAM or CRAM) and return its parsed header.
/// For CRAM, the caller must pass a path to an indexed FASTA so the
/// reader can reconstruct read sequences from reference-compressed
/// records. For BAM, `ref_fasta` is ignored and may be `None`.
pub fn open(path: &Path, ref_fasta: Option<&Path>) -> Result<(Header, AlignmentReader)> {
    let format = AlignmentFormat::from_path(path)?;
    match format {
        AlignmentFormat::Bam => {
            let f = std::fs::File::open(path)
                .with_context(|| format!("open BAM {}", path.display()))?;
            let buf = std::io::BufReader::new(f);
            let inner: Box<dyn std::io::Read> = Box::new(buf);
            let mut reader = noodles::bam::io::Reader::new(inner);
            let header = reader.read_header().context("read BAM header")?;
            Ok((header, AlignmentReader::Bam(reader)))
        }
        AlignmentFormat::Cram => {
            use noodles::fasta::{self, repository::adapters::IndexedReader};
            let ref_path = ref_fasta
                .ok_or_else(|| anyhow!("CRAM input requires --ref-fasta to decompress sequences"))?;
            // CRAM needs an indexed FASTA — the .fai index file must
            // sit next to the FASTA. We let noodles handle the index
            // discovery via its IndexedReader builder.
            let fa_reader = fasta::indexed_reader::Builder::default()
                .build_from_path(ref_path)
                .with_context(|| format!("open ref FASTA {}", ref_path.display()))?;
            let repo = fasta::Repository::new(IndexedReader::new(fa_reader));
            let mut reader = noodles::cram::io::reader::Builder::default()
                .set_reference_sequence_repository(repo)
                .build_from_path(path)
                .with_context(|| format!("open CRAM {}", path.display()))?;
            let header = reader.read_header().context("read CRAM header")?;
            Ok((header, AlignmentReader::Cram(reader)))
        }
    }
}

impl AlignmentReader {
    /// Iterate every record in the file, calling `f` on each. The
    /// callback receives `&dyn Record` so the call sites work
    /// regardless of whether the source is BAM or CRAM. CRAM records
    /// arrive as `RecordBuf`s (owned), BAM as `bam::Record` (owned
    /// per-iteration). Returns the number of records visited.
    pub fn for_each_record<F>(&mut self, header: &Header, mut f: F) -> Result<usize>
    where
        F: FnMut(&dyn Record) -> Result<()>,
    {
        let mut n = 0usize;
        match self {
            AlignmentReader::Bam(r) => {
                for rec in r.records() {
                    let rec = rec.context("read BAM record")?;
                    f(&rec)?;
                    n += 1;
                }
            }
            AlignmentReader::Cram(r) => {
                for rec in r.records(header) {
                    let rec = rec.context("read CRAM record")?;
                    f(&rec)?;
                    n += 1;
                }
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn format_from_path_dispatches_extensions() {
        assert_eq!(
            AlignmentFormat::from_path(&PathBuf::from("a.bam")).unwrap(),
            AlignmentFormat::Bam
        );
        assert_eq!(
            AlignmentFormat::from_path(&PathBuf::from("a.cram")).unwrap(),
            AlignmentFormat::Cram
        );
        assert_eq!(
            AlignmentFormat::from_path(&PathBuf::from("A.BAM")).unwrap(),
            AlignmentFormat::Bam
        );
        assert!(AlignmentFormat::from_path(&PathBuf::from("a.sam")).is_err());
        assert!(AlignmentFormat::from_path(&PathBuf::from("a")).is_err());
    }
}

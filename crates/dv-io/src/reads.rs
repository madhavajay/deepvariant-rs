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
//! Region queries: when a sibling `.bai` (or `.csi`) is found next to
//! a BAM, `open` returns an indexed reader and `for_each_record_in_region`
//! decompresses only the BGZF blocks that overlap the requested region.
//! Without an index it falls back to a full scan + filter.

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
type BamIdxRdr = noodles::bam::io::IndexedReader<noodles::bgzf::Reader<std::fs::File>>;
type CramRdr = noodles::cram::io::Reader<std::fs::File>;
type CramIdxRdr = noodles::cram::io::IndexedReader<std::fs::File>;

/// Format-dispatched alignment reader.
pub enum AlignmentReader {
    Bam(BamRdr),
    BamIndexed(BamIdxRdr),
    Cram(CramRdr),
    CramIndexed(CramIdxRdr),
}

/// Open an alignment file (BAM or CRAM) and return its parsed header.
/// For CRAM, the caller must pass a path to an indexed FASTA so the
/// reader can reconstruct read sequences from reference-compressed
/// records. For BAM, `ref_fasta` is ignored and may be `None`.
///
/// For BAM, if a sibling index (`<file>.bai` or `<file>.csi`) exists,
/// returns an `IndexedReader` so callers can use
/// `for_each_record_in_region` for fast region queries.
pub fn open(path: &Path, ref_fasta: Option<&Path>) -> Result<(Header, AlignmentReader)> {
    let format = AlignmentFormat::from_path(path)?;
    match format {
        AlignmentFormat::Bam => {
            let bai = path.with_extension(format!(
                "{}.bai",
                path.extension().and_then(|e| e.to_str()).unwrap_or("bam")
            ));
            let csi = path.with_extension(format!(
                "{}.csi",
                path.extension().and_then(|e| e.to_str()).unwrap_or("bam")
            ));
            if bai.exists() || csi.exists() {
                let mut reader = noodles::bam::io::indexed_reader::Builder::default()
                    .build_from_path(path)
                    .with_context(|| format!("open indexed BAM {}", path.display()))?;
                let header = reader.read_header().context("read BAM header")?;
                Ok((header, AlignmentReader::BamIndexed(reader)))
            } else {
                let f = std::fs::File::open(path)
                    .with_context(|| format!("open BAM {}", path.display()))?;
                let buf = std::io::BufReader::new(f);
                let inner: Box<dyn std::io::Read> = Box::new(buf);
                let mut reader = noodles::bam::io::Reader::new(inner);
                let header = reader.read_header().context("read BAM header")?;
                Ok((header, AlignmentReader::Bam(reader)))
            }
        }
        AlignmentFormat::Cram => {
            use noodles::fasta::{self, repository::adapters::IndexedReader as FastaIdx};
            let ref_path = ref_fasta
                .ok_or_else(|| anyhow!("CRAM input requires --ref-fasta to decompress sequences"))?;
            let fa_reader = fasta::indexed_reader::Builder::default()
                .build_from_path(ref_path)
                .with_context(|| format!("open ref FASTA {}", ref_path.display()))?;
            let repo = fasta::Repository::new(FastaIdx::new(fa_reader));

            // Sibling `.crai` → indexed reader for fast region queries.
            let crai = path.with_extension(format!(
                "{}.crai",
                path.extension().and_then(|e| e.to_str()).unwrap_or("cram")
            ));
            if crai.exists() {
                let mut reader = noodles::cram::io::indexed_reader::Builder::default()
                    .set_reference_sequence_repository(repo)
                    .build_from_path(path)
                    .with_context(|| format!("open indexed CRAM {}", path.display()))?;
                let header = reader.read_header().context("read CRAM header")?;
                Ok((header, AlignmentReader::CramIndexed(reader)))
            } else {
                let mut reader = noodles::cram::io::reader::Builder::default()
                    .set_reference_sequence_repository(repo)
                    .build_from_path(path)
                    .with_context(|| format!("open CRAM {}", path.display()))?;
                let header = reader.read_header().context("read CRAM header")?;
                Ok((header, AlignmentReader::Cram(reader)))
            }
        }
    }
}

impl AlignmentReader {
    /// Iterate every record in the file, calling `f` on each. The
    /// callback receives `&dyn Record` so the call sites work
    /// regardless of whether the source is BAM or CRAM. Returns the
    /// number of records visited.
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
            AlignmentReader::BamIndexed(r) => {
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
            AlignmentReader::CramIndexed(r) => {
                for rec in r.records(header) {
                    let rec = rec.context("read CRAM record")?;
                    f(&rec)?;
                    n += 1;
                }
            }
        }
        Ok(n)
    }

    /// Iterate records overlapping `ref_name:start-end` (1-based,
    /// inclusive). For indexed BAM, uses the `.bai`/`.csi` to fetch
    /// only overlapping BGZF blocks. For unindexed BAM and CRAM,
    /// falls back to a full scan with a coordinate filter.
    ///
    /// `start`/`end` are 0-based half-open from the caller's
    /// perspective (matching the rest of the dv-io API); we convert
    /// to 1-based inclusive for noodles internally.
    pub fn for_each_record_in_region<F>(
        &mut self,
        header: &Header,
        ref_name: &str,
        start: i64,
        end: i64,
        mut f: F,
    ) -> Result<usize>
    where
        F: FnMut(&dyn Record) -> Result<()>,
    {
        match self {
            AlignmentReader::BamIndexed(r) => {
                use noodles::core::{Position, Region};
                // 0-based half-open [start, end) → 1-based inclusive [start+1, end].
                let s = Position::try_from(usize::try_from(start.max(0) + 1)?)?;
                let e = Position::try_from(usize::try_from(end.max(start + 1))?)?;
                let region = Region::new(ref_name.to_string(), s..=e);
                let mut n = 0usize;
                let query = r
                    .query(header, &region)
                    .with_context(|| format!("query indexed BAM {ref_name}:{start}-{end}"))?;
                for rec in query {
                    let rec = rec.context("read BAM record")?;
                    f(&rec)?;
                    n += 1;
                }
                Ok(n)
            }
            AlignmentReader::CramIndexed(r) => {
                use noodles::core::{Position, Region};
                let s = Position::try_from(usize::try_from(start.max(0) + 1)?)?;
                let e = Position::try_from(usize::try_from(end.max(start + 1))?)?;
                let region = Region::new(ref_name.to_string(), s..=e);
                let mut n = 0usize;
                let query = r
                    .query(header, &region)
                    .with_context(|| format!("query indexed CRAM {ref_name}:{start}-{end}"))?;
                for rec in query {
                    let rec = rec.context("read CRAM record")?;
                    f(&rec)?;
                    n += 1;
                }
                Ok(n)
            }
            // No index → full scan + caller-side filter (caller already
            // filters by region in its callback).
            AlignmentReader::Bam(_) | AlignmentReader::Cram(_) => self.for_each_record(header, f),
        }
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

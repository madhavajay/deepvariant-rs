//! Indexed FASTA reader via noodles. Wraps the path-based open + per-base
//! and per-range fetch helpers used by allele counter / pileup builder /
//! gVCF emitter.

use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result};

pub use noodles::fasta;

/// Type-erased reader handle. We re-borrow on each query.
pub struct Indexed {
    inner: RefCell<Box<dyn FastaReader>>,
}

trait FastaReader {
    fn query_bases(&mut self, region: &noodles::core::Region) -> Option<Vec<u8>>;
}

impl<R: std::io::BufRead + std::io::Seek + 'static> FastaReader
    for noodles::fasta::IndexedReader<R>
{
    fn query_bases(&mut self, region: &noodles::core::Region) -> Option<Vec<u8>> {
        self.query(region)
            .ok()
            .map(|rec| rec.sequence().as_ref().iter().map(|b| b.to_ascii_uppercase()).collect())
    }
}

pub fn open_indexed(path: impl AsRef<Path>) -> Result<Indexed> {
    let path = path.as_ref();
    let reader = noodles::fasta::indexed_reader::Builder::default()
        .build_from_path(path)
        .with_context(|| format!("open FASTA {}", path.display()))?;
    Ok(Indexed {
        inner: RefCell::new(Box::new(reader)),
    })
}

impl Indexed {
    pub fn fetch_base(&self, contig: &str, pos: i64) -> Option<u8> {
        let region = format!("{}:{}-{}", contig, pos + 1, pos + 1).parse().ok()?;
        self.inner.borrow_mut().query_bases(&region).and_then(|v| v.into_iter().next())
    }

    pub fn fetch_range(&self, contig: &str, start: i64, end: i64) -> Option<Vec<u8>> {
        if end <= start {
            return Some(Vec::new());
        }
        let region = format!("{}:{}-{}", contig, start + 1, end).parse().ok()?;
        self.inner.borrow_mut().query_bases(&region)
    }
}

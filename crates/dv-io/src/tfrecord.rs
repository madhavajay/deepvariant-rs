//! TFRecord codec.
//!
//! Frame layout (little-endian):
//!   u64 length
//!   u32 masked-CRC32C(length)
//!   length × u8 payload
//!   u32 masked-CRC32C(payload)
//!
//! Mask transform from TF source (`tensorflow/core/lib/hash/crc32c.h`):
//!   masked = ((crc >> 15) | (crc << 17)) + 0xa282ead8
//!
//! Upstream DeepVariant emits `*.tfrecord.gz` shards (GZIP-wrapped). The
//! `open_reader`/`open_writer` helpers transparently handle `.gz` paths.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;

const MASK_DELTA: u32 = 0xa282_ead8;

#[inline]
fn mask(crc: u32) -> u32 {
    ((crc >> 15) | (crc << 17)).wrapping_add(MASK_DELTA)
}

#[cfg(test)]
#[inline]
fn unmask(masked: u32) -> u32 {
    let rot = masked.wrapping_sub(MASK_DELTA);
    (rot >> 17) | (rot << 15)
}

pub struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    /// Read one record. Returns `Ok(None)` at clean EOF.
    pub fn read_record(&mut self) -> io::Result<Option<Vec<u8>>> {
        let mut len_buf = [0u8; 8];
        match read_exact_or_eof(&mut self.inner, &mut len_buf)? {
            ReadOutcome::Eof => return Ok(None),
            ReadOutcome::Full => {}
        }
        let len = u64::from_le_bytes(len_buf);

        let mut len_crc_buf = [0u8; 4];
        self.inner.read_exact(&mut len_crc_buf)?;
        let len_crc_actual = u32::from_le_bytes(len_crc_buf);
        let len_crc_expected = mask(crc32c::crc32c(&len_buf));
        if len_crc_actual != len_crc_expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "TFRecord length CRC mismatch: got {len_crc_actual:#010x}, expected {len_crc_expected:#010x}"
                ),
            ));
        }

        let mut payload = vec![0u8; len as usize];
        self.inner.read_exact(&mut payload)?;

        let mut payload_crc_buf = [0u8; 4];
        self.inner.read_exact(&mut payload_crc_buf)?;
        let payload_crc_actual = u32::from_le_bytes(payload_crc_buf);
        let payload_crc_expected = mask(crc32c::crc32c(&payload));
        if payload_crc_actual != payload_crc_expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "TFRecord payload CRC mismatch: got {payload_crc_actual:#010x}, expected {payload_crc_expected:#010x}"
                ),
            ));
        }

        Ok(Some(payload))
    }

    pub fn records(self) -> Records<R> {
        Records { reader: self }
    }
}

pub struct Records<R: Read> {
    reader: Reader<R>,
}

impl<R: Read> Iterator for Records<R> {
    type Item = io::Result<Vec<u8>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.reader.read_record().transpose()
    }
}

pub struct Writer<W: Write> {
    inner: W,
}

impl<W: Write> Writer<W> {
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    pub fn write_record(&mut self, payload: &[u8]) -> io::Result<()> {
        let len_buf = (payload.len() as u64).to_le_bytes();
        let len_crc = mask(crc32c::crc32c(&len_buf));
        let payload_crc = mask(crc32c::crc32c(payload));

        self.inner.write_all(&len_buf)?;
        self.inner.write_all(&len_crc.to_le_bytes())?;
        self.inner.write_all(payload)?;
        self.inner.write_all(&payload_crc.to_le_bytes())?;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}

/// Open a TFRecord file, transparently handling `.gz` shards.
pub fn open_reader(path: impl AsRef<Path>) -> io::Result<Reader<Box<dyn Read>>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let buf = BufReader::new(file);
    let inner: Box<dyn Read> = if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        Box::new(MultiGzDecoder::new(buf))
    } else {
        Box::new(buf)
    };
    Ok(Reader::new(inner))
}

/// Create a TFRecord file, transparently GZIP-wrapping `.gz` paths.
pub fn open_writer(path: impl AsRef<Path>) -> io::Result<Writer<Box<dyn Write>>> {
    let path = path.as_ref();
    let file = File::create(path)?;
    let buf = BufWriter::new(file);
    let inner: Box<dyn Write> = if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        Box::new(GzEncoder::new(buf, Compression::default()))
    } else {
        Box::new(buf)
    };
    Ok(Writer::new(inner))
}

enum ReadOutcome {
    Full,
    Eof,
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<ReadOutcome> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => {
                return if filled == 0 {
                    Ok(ReadOutcome::Eof)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "TFRecord truncated mid-frame",
                    ))
                };
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(ReadOutcome::Full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn mask_roundtrip() {
        for crc in [0u32, 1, 0xdead_beef, 0xffff_ffff, 0x1234_5678] {
            assert_eq!(unmask(mask(crc)), crc);
        }
    }

    /// Mask of CRC32C of an empty buffer.
    /// CRC32C("") = 0; masked = ((0>>15)|(0<<17)) + 0xa282ead8 = 0xa282ead8.
    #[test]
    fn known_mask_vector() {
        assert_eq!(mask(crc32c::crc32c(&[])), 0xa282_ead8);
    }

    #[test]
    fn roundtrip_payloads() {
        let mut buf = Vec::new();
        let mut w = Writer::new(&mut buf);
        let payloads: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"a".to_vec(),
            b"hello, world".to_vec(),
            (0u8..=255).collect(),
            vec![0xab; 100_000],
        ];
        for p in &payloads {
            w.write_record(p).unwrap();
        }
        drop(w);

        let mut r = Reader::new(Cursor::new(buf));
        for expected in &payloads {
            let got = r.read_record().unwrap().unwrap();
            assert_eq!(&got, expected);
        }
        assert!(r.read_record().unwrap().is_none());
    }

    #[test]
    fn detects_payload_crc_corruption() {
        let mut buf = Vec::new();
        let mut w = Writer::new(&mut buf);
        w.write_record(b"hello").unwrap();
        // Flip a payload byte.
        let payload_offset = 8 + 4; // length + len_crc
        buf[payload_offset] ^= 0x01;
        let mut r = Reader::new(Cursor::new(buf));
        let err = r.read_record().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn detects_length_crc_corruption() {
        let mut buf = Vec::new();
        let mut w = Writer::new(&mut buf);
        w.write_record(b"hello").unwrap();
        // Flip a length byte.
        buf[0] ^= 0x01;
        let mut r = Reader::new(Cursor::new(buf));
        let err = r.read_record().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn truncated_mid_frame_errors() {
        let mut buf = Vec::new();
        let mut w = Writer::new(&mut buf);
        w.write_record(b"hello").unwrap();
        buf.truncate(buf.len() - 4); // drop trailing payload CRC
        let mut r = Reader::new(Cursor::new(buf));
        let err = r.read_record().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}

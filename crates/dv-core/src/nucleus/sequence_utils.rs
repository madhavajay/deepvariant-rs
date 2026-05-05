//! Port of `third_party/nucleus/util/sequence_utils.py`.

const COMPLEMENT_TABLE: [u8; 128] = {
    let mut t = [b'N'; 128];
    let pairs: [(u8, u8); 11] = [
        (b'A', b'T'),
        (b'T', b'A'),
        (b'C', b'G'),
        (b'G', b'C'),
        (b'N', b'N'),
        (b'a', b't'),
        (b't', b'a'),
        (b'c', b'g'),
        (b'g', b'c'),
        (b'n', b'n'),
        (b'-', b'-'),
    ];
    let mut i = 0;
    while i < pairs.len() {
        t[pairs[i].0 as usize] = pairs[i].1;
        i += 1;
    }
    t
};

/// In-place complement of a DNA sequence (A↔T, C↔G, case preserved, N→N).
pub fn complement(seq: &mut [u8]) {
    for b in seq {
        *b = COMPLEMENT_TABLE[(*b as usize) & 0x7f];
    }
}

/// Reverse complement of a DNA sequence.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = seq.iter().rev().copied().collect();
    complement(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rc_basic() {
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
        assert_eq!(reverse_complement(b"AAAA"), b"TTTT");
        assert_eq!(reverse_complement(b"GATC"), b"GATC"); // palindrome
        assert_eq!(reverse_complement(b"AAGG"), b"CCTT");
    }

    #[test]
    fn rc_preserves_case() {
        assert_eq!(reverse_complement(b"acgT"), b"Acgt");
    }

    #[test]
    fn rc_handles_n() {
        assert_eq!(reverse_complement(b"ACNGT"), b"ACNGT");
    }

    #[test]
    fn complement_in_place() {
        let mut s = b"ACGT".to_vec();
        complement(&mut s);
        assert_eq!(s, b"TGCA");
    }
}

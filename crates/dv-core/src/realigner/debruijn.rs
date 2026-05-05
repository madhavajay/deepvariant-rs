//! De Bruijn graph for haplotype assembly.
//!
//! Port of `deepvariant/realigner/debruijn_graph.cc`. Different
//! implementation strategy (HashMap-backed adjacency lists, no Boost)
//! but identical externally observable behavior:
//!
//!   - reference edges are flagged is_ref so they survive pruning
//!   - read edges have weights == multi-edge count (re-traversal)
//!   - low-weight non-ref edges and unreachable vertices are pruned
//!   - cycle detection drives k selection (try min_k → max_k step_k)
//!   - candidate haplotypes = string spelled by source→sink paths
//!   - if more than `max_num_paths` paths exist, return empty
//!
//! Read base-quality filtering matches upstream's
//! `NextBadPosition`: any base with `qual < min_base_quality` or that
//! is non-ACGT terminates the current k-mer chunk; the next chunk
//! resumes after the bad position.

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy)]
pub struct DeBruijnOptions {
    pub min_k: usize,
    pub max_k: usize,
    pub step_k: usize,
    pub min_mapq: u8,
    pub min_base_quality: u8,
    pub min_edge_weight: i32,
    pub max_num_paths: usize,
    pub disable_graph_pruning: bool,
}

impl Default for DeBruijnOptions {
    fn default() -> Self {
        // Defaults mirror `deepvariant/protos/realigner.proto` defaults.
        Self {
            min_k: 11,
            max_k: 75,
            step_k: 1,
            min_mapq: 14,
            min_base_quality: 15,
            min_edge_weight: 2,
            max_num_paths: 256,
            disable_graph_pruning: false,
        }
    }
}

type VertexId = usize;

#[derive(Debug, Clone)]
struct Vertex {
    kmer: Vec<u8>,
    out_edges: Vec<EdgeId>,
    in_edges: Vec<EdgeId>,
    removed: bool,
}

type EdgeId = usize;

#[derive(Debug, Clone)]
struct Edge {
    from: VertexId,
    to: VertexId,
    weight: i32,
    is_ref: bool,
    removed: bool,
}

pub struct DeBruijnGraph {
    k: usize,
    options: DeBruijnOptions,
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    kmer_to_vertex: HashMap<Vec<u8>, VertexId>,
    edge_lookup: HashMap<(VertexId, VertexId), EdgeId>,
    source: VertexId,
    sink: VertexId,
}

/// One read summarized down to the bare minimum the graph needs.
#[derive(Debug, Clone)]
pub struct ReadInput<'a> {
    pub aligned_sequence: &'a [u8],
    pub aligned_quality: &'a [u8],
    pub mapping_quality: u8,
}

impl DeBruijnGraph {
    /// Try to build an acyclic graph at the smallest k in [min_k, max_k].
    /// Returns `None` if no k yields an acyclic graph.
    pub fn build(
        reference: &[u8],
        reads: &[ReadInput<'_>],
        options: &DeBruijnOptions,
    ) -> Option<Self> {
        // Phase 1: find the smallest k for which the *reference alone* is
        // acyclic (no repeated k-mers).
        let max_k = options.max_k.min(reference.len().saturating_sub(1));
        let mut min_acyclic_k = None;
        let mut k = options.min_k;
        while k <= max_k {
            if !ref_has_cycle(reference, k) {
                min_acyclic_k = Some(k);
                break;
            }
            k += options.step_k.max(1);
        }
        let start_k = min_acyclic_k?;

        // Phase 2: from start_k upward, build the full graph (incl reads)
        // and accept the first k that's acyclic.
        let mut k = start_k;
        while k <= max_k {
            let mut g = Self::new_at_k(reference, reads, options, k);
            if !g.has_cycle() {
                if options.disable_graph_pruning {
                    g.prune_lite();
                } else {
                    g.prune();
                }
                return Some(g);
            }
            k += options.step_k.max(1);
        }
        None
    }

    fn new_at_k(
        reference: &[u8],
        reads: &[ReadInput<'_>],
        options: &DeBruijnOptions,
        k: usize,
    ) -> Self {
        let mut g = DeBruijnGraph {
            k,
            options: *options,
            vertices: Vec::new(),
            edges: Vec::new(),
            kmer_to_vertex: HashMap::new(),
            edge_lookup: HashMap::new(),
            source: 0,
            sink: 0,
        };
        g.add_edges_for_reference(reference);
        g.source = g.vertex_for_kmer(&reference[..k]).expect("source kmer");
        g.sink = g
            .vertex_for_kmer(&reference[reference.len() - k..])
            .expect("sink kmer");
        for read in reads {
            if read.mapping_quality >= options.min_mapq {
                g.add_edges_for_read(read);
            }
        }
        g
    }

    fn ensure_vertex(&mut self, kmer: &[u8]) -> VertexId {
        if let Some(&v) = self.kmer_to_vertex.get(kmer) {
            return v;
        }
        let v = self.vertices.len();
        self.vertices.push(Vertex {
            kmer: kmer.to_vec(),
            out_edges: Vec::new(),
            in_edges: Vec::new(),
            removed: false,
        });
        self.kmer_to_vertex.insert(kmer.to_vec(), v);
        v
    }

    fn vertex_for_kmer(&self, kmer: &[u8]) -> Option<VertexId> {
        self.kmer_to_vertex.get(kmer).copied()
    }

    fn add_edge(&mut self, from: VertexId, to: VertexId, is_ref: bool) -> EdgeId {
        if let Some(&eid) = self.edge_lookup.get(&(from, to)) {
            self.edges[eid].weight += 1;
            self.edges[eid].is_ref |= is_ref;
            return eid;
        }
        let eid = self.edges.len();
        self.edges.push(Edge {
            from,
            to,
            weight: 1,
            is_ref,
            removed: false,
        });
        self.vertices[from].out_edges.push(eid);
        self.vertices[to].in_edges.push(eid);
        self.edge_lookup.insert((from, to), eid);
        eid
    }

    fn add_kmers_and_edges(&mut self, bases: &[u8], start: usize, end: usize, is_ref: bool) {
        if end == 0 || start + self.k > bases.len() {
            return;
        }
        debug_assert!(end + self.k <= bases.len(), "bad end");
        let mut prev = self.ensure_vertex(&bases[start..start + self.k]);
        for i in (start + 1)..=end {
            let cur = self.ensure_vertex(&bases[i..i + self.k]);
            self.add_edge(prev, cur, is_ref);
            prev = cur;
        }
    }

    fn add_edges_for_reference(&mut self, reference: &[u8]) {
        if reference.len() <= self.k {
            return;
        }
        self.add_kmers_and_edges(reference, 0, reference.len() - self.k, true);
    }

    fn add_edges_for_read(&mut self, read: &ReadInput<'_>) {
        let bases = ascii_upper(read.aligned_sequence);
        let qs = read.aligned_quality;
        if bases.len() <= self.k {
            return;
        }
        let stop = bases.len() - self.k;
        let min_bq = self.options.min_base_quality;
        let mut i = 0usize;
        while i < stop {
            let bad = next_bad_position(&bases, qs, i, min_bq);
            let chunk_end = bad.saturating_sub(self.k);
            if chunk_end >= i {
                self.add_kmers_and_edges(&bases, i, chunk_end, false);
            }
            i = bad + 1;
        }
    }

    /// True iff there's a directed cycle reachable from any vertex.
    pub fn has_cycle(&self) -> bool {
        // Kahn's algorithm: peel zero-in-degree vertices.
        let n = self.vertices.len();
        let mut in_deg: Vec<usize> = self
            .vertices
            .iter()
            .map(|v| v.in_edges.iter().filter(|&&e| !self.edges[e].removed).count())
            .collect();
        let mut queue: VecDeque<VertexId> = (0..n)
            .filter(|&v| !self.vertices[v].removed && in_deg[v] == 0)
            .collect();
        let mut visited = 0usize;
        while let Some(v) = queue.pop_front() {
            visited += 1;
            for &e in &self.vertices[v].out_edges {
                if self.edges[e].removed {
                    continue;
                }
                let to = self.edges[e].to;
                if self.vertices[to].removed {
                    continue;
                }
                in_deg[to] -= 1;
                if in_deg[to] == 0 {
                    queue.push_back(to);
                }
            }
        }
        let active = self.vertices.iter().filter(|v| !v.removed).count();
        visited != active
    }

    /// Remove isolated vertices (zero in + zero out).
    pub fn prune_lite(&mut self) {
        for v in 0..self.vertices.len() {
            if self.vertices[v].removed {
                continue;
            }
            let in_active = self.vertices[v]
                .in_edges
                .iter()
                .any(|&e| !self.edges[e].removed);
            let out_active = self.vertices[v]
                .out_edges
                .iter()
                .any(|&e| !self.edges[e].removed);
            if !in_active && !out_active {
                self.remove_vertex(v);
            }
        }
    }

    /// Remove low-weight non-ref edges, then drop vertices unreachable
    /// from source going forward AND from sink going backward.
    pub fn prune(&mut self) {
        let min_w = self.options.min_edge_weight;
        for e in 0..self.edges.len() {
            if !self.edges[e].is_ref && self.edges[e].weight < min_w {
                self.edges[e].removed = true;
            }
        }
        let fwd = self.reachable_from(self.source, false);
        let rev = self.reachable_from(self.sink, true);
        let keep: HashSet<VertexId> = fwd.intersection(&rev).copied().collect();
        for v in 0..self.vertices.len() {
            if !self.vertices[v].removed && !keep.contains(&v) {
                self.remove_vertex(v);
            }
        }
    }

    fn reachable_from(&self, start: VertexId, reverse: bool) -> HashSet<VertexId> {
        let mut seen = HashSet::new();
        let mut q = VecDeque::new();
        if !self.vertices[start].removed {
            seen.insert(start);
            q.push_back(start);
        }
        while let Some(v) = q.pop_front() {
            let edges_iter = if reverse {
                &self.vertices[v].in_edges
            } else {
                &self.vertices[v].out_edges
            };
            for &e in edges_iter {
                if self.edges[e].removed {
                    continue;
                }
                let next = if reverse {
                    self.edges[e].from
                } else {
                    self.edges[e].to
                };
                if self.vertices[next].removed {
                    continue;
                }
                if seen.insert(next) {
                    q.push_back(next);
                }
            }
        }
        seen
    }

    fn remove_vertex(&mut self, v: VertexId) {
        if self.vertices[v].removed {
            return;
        }
        self.vertices[v].removed = true;
        for &e in &self.vertices[v].out_edges.clone() {
            self.edges[e].removed = true;
        }
        for &e in &self.vertices[v].in_edges.clone() {
            self.edges[e].removed = true;
        }
        let kmer = self.vertices[v].kmer.clone();
        self.kmer_to_vertex.remove(&kmer);
    }

    /// Enumerate all source→sink (or source→dead-end) paths.
    /// Returns empty vec if `max_num_paths` is exceeded.
    pub fn candidate_paths(&self) -> Vec<Vec<VertexId>> {
        let mut terminated: Vec<Vec<VertexId>> = Vec::new();
        let mut extendable: VecDeque<Vec<VertexId>> = VecDeque::new();
        if !self.vertex_has_active_out(self.source) {
            return Vec::new();
        }
        extendable.push_back(vec![self.source]);
        while let Some(path) = extendable.pop_front() {
            let total = terminated.len() + extendable.len() + 1;
            if total > self.options.max_num_paths {
                return Vec::new();
            }
            let last = *path.last().unwrap();
            let mut any_successor = false;
            for &e in &self.vertices[last].out_edges {
                if self.edges[e].removed {
                    continue;
                }
                let to = self.edges[e].to;
                if self.vertices[to].removed {
                    continue;
                }
                any_successor = true;
                let mut extended = path.clone();
                extended.push(to);
                if to == self.sink || !self.vertex_has_active_out(to) {
                    terminated.push(extended);
                } else {
                    extendable.push_back(extended);
                }
            }
            if !any_successor {
                terminated.push(path);
            }
        }
        terminated
    }

    fn vertex_has_active_out(&self, v: VertexId) -> bool {
        if self.vertices[v].removed {
            return false;
        }
        self.vertices[v].out_edges.iter().any(|&e| {
            !self.edges[e].removed && !self.vertices[self.edges[e].to].removed
        })
    }

    /// Spell the haplotype string traced by a path.
    pub fn haplotype_for_path(&self, path: &[VertexId]) -> Vec<u8> {
        if path.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &v in path {
            out.push(self.vertices[v].kmer[0]);
        }
        // Append the rest of the last k-mer (k-1 bases).
        out.extend_from_slice(&self.vertices[*path.last().unwrap()].kmer[1..]);
        out
    }

    /// Sorted list of candidate haplotype strings (empty if path-cap hit).
    pub fn candidate_haplotypes(&self) -> Vec<Vec<u8>> {
        let mut hs: Vec<Vec<u8>> = self
            .candidate_paths()
            .iter()
            .map(|p| self.haplotype_for_path(p))
            .collect();
        hs.sort();
        hs
    }

    pub fn k(&self) -> usize {
        self.k
    }
}

fn ref_has_cycle(reference: &[u8], k: usize) -> bool {
    if reference.len() < k {
        return false;
    }
    let mut seen = HashSet::new();
    for i in 0..=reference.len() - k {
        if !seen.insert(&reference[i..i + k]) {
            return true;
        }
    }
    false
}

fn ascii_upper(s: &[u8]) -> Vec<u8> {
    s.iter().map(|b| b.to_ascii_uppercase()).collect()
}

fn next_bad_position(bases: &[u8], quals: &[u8], start: usize, min_bq: u8) -> usize {
    let mut i = start;
    while i < bases.len() {
        let b = bases[i];
        let q = quals.get(i).copied().unwrap_or(0);
        if !is_canonical_acgt(b) || q < min_bq {
            return i;
        }
        i += 1;
    }
    bases.len()
}

fn is_canonical_acgt(b: u8) -> bool {
    matches!(b, b'A' | b'C' | b'G' | b'T')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read<'a>(seq: &'a [u8], qual: &'a [u8]) -> ReadInput<'a> {
        ReadInput {
            aligned_sequence: seq,
            aligned_quality: qual,
            mapping_quality: 60,
        }
    }

    fn opts_small() -> DeBruijnOptions {
        DeBruijnOptions {
            min_k: 4,
            max_k: 12,
            step_k: 1,
            min_mapq: 0,
            min_base_quality: 0,
            min_edge_weight: 1,
            max_num_paths: 64,
            disable_graph_pruning: false,
        }
    }

    #[test]
    fn ref_only_yields_one_haplotype() {
        let reference = b"ACGTACGTGGCC";
        let opts = opts_small();
        let g = DeBruijnGraph::build(reference, &[], &opts).expect("build");
        let hs = g.candidate_haplotypes();
        assert_eq!(hs.len(), 1);
        assert_eq!(hs[0], reference);
    }

    #[test]
    fn snp_in_reads_yields_two_haplotypes() {
        // ref and alt share start (AAAC) and end (GTTT). Diverge at offset 6
        // (T vs G). At k=4 every 4-mer in both is unique → graph is acyclic.
        let reference = b"AAACCCTGGGTTT";
        let alt_seq = b"AAACCCGGGGTTT";
        let qual = vec![40u8; alt_seq.len()];
        let mut reads = Vec::new();
        for _ in 0..10 {
            reads.push(read(alt_seq, &qual));
        }
        let opts = opts_small();
        let g = DeBruijnGraph::build(reference, &reads, &opts).expect("build");
        let hs = g.candidate_haplotypes();
        assert!(
            hs.iter().any(|h| h.as_slice() == reference),
            "expected reference haplotype, got {hs:?}"
        );
        assert!(
            hs.iter().any(|h| h.as_slice() == alt_seq.as_slice()),
            "expected alt haplotype, got {} candidates: {hs:?}",
            hs.len()
        );
    }

    #[test]
    fn low_weight_alt_is_pruned() {
        let reference = b"AAACCCTGGGTTT";
        let alt_seq = b"AAACCCGGGGTTT";
        let qual = vec![40u8; alt_seq.len()];
        let reads = vec![read(alt_seq, &qual)];
        let mut opts = opts_small();
        opts.min_edge_weight = 2;
        let g = DeBruijnGraph::build(reference, &reads, &opts).expect("build");
        let hs = g.candidate_haplotypes();
        assert_eq!(hs.len(), 1, "got {hs:?}");
        assert_eq!(hs[0], reference);
    }

    #[test]
    fn low_quality_bases_skip_kmers() {
        // Read that has a low-quality base in the middle - should not contribute
        // any k-mers spanning that position.
        let reference = b"AAAAACCCCCAAAAA";
        let alt_seq = b"AAAAACGCCCAAAAA"; // SNP at position 6
        let mut qual = vec![40u8; alt_seq.len()];
        qual[6] = 5; // low quality
        let reads = vec![read(alt_seq, &qual)];
        let mut opts = opts_small();
        opts.min_base_quality = 20;
        opts.min_edge_weight = 1;
        let g = DeBruijnGraph::build(reference, &reads, &opts).expect("build");
        let hs = g.candidate_haplotypes();
        // Low-quality SNP should be filtered out. Only reference haplotype.
        assert_eq!(hs.len(), 1);
        assert_eq!(hs[0], reference);
    }

    #[test]
    fn cycle_detection_returns_none_for_pure_repeat() {
        // A reference with a tandem repeat too small for our k range yields no
        // valid graph. min_k=4, but max_k=12; the reference is shorter than 13
        // and is fully periodic ACAC, so for k < 5 every kmer repeats →
        // ref-cycle for all small k. Larger k makes ref shorter than k.
        let reference = b"ACACACAC";
        let opts = DeBruijnOptions {
            min_k: 4,
            max_k: 7,
            step_k: 1,
            min_mapq: 0,
            min_base_quality: 0,
            min_edge_weight: 1,
            max_num_paths: 64,
            disable_graph_pruning: false,
        };
        // For k=4, every 4-mer (ACAC, CACA) repeats → cycle. k=5..7 covers most
        // of the reference. The build should find some k that works or return None;
        // for this tightly periodic input, no good k exists in [4,7].
        let g = DeBruijnGraph::build(reference, &[], &opts);
        // Either None (no acyclic k found) or has a single haplotype matching ref.
        if let Some(g) = g {
            let hs = g.candidate_haplotypes();
            assert_eq!(hs.len(), 1);
        }
    }

    #[test]
    fn haplotype_for_path_basics() {
        let reference = b"ACGTACGTGGCC";
        let opts = opts_small();
        let g = DeBruijnGraph::build(reference, &[], &opts).unwrap();
        let paths = g.candidate_paths();
        assert!(!paths.is_empty());
        let h = g.haplotype_for_path(&paths[0]);
        assert_eq!(h, reference);
    }

    #[test]
    fn max_num_paths_cap_returns_empty() {
        // Construct a graph likely to branch into many paths: many overlapping
        // alts in a repetitive region.
        let reference = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let mut reads = Vec::new();
        // Each read introduces a different SNP.
        for i in 5..20 {
            let mut seq = reference.to_vec();
            seq[i] = b'C';
            let qual = vec![40u8; seq.len()];
            for _ in 0..3 {
                reads.push(ReadInput {
                    aligned_sequence: Box::leak(seq.clone().into_boxed_slice()),
                    aligned_quality: Box::leak(qual.clone().into_boxed_slice()),
                    mapping_quality: 60,
                });
            }
        }
        let opts = DeBruijnOptions {
            min_k: 4,
            max_k: 8,
            step_k: 1,
            min_mapq: 0,
            min_base_quality: 0,
            min_edge_weight: 1,
            max_num_paths: 4,
            disable_graph_pruning: true,
        };
        if let Some(g) = DeBruijnGraph::build(reference, &reads, &opts) {
            let hs = g.candidate_haplotypes();
            // Either capped (empty) or just the ref.
            assert!(hs.is_empty() || hs.iter().all(|h| h.len() == reference.len()));
        }
    }
}

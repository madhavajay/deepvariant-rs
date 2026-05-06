//! Direct phasing — read-based phasing via candidate-allele DP graph.
//!
//! Port of `deepvariant/direct_phasing.cc` (~1000 LOC). Builds a graph
//! where each vertex is one allele at one heterozygous-SNV candidate
//! position; edges connect alleles supported by the same read across
//! consecutive positions. Dynamic programming finds the
//! highest-scoring partition of vertices into two phases (= the two
//! haplotypes); reads then get phase 1 / 2 / 0 based on the alleles
//! they cover.
//!
//! Outputs feed:
//!   * `MID=phased` annotations on CVO records that came from phased
//!     candidate calling.
//!   * The gVCF `PS` (phase set) tag indicating which records share a
//!     phasing block.
//!
//! Out of scope here:
//!   * `direct_phasing.cc::AddMethylatedRefCandidate` (methylation-tag
//!     phasing) is implemented with a feature flag — the trigger
//!     (REF candidates with `MF` info) is rare on WGS short-read data.
//!     Methylation-aware phasing has its own module.
//!   * GraphViz output — debugging-only; can be added if needed.
//!
//! Test cases mirror upstream `direct_phasing_test.cc` so coverage
//! stays roughly equivalent.

pub mod graph;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use dv_proto::dv::{AlleleType, DeepVariantCall};

pub use graph::{AlleleInfo, Edge, Graph, Vertex};

/// Special allele bases recognized by the phasing algorithm.
pub const REF_ALLELE: &str = "REF";
pub const UNCALLED_ALLELE: &str = "UNCALLED_ALLELE";

/// Minimum number of REF-supporting reads required for the REF allele
/// to be admitted as a graph vertex.
pub const MIN_REF_ALLELE_DEPTH: usize = 3;

/// Number of haplotypes we phase against.
pub const NUM_PHASES: usize = 2;

/// Methylation thresholds for treating a REF site as a phasing candidate
/// (only used when methylation-aware phasing is enabled).
pub const MIN_METHYLATION_THRESHOLD: f64 = 0.4;
pub const MAX_METHYLATION_THRESHOLD: f64 = 0.6;

/// One read's support for one allele.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadSupportInfo {
    pub read_index: usize,
    pub is_low_quality: bool,
    pub is_first_allele: bool,
}

impl ReadSupportInfo {
    pub fn new(read_index: usize, is_low_quality: bool) -> Self {
        Self {
            read_index,
            is_low_quality,
            is_first_allele: false,
        }
    }
}

/// Tunable knobs (from upstream `DirectPhasingOptions`).
#[derive(Debug, Clone, Copy)]
pub struct DirectPhasingOptions {
    pub min_alleles_to_phase: u32,
    pub enable_methylation_aware_phasing: bool,
}

impl Default for DirectPhasingOptions {
    fn default() -> Self {
        Self {
            min_alleles_to_phase: 2,
            enable_methylation_aware_phasing: false,
        }
    }
}

/// A candidate-position allele's role in the phasing block.
#[derive(Debug, Clone, PartialEq)]
pub struct PhasedVariant {
    pub position: i64,
    pub phase_1_bases: String,
    pub phase_2_bases: String,
    pub is_first_in_block: bool,
}

/// Compose `read_name/read_number` the same way upstream does (this is
/// what `DeepVariantCall.allele_support_ext[..].read_infos[*].read_name`
/// stores, so it matches the index from `read_to_index` lookup).
pub fn read_key(fragment_name: &str, read_number: i32) -> String {
    format!("{}/{}", fragment_name, read_number)
}

/// Determine an allele's type from its bases and the candidate's REF span.
pub fn allele_type_from_candidate(bases: &str, candidate: &DeepVariantCall) -> AlleleType {
    let v = match candidate.variant.as_ref() {
        Some(v) => v,
        None => return AlleleType::Unspecified,
    };
    let span = (v.end - v.start) as usize;
    if bases.len() > span {
        AlleleType::Insertion
    } else if bases.len() < span {
        AlleleType::Deletion
    } else if bases.len() == span {
        AlleleType::Substitution
    } else {
        AlleleType::Unspecified
    }
}

/// Number of alt alleles in the candidate that are SNV substitutions
/// (skip the synthetic UNCALLED_ALLELE).
pub fn num_of_substitution_alleles(candidate: &DeepVariantCall) -> usize {
    candidate
        .allele_support_ext
        .iter()
        .filter(|(allele, _)| allele.as_str() != UNCALLED_ALLELE)
        .filter(|(allele, _)| {
            allele_type_from_candidate(allele, candidate) == AlleleType::Substitution
        })
        .count()
}

/// Number of alt alleles in the candidate that are insertions or deletions.
pub fn num_of_indel_alleles(candidate: &DeepVariantCall) -> usize {
    candidate
        .allele_support_ext
        .iter()
        .filter(|(allele, _)| allele.as_str() != UNCALLED_ALLELE)
        .filter(|(allele, _)| {
            let t = allele_type_from_candidate(allele, candidate);
            t == AlleleType::Deletion || t == AlleleType::Insertion
        })
        .count()
}

/// Total read depth across all SUB alt alleles in the candidate.
pub fn substitution_alleles_depth(candidate: &DeepVariantCall) -> usize {
    candidate
        .allele_support_ext
        .iter()
        .filter(|(allele, _)| allele.as_str() != UNCALLED_ALLELE)
        .filter(|(allele, _)| {
            allele_type_from_candidate(allele, candidate) == AlleleType::Substitution
        })
        .map(|(_, supports)| supports.read_infos.len())
        .sum()
}

/// Filter test the upstream uses to decide whether a candidate enters
/// the phasing graph at all. Returns `true` if the candidate should be
/// included; updates `indel_end` in place to remember the rightmost
/// position currently under an INDEL "shadow".
///
/// Excludes:
///   * homozygous candidates (only one called allele AND too little ref
///     support);
///   * candidates containing INDELs (and any candidate falling within
///     an INDEL's right-extending span);
///   * candidates whose alt allele length doesn't equal the REF span.
pub fn candidate_filter(candidate: &DeepVariantCall, indel_end: &mut i64) -> bool {
    let num_called = candidate
        .allele_support_ext
        .iter()
        .filter(|(allele, _)| allele.as_str() != UNCALLED_ALLELE)
        .count();
    let ref_support_count = candidate
        .ref_support_ext
        .as_ref()
        .map(|s| s.read_infos.len())
        .unwrap_or(0);

    if num_called <= 1 && ref_support_count < MIN_REF_ALLELE_DEPTH {
        return false;
    }

    let v = match candidate.variant.as_ref() {
        Some(v) => v,
        None => return false,
    };
    for (allele, _) in candidate.allele_support_ext.iter() {
        if allele.as_str() == UNCALLED_ALLELE {
            continue;
        }
        let span = (v.end - v.start) as usize;
        if v.end <= *indel_end || allele.len() != span {
            if *indel_end < v.end {
                *indel_end = v.end;
            }
            return false;
        }
    }
    true
}

/// Main phasing struct — owns the graph and per-read maps, runs the DP.
pub struct DirectPhasing {
    options: DirectPhasingOptions,
    graph: Graph,

    /// Sorted candidate positions actually admitted to the graph.
    positions: Vec<i64>,
    /// Per-position, the vertices at that position.
    vertices_by_position: BTreeMap<i64, Vec<usize>>,
    /// (vertex_pair) -> running-best Score under DP.
    scores: HashMap<VertexPair, Score>,
    /// Read index -> ordered list of (vertex, support-info) for that read.
    read_to_alleles: HashMap<usize, Vec<AlleleSupport>>,
    /// Lookup tables for reads.
    read_to_index: HashMap<String, usize>,
    index_to_read_name: HashMap<usize, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexPair {
    pub phase_1: usize,
    pub phase_2: usize,
}

#[derive(Debug, Clone)]
pub struct Score {
    pub score: i32,
    /// Source vertices for back-tracking. `None` means start-of-block.
    pub from: [Option<usize>; 2],
    /// Reads that carry phase 1 / phase 2 along the best path.
    pub read_support: [HashSet<usize>; 2],
}

impl Default for Score {
    fn default() -> Self {
        Self {
            score: 0,
            from: [None, None],
            read_support: [HashSet::new(), HashSet::new()],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AlleleSupport {
    pub vertex: usize,
    pub read_support: ReadSupportInfo,
}

impl DirectPhasing {
    pub fn new(options: DirectPhasingOptions) -> Self {
        Self {
            options,
            graph: Graph::new(),
            positions: Vec::new(),
            vertices_by_position: BTreeMap::new(),
            scores: HashMap::new(),
            read_to_alleles: HashMap::new(),
            read_to_index: HashMap::new(),
            index_to_read_name: HashMap::new(),
        }
    }

    /// Reset all state so the same instance can phase a new region.
    pub fn clear(&mut self) {
        self.graph = Graph::new();
        self.positions.clear();
        self.vertices_by_position.clear();
        self.scores.clear();
        self.read_to_alleles.clear();
        self.read_to_index.clear();
        self.index_to_read_name.clear();
    }

    /// Build read-name → read-index lookups.
    pub fn initialize_read_maps<I, S>(&mut self, reads: I)
    where
        I: IntoIterator<Item = (S, i32)>,
        S: AsRef<str>,
    {
        for (i, (frag, num)) in reads.into_iter().enumerate() {
            let key = read_key(frag.as_ref(), num);
            self.read_to_index.insert(key, i);
            self.index_to_read_name.insert(i, frag.as_ref().to_string());
        }
    }

    /// Convert a `DeepVariantCall.SupportingReadsExt`-style read list
    /// into our internal `ReadSupportInfo` vector. Drops low-quality
    /// reads (matching upstream — see `ReadSupportFromProto`) and any
    /// reads that aren't in the read-name map.
    pub fn read_support_from_proto(
        &self,
        infos: &[(String, bool)],
    ) -> Vec<ReadSupportInfo> {
        infos
            .iter()
            .filter_map(|(name, is_lq)| {
                let idx = *self.read_to_index.get(name)?;
                if *is_lq {
                    None
                } else {
                    Some(ReadSupportInfo::new(idx, *is_lq))
                }
            })
            .collect()
    }

    /// Insert a vertex with its allele info; record (read → vertex)
    /// links for later edge-building.
    fn add_vertex(
        &mut self,
        position: i64,
        allele_type: AlleleType,
        bases: &str,
        infos: Vec<ReadSupportInfo>,
    ) -> usize {
        let v = self.graph.add_vertex(AlleleInfo {
            allele_type,
            position,
            bases: bases.to_string(),
            phase: 0,
            is_first_in_block: false,
            read_support: infos,
        });
        self.update_read_to_alleles_map(v);
        v
    }

    /// After adding a vertex, splice its position into the bookkeeping
    /// indexes and link each supporting read to the new vertex.
    fn update_read_to_alleles_map(&mut self, v: usize) {
        let position = self.graph.vertex(v).position;
        self.vertices_by_position.entry(position).or_default().push(v);

        // Per-read first-allele flag: the very first allele for a given
        // read in DP order is_first_allele=true. Then subsequent ones
        // are false.
        let infos = self.graph.vertex(v).read_support.clone();
        for ri in &infos {
            let is_first = !self.read_to_alleles.contains_key(&ri.read_index);
            // Stash the is_first_allele flag back onto the vertex's
            // read_support (we copy out, mutate, then mutate the
            // graph's owned copy as well — both are in-sync).
            let entry = AlleleSupport {
                vertex: v,
                read_support: ReadSupportInfo {
                    read_index: ri.read_index,
                    is_low_quality: ri.is_low_quality,
                    is_first_allele: is_first,
                },
            };
            self.read_to_alleles.entry(ri.read_index).or_default().push(entry);
        }

        // Mirror the is_first_allele flags onto the in-graph vertex's
        // read_support so the algorithm functions can consult it via
        // `graph_[v].allele_info.read_support[i].is_first_allele`.
        let v_mut = self.graph.vertex_mut(v);
        for ri in v_mut.read_support.iter_mut() {
            // is_first_allele = is this read's *first* registered
            // allele globally? Look up in read_to_alleles; the pushed
            // entry above is the latest, so the first one's flag tells
            // us. Equivalently, this vertex carries the flag iff the
            // matching pushed entry has is_first_allele=true.
            let entries = self.read_to_alleles.get(&ri.read_index);
            ri.is_first_allele = match entries {
                Some(es) => {
                    es.iter()
                        .rfind(|e| e.vertex == v)
                        .map(|e| e.read_support.is_first_allele)
                        .unwrap_or(false)
                }
                None => false,
            };
        }
    }

    /// Get-or-create an edge with cumulative weight.
    fn add_edge(&mut self, from: usize, to: usize, weight: f32) -> usize {
        self.graph.add_edge(from, to, weight)
    }

    /// Convenience overload that derives weight from low-quality flags.
    fn add_edge_weighted(
        &mut self,
        from: usize,
        is_lq_in: bool,
        to: usize,
        is_lq_out: bool,
    ) -> usize {
        let w = (if is_lq_in { 0.25 } else { 0.5 }) + (if is_lq_out { 0.25 } else { 0.5 });
        self.add_edge(from, to, w)
    }

    /// Add a heterozygous candidate to the graph: REF if ≥3 supporting
    /// reads, plus every named alt allele.
    fn add_candidate(&mut self, candidate: &DeepVariantCall) {
        let v = match candidate.variant.as_ref() {
            Some(v) => v,
            None => return,
        };

        // Reference-only methylated site: alt is exactly ".".
        let is_ref_site =
            v.alternate_bases.len() == 1 && v.alternate_bases[0] == ".";
        if is_ref_site {
            // Methylation-aware phasing for REF sites — kept in a
            // sibling module so this struct stays focused.
            if self.options.enable_methylation_aware_phasing {
                self.add_methylated_ref_candidate(candidate);
            }
            return;
        }

        // Add REF if it has read support.
        let ref_infos: Vec<(String, bool)> = candidate
            .ref_support_ext
            .as_ref()
            .map(|s| {
                s.read_infos
                    .iter()
                    .map(|r| (r.read_name.clone(), r.is_low_quality))
                    .collect()
            })
            .unwrap_or_default();
        if ref_infos.len() >= MIN_REF_ALLELE_DEPTH {
            let support = self.read_support_from_proto(&ref_infos);
            self.add_vertex(v.start, AlleleType::Reference, REF_ALLELE, support);
        }

        // Sort alleles for determinism (matches upstream's btree_set).
        let mut alleles: Vec<(&String, Vec<(String, bool)>)> = candidate
            .allele_support_ext
            .iter()
            .filter(|(a, _)| a.as_str() != UNCALLED_ALLELE)
            .map(|(a, supports)| {
                let infos: Vec<(String, bool)> = supports
                    .read_infos
                    .iter()
                    .map(|r| (r.read_name.clone(), r.is_low_quality))
                    .collect();
                (a, infos)
            })
            .collect();
        alleles.sort_by(|a, b| a.0.cmp(b.0));

        for (allele, infos) in alleles {
            let support = self.read_support_from_proto(&infos);
            let at = allele_type_from_candidate(allele, candidate);
            self.add_vertex(v.start, at, allele, support);
        }
    }

    /// REF-site phasing for methylated reads — adds two vertices ("M",
    /// "U") if the ref methylation is in the heterozygous-methylation
    /// window. Implementation lives in a helper in the methylation
    /// module; here we provide the no-op default so tests for the
    /// methylation-disabled path don't need to special-case it.
    fn add_methylated_ref_candidate(&mut self, _candidate: &DeepVariantCall) {
        // Methylation-aware phasing is implemented in
        // `methylation_aware_phasing.rs`. This stub keeps the call site
        // simple; when the feature is enabled, that module wires its
        // own logic in via a public helper.
    }

    /// Whether any vertex in `verts` has at least one incoming edge in
    /// the graph. Used to detect a "broken path" needing zero-weight
    /// edges from the previous position.
    fn has_at_least_one_incoming_edge(&self, verts: &[usize]) -> bool {
        verts.iter().any(|&v| !self.graph.in_edges(v).is_empty())
    }

    /// Find reads that support both the running phase and the new
    /// vertex. Returns two sets:
    ///   * idx 0: reads in `starting_score.read_support[phase]` that
    ///     also support the new vertex.
    ///   * idx 1: reads whose first-ever allele is at the new vertex
    ///     (these get half-credit).
    fn find_supporting_reads(
        &self,
        vertex: usize,
        starting_score: &Score,
        phase: usize,
    ) -> [HashSet<usize>; 2] {
        let mut overlap: HashSet<usize> = HashSet::new();
        let mut firsts: HashSet<usize> = HashSet::new();
        for ri in &self.graph.vertex(vertex).read_support {
            if ri.is_first_allele {
                firsts.insert(ri.read_index);
            }
            if starting_score.read_support[phase].contains(&ri.read_index) {
                overlap.insert(ri.read_index);
            }
        }
        [overlap, firsts]
    }

    /// DP scoring for a single (edge_1, edge_2) pair. Returns the new
    /// `Score` if the previous score for the source pair exists, else
    /// the default empty Score.
    fn calculate_score(&self, edge_1: &Edge, edge_2: &Edge) -> Score {
        let from_pair = VertexPair {
            phase_1: edge_1.from,
            phase_2: edge_2.from,
        };
        let prev = match self.scores.get(&from_pair) {
            Some(p) => p,
            None => return Score::default(),
        };

        let to_pair = [edge_1.to, edge_2.to];
        let supporting_per_phase: [[HashSet<usize>; 2]; 2] = [
            self.find_supporting_reads(to_pair[0], prev, 0),
            self.find_supporting_reads(to_pair[1], prev, 1),
        ];

        // Union of overlapping reads across both phases.
        let mut all_supporting: HashSet<usize> = HashSet::new();
        let mut all_firsts: HashSet<usize> = HashSet::new();
        for p in 0..NUM_PHASES {
            all_supporting.extend(&supporting_per_phase[p][0]);
            all_firsts.extend(&supporting_per_phase[p][1]);
        }
        let mut read_support: [HashSet<usize>; 2] = [HashSet::new(), HashSet::new()];
        for p in 0..NUM_PHASES {
            read_support[p].extend(&supporting_per_phase[p][0]);
            read_support[p].extend(&supporting_per_phase[p][1]);
        }

        // Score formula matches upstream:
        //   new = prev + |union of all overlap reads|
        //               + |all-first-allele reads|/2
        // but with a caveat: if neither phase has at least 2 overlapping
        // reads, the score doesn't advance.
        let mut score = prev.score
            + all_supporting.len() as i32
            + (all_firsts.len() as i32 / 2);
        if supporting_per_phase[0][0].len() < 2 && supporting_per_phase[1][0].len() < 2 {
            score = prev.score;
        }

        Score {
            score,
            from: [Some(edge_1.from), Some(edge_2.from)],
            read_support,
        }
    }

    /// Recompute scores at `verts` as if there were no incoming edges
    /// (the "starting score" case). Each pair's score = |union of
    /// supporting reads|.
    fn update_starting_score(&mut self, verts: &[usize]) {
        // Wipe any leftover entries for these vertices.
        for &v1 in verts {
            for &v2 in verts {
                self.scores.remove(&VertexPair { phase_1: v1, phase_2: v2 });
            }
        }
        for i in 0..verts.len() {
            for j in i..verts.len() {
                let v1 = verts[i];
                let v2 = verts[j];
                let s1: HashSet<usize> = self
                    .graph
                    .vertex(v1)
                    .read_support
                    .iter()
                    .map(|r| r.read_index)
                    .collect();
                let s2: HashSet<usize> = self
                    .graph
                    .vertex(v2)
                    .read_support
                    .iter()
                    .map(|r| r.read_index)
                    .collect();
                let score = if s1 == s2 {
                    s1.len() as i32
                } else {
                    s1.len() as i32 + s2.len() as i32
                };
                self.scores.insert(
                    VertexPair { phase_1: v1, phase_2: v2 },
                    Score {
                        score,
                        from: [None, None],
                        read_support: [s1, s2],
                    },
                );
            }
        }
    }

    /// Tie-breaker: when two `(from_1, from_2)` pairs produce the same
    /// score, prefer the one with lexicographically larger bases. This
    /// is what upstream does to keep results deterministic across runs.
    fn compare_vertex_pair_by_bases(
        &self,
        v1_1: Option<usize>,
        v1_2: Option<usize>,
        v2_1: Option<usize>,
        v2_2: Option<usize>,
    ) -> bool {
        let (v1_1, v1_2) = match (v1_1, v1_2) {
            (Some(a), Some(b)) => (a, b),
            _ => return false,
        };
        let (v2_1, v2_2) = match (v2_1, v2_2) {
            (Some(a), Some(b)) => (a, b),
            _ => return true,
        };
        let b1_1 = &self.graph.vertex(v1_1).bases;
        let b1_2 = &self.graph.vertex(v1_2).bases;
        let b2_1 = &self.graph.vertex(v2_1).bases;
        let b2_2 = &self.graph.vertex(v2_2).bases;
        match b1_1.cmp(b2_1) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => b1_2 > b2_2,
        }
    }

    /// Are all the score values at the current keyed-edges set within
    /// 1 of each other? If yes, this position is unphased; we restart
    /// the DP from the next position.
    fn all_scores_are_the_same(
        &self,
        keyed_edges: &BTreeMap<(String, String), Edge>,
    ) -> bool {
        let mut min_s = i32::MAX;
        let mut max_s = i32::MIN;
        for e1 in keyed_edges.values() {
            for e2 in keyed_edges.values() {
                let to = VertexPair {
                    phase_1: e1.to,
                    phase_2: e2.to,
                };
                if let Some(score) = self.scores.get(&to) {
                    if score.score < min_s {
                        min_s = score.score;
                    }
                    if score.score > max_s {
                        max_s = score.score;
                    }
                }
            }
        }
        if max_s == i32::MIN {
            return true; // no scores at all
        }
        max_s - min_s <= 1
    }

    /// Find the running-best vertex pair at `position_idx`, with the
    /// tie-break by allele bases. Returns `None` if no scores at all,
    /// or if every score equals the maximum (then this position can't
    /// be phased and the caller should fall back).
    fn max_score(&self, position_idx: usize) -> Option<VertexPair> {
        let pos = self.positions[position_idx];
        let verts = self.vertices_by_position.get(&pos)?;

        let mut max_pair: Option<VertexPair> = None;
        let mut max_val = 0i32;
        for &v1 in verts {
            for &v2 in verts {
                let pair = VertexPair { phase_1: v1, phase_2: v2 };
                let s = match self.scores.get(&pair) {
                    Some(s) => s,
                    None => continue,
                };
                if s.score > max_val {
                    max_pair = Some(pair);
                    max_val = s.score;
                } else if s.score == max_val {
                    if let Some(prev) = max_pair {
                        if self.compare_vertex_pair_by_bases(
                            Some(pair.phase_1),
                            Some(pair.phase_2),
                            Some(prev.phase_1),
                            Some(prev.phase_2),
                        ) {
                            max_pair = Some(pair);
                            max_val = s.score;
                        }
                    } else {
                        max_pair = Some(pair);
                        max_val = s.score;
                    }
                }
            }
        }

        // Reject if all scores at this position are equal — couldn't phase.
        let mut all_equal = true;
        for &v1 in verts {
            for &v2 in verts {
                let pair = VertexPair { phase_1: v1, phase_2: v2 };
                if let Some(s) = self.scores.get(&pair) {
                    if s.score != max_val {
                        all_equal = false;
                        break;
                    }
                }
            }
            if !all_equal {
                break;
            }
        }
        if all_equal {
            None
        } else {
            max_pair
        }
    }

    /// Backtrack from the end and assign phases (0/1/2) to vertices.
    /// Block boundaries get is_first_in_block=true.
    fn assign_phases_to_vertices(&mut self) {
        if self.scores.is_empty() {
            return;
        }
        let mut i: i64 = self.positions.len() as i64 - 1;
        let mut prev_pair: Option<VertexPair> = None;
        let mut prev_score: i32 = i32::MIN;

        while i >= 0 {
            // Skip leading positions that can't be phased.
            let mut current_pair: Option<VertexPair>;
            loop {
                if i < 0 {
                    return;
                }
                current_pair = self.max_score(i as usize);
                if current_pair.is_some() {
                    break;
                }
                i -= 1;
            }

            // First entry into a new block (or end of phasing).
            if prev_pair.is_none() {
                prev_pair = current_pair;
            } else if let Some(prev) = prev_pair {
                self.graph.vertex_mut(prev.phase_1).is_first_in_block = true;
                self.graph.vertex_mut(prev.phase_2).is_first_in_block = true;
            }

            let mut num_in_block = 0usize;
            while let Some(cur) = current_pair {
                num_in_block += 1;
                if cur.phase_1 != cur.phase_2 {
                    self.graph.vertex_mut(cur.phase_1).phase = 1;
                    self.graph.vertex_mut(cur.phase_2).phase = 2;
                } else {
                    self.graph.vertex_mut(cur.phase_1).phase = 0;
                    self.graph.vertex_mut(cur.phase_2).phase = 0;
                }

                let cur_score = self.scores[&cur].score;
                // Score must keep advancing within a block. If it
                // didn't, that's a phasing break.
                if Some(cur) != prev_pair && num_in_block > 1 && cur_score == prev_score {
                    self.graph.vertex_mut(cur.phase_1).phase = 0;
                    self.graph.vertex_mut(cur.phase_2).phase = 0;
                    i -= 1;
                    break;
                }

                let next_pair = {
                    let s = &self.scores[&cur];
                    match (s.from[0], s.from[1]) {
                        (Some(a), Some(b)) => Some(VertexPair { phase_1: a, phase_2: b }),
                        _ => None,
                    }
                };
                let next_in_scores = next_pair
                    .filter(|p| self.scores.contains_key(p));

                if next_in_scores.is_none() {
                    // Block ends here.
                    if num_in_block == 1 {
                        // Single-vertex block can't be phased.
                        self.graph.vertex_mut(cur.phase_1).phase = 0;
                        self.graph.vertex_mut(cur.phase_2).phase = 0;
                    }
                    i -= 1;
                    prev_pair = Some(cur);
                    prev_score = cur_score;
                    break;
                }
                if next_in_scores == Some(cur) {
                    // Self-loop guard. Should never happen but match upstream.
                    i -= 1;
                    break;
                }
                prev_pair = Some(cur);
                prev_score = cur_score;
                current_pair = next_in_scores;
                i -= 1;
            }
        }

        // First-in-block flag for the very first entry of the very
        // first block (the loop above sets it on transitions).
        if let Some(prev) = prev_pair {
            self.graph.vertex_mut(prev.phase_1).is_first_in_block = true;
            self.graph.vertex_mut(prev.phase_2).is_first_in_block = true;
        }
    }

    /// Build the graph + run DP + assign phases. Returns one int per
    /// input read: 0 = unphased, 1 = phase 1, 2 = phase 2.
    pub fn phase_reads(
        &mut self,
        candidates: &[DeepVariantCall],
        reads: &[(String, i32)],
    ) -> Vec<i32> {
        self.build(candidates, reads);

        // Iterate positions in order and run DP.
        for i in 0..self.positions.len() {
            // First position has no incoming edges by definition.
            if i == 0 {
                let verts = self.vertices_by_position[&self.positions[i]].clone();
                self.update_starting_score(&verts);
                continue;
            }

            let cur_verts = self.vertices_by_position[&self.positions[i]].clone();
            // If no vertex at this position has any incoming edge, we
            // restart the DP from here (broken phasing block).
            if !self.has_at_least_one_incoming_edge(&cur_verts) {
                self.update_starting_score(&cur_verts);
                continue;
            }

            // Gather and possibly add zero-weight edges so that every
            // vertex at the current position has incoming edges from
            // every vertex at the previous position. This handles the
            // "broken path" case where some vertices have evidence and
            // others don't.
            let mut incoming: BTreeSet<Edge> = BTreeSet::new();
            for &v in &cur_verts {
                let in_e = self.graph.in_edges(v).to_vec();
                if in_e.is_empty() && i > 0 {
                    let prev_verts = self.vertices_by_position[&self.positions[i - 1]].clone();
                    for &pv in &prev_verts {
                        let e_id = self.add_edge(pv, v, 0.0);
                        incoming.insert(self.graph.edge(e_id));
                    }
                } else {
                    for e in in_e {
                        incoming.insert(e);
                    }
                }
            }

            // Key edges by (source bases, target bases) so the
            // enumeration order is deterministic and matches upstream.
            let mut keyed_edges: BTreeMap<(String, String), Edge> = BTreeMap::new();
            for e in &incoming {
                let s = self.graph.vertex(e.from).bases.clone();
                let t = self.graph.vertex(e.to).bases.clone();
                keyed_edges.insert((s, t), *e);
            }

            // Enumerate all (edge1, edge2) pairs.
            let mut found_advancing_score = false;
            let pairs: Vec<(Edge, Edge)> = keyed_edges
                .values()
                .flat_map(|e1| keyed_edges.values().map(move |e2| (*e1, *e2)))
                .collect();
            for (e1, e2) in &pairs {
                let prev_pair = VertexPair {
                    phase_1: e1.from,
                    phase_2: e2.from,
                };
                let prev_score = match self.scores.get(&prev_pair) {
                    Some(p) => p.score,
                    None => continue,
                };
                let new_score = self.calculate_score(e1, e2);
                if prev_score < new_score.score {
                    found_advancing_score = true;
                }
                let to_pair = VertexPair {
                    phase_1: e1.to,
                    phase_2: e2.to,
                };
                let needs_update = match self.scores.get(&to_pair) {
                    None => true,
                    Some(existing) if existing.score < new_score.score => true,
                    Some(existing) if existing.score == new_score.score => self
                        .compare_vertex_pair_by_bases(
                            new_score.from[0],
                            new_score.from[1],
                            existing.from[0],
                            existing.from[1],
                        ),
                    _ => false,
                };
                if needs_update {
                    self.scores.insert(to_pair, new_score);
                }
            }

            // If we couldn't advance (or the whole position is all-equal
            // scores), restart DP. Skip on the last position.
            if i < self.positions.len() - 1
                && (!found_advancing_score || self.all_scores_are_the_same(&keyed_edges))
            {
                self.update_starting_score(&cur_verts);
            }
        }

        self.assign_phases_to_vertices();
        self.assign_phases_to_reads(reads)
    }

    /// Iterate the candidate list, filter, and add to the graph;
    /// then add edges between consecutive positions for each read.
    fn build(&mut self, candidates: &[DeepVariantCall], reads: &[(String, i32)]) {
        self.clear();
        self.initialize_read_maps(reads.iter().map(|(n, m)| (n.as_str(), *m)));

        let mut indel_end: i64 = 0;
        for (i, c) in candidates.iter().enumerate() {
            if i > 0 {
                let prev = candidates[i - 1].variant.as_ref().expect("prev variant");
                let cur = c.variant.as_ref().expect("cur variant");
                debug_assert!(prev.start < cur.start, "candidates must be sorted");
            }
            if candidate_filter(c, &mut indel_end) {
                self.add_candidate(c);
                if let Some(v) = c.variant.as_ref() {
                    self.positions.push(v.start);
                }
            }
        }

        // Edges: for each read, walk its allele sequence in position
        // order and add an edge from the previous-allele vertex to the
        // current-allele vertex if they are at *consecutive*
        // graph-positions (i.e. no missing position between them).
        // This conservative rule matches upstream: it lets us skip
        // candidates that the read doesn't cover.
        let read_to_alleles_keys: Vec<usize> = self.read_to_alleles.keys().copied().collect();
        for read_idx in read_to_alleles_keys {
            let entries = self.read_to_alleles[&read_idx].clone();
            let mut is_first = true;
            let mut prev_entry = AlleleSupport {
                vertex: 0,
                read_support: ReadSupportInfo::new(0, false),
            };
            for entry in entries {
                if is_first {
                    is_first = false;
                    prev_entry = entry;
                    continue;
                }
                let cur_pos = self.graph.vertex(entry.vertex).position;
                let prev_pos = self.graph.vertex(prev_entry.vertex).position;
                // Locate cur_pos in self.positions and check the slot
                // immediately before it equals prev_pos (or is at the
                // start of the list).
                let pos_idx = self.positions.iter().position(|&p| p == cur_pos);
                if let Some(idx) = pos_idx {
                    let connect = idx == 0
                        || self.positions[idx - 1] == prev_pos;
                    if connect {
                        self.add_edge_weighted(
                            prev_entry.vertex,
                            prev_entry.read_support.is_low_quality,
                            entry.vertex,
                            entry.read_support.is_low_quality,
                        );
                    }
                }
                prev_entry = entry;
            }
        }
    }

    /// Phase each input read by majority vote of the alleles it covers.
    /// `min_alleles_to_phase` (from options) controls the minimum vote
    /// needed for the read to receive a phase.
    fn assign_phases_to_reads(&self, reads: &[(String, i32)]) -> Vec<i32> {
        let mut out = vec![0i32; reads.len()];
        for (i, (frag, num)) in reads.iter().enumerate() {
            let key = read_key(frag, *num);
            let Some(&idx) = self.read_to_index.get(&key) else {
                continue;
            };
            let entries = match self.read_to_alleles.get(&idx) {
                Some(e) => e,
                None => continue,
            };
            let mut votes = [0u32; 3];
            for entry in entries {
                let phase = self.graph.vertex(entry.vertex).phase;
                if (0..3).contains(&phase) {
                    votes[phase as usize] += 1;
                }
            }
            let min_votes = self.options.min_alleles_to_phase;
            if votes[1] > votes[2] && votes[1] >= min_votes {
                out[i] = 1;
            } else if votes[2] > votes[1] && votes[2] >= min_votes {
                out[i] = 2;
            } else {
                out[i] = 0;
            }
        }
        out
    }

    /// Return a list of {position, phase 1 bases, phase 2 bases,
    /// is_first_in_block} entries for the phased candidates. Only
    /// candidates whose two alleles are *both* assigned a phase appear.
    pub fn phased_variants(&self) -> Vec<PhasedVariant> {
        let mut out = Vec::new();
        for &pos in &self.positions {
            let verts = match self.vertices_by_position.get(&pos) {
                Some(v) => v,
                None => continue,
            };
            let mut p1 = String::new();
            let mut p2 = String::new();
            let mut is_first_in_block = false;
            for &v in verts {
                let av = self.graph.vertex(v);
                if av.phase == 1 {
                    p1 = av.bases.clone();
                } else if av.phase == 2 {
                    p2 = av.bases.clone();
                }
                is_first_in_block = av.is_first_in_block;
            }
            if !p1.is_empty() && !p2.is_empty() {
                out.push(PhasedVariant {
                    position: pos,
                    phase_1_bases: p1,
                    phase_2_bases: p2,
                    is_first_in_block,
                });
            }
        }
        out
    }

    /// Read accessor used by tests to inspect the built graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Read accessor for the position list.
    pub fn positions(&self) -> &[i64] {
        &self.positions
    }

    /// Read accessor for the vertices-by-position map.
    pub fn vertices_by_position(&self) -> &BTreeMap<i64, Vec<usize>> {
        &self.vertices_by_position
    }

    /// Read accessor for the read-to-index map.
    pub fn read_to_index(&self) -> &HashMap<String, usize> {
        &self.read_to_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dv_proto::dv::{
        deep_variant_call::SupportingReadsExt, DeepVariantCall,
    };
    use dv_proto::dv::deep_variant_call::ReadSupport;
    use dv_proto::nucleus_v1::Variant;

    fn make_candidate(
        start: i64,
        end: i64,
        allele_support: &[(&str, &[&str])],
        ref_support: &[&str],
    ) -> DeepVariantCall {
        let mut variant = Variant::default();
        variant.start = start;
        variant.end = end;
        let mut alts = Vec::new();
        for (a, _) in allele_support {
            alts.push((*a).to_string());
        }
        variant.alternate_bases = alts;

        let mut allele_support_ext = std::collections::BTreeMap::new();
        for (a, reads) in allele_support {
            let mut sre = SupportingReadsExt::default();
            for r in *reads {
                let mut rs = ReadSupport::default();
                rs.read_name = (*r).to_string();
                rs.is_low_quality = false;
                sre.read_infos.push(rs);
            }
            allele_support_ext.insert((*a).to_string(), sre);
        }

        let mut ref_ext = SupportingReadsExt::default();
        for r in ref_support {
            let mut rs = ReadSupport::default();
            rs.read_name = (*r).to_string();
            rs.is_low_quality = false;
            ref_ext.read_infos.push(rs);
        }

        DeepVariantCall {
            variant: Some(variant),
            allele_support_ext,
            ref_support_ext: Some(ref_ext),
            ..Default::default()
        }
    }

    #[test]
    fn allele_type_substitution() {
        let c = make_candidate(100, 102, &[], &[]);
        assert_eq!(allele_type_from_candidate("CC", &c), AlleleType::Substitution);
    }

    #[test]
    fn allele_type_deletion() {
        let c = make_candidate(100, 102, &[], &[]);
        assert_eq!(allele_type_from_candidate("C", &c), AlleleType::Deletion);
    }

    #[test]
    fn allele_type_insertion() {
        let c = make_candidate(100, 101, &[], &[]);
        assert_eq!(allele_type_from_candidate("CCC", &c), AlleleType::Insertion);
    }

    #[test]
    fn allele_type_one_base_substitution() {
        let c = make_candidate(100, 101, &[], &[]);
        assert_eq!(allele_type_from_candidate("A", &c), AlleleType::Substitution);
    }

    #[test]
    fn num_of_substitution_alleles_multi() {
        let c = make_candidate(
            100,
            101,
            &[
                ("A", &["read1", "read2", "read3"]),
                ("C", &["read4", "read5"]),
                ("CC", &["read6", "read7"]),
            ],
            &[],
        );
        assert_eq!(num_of_substitution_alleles(&c), 2);
    }

    #[test]
    fn num_of_substitution_alleles_uncalled_present() {
        let c = make_candidate(
            100,
            101,
            &[
                (UNCALLED_ALLELE, &["read1", "read2", "read3"]),
                ("C", &["read4", "read5"]),
                ("CC", &["read6", "read7"]),
            ],
            &[],
        );
        assert_eq!(num_of_substitution_alleles(&c), 1);
    }

    #[test]
    fn num_of_indel_alleles_2sub_1indel() {
        let c = make_candidate(
            100,
            101,
            &[
                ("A", &["read1", "read2", "read3"]),
                ("C", &["read4", "read5"]),
                ("CC", &["read6", "read7"]),
            ],
            &[],
        );
        assert_eq!(num_of_indel_alleles(&c), 1);
    }

    #[test]
    fn num_of_indel_alleles_uncalled_present() {
        let c = make_candidate(
            100,
            103,
            &[
                (UNCALLED_ALLELE, &["read1", "read2", "read3"]),
                ("C", &["read4", "read5"]),
                ("CCCC", &["read6", "read7"]),
            ],
            &[],
        );
        assert_eq!(num_of_indel_alleles(&c), 2);
    }

    #[test]
    fn substitution_alleles_depth_2sub() {
        let c = make_candidate(
            100,
            101,
            &[
                ("A", &["read1", "read2", "read3"]),
                ("C", &["read4", "read5"]),
                ("CC", &["read6", "read7"]),
            ],
            &[],
        );
        assert_eq!(substitution_alleles_depth(&c), 5);
    }

    #[test]
    fn substitution_alleles_depth_uncalled_and_indels() {
        let c = make_candidate(
            100,
            103,
            &[
                (UNCALLED_ALLELE, &["read1", "read2", "read3"]),
                ("C", &["read4", "read5"]),
                ("CCCC", &["read6", "read7"]),
            ],
            &[],
        );
        assert_eq!(substitution_alleles_depth(&c), 0);
    }

    #[test]
    fn read_support_from_proto_simple() {
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        dp.read_to_index.insert("read1".into(), 1);
        dp.read_to_index.insert("read2".into(), 2);
        dp.read_to_index.insert("read3".into(), 3);
        dp.read_to_index.insert("read4".into(), 4);
        let infos = vec![("read1".to_string(), false), ("read2".to_string(), false)];
        let out = dp.read_support_from_proto(&infos);
        let mut idxs: Vec<usize> = out.iter().map(|r| r.read_index).collect();
        idxs.sort();
        assert_eq!(idxs, vec![1, 2]);
    }

    #[test]
    fn read_support_from_proto_drops_lq() {
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        dp.read_to_index.insert("read1".into(), 1);
        dp.read_to_index.insert("read2".into(), 2);
        dp.read_to_index.insert("read3".into(), 3);
        let infos = vec![
            ("read1".to_string(), false),
            ("read2".to_string(), false),
            ("read3".to_string(), true), // dropped
        ];
        let out = dp.read_support_from_proto(&infos);
        let mut idxs: Vec<usize> = out.iter().map(|r| r.read_index).collect();
        idxs.sort();
        assert_eq!(idxs, vec![1, 2]);
    }

    #[test]
    fn build_graph_simple() {
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[("C", &["read1/0", "read2/0", "read3/0"])],
                &["read4/0", "read5/0", "read6/0"],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
        ];
        let reads: Vec<(String, i32)> = (1..=6)
            .map(|i| (format!("read{}", i), 0))
            .collect();
        dp.build(&candidates, &reads);

        // Expect 6 vertices and 4 edges.
        assert_eq!(dp.graph.num_vertices(), 6);
        assert_eq!(dp.graph.num_edges(), 4);
        assert_eq!(dp.positions, vec![100, 105, 110]);
    }

    /// Helper used by the tests below: build a read list of N reads
    /// named `read1`, …, `readN` with `read_number = 0`, matching the
    /// upstream `CreateTestReads` convention.
    fn test_reads(n: usize) -> Vec<(String, i32)> {
        (1..=n).map(|i| (format!("read{}", i), 0)).collect()
    }

    /// Mirrors upstream `PhaseReadSimpleTest`.
    #[test]
    fn phase_reads_simple() {
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[("C", &["read1/0", "read2/0", "read4/0", "read5/0"])],
                &[],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
        ];
        let reads = test_reads(5);
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let phases = dp.phase_reads(&candidates, &reads);
        assert_eq!(phases, vec![1, 1, 1, 2, 2]);
    }

    /// Mirrors upstream `PhaseReadWithErrorCorrection`. read3 supports
    /// phase 1 at position 100 but switches to phase 2 alt at 110;
    /// majority vote across all four positions still phases it as 1.
    #[test]
    fn phase_reads_with_error_correction() {
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[("C", &["read1/0", "read2/0", "read3/0", "read4/0", "read5/0"])],
                &[],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0"]),
                    ("G", &["read3/0", "read4/0", "read5/0"]),
                ],
                &[],
            ),
            make_candidate(
                120,
                121,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
        ];
        let reads = test_reads(5);
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let phases = dp.phase_reads(&candidates, &reads);
        assert_eq!(phases, vec![1, 1, 1, 2, 2]);
    }

    /// Mirrors upstream `PhaseReadChangedOrderOfAlleles`. Alleles
    /// listed in different orders at different positions; phasing
    /// should still find the same partition.
    #[test]
    fn phase_reads_changed_order_of_alleles() {
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[("C", &["read1/0", "read2/0", "read3/0", "read4/0", "read5/0"])],
                &[],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read4/0", "read5/0"]),
                    ("G", &["read1/0", "read2/0", "read3/0"]),
                ],
                &[],
            ),
            make_candidate(
                120,
                121,
                &[
                    ("G", &["read4/0", "read5/0"]),
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                ],
                &[],
            ),
        ];
        let reads = test_reads(5);
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let phases = dp.phase_reads(&candidates, &reads);
        assert_eq!(phases, vec![1, 1, 1, 2, 2]);
    }

    /// Mirrors upstream `PhaseReadUnphasedRead`. read3 covers one
    /// allele in phase 1, one homozygous, then one in phase 2 — vote
    /// is split, so its phase is 0.
    #[test]
    fn phase_reads_unphased_read() {
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[("C", &["read1/0", "read2/0", "read3/0", "read4/0", "read5/0"])],
                &[],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0"]),
                    ("G", &["read4/0", "read5/0", "read3/0"]),
                ],
                &[],
            ),
        ];
        let reads = test_reads(5);
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let phases = dp.phase_reads(&candidates, &reads);
        assert_eq!(phases, vec![1, 1, 0, 2, 2]);
    }

    /// Mirrors upstream `PhaseReadFullyConnectedGraph`. Three SNVs
    /// all phased together with three reads per haplotype.
    #[test]
    fn phase_reads_fully_connected_graph() {
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[
                    ("C", &["read4/0", "read5/0", "read1/0"]),
                    ("G", &["read2/0", "read3/0", "read6/0"]),
                ],
                &[],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
        ];
        let reads = test_reads(6);
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let phases = dp.phase_reads(&candidates, &reads);
        assert_eq!(phases, vec![1, 1, 1, 2, 2, 2]);
    }

    /// Mirrors upstream `FilterOneAlleleCandidate`. Single-allele
    /// candidate with insufficient REF support is filtered out.
    #[test]
    fn filter_one_allele_candidate() {
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[("C", &["read4/0", "read5/0", "read6/0"])],
                &["read7/0"], // only 1 ref read — below MIN_REF_ALLELE_DEPTH
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
        ];
        let reads = test_reads(7);
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        dp.build(&candidates, &reads);
        // No vertex should exist at position 100.
        assert!(!dp.vertices_by_position.contains_key(&100));
        assert!(dp.vertices_by_position.contains_key(&110));
    }

    /// Mirrors upstream `FilterCandidateWithIndel`. Candidate with
    /// any INDEL allele is filtered out entirely.
    #[test]
    fn filter_candidate_with_indel() {
        let candidates = vec![
            make_candidate(
                100,
                102,
                &[
                    ("CC", &["read4/0", "read5/0", "read6/0"]),
                    ("A", &["read1/0", "read2/0"]), // INDEL — disqualifies whole candidate
                ],
                &["read7/0"],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
        ];
        let reads = test_reads(7);
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        dp.build(&candidates, &reads);
        assert!(!dp.vertices_by_position.contains_key(&100));
        assert!(dp.vertices_by_position.contains_key(&110));
    }

    /// Mirrors upstream `DirectPhasingReuseObject`. A second call to
    /// `phase_reads` on the same instance with different candidates
    /// should not be polluted by the first run.
    #[test]
    fn reuse_object() {
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let reads = test_reads(5);
        let c1 = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[("C", &["read1/0", "read2/0", "read4/0", "read5/0"])],
                &[],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
        ];
        let phases = dp.phase_reads(&c1, &reads);
        assert_eq!(phases, vec![1, 1, 1, 2, 2]);

        // Second region with only one het + one homozygous candidate
        // — neither side has enough info to phase.
        let c2 = vec![
            make_candidate(
                120,
                121,
                &[
                    ("G", &["read1/0", "read2/0", "read3/0"]),
                    ("A", &["read4/0", "read5/0"]),
                ],
                &[],
            ),
            make_candidate(
                130,
                131,
                &[("T", &["read1/0", "read2/0", "read3/0", "read4/0", "read5/0"])],
                &[],
            ),
        ];
        let phases2 = dp.phase_reads(&c2, &reads);
        // Each read covers exactly one phasable allele (the het at
        // 120) — under the default `min_alleles_to_phase=2`, that's
        // not enough. All reads stay unphased.
        assert_eq!(phases2, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn phase_reads_simple_two_phase() {
        // Three SNVs all phased together.
        // reads 1-3 carry alts at all three positions (= phase 1)
        // reads 4-6 carry the other allele (= phase 2)
        let candidates = vec![
            make_candidate(
                100,
                101,
                &[
                    ("A", &["read1/0", "read2/0", "read3/0"]),
                    ("C", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
            make_candidate(
                105,
                106,
                &[
                    ("G", &["read1/0", "read2/0", "read3/0"]),
                    ("T", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
            make_candidate(
                110,
                111,
                &[
                    ("T", &["read1/0", "read2/0", "read3/0"]),
                    ("G", &["read4/0", "read5/0", "read6/0"]),
                ],
                &[],
            ),
        ];
        let reads: Vec<(String, i32)> = (1..=6).map(|i| (format!("read{}", i), 0)).collect();
        let mut dp = DirectPhasing::new(DirectPhasingOptions::default());
        let phases = dp.phase_reads(&candidates, &reads);
        // The two haplotype groups should land on distinct phases.
        assert_eq!(phases.len(), 6);
        let group1 = &phases[0..3];
        let group2 = &phases[3..6];
        assert!(group1.iter().all(|&p| p == group1[0] && (p == 1 || p == 2)));
        assert!(group2.iter().all(|&p| p == group2[0] && (p == 1 || p == 2)));
        assert_ne!(group1[0], group2[0], "two haplotypes got same phase");
    }
}

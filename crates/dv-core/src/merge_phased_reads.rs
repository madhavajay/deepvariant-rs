//! Merge per-shard direct-phasing output into one consistent global
//! phasing. Port of `deepvariant/merge_phased_reads.cc` (~362 LOC).
//!
//! Upstream's `dv make_examples` is sharded by region — each shard
//! runs direct phasing independently and emits a CSV with rows
//! `(fragment_name, phase, region_order)`. Phases assigned in
//! different shards aren't directly comparable: shard A might call a
//! read "phase 1" while shard B calls the same read "phase 2" simply
//! because each shard chose its haplotype labeling independently.
//!
//! `Merger` walks the per-(shard, region) groups in stride order,
//! checks whether the running group agrees or disagrees with the
//! previous group (by counting reads with identical / opposite phase
//! across the overlap), and flips the new group's phases if there's
//! a clear majority disagreement. `CorrectPhasing` then takes the
//! majority vote per read across all the merged groups.
//!
//! Out-of-scope CSV / sharded-file-spec wrapping is provided as plain
//! Rust helpers so the same logic is callable from a CLI or as a
//! library.

use std::collections::HashMap;

/// One read's phase assignment from a single (shard, region) group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnmergedRead {
    pub fragment_name: String,
    /// Phase: 0 = unphased, 1 = haplotype 1, 2 = haplotype 2.
    pub phase: i32,
    /// Order of the region within its shard.
    pub region_order: i32,
    /// Shard index (0-based).
    pub shard: i32,
    /// Internal id assigned during loading; matches the index into
    /// `Merger::merged_reads`.
    pub id: i32,
}

/// One read's merged phase plus a histogram of the per-group phase
/// assignments it received. After `correct_phasing`, `phase` is the
/// majority vote.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergedPhaseRead {
    pub fragment_name: String,
    pub phase: i32,
    /// phase value → count (e.g. {1: 3, 2: 1, 0: 2}).
    pub phase_dist: HashMap<i32, i32>,
}

/// Group key = (shard, region_order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShardRegion {
    pub shard: i32,
    pub region: i32,
}

/// A phase-comparison outcome between two groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonResult {
    Match = 0,
    Switch = 1,
    NotEnoughOverlap = 2,
}

/// Sharded file spec: `<basename>@<nshards>[.<suffix>]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedFileSpec {
    pub basename: String,
    pub nshards: i32,
    pub suffix: String,
}

/// Parse a sharded file spec. Mirrors upstream's regex
/// `(.*)\@(\d*[1-9]\d*)(?:\.(.+))?`. Returns `Err` on a malformed
/// spec or a non-positive shard count.
pub fn parse_sharded_file_spec(spec: &str) -> Result<ShardedFileSpec, String> {
    let at_idx = spec.rfind('@').ok_or_else(|| {
        format!("'{spec}' is not a valid sharded file spec (no '@' found)")
    })?;
    let basename = &spec[..at_idx];
    let after_at = &spec[at_idx + 1..];
    // Split optional suffix off the right.
    let (nshards_str, suffix) = match after_at.find('.') {
        Some(dot) => (&after_at[..dot], &after_at[dot + 1..]),
        None => (after_at, ""),
    };
    if nshards_str.is_empty() {
        return Err(format!("'{spec}': missing shard count after '@'"));
    }
    let nshards: i32 = nshards_str
        .parse()
        .map_err(|_| format!("'{spec}': '{nshards_str}' is not a valid shard count"))?;
    if nshards <= 0 {
        return Err(format!(
            "'{spec}': shard count must be > 0 (got {nshards})"
        ));
    }
    Ok(ShardedFileSpec {
        basename: basename.to_string(),
        nshards,
        suffix: suffix.to_string(),
    })
}

fn shard_width(num_shards: i32) -> usize {
    if num_shards < 100_000 {
        5
    } else if num_shards < 1_000_000 {
        6
    } else if num_shards < 10_000_000 {
        7
    } else if num_shards < 100_000_000 {
        8
    } else {
        9
    }
}

/// Generate `<basename>-<shard>-of-<nshards>[.<suffix>]`. Mirrors
/// upstream `generate_sharded_filename`. Suffix is appended verbatim
/// (including its leading `.`) when non-empty.
pub fn generate_sharded_filename(spec: &ShardedFileSpec, shard: i32) -> String {
    let w = shard_width(spec.nshards);
    if spec.suffix.is_empty() {
        format!(
            "{}-{:0w$}-of-{:05}",
            spec.basename,
            shard,
            spec.nshards,
            w = w
        )
    } else {
        format!(
            "{}-{:0w$}-of-{:05}.{}",
            spec.basename,
            shard,
            spec.nshards,
            spec.suffix,
            w = w
        )
    }
}

/// Read a single shard's per-read phasing CSV. Each line after the
/// header is `fragment_name\tphase\tregion_order` (extra fields are
/// ignored). `region_order` must be > 0 (mirrors upstream's CHECK).
pub fn parse_phasing_csv(
    csv_text: &str,
    shard: i32,
    next_id: &mut i32,
    name_to_id: &mut HashMap<String, i32>,
    merged_reads: &mut Vec<MergedPhaseRead>,
    unmerged_reads: &mut Vec<UnmergedRead>,
) -> Result<(), String> {
    for (line_idx, line) in csv_text.lines().enumerate() {
        if line_idx == 0 {
            continue; // header
        }
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            return Err(format!("malformed CSV line: {line:?}"));
        }
        let fragment_name = parts[0].to_string();
        let phase: i32 = parts[1]
            .parse()
            .map_err(|_| format!("phase parse error on line {line_idx}"))?;
        let region: i32 = parts[2]
            .parse()
            .map_err(|_| format!("region parse error on line {line_idx}"))?;
        if region <= 0 {
            return Err(format!("region must be > 0 (got {region})"));
        }
        let id = if let Some(&id) = name_to_id.get(&fragment_name) {
            id
        } else {
            let id = *next_id;
            name_to_id.insert(fragment_name.clone(), id);
            merged_reads.push(MergedPhaseRead {
                fragment_name: fragment_name.clone(),
                phase: 0,
                phase_dist: HashMap::new(),
            });
            *next_id += 1;
            id
        };
        unmerged_reads.push(UnmergedRead {
            fragment_name,
            phase,
            region_order: region,
            shard,
            id,
        });
    }
    Ok(())
}

/// Per-(shard, region) group: maps merged_read id → unmerged_read index.
#[derive(Debug, Clone, Default)]
pub struct Group {
    pub merged_id_to_unmerged_id: HashMap<i32, usize>,
}

/// Merger drives the merging algorithm.
#[derive(Debug, Default)]
pub struct Merger {
    pub unmerged_reads: Vec<UnmergedRead>,
    pub merged_reads: Vec<MergedPhaseRead>,
    pub merged_reads_map: HashMap<String, i32>,
    pub groups: HashMap<ShardRegion, Group>,
    pub num_shards: i32,
    pub num_groups: i32,
}

impl Merger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an unmerged read directly (used by tests and CSV-loaded
    /// flows alike). Updates `merged_reads_map` and seeds an empty
    /// `merged_reads` entry on first sight of a fragment_name.
    pub fn add_unmerged_read(&mut self, mut r: UnmergedRead) {
        let id = if let Some(&id) = self.merged_reads_map.get(&r.fragment_name) {
            id
        } else {
            let id = self.merged_reads.len() as i32;
            self.merged_reads_map
                .insert(r.fragment_name.clone(), id);
            self.merged_reads.push(MergedPhaseRead {
                fragment_name: r.fragment_name.clone(),
                phase: 0,
                phase_dist: HashMap::new(),
            });
            id
        };
        r.id = id;
        if r.shard + 1 > self.num_shards {
            self.num_shards = r.shard + 1;
        }
        self.unmerged_reads.push(r);
    }

    /// Group unmerged reads by (shard, region). Mirrors upstream's
    /// `GroupReads`.
    pub fn group_reads(&mut self) {
        for (idx, r) in self.unmerged_reads.iter().enumerate() {
            let key = ShardRegion {
                shard: r.shard,
                region: r.region_order,
            };
            let merged_id = *self
                .merged_reads_map
                .get(&r.fragment_name)
                .expect("read not in map");
            self.groups
                .entry(key)
                .or_default()
                .merged_id_to_unmerged_id
                .insert(merged_id, idx);
        }
        self.num_groups = self.groups.len() as i32;
    }

    /// Compare two groups. Mirrors upstream's `CompareGroups` decision
    /// (margin must exceed 1 for MATCH / SWITCH; otherwise NOT_ENOUGH_OVERLAP).
    pub fn compare_groups(&self, g1: ShardRegion, g2: ShardRegion) -> ComparisonResult {
        let (Some(group_1), Some(group_2)) = (self.groups.get(&g1), self.groups.get(&g2)) else {
            return ComparisonResult::NotEnoughOverlap;
        };
        let mut matching = 0i32;
        let mut not_matching = 0i32;
        for (&merged_id_2, &um_idx_2) in &group_2.merged_id_to_unmerged_id {
            let um_idx_1 = match group_1.merged_id_to_unmerged_id.get(&merged_id_2) {
                Some(&i) => i,
                None => continue,
            };
            let p1 = self.unmerged_reads[um_idx_1].phase;
            let p2 = self.unmerged_reads[um_idx_2].phase;
            if p1 == 0 || p2 == 0 {
                continue;
            }
            if p1 == p2 {
                matching += 1;
            } else {
                not_matching += 1;
            }
        }
        if (not_matching - matching).abs() < 2 {
            return ComparisonResult::NotEnoughOverlap;
        }
        if not_matching > matching {
            ComparisonResult::Switch
        } else {
            ComparisonResult::Match
        }
    }

    /// Reverse phase 1↔2 for every read in `group`. Phase 0 stays 0.
    pub fn reverse_phasing(&mut self, group: ShardRegion) {
        let g = match self.groups.get(&group) {
            Some(g) => g.clone(),
            None => return,
        };
        for &um_idx in g.merged_id_to_unmerged_id.values() {
            let p = self.unmerged_reads[um_idx].phase;
            if p > 0 {
                self.unmerged_reads[um_idx].phase = 3 - p;
            }
        }
    }

    /// Merge a group into `merged_reads`: first non-zero phase wins
    /// for the read's `phase` field; the per-phase histogram is
    /// always updated. Mirrors upstream's `MergeGroup`.
    pub fn merge_group(&mut self, group: ShardRegion) {
        let g = match self.groups.get(&group) {
            Some(g) => g.clone(),
            None => return,
        };
        for (&merged_id, &um_idx) in &g.merged_id_to_unmerged_id {
            let merged = &mut self.merged_reads[merged_id as usize];
            let p = self.unmerged_reads[um_idx].phase;
            if merged.phase == 0 {
                merged.phase = p;
            }
            *merged.phase_dist.entry(p).or_insert(0) += 1;
        }
    }

    /// Drive the merge: walk every (shard, region) in canonical order
    /// (region outer, shard inner). For each group, compare against
    /// the previous group; flip phases on SWITCH; merge.
    /// Returns the per-group comparison result so callers can log
    /// switches if they want.
    pub fn merge_reads(&mut self) -> Vec<(ShardRegion, ComparisonResult)> {
        if self.groups.is_empty() {
            self.group_reads();
        }
        let mut switches: Vec<(ShardRegion, ComparisonResult)> = Vec::new();
        let mut cur_region = 1i32;
        let mut processed: i32 = 0;
        let mut prev = ShardRegion {
            shard: 0,
            region: 0,
        };
        // Compute the highest region seen so the loop terminates even
        // if regions aren't 1..=N contiguous.
        let max_region: i32 = self
            .unmerged_reads
            .iter()
            .map(|r| r.region_order)
            .max()
            .unwrap_or(0);
        while processed < self.num_groups && cur_region <= max_region {
            for shard in 0..self.num_shards {
                let key = ShardRegion {
                    shard,
                    region: cur_region,
                };
                if !self.groups.contains_key(&key) {
                    continue;
                }
                let cmp = self.compare_groups(prev, key);
                if cmp == ComparisonResult::Switch {
                    self.reverse_phasing(key);
                }
                switches.push((key, cmp));
                self.merge_group(key);
                processed += 1;
                prev = key;
            }
            cur_region += 1;
        }
        switches
    }

    /// Take the majority phase per read across all merged groups. If
    /// phase 1 and phase 2 counts tie, the read becomes unphased.
    /// Returns the number of reads whose phase was changed by the
    /// correction. Mirrors upstream's `CorrectPhasing`.
    pub fn correct_phasing(&mut self) -> usize {
        let mut corrected = 0;
        for r in &mut self.merged_reads {
            let p1 = r.phase_dist.get(&1).copied().unwrap_or(0);
            let p2 = r.phase_dist.get(&2).copied().unwrap_or(0);
            let new_phase = match p1.cmp(&p2) {
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => 2,
            };
            if r.phase != new_phase {
                corrected += 1;
            }
            r.phase = new_phase;
        }
        corrected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ur(name: &str, phase: i32, region: i32, shard: i32) -> UnmergedRead {
        UnmergedRead {
            fragment_name: name.into(),
            phase,
            region_order: region,
            shard,
            id: 0,
        }
    }

    #[test]
    fn parse_sharded_file_spec_basic() {
        let s = parse_sharded_file_spec("foo@10.csv").unwrap();
        assert_eq!(s.basename, "foo");
        assert_eq!(s.nshards, 10);
        assert_eq!(s.suffix, "csv");
    }

    #[test]
    fn parse_sharded_file_spec_no_suffix() {
        let s = parse_sharded_file_spec("foo@5").unwrap();
        assert_eq!(s.basename, "foo");
        assert_eq!(s.nshards, 5);
        assert_eq!(s.suffix, "");
    }

    #[test]
    fn parse_sharded_file_spec_invalid() {
        assert!(parse_sharded_file_spec("noatsign").is_err());
        assert!(parse_sharded_file_spec("foo@").is_err());
        assert!(parse_sharded_file_spec("foo@0.csv").is_err());
    }

    #[test]
    fn generate_sharded_filename_format() {
        let s = ShardedFileSpec {
            basename: "phasing".into(),
            nshards: 10,
            suffix: "csv".into(),
        };
        assert_eq!(generate_sharded_filename(&s, 0), "phasing-00000-of-00010.csv");
        assert_eq!(generate_sharded_filename(&s, 9), "phasing-00009-of-00010.csv");

        // No suffix variant.
        let s2 = ShardedFileSpec {
            basename: "phasing".into(),
            nshards: 10,
            suffix: "".into(),
        };
        assert_eq!(generate_sharded_filename(&s2, 3), "phasing-00003-of-00010");
    }

    #[test]
    fn merge_two_consistent_shards_does_not_flip() {
        // Same reads in shard 0 and shard 1, region 1, same phases.
        let reads = vec![
            ur("r1", 1, 1, 0),
            ur("r2", 2, 1, 0),
            ur("r3", 1, 1, 0),
            ur("r1", 1, 1, 1),
            ur("r2", 2, 1, 1),
            ur("r3", 1, 1, 1),
        ];
        let mut m = Merger::new();
        for r in reads {
            m.add_unmerged_read(r);
        }
        let switches = m.merge_reads();
        // No flips.
        for (_, cmp) in &switches {
            assert_ne!(*cmp, ComparisonResult::Switch);
        }
        m.correct_phasing();
        let by_name: HashMap<&str, i32> = m
            .merged_reads
            .iter()
            .map(|r| (r.fragment_name.as_str(), r.phase))
            .collect();
        assert_eq!(by_name["r1"], 1);
        assert_eq!(by_name["r2"], 2);
        assert_eq!(by_name["r3"], 1);
    }

    #[test]
    fn merge_inverted_shard_flips_phases() {
        // shard 0: r1=1, r2=2, r3=1
        // shard 1: r1=2, r2=1, r3=2  (inverted) — should be flipped on merge
        let reads = vec![
            ur("r1", 1, 1, 0),
            ur("r2", 2, 1, 0),
            ur("r3", 1, 1, 0),
            ur("r1", 2, 1, 1),
            ur("r2", 1, 1, 1),
            ur("r3", 2, 1, 1),
        ];
        let mut m = Merger::new();
        for r in reads {
            m.add_unmerged_read(r);
        }
        let switches = m.merge_reads();
        // Expect at least one Switch outcome.
        assert!(switches.iter().any(|(_, c)| *c == ComparisonResult::Switch));
        m.correct_phasing();
        let by_name: HashMap<&str, i32> = m
            .merged_reads
            .iter()
            .map(|r| (r.fragment_name.as_str(), r.phase))
            .collect();
        // After flipping shard 1, both shards agree → phases are
        // correctly propagated.
        assert_eq!(by_name["r1"], 1);
        assert_eq!(by_name["r2"], 2);
        assert_eq!(by_name["r3"], 1);
    }

    #[test]
    fn correct_phasing_handles_ties() {
        let mut m = Merger::new();
        m.merged_reads.push(MergedPhaseRead {
            fragment_name: "r1".into(),
            phase: 1,
            phase_dist: [(1, 2), (2, 2)].into_iter().collect(),
        });
        m.merged_reads.push(MergedPhaseRead {
            fragment_name: "r2".into(),
            phase: 2,
            phase_dist: [(1, 1), (2, 3)].into_iter().collect(),
        });
        m.correct_phasing();
        // r1: tie → phase 0
        assert_eq!(m.merged_reads[0].phase, 0);
        // r2: phase 2 majority → phase 2
        assert_eq!(m.merged_reads[1].phase, 2);
    }

    #[test]
    fn parse_phasing_csv_basic() {
        let csv = "fragment\tphase\tregion\nr1\t1\t1\nr2\t2\t1\nr3\t1\t1\n";
        let mut next_id = 0i32;
        let mut name_to_id = HashMap::new();
        let mut merged = Vec::new();
        let mut unmerged = Vec::new();
        parse_phasing_csv(csv, 0, &mut next_id, &mut name_to_id, &mut merged, &mut unmerged)
            .unwrap();
        assert_eq!(unmerged.len(), 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(unmerged[0].fragment_name, "r1");
        assert_eq!(unmerged[0].phase, 1);
        assert_eq!(unmerged[0].region_order, 1);
    }

    #[test]
    fn parse_phasing_csv_rejects_zero_region() {
        let csv = "fragment\tphase\tregion\nr1\t1\t0\n";
        let mut next_id = 0i32;
        let mut name_to_id = HashMap::new();
        let mut merged = Vec::new();
        let mut unmerged = Vec::new();
        let res = parse_phasing_csv(
            csv,
            0,
            &mut next_id,
            &mut name_to_id,
            &mut merged,
            &mut unmerged,
        );
        assert!(res.is_err());
    }

    #[test]
    fn not_enough_overlap_when_groups_match_close_to_50_50() {
        // 3 reads agree, 3 disagree → margin=0 < 2 → NOT_ENOUGH_OVERLAP.
        let reads = vec![
            ur("a", 1, 1, 0), ur("a", 1, 1, 1), // match
            ur("b", 2, 1, 0), ur("b", 2, 1, 1), // match
            ur("c", 1, 1, 0), ur("c", 1, 1, 1), // match
            ur("d", 1, 1, 0), ur("d", 2, 1, 1), // mismatch
            ur("e", 1, 1, 0), ur("e", 2, 1, 1), // mismatch
            ur("f", 1, 1, 0), ur("f", 2, 1, 1), // mismatch
        ];
        let mut m = Merger::new();
        for r in reads {
            m.add_unmerged_read(r);
        }
        m.group_reads();
        let cmp = m.compare_groups(
            ShardRegion { shard: 0, region: 1 },
            ShardRegion { shard: 1, region: 1 },
        );
        assert_eq!(cmp, ComparisonResult::NotEnoughOverlap);
    }
}

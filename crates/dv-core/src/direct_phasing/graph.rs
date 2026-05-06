//! Hand-rolled directed graph for direct phasing. Upstream uses
//! boost::adjacency_list with bidirectional edges; we just keep parallel
//! per-vertex `out_edges` and `in_edges` lists. Edges are deduplicated
//! by (from, to) — re-adding the same pair accumulates weight rather
//! than producing parallel edges. That mirrors upstream's
//! `boost::add_edge` + `setS` behavior.

use dv_proto::dv::AlleleType;

use super::ReadSupportInfo;

/// One graph vertex = one allele at one candidate position.
#[derive(Debug, Clone)]
pub struct AlleleInfo {
    pub allele_type: AlleleType,
    pub position: i64,
    pub bases: String,
    /// 0 = unphased, 1 = phase 1, 2 = phase 2.
    pub phase: i32,
    pub is_first_in_block: bool,
    pub read_support: Vec<ReadSupportInfo>,
}

#[derive(Debug, Clone)]
pub struct Vertex {
    pub info: AlleleInfo,
    /// Edge ids where this vertex is the source.
    pub out_edges: Vec<usize>,
    /// Edge ids where this vertex is the target.
    pub in_edges: Vec<usize>,
}

/// One graph edge = "this read connects allele A at position P to
/// allele B at position P+1". Multiple reads contribute additive weight.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
}

impl Eq for Edge {}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lexicographic on (from, to). Weight is ignored for set
        // membership — matches upstream where `boost::edge` returns
        // the existing edge regardless of weight.
        match self.from.cmp(&other.from) {
            std::cmp::Ordering::Equal => self.to.cmp(&other.to),
            o => o,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn add_vertex(&mut self, info: AlleleInfo) -> usize {
        let id = self.vertices.len();
        self.vertices.push(Vertex {
            info,
            out_edges: Vec::new(),
            in_edges: Vec::new(),
        });
        id
    }

    /// Add or update an edge from `from` to `to`. If the edge exists,
    /// its weight is incremented by `weight` (matches upstream
    /// `AddEdge` which calls `boost::edge` first then `weight +=`).
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f32) -> usize {
        for &eid in &self.vertices[from].out_edges {
            if self.edges[eid].to == to {
                self.edges[eid].weight += weight;
                return eid;
            }
        }
        let id = self.edges.len();
        self.edges.push(Edge { from, to, weight });
        self.vertices[from].out_edges.push(id);
        self.vertices[to].in_edges.push(id);
        id
    }

    pub fn vertex(&self, id: usize) -> &AlleleInfo {
        &self.vertices[id].info
    }

    pub fn vertex_mut(&mut self, id: usize) -> &mut AlleleInfo {
        &mut self.vertices[id].info
    }

    pub fn edge(&self, id: usize) -> Edge {
        self.edges[id]
    }

    pub fn in_edges(&self, v: usize) -> Vec<Edge> {
        self.vertices[v]
            .in_edges
            .iter()
            .map(|&e| self.edges[e])
            .collect()
    }

    pub fn out_edges(&self, v: usize) -> Vec<Edge> {
        self.vertices[v]
            .out_edges
            .iter()
            .map(|&e| self.edges[e])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(pos: i64, bases: &str) -> AlleleInfo {
        AlleleInfo {
            allele_type: AlleleType::Substitution,
            position: pos,
            bases: bases.to_string(),
            phase: 0,
            is_first_in_block: false,
            read_support: Vec::new(),
        }
    }

    #[test]
    fn add_vertex_returns_unique_ids() {
        let mut g = Graph::new();
        let a = g.add_vertex(info(100, "A"));
        let b = g.add_vertex(info(100, "C"));
        assert_ne!(a, b);
        assert_eq!(g.num_vertices(), 2);
    }

    #[test]
    fn add_edge_idempotent_with_weight_accumulation() {
        let mut g = Graph::new();
        let a = g.add_vertex(info(100, "A"));
        let b = g.add_vertex(info(105, "C"));
        let e1 = g.add_edge(a, b, 0.5);
        let e2 = g.add_edge(a, b, 0.5);
        // Same edge id, weight accumulated.
        assert_eq!(e1, e2);
        assert_eq!(g.num_edges(), 1);
        assert!((g.edge(e1).weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn in_and_out_edges() {
        let mut g = Graph::new();
        let a = g.add_vertex(info(100, "A"));
        let b = g.add_vertex(info(105, "C"));
        let _ = g.add_edge(a, b, 1.0);
        assert_eq!(g.out_edges(a).len(), 1);
        assert_eq!(g.in_edges(b).len(), 1);
        assert_eq!(g.in_edges(a).len(), 0);
        assert_eq!(g.out_edges(b).len(), 0);
    }
}

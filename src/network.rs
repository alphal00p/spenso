use ahash::AHashMap;
#[cfg(feature = "shadowing")]
use ahash::{AHashSet, HashMap};
#[cfg(feature = "shadowing")]
use anyhow::anyhow;
#[cfg(feature = "shadowing")]
use std::sync::Arc;

#[cfg(feature = "shadowing")]
use symbolica::atom::PowView;

// use log::trace;
use serde::{Deserialize, Serialize};
use slotmap::{new_key_type, DenseSlotMap, Key, SecondaryMap};
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{representation::FunView, AddView, Atom, AtomView, MulView, Symbol},
    coefficient::ConvertToRing,
    domains::{
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
        },
        float::{Complex as SymComplex, NumericalFloatLike, Real, SingleFloat},
        rational::Rational,
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
        EuclideanDomain,
    },
    evaluate::{
        CompileOptions, CompiledCode, CompiledEvaluator, EvalTree, EvaluationFn, ExportedCode,
        ExpressionEvaluator, FunctionMap, InlineASM,
    },
    id::{Condition, MatchSettings, Pattern, PatternRestriction},
    poly::{factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Variable},
};

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{AtomCore, KeyLookup},
    id::{BorrowPatternOrMap, BorrowReplacement},
    poly::PositiveExponent,
};

#[cfg(feature = "shadowing")]
use crate::{
    complex::{Complex, RealOrComplexTensor},
    contraction::RefZero,
    data::{DataIterator, DenseTensor, SetTensorData, SparseTensor},
    iterators::IteratableTensor,
    parametric::{
        atomcore::PatternReplacement, AtomViewOrConcrete, CompiledEvalTensor, EvalTensor,
        EvalTreeTensor, MixedTensor, ParamTensor, SerializableCompiledCode,
        SerializableCompiledEvaluator, SerializableExportedCode,
    },
    shadowing::{ShadowMapping, Shadowable},
    structure::representation::Rep,
    structure::slot::IsAbstractSlot,
    structure::{StructureContract, ToSymbolic},
    symbolica_utils::{IntoArgs, IntoSymbol, SerializableAtom},
    upgrading_arithmetic::{FallibleAdd, TrySmallestUpgrade},
};

use crate::{
    arithmetic::ScalarMul,
    contraction::{Contract, ContractionError, Trace},
    data::{DataTensor, GetTensorData, HasTensorData},
    structure::{
        slot::DualSlotTo, CastStructure, HasName, HasStructure, ScalarTensor, TensorStructure,
        TracksCount,
    },
    upgrading_arithmetic::FallibleMul,
};

#[cfg(feature = "shadowing")]
use crate::{
    data::StorageTensor,
    parametric::atomcore::{TensorAtomMaps, TensorAtomOps},
};

use anyhow::Result;

use smartstring::alias::String;
use std::fmt::{Debug, Display};

new_key_type! {
    pub struct NodeId;
    pub struct HedgeId;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Direction {
    None,
    Source,
    Sink,
}

impl Direction {
    pub fn reverse(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Sink => Self::Source,
            Self::Source => Self::Sink,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invel<T> {
    pub dir: Direction,
    pub data: T,
}

impl<T> Invel<T> {
    pub fn undirected(data: T) -> Self {
        Invel {
            dir: Direction::None,
            data,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DisplayOption<T> {
    None,
    Some(T),
}

impl<T: Display> Display for DisplayOption<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DisplayOption::None => write!(f, ""),
            DisplayOption::Some(t) => write!(f, "{}", t),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalfEdgeGraph<N, E> {
    pub edges: DenseSlotMap<HedgeId, E>,
    pub involution: SecondaryMap<HedgeId, Invel<HedgeId>>,
    pub neighbors: SecondaryMap<HedgeId, HedgeId>,
    pub nodemap: SecondaryMap<HedgeId, NodeId>,
    pub nodes: DenseSlotMap<NodeId, N>,
    pub reverse_nodemap: SecondaryMap<NodeId, HedgeId>,
}

impl<N, E> HalfEdgeGraph<N, E> {
    pub fn map_nodes<U, F>(self, f: F) -> HalfEdgeGraph<U, E>
    where
        F: Fn((NodeId, N)) -> U,
    {
        let mut nodeidmap: SecondaryMap<NodeId, NodeId> = SecondaryMap::new();
        let mut newnodes: DenseSlotMap<NodeId, U> = DenseSlotMap::with_key();

        for (i, n) in self.nodes {
            let nid = newnodes.insert(f((i, n)));
            nodeidmap.insert(i, nid);
        }

        let mut newnodemap: SecondaryMap<HedgeId, NodeId> = SecondaryMap::new();
        for (i, n) in self.nodemap {
            newnodemap.insert(i, nodeidmap[n]);
        }

        let mut newreverse_nodemap: SecondaryMap<NodeId, HedgeId> = SecondaryMap::new();
        for (i, n) in self.reverse_nodemap {
            newreverse_nodemap.insert(nodeidmap[i], n);
        }

        HalfEdgeGraph {
            edges: self.edges,
            involution: self.involution,
            nodes: newnodes,
            nodemap: newnodemap,
            reverse_nodemap: newreverse_nodemap,
            neighbors: self.neighbors,
        }
    }

    pub fn map_nodes_ref<U, F>(&self, f: F) -> HalfEdgeGraph<U, E>
    where
        F: Fn((NodeId, &N)) -> U,
        E: Clone,
    {
        let mut nodeidmap: SecondaryMap<NodeId, NodeId> = SecondaryMap::new();
        let mut newnodes: DenseSlotMap<NodeId, U> = DenseSlotMap::with_key();

        for (i, n) in &self.nodes {
            let nid = newnodes.insert(f((i, n)));
            nodeidmap.insert(i, nid);
        }

        let mut newnodemap: SecondaryMap<HedgeId, NodeId> = SecondaryMap::new();
        for (i, &n) in &self.nodemap {
            newnodemap.insert(i, nodeidmap[n]);
        }

        let mut newreverse_nodemap: SecondaryMap<NodeId, HedgeId> = SecondaryMap::new();
        for (i, &n) in &self.reverse_nodemap {
            newreverse_nodemap.insert(nodeidmap[i], n);
        }

        HalfEdgeGraph {
            edges: self.edges.clone(),
            involution: self.involution.clone(),
            nodes: newnodes,
            nodemap: newnodemap,
            reverse_nodemap: newreverse_nodemap,
            neighbors: self.neighbors.clone(),
        }
    }

    pub fn map_nodes_ref_option<U, F>(&self, f: F) -> Option<HalfEdgeGraph<U, E>>
    where
        F: Fn((NodeId, &N)) -> Option<U>,
        E: Clone,
    {
        let mut nodeidmap: SecondaryMap<NodeId, NodeId> = SecondaryMap::new();
        let mut newnodes: DenseSlotMap<NodeId, U> = DenseSlotMap::with_key();

        for (i, n) in &self.nodes {
            let nid = newnodes.insert(f((i, n))?);
            nodeidmap.insert(i, nid);
        }

        let mut newnodemap: SecondaryMap<HedgeId, NodeId> = SecondaryMap::new();
        for (i, &n) in &self.nodemap {
            newnodemap.insert(i, nodeidmap[n]);
        }

        let mut newreverse_nodemap: SecondaryMap<NodeId, HedgeId> = SecondaryMap::new();
        for (i, &n) in &self.reverse_nodemap {
            newreverse_nodemap.insert(nodeidmap[i], n);
        }

        Some(HalfEdgeGraph {
            edges: self.edges.clone(),
            involution: self.involution.clone(),
            nodes: newnodes,
            nodemap: newnodemap,
            reverse_nodemap: newreverse_nodemap,
            neighbors: self.neighbors.clone(),
        })
    }

    pub fn map_nodes_ref_result<U, F, Er>(&self, f: F) -> Result<HalfEdgeGraph<U, E>, Er>
    where
        F: Fn((NodeId, &N)) -> Result<U, Er>,
        E: Clone,
    {
        let mut nodeidmap: SecondaryMap<NodeId, NodeId> = SecondaryMap::new();
        let mut newnodes: DenseSlotMap<NodeId, U> = DenseSlotMap::with_key();

        for (i, n) in &self.nodes {
            let nid = newnodes.insert(f((i, n))?);
            nodeidmap.insert(i, nid);
        }

        let mut newnodemap: SecondaryMap<HedgeId, NodeId> = SecondaryMap::new();
        for (i, &n) in &self.nodemap {
            newnodemap.insert(i, nodeidmap[n]);
        }

        let mut newreverse_nodemap: SecondaryMap<NodeId, HedgeId> = SecondaryMap::new();
        for (i, &n) in &self.reverse_nodemap {
            newreverse_nodemap.insert(nodeidmap[i], n);
        }

        Ok(HalfEdgeGraph {
            edges: self.edges.clone(),
            involution: self.involution.clone(),
            nodes: newnodes,
            nodemap: newnodemap,
            reverse_nodemap: newreverse_nodemap,
            neighbors: self.neighbors.clone(),
        })
    }

    pub fn map_nodes_ref_mut<U, F>(&mut self, mut f: F) -> HalfEdgeGraph<U, E>
    where
        F: FnMut((NodeId, &mut N)) -> U,
        E: Clone,
    {
        let mut nodeidmap: SecondaryMap<NodeId, NodeId> = SecondaryMap::new();
        let mut newnodes: DenseSlotMap<NodeId, U> = DenseSlotMap::with_key();

        for (i, n) in &mut self.nodes {
            let nid = newnodes.insert(f((i, n)));
            nodeidmap.insert(i, nid);
        }

        let mut newnodemap: SecondaryMap<HedgeId, NodeId> = SecondaryMap::new();
        for (i, &n) in &self.nodemap {
            newnodemap.insert(i, nodeidmap[n]);
        }

        let mut newreverse_nodemap: SecondaryMap<NodeId, HedgeId> = SecondaryMap::new();
        for (i, &n) in &self.reverse_nodemap {
            newreverse_nodemap.insert(nodeidmap[i], n);
        }

        HalfEdgeGraph {
            edges: self.edges.clone(),
            involution: self.involution.clone(),
            nodes: newnodes,
            nodemap: newnodemap,
            reverse_nodemap: newreverse_nodemap,
            neighbors: self.neighbors.clone(),
        }
    }

    pub fn map_nodes_mut<F>(&mut self, mut f: F)
    where
        F: FnMut((NodeId, &mut N)),
        E: Clone,
    {
        for (i, n) in &mut self.nodes {
            f((i, n));
        }
    }
}

struct IncidentIterator<'a> {
    neighbors: &'a SecondaryMap<HedgeId, HedgeId>,
    current: Option<HedgeId>,
    start: HedgeId,
}

impl<'a> Iterator for IncidentIterator<'a> {
    type Item = HedgeId;
    fn next(&mut self) -> Option<HedgeId> {
        let current = self.current?;

        self.current = Some(self.neighbors[current]);

        if self.current == Some(self.start) {
            self.current = None;
        }

        Some(current)
    }
}

impl<'a> IncidentIterator<'a> {
    fn new<N, E>(graph: &'a HalfEdgeGraph<N, E>, initial: HedgeId) -> Self {
        IncidentIterator {
            neighbors: &graph.neighbors,
            current: Some(initial),
            start: initial,
        }
    }
}

#[allow(dead_code)]
impl<N, E> HalfEdgeGraph<N, E> {
    fn new() -> Self {
        HalfEdgeGraph {
            involution: SecondaryMap::new(),
            nodemap: SecondaryMap::new(),
            neighbors: SecondaryMap::new(),
            reverse_nodemap: SecondaryMap::new(),
            nodes: DenseSlotMap::with_key(),
            edges: DenseSlotMap::with_key(),
        }
    }

    pub fn dot(&self) -> std::string::String
    where
        E: Display,
    {
        let mut out = "digraph {\n".to_string();
        out.push_str("  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\";");

        // for (i, n) in &self.nodes {
        //     out.push_str(&format!("\n {}", i.data().as_ffi()));
        // }
        for (i, _) in &self.neighbors {
            match i.cmp(&self.involution[i].data) {
                std::cmp::Ordering::Greater => {
                    match self.involution[i] {
                        Invel {
                            dir: Direction::None,
                            data,
                        } => out.push_str(&format!(
                            "\n {} -> {} [label=\" {} = {}\" dir=none];",
                            self.nodemap[i].data().as_ffi(),
                            self.nodemap[self.involution[i].data].data().as_ffi(),
                            self.edges[data],
                            self.edges[i]
                        )),
                        Invel {
                            dir: Direction::Sink,
                            data,
                        } => out.push_str(&format!(
                            "\n {} -> {} [label=\" {}={} \"];",
                            self.nodemap[self.involution[i].data].data().as_ffi(),
                            self.nodemap[i].data().as_ffi(),
                            self.edges[i],
                            self.edges[data],
                        )),
                        Invel {
                            dir: Direction::Source,
                            data,
                        } => out.push_str(&format!(
                            "\n {} -> {} [label=\" {}={} \"];",
                            self.nodemap[i].data().as_ffi(),
                            self.nodemap[self.involution[i].data].data().as_ffi(),
                            self.edges[data],
                            self.edges[i]
                        )),
                    };
                }
                std::cmp::Ordering::Equal => {
                    out.push_str(&format!(
                        " \n ext{} [shape=none, label=\"\"];",
                        i.data().as_ffi()
                    ));
                    match self.involution[i].dir {
                        Direction::None => out.push_str(&format!(
                            "\n {} -> ext{} [label =\" {}\" dir=none];",
                            self.nodemap[i].data().as_ffi(),
                            i.data().as_ffi(),
                            self.edges[i]
                        )),
                        Direction::Sink => out.push_str(&format!(
                            "\n ext{} -> {} [label =\" {}\"];",
                            i.data().as_ffi(),
                            self.nodemap[i].data().as_ffi(),
                            self.edges[i]
                        )),
                        Direction::Source => out.push_str(&format!(
                            "\n {} -> ext{} [label =\" {}\"];",
                            self.nodemap[i].data().as_ffi(),
                            i.data().as_ffi(),
                            self.edges[i]
                        )),
                    };
                }
                _ => {}
            }
        }

        out += "}";
        out
    }

    pub fn dot_nodes(&self) -> std::string::String
    where
        E: Display,
        N: Display,
    {
        let mut out = "digraph {\n".to_string();
        out.push_str("  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\";");

        for (i, n) in &self.nodes {
            out.push_str(&format!("\n {} [label=\" {} \"]", i.data().as_ffi(), n));
        }
        for (i, _) in &self.neighbors {
            match i.cmp(&self.involution[i].data) {
                std::cmp::Ordering::Greater => {
                    match self.involution[i] {
                        Invel {
                            dir: Direction::None,
                            data,
                        } => out.push_str(&format!(
                            "\n {} -> {} [label=\" {} = {}\" dir=none];",
                            self.nodemap[i].data().as_ffi(),
                            self.nodemap[self.involution[i].data].data().as_ffi(),
                            self.edges[data],
                            self.edges[i]
                        )),
                        Invel {
                            dir: Direction::Sink,
                            data,
                        } => out.push_str(&format!(
                            "\n {} -> {} [label=\" {}={} \" penwidth=2.];",
                            self.nodemap[self.involution[i].data].data().as_ffi(),
                            self.nodemap[i].data().as_ffi(),
                            self.edges[i],
                            self.edges[data],
                        )),
                        Invel {
                            dir: Direction::Source,
                            data,
                        } => out.push_str(&format!(
                            "\n {} -> {} [label=\" {}={} \" penwidth=2.];",
                            self.nodemap[i].data().as_ffi(),
                            self.nodemap[self.involution[i].data].data().as_ffi(),
                            self.edges[data],
                            self.edges[i]
                        )),
                    };
                }
                std::cmp::Ordering::Equal => {
                    out.push_str(&format!(
                        " \n ext{} [shape=none, label=\"\"];",
                        i.data().as_ffi()
                    ));
                    match self.involution[i].dir {
                        Direction::None => out.push_str(&format!(
                            "\n {} -> ext{} [label =\" {}\" dir=none];",
                            self.nodemap[i].data().as_ffi(),
                            i.data().as_ffi(),
                            self.edges[i]
                        )),
                        Direction::Sink => out.push_str(&format!(
                            "\n ext{} -> {} [label =\" {}\" penwidth=2.];",
                            i.data().as_ffi(),
                            self.nodemap[i].data().as_ffi(),
                            self.edges[i]
                        )),
                        Direction::Source => out.push_str(&format!(
                            "\n {} -> ext{} [label =\" {}\" penwidth=2.];",
                            self.nodemap[i].data().as_ffi(),
                            i.data().as_ffi(),
                            self.edges[i]
                        )),
                    };
                }
                _ => {}
            }
        }

        out += "}";
        out
    }

    fn add_node(&mut self, data: N) -> NodeId {
        self.nodes.insert(data)
    }

    fn node_indices(&self) -> slotmap::dense::Keys<'_, NodeId, N> {
        self.nodes.keys()
    }

    /// Add a node with a list of edget with associated data. Matches edges by equality.
    fn add_node_with_edges(&mut self, data: N, edges: &[E]) -> NodeId
    where
        E: Eq + Clone,
    {
        self.add_node_with_edges_fn(data, edges, |e, eo| (*e == *eo, Direction::None))
    }

    /// Add a node with a list of edget with associated data. Matches edges by equality.
    fn add_node_with_edges_fn<F>(&mut self, data: N, edges: &[E], f: F) -> NodeId
    where
        E: Eq + Clone,
        F: Fn(&E, &E) -> (bool, Direction),
    {
        let idx = self.add_node(data);
        for e in edges {
            let mut found_match = false;
            for (i, other_e) in &self.edges {
                let (matches, dir) = f(e, other_e);
                if matches && self.involution[i].data == i {
                    found_match = true;
                    let eid = self.edges.insert(e.clone());
                    self.involution.insert(eid, Invel { dir, data: i });
                    self.involution.insert(i, Invel { dir, data: eid });
                    self.nodemap.insert(eid, idx);
                    if let Some(prev_eid) = self.reverse_nodemap.insert(idx, eid) {
                        let next_eid = self.neighbors.insert(prev_eid, eid).unwrap();
                        self.neighbors.insert(eid, next_eid);
                    } else {
                        self.neighbors.insert(eid, eid);
                    }
                    break;
                }
            }
            if !found_match {
                let eid = self.edges.insert(e.clone());
                self.involution.insert(
                    eid,
                    Invel {
                        dir: Direction::None,
                        data: eid,
                    },
                );
                self.nodemap.insert(eid, idx);
                if let Some(prev_eid) = self.reverse_nodemap.insert(idx, eid) {
                    let next_eid = self.neighbors.insert(prev_eid, eid).unwrap();
                    self.neighbors.insert(eid, next_eid);
                } else {
                    self.neighbors.insert(eid, eid);
                }
            }
        }

        idx
    }

    pub fn validate_neighbors(&self) -> bool {
        for (i, n) in &self.reverse_nodemap {
            for j in IncidentIterator::new(self, *n) {
                if self.nodemap[j] != i {
                    return false;
                }
            }
        }
        true
    }

    fn node_labels(&self) -> String
    where
        N: Display,
    {
        let mut out = String::new();
        for (i, n) in &self.nodes {
            out.push_str(&format!("{}[label= \"{}\"]\n", i.data().as_ffi(), n));
        }
        out
    }

    fn remove_edge(&mut self, edge: HedgeId) {
        self.edges.remove(edge);
        self.edges.remove(self.involution[edge].data);
        self.nodemap.remove(edge);
        self.nodemap.remove(self.involution[edge].data);
        self.involution.remove(self.involution[edge].data);
        self.involution.remove(edge);
    }

    #[allow(clippy::too_many_lines)]
    fn merge_nodes(&mut self, a: NodeId, b: NodeId, data: N) -> NodeId {
        let c = self.nodes.insert(data);

        // New initial edge for reverse_nodemap, that does not link to b
        // if none is found, all incident edges are link to b and must be removed from the neighbors list
        let mut new_initial_a = self.edges_incident(a).find(|x| {
            self.nodemap[self.involution[*x].data] != b && self.involution[*x].data != *x
        });

        if new_initial_a.is_none() {
            new_initial_a = self
                .edges_incident(a)
                .find(|x| self.nodemap[self.involution[*x].data] != b);
        }

        if let Some(initial) = new_initial_a {
            let mut current = Ok(initial);

            while let Ok(cur) = current {
                let mut next = self.neighbors[cur];
                while self.nodemap[self.involution[next].data] == b {
                    next = self.neighbors.remove(next).unwrap();
                }
                self.neighbors[cur] = next;

                if next == initial {
                    current = Err(cur);
                } else {
                    current = Ok(next);
                }
            }
        } else {
            // all edges link to b, and must be removed
            let initial = self.reverse_nodemap[a];
            let mut current = Some(initial);

            while let Some(c) = current {
                let next = self.neighbors.remove(c);
                // self.remove_edge(c);
                if next == Some(initial) {
                    current = None;
                } else {
                    current = next;
                }
            }
        }

        let mut new_initial_b = self.edges_incident(b).find(|x| {
            self.nodemap[self.involution[*x].data] != a && self.involution[*x].data != *x
        });

        if new_initial_b.is_none() {
            new_initial_b = self
                .edges_incident(b)
                .find(|x| self.nodemap[self.involution[*x].data] != a);
        }

        let mut edge_leading_to_start_b = None;

        if let Some(initial) = new_initial_b {
            let mut current = Ok(initial);
            while let Ok(cur) = current {
                let mut next = self.neighbors[cur];
                while self.nodemap[self.involution[next].data] == a {
                    self.remove_edge(next);
                    next = self.neighbors.remove(next).unwrap();
                }
                self.neighbors[cur] = next;

                if next == initial {
                    current = Err(cur);
                } else {
                    current = Ok(next);
                }
            }

            if let Err(cur) = current {
                edge_leading_to_start_b = Some(cur);
            }
        } else {
            // all edges link to b, and must be removed
            let initial = self.reverse_nodemap[b];
            let mut current = Some(initial);

            while let Some(c) = current {
                let next = self.neighbors.remove(c);
                self.remove_edge(c);
                if next == Some(initial) {
                    current = None;
                } else {
                    current = next;
                }
            }
        }

        match (new_initial_a, new_initial_b) {
            (Some(new_edge_a), Some(new_edge_b)) => {
                self.reverse_nodemap.insert(c, new_edge_a);
                self.reverse_nodemap.remove(a);
                self.reverse_nodemap.remove(b);
                let old_neig = self.neighbors.insert(new_edge_a, new_edge_b).unwrap();
                if let Some(next) = edge_leading_to_start_b {
                    self.neighbors.insert(next, old_neig).unwrap();
                } else {
                    self.neighbors.insert(new_edge_b, old_neig);
                }
            }
            (Some(new_edge_a), None) => {
                self.reverse_nodemap.insert(c, new_edge_a);
                self.reverse_nodemap.remove(a);
                self.reverse_nodemap.remove(b);
            }
            (None, Some(new_edge_b)) => {
                self.reverse_nodemap.insert(c, new_edge_b);
                self.reverse_nodemap.remove(a);
                self.reverse_nodemap.remove(b);
            }
            (None, None) => {
                self.reverse_nodemap.remove(b);
                self.reverse_nodemap.remove(a);
            }
        }

        if let Some(&init) = self.reverse_nodemap.get(c) {
            let mut current = Some(init);
            while let Some(cur) = current {
                self.nodemap.insert(cur, c);
                let next = self.neighbors[cur];
                if next == init {
                    current = None;
                } else {
                    current = Some(next);
                }
            }
        }

        self.nodes.remove(a);
        self.nodes.remove(b);
        c
    }

    /// Add an internal edge between two nodes.
    fn add_edge(&mut self, a: NodeId, b: NodeId, data: E, dir: Direction) -> HedgeId
    where
        E: Clone,
    {
        let hedge_id_a = self.edges.insert(data.clone());
        let hedge_id_b = self.edges.insert(data);
        self.involution.insert(
            hedge_id_a,
            Invel {
                data: hedge_id_b,
                dir,
            },
        );
        self.involution.insert(
            hedge_id_b,
            Invel {
                data: hedge_id_a,
                dir: dir.reverse(),
            },
        );
        self.nodemap.insert(hedge_id_a, a);
        if let Some(prev_eid) = self.reverse_nodemap.insert(a, hedge_id_a) {
            let next_eid = self.neighbors.insert(prev_eid, hedge_id_a).unwrap();
            self.neighbors.insert(hedge_id_a, next_eid);
        } else {
            self.neighbors.insert(hedge_id_a, hedge_id_a);
        }
        self.nodemap.insert(hedge_id_b, b);
        if let Some(prev_eid) = self.reverse_nodemap.insert(b, hedge_id_b) {
            let next_eid = self.neighbors.insert(prev_eid, hedge_id_b).unwrap();
            self.neighbors.insert(hedge_id_b, next_eid);
        } else {
            self.neighbors.insert(hedge_id_b, hedge_id_b);
        }
        hedge_id_a
    }

    /// Add external, as a fixed point involution half edge.
    fn add_external(&mut self, a: NodeId, data: E, dir: Direction) -> HedgeId {
        let id = self.edges.insert(data);
        self.involution.insert(id, Invel { data: id, dir });
        self.nodemap.insert(id, a);
        if let Some(prev_eid) = self.reverse_nodemap.insert(a, id) {
            let next_eid = self.neighbors.insert(prev_eid, id).unwrap();
            self.neighbors.insert(id, next_eid).unwrap();
        } else {
            self.neighbors.insert(id, id);
        }
        id
    }

    fn edges_incident(&self, node: NodeId) -> impl Iterator<Item = HedgeId> + '_ {
        IncidentIterator::new(self, self.reverse_nodemap[node])
    }

    fn edges_between(&self, a: NodeId, b: NodeId) -> impl Iterator<Item = HedgeId> + '_ {
        self.edges_incident(a)
            .filter(move |&i| self.nodemap[self.involution[i].data] == b)
    }

    fn internal_edges_incident(&self, node: NodeId) -> impl Iterator<Item = HedgeId> + '_ {
        self.edges_incident(node)
            .filter(move |&i| self.nodemap[self.involution[i].data] != node)
    }

    fn external_edges_incident(&self, node: NodeId) -> impl Iterator<Item = HedgeId> + '_ {
        self.edges_incident(node)
            .filter(move |&i| self.nodemap[self.involution[i].data] == node)
    }

    fn degree(&self, node: NodeId) -> usize {
        self.edges_incident(node).collect::<Vec<_>>().len()
    }

    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.edges_incident(node)
            .map(move |i| self.nodemap[self.involution[i].data])
    }

    // fn map_nodes<F, U>(&self, f: F) -> HalfEdgeGraph<U, E>
    // where
    //     F: Fn(&N) -> U,
    //     E: Clone,
    // {
    //     let edges = self.edges.clone();
    //     let involution = self.involution.clone();

    //     let mut nodes = DenseSlotMap::with_key();
    //     let mut nodemap = SecondaryMap::new();

    //     for n in &self.nodes {
    //         let nid = nodes.insert(f(n.1));
    //         for e in self.edges_incident(n.0) {
    //             nodemap.insert(e, nid);
    //         }
    //     }

    //     HalfEdgeGraph {
    //         edges,
    //         involution,
    //         nodes,
    //         nodemap,
    //     }
    // }
}

#[test]
fn merge() {
    let mut graph = HalfEdgeGraph::new();
    let a = graph.add_node_with_edges_fn(1, &[1, -2, 3, 4, 5], |a, b| (*a == -b, Direction::None));
    let b = graph.add_node_with_edges_fn(2, &[-1, 2, -6, 7, 8], |a, b| (*a == -b, Direction::None));
    let c =
        graph.add_node_with_edges_fn(4, &[-4, 6, 9, 10, 11], |a, b| (*a == -b, Direction::None));

    println!("{}", graph.dot());
    println!("{}", graph.degree(a));
    println!("{}", graph.degree(b));

    for (i, n) in &graph.neighbors {
        println!("{} {}", graph.edges[i], graph.edges[*n]);
    }

    let d = graph.merge_nodes(a, b, 3);
    // print!("merge");

    // for (i, n) in &graph.neighbors {
    //     println!("{} {}", graph.edges[i], graph.edges[*n]);
    // }

    // // println!("{:#?}", graph);

    println!("{}", graph.dot());
    println!("{}", graph.degree(c));
    println!("{}", graph.neighbors.len());

    let _e = graph.merge_nodes(c, d, 5);

    println!("{}", graph.dot());
    println!("neighbors");
    for (i, n) in &graph.neighbors {
        println!("{} {}", graph.edges[i], graph.edges[*n]);
    }
    println!("edges");
    for (i, n) in &graph.edges {
        println!("{:?} {}", i, n);
    }
    println!("involution");

    for (i, n) in &graph.involution {
        println!("{} {}", graph.edges[i], graph.edges[n.data]);
    }
    println!("nodemap");

    for (i, n) in &graph.nodemap {
        println!("{} {}", graph.edges[i], graph.nodes[*n]);
    }
    println!("reverse_nodemap");

    for (i, n) in &graph.reverse_nodemap {
        println!("{} {}", graph.nodes[i], graph.edges[*n]);
    }
    println!("nodes");

    for (i, n) in &graph.nodes {
        println!("{:?} {}", i, n);
    }
    // println!("{}", graph.degree(e));
    // println!("{}", graph.neighbors.len());

    // let mut graph = HalfEdgeGraph::new();
    // let a = graph.add_node_with_edges("a", &[10, 2, 3]);
    // let b = graph.add_node_with_edges("b", &[20, 3, 4]);
    // let c = graph.add_node_with_edges("c", &[30, 4, 2]);
    // let d = graph.add_node_with_edges("d", &[20]);
    // let e = graph.add_node_with_edges("e", &[30]);

    // println!("Test {}", graph.dot());
    // println!("{}", graph.degree(a));
    // println!("{}", graph.degree(b));

    // for (i, n) in &graph.neighbors {
    //     println!("{} {}", graph.edges[i], graph.edges[*n]);
    // }

    // let d = graph.merge_nodes(d, b, "bd");

    // // for (i, n) in &graph.neighbors {
    // //     println!("{} {}", graph.edges[i], graph.edges[*n]);
    // // }

    // println!("{}", graph.degree(c));
    // println!("{}", graph.neighbors.len());

    // println!("{}", graph.dot());

    // let e = graph.merge_nodes(c, e, "ce");

    // if graph.validate_neighbors() {
    //     println!("valid");
    // } else {
    //     println!("invalid");
    // }

    // println!("{}", graph.dot());
    // let f = graph.merge_nodes(d, e, "de");

    // if graph.validate_neighbors() {
    //     println!("valid");
    // } else {
    //     println!("invalid");
    // }

    // println!("{}", graph.dot());
    // println!("{}", graph.node_labels());
    // println!("{}", graph.degree(a));
    // println!("{}", graph.neighbors.len());

    // let g = graph.merge_nodes(a, f, "af");

    // if graph.validate_neighbors() {
    //     println!("valid");
    // } else {
    //     println!("invalid");
    // }

    // println!("{}", graph.dot());
    // println!("{}", graph.neighbors.len());
    // println!("{}", graph.degree(g));

    // println!("{}", graph.degree(b));
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TensorNetwork<T: TensorStructure, S>
where
    T::Slot: Serialize + for<'a> Deserialize<'a>,
{
    pub graph: HalfEdgeGraph<T, <T as TensorStructure>::Slot>,
    // pub params: AHashSet<Atom>,
    pub scalar: Option<S>,
}

impl<T: TensorStructure, S> TensorNetwork<T, S>
where
    T::Slot: Serialize + for<'a> Deserialize<'a>,
{
    pub fn map_scalar<F, U>(self, f: F) -> TensorNetwork<T, U>
    where
        F: FnOnce(S) -> U,
    {
        TensorNetwork {
            graph: self.graph,
            scalar: self.scalar.map(f),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TensorNetworkSet<T: TensorStructure, S>
where
    T::Slot: Serialize + for<'a> Deserialize<'a>,
{
    pub networks: Vec<TensorNetwork<T, S>>,
}

impl<T, S> TensorNetworkSet<T, S>
where
    T: TensorStructure,
    T::Slot: Serialize + for<'a> Deserialize<'a>,
{
    pub fn new() -> Self {
        TensorNetworkSet {
            networks: vec![],
            // scalars: vec![],
        }
    }

    pub fn push(&mut self, network: TensorNetwork<T, S>) {
        // self.scalars.push(network.scalar);
        self.networks.push(network);
    }
}

impl<T, S> Default for TensorNetworkSet<T, S>
where
    T: TensorStructure,
    T::Slot: Serialize + for<'a> Deserialize<'a>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "shadowing")]
pub type EvalTreeTensorNetworkSet<T, S> = SharedTensorNetworkSet<EvalTree<T>, S>;

#[cfg(feature = "shadowing")]
pub type EvalTensorNetworkSet<T, S> = SharedTensorNetworkSet<ExpressionEvaluator<T>, S>;

#[cfg(feature = "shadowing")]
pub type CompiledTensorNetworkSet<S> = SharedTensorNetworkSet<CompiledEvaluator, S>;

#[derive(Debug, Clone)]
pub struct SharedTensorNetworkSet<D, S: TensorStructure>
where
    S::Slot: Serialize + for<'a> Deserialize<'a>,
{
    pub networks: Vec<HalfEdgeGraph<DataTensor<usize, S>, <S as TensorStructure>::Slot>>,
    pub shared_data: D,
    pub len: usize,
}

impl<'a, T: HasTensorData + GetTensorData<GetDataOwned = T::Data>> TensorNetworkSet<T, T::Data>
where
    T: Clone + Contract<LCM = T> + Trace,
    T::Structure: TensorStructure<Slot: Serialize + for<'c> Deserialize<'c>>,
    T::Data: Clone,
    // T::GetData<'a>: &'a T::Data,
{
    pub fn result(&'a self) -> Result<Vec<T::Data>, TensorNetworkError> {
        let mut data = vec![];
        for n in &self.networks {
            data.push(n.result_scalar()?);
        }
        Ok(data)
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone>
    TensorNetworkSet<ParamTensor<S>, Atom>
{
    pub fn eval_tree(
        &self,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<EvalTreeTensorNetworkSet<Rational, S>>
    where
        S: TensorStructure,
    {
        let mut networks = vec![];

        let mut atoms = vec![];
        let mut id = 0;

        let one = Atom::new_num(1);

        for s in self.networks.iter().map(|x| x.scalar.as_ref()) {
            if let Some(a) = s {
                atoms.push(a.as_view());
                // trace!("Scalar is Some {}", a);
            } else {
                atoms.push(one.as_view());
                // trace!("Scalar is None");
            }
        }

        id += self.networks.len();
        for net in &self.networks {
            let mut usize_net = HalfEdgeGraph::new();
            for (_, p) in &net.graph.nodes {
                let structure = p.structure().clone();
                let usize_tensor = match &p.tensor {
                    DataTensor::Dense(d) => {
                        let oldid = id;
                        id += d.size().unwrap();
                        for (_, a) in d.flat_iter() {
                            atoms.push(a.as_view());
                        }
                        DataTensor::Dense(DenseTensor::from_data(
                            Vec::from_iter(oldid..id),
                            structure,
                        )?)
                    }
                    DataTensor::Sparse(s) => {
                        let mut t = SparseTensor::empty(structure);
                        for (i, a) in s.flat_iter() {
                            t.set_flat(i, id)?;
                            atoms.push(a.as_view());
                            id += 1;
                        }
                        DataTensor::Sparse(t)
                    }
                };

                let slots = usize_tensor.external_structure().to_vec();
                usize_net.add_node_with_edges_fn(usize_tensor, &slots, |s, so| {
                    (s.matches(so), Direction::None)
                });
            }
            networks.push(usize_net);
        }

        Ok(EvalTreeTensorNetworkSet {
            networks,
            shared_data: AtomView::to_eval_tree_multiple(&atoms, fn_map, params)
                .map_err(|s| anyhow!(s))?,
            len: atoms.len(),
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: Clone + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>>
    EvalTreeTensorNetworkSet<Rational, S>
{
    pub fn horner_scheme(&mut self) {
        self.shared_data.horner_scheme();
    }
}

#[cfg(feature = "shadowing")]
impl<T, S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone>
    EvalTreeTensorNetworkSet<T, S>
{
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTreeTensorNetworkSet<T2, S>
    where
        T: Clone + PartialEq,
    {
        EvalTreeTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.map_coeff(f),
            len: self.len,
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    pub fn linearize(self, cpe_rounds: Option<usize>) -> EvalTensorNetworkSet<T, S>
    where
        T: Clone + Default + PartialEq,
    {
        EvalTensorNetworkSet {
            networks: self.networks,
            shared_data: self.shared_data.linearize(cpe_rounds),
            len: self.len,
        }
    }

    pub fn common_subexpression_elimination(&mut self)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
    {
        self.shared_data.common_subexpression_elimination()
    }

    pub fn evaluate(&mut self, params: &[T]) -> TensorNetworkSet<DataTensor<T, S>, T>
    where
        T: Real + SingleFloat,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let mut data = vec![zero; self.len];

        let mut networks = vec![];

        self.shared_data.evaluate(params, &mut data);

        let scalars: Vec<Option<T>> = data
            .iter()
            .take(self.networks.len())
            .map(|x| Some(x.clone()))
            .collect();
        for (i, net) in self.networks.iter().enumerate() {
            let mut data_net = HalfEdgeGraph::new();
            for (_, p) in net.nodes.iter() {
                let structure = p.structure().clone();
                let data_tensor = match &p {
                    DataTensor::Dense(d) => {
                        let mut t_data = vec![];
                        for (_, &a) in d.flat_iter() {
                            t_data.push(data[a].clone());
                        }
                        DataTensor::Dense(DenseTensor::from_data(t_data, structure).unwrap())
                    }
                    DataTensor::Sparse(s) => {
                        let mut t = SparseTensor::empty(structure);
                        for (i, &a) in s.flat_iter() {
                            t.set_flat(i, data[a].clone()).unwrap();
                        }
                        DataTensor::Sparse(t)
                    }
                };

                let slots = data_tensor.external_structure().to_vec();
                data_net.add_node_with_edges_fn(data_tensor, &slots, |s, so| {
                    (s.matches(so), Direction::None)
                });
            }
            networks.push(TensorNetwork {
                graph: data_net,
                scalar: scalars[i].clone(),
            });
        }

        TensorNetworkSet { networks }
    }
}

#[cfg(feature = "shadowing")]
impl<T, S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone>
    EvalTensorNetworkSet<T, S>
{
    pub fn evaluate(&mut self, params: &[T]) -> TensorNetworkSet<DataTensor<T, S>, T>
    where
        T: Real + SingleFloat,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let mut data = vec![zero; self.len];

        let mut networks = vec![];

        self.shared_data.evaluate(params, &mut data);

        let scalars: Vec<Option<T>> = data
            .iter()
            .take(self.networks.len())
            .map(|x| Some(x.clone()))
            .collect();
        for (i, net) in self.networks.iter().enumerate() {
            let mut data_net = HalfEdgeGraph::new();
            for (_, p) in net.nodes.iter() {
                let structure = p.structure().clone();
                let data_tensor = match &p {
                    DataTensor::Dense(d) => {
                        let mut t_data = vec![];
                        for (_, &a) in d.flat_iter() {
                            t_data.push(data[a].clone());
                        }
                        DataTensor::Dense(DenseTensor::from_data(t_data, structure).unwrap())
                    }
                    DataTensor::Sparse(s) => {
                        let mut t = SparseTensor::empty(structure);
                        for (i, &a) in s.flat_iter() {
                            t.set_flat(i, data[a].clone()).unwrap();
                        }
                        DataTensor::Sparse(t)
                    }
                };

                let slots = data_tensor.external_structure().to_vec();
                data_net.add_node_with_edges_fn(data_tensor, &slots, |s, so| {
                    (s.matches(so), Direction::None)
                });
            }
            networks.push(TensorNetwork {
                graph: data_net,
                scalar: scalars[i].clone(),
            });
        }

        TensorNetworkSet { networks }
    }

    /// Create a C++ code representation of the evaluation tree tensor.
    /// With `inline_asm` set to any value other than `None`,
    /// high-performance inline ASM code will be generated for most
    /// evaluation instructions. This often gives better performance than
    /// the `O3` optimization level and results in very fast compilation.
    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
        inline_asm: InlineASM,
    ) -> Result<SharedTensorNetworkSet<ExportedCode, S>, std::io::Error>
    where
        T: Display,
    {
        Ok(SharedTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.export_cpp(
                filename,
                function_name,
                include_header,
                inline_asm,
            )?,
            len: self.len,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone>
    SharedTensorNetworkSet<ExportedCode, S>
{
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<SharedTensorNetworkSet<CompiledCode, S>, std::io::Error> {
        Ok(SharedTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.compile(out, options)?,
            len: self.len,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone>
    SharedTensorNetworkSet<CompiledCode, S>
{
    pub fn load(&self) -> Result<SharedTensorNetworkSet<CompiledEvaluator, S>, String> {
        Ok(SharedTensorNetworkSet {
            networks: self.networks.clone(),
            shared_data: self.shared_data.load()?,
            len: self.len,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone>
    CompiledTensorNetworkSet<S>
{
    pub fn evaluate<T: symbolica::evaluate::CompiledEvaluatorFloat + Default + Clone>(
        &mut self,
        params: &[T],
    ) -> TensorNetworkSet<DataTensor<T, S>, T>
    where
        S: TensorStructure + Clone,
    {
        let zero = T::default();
        let mut data = vec![zero; self.len];

        let mut networks = vec![];

        self.shared_data.evaluate(params, &mut data);

        let scalars: Vec<Option<T>> = data
            .iter()
            .take(self.networks.len())
            .map(|x| Some(x.clone()))
            .collect();

        for (i, net) in self.networks.iter().enumerate() {
            let mut data_net = HalfEdgeGraph::new();
            for (_, p) in net.nodes.iter() {
                let structure = p.structure().clone();
                let data_tensor = match &p {
                    DataTensor::Dense(d) => {
                        let mut t_data = vec![];
                        for (_, &a) in d.flat_iter() {
                            t_data.push(data[a].clone());
                        }
                        DataTensor::Dense(DenseTensor::from_data(t_data, structure).unwrap())
                    }
                    DataTensor::Sparse(s) => {
                        let mut t = SparseTensor::empty(structure);
                        for (i, &a) in s.flat_iter() {
                            t.set_flat(i, data[a].clone()).unwrap();
                        }
                        DataTensor::Sparse(t)
                    }
                };

                let slots = data_tensor.external_structure().to_vec();
                data_net.add_node_with_edges_fn(data_tensor, &slots, |s, so| {
                    (s.matches(so), Direction::None)
                });
            }
            networks.push(TensorNetwork {
                graph: data_net,
                scalar: scalars[i].clone(),
            });
        }

        TensorNetworkSet { networks }
    }
}

// impl<T: TensorStructure + Serialize, Sc: Serialize> Serialize for TensorNetwork<T, Sc>
// where
//     T::Slot: Serialize,
// {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         let mut state = serializer.serialize_struct("TensorNetwork", 2)?;
//         state.serialize_field("graph", &self.graph)?;
//         state.serialize_field("scalar", &self.scalar)?;
//         state.end()
//     }
// }

impl<T: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>, S> TensorNetwork<T, S> {
    pub fn scalar_mul(&mut self, scalar: S)
    where
        S: FallibleMul<S, Output = S>,
    {
        if let Some(ref mut s) = self.scalar {
            *s = scalar.mul_fallible(s).unwrap();
        } else {
            self.scalar = Some(scalar);
        }
    }

    pub fn edge_to_min_degree_node(&self) -> Option<HedgeId> {
        let mut neighs = self.graph.reverse_nodemap.clone();
        if neighs.is_empty() {
            return None;
        }

        let mut all_ext = true;

        for (h, g) in &self.graph.involution {
            if h != g.data {
                all_ext = false;
                break;
            }
        }

        if all_ext {
            return None;
        }

        loop {
            for (node, initial) in &mut neighs {
                *initial = self.graph.neighbors[*initial]; //neighbors is a linked list that is cyclic, here we essentially move all the initial pointers to the next node

                let start = self.graph.reverse_nodemap[node]; // get original start

                if *initial == start {
                    // first neighborhood to completely cycle
                    let mut all_ext_in_cycle = false;
                    while self.graph.involution[*initial].data == *initial && !all_ext_in_cycle {
                        //loop through the cycle till we find a non self involution i.e. an internal edge
                        *initial = self.graph.neighbors[*initial];
                        if *initial == start {
                            // if we're back at the start then we have cycled through all the edges and they are all external
                            all_ext_in_cycle = true;
                        }
                    }
                    if !all_ext_in_cycle {
                        return Some(*initial);
                    }
                }
            }
        }
    }

    pub fn to_vec(&self) -> Vec<&T> {
        self.graph.nodes.values().collect()
    }
}

#[cfg(feature = "shadowing")]
impl<T, S> TensorNetwork<MixedTensor<T, S>, SerializableAtom>
where
    S: Clone + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Debug,
    T: Clone,
{
    #[cfg(feature = "shadowing")]
    pub fn to_symbolic_tensor_vec(mut self) -> Vec<ParamTensor<S>> {
        self.graph
            .nodes
            .drain()
            .flat_map(|(_, n)| n.try_into_parametric()) //filters out all parametric tensors
            .collect()
    }

    // pub fn evaluate<'a, D>(
    //     &'a self,
    //     const_map: &AHashMap<AtomView<'a>, D>,
    // ) -> TensorNetwork<DataTensor<D, N>>
    // where
    //     D: Clone,
    //     N: Clone + TensorStructure,
    // {
    //     // let mut evaluated_net = TensorNetwork::new();
    //     // for (id,t) in &self.graph.nodes {

    //     //     let evaluated_tensor = match t{
    //     //         MixedTensor::Complex(t)=> t.into(),
    //     //         MixedTensor::Symbolic(t)=> t.evaluate(const_map),
    //     //         MixedTensor::Float(t)=> t.into(),
    //     //     }
    //     //     evaluated_net.push(evaluated_tensor);
    //     // }

    //     // evaluated_net
    // }

    #[cfg(feature = "shadowing")]
    pub fn evaluate_real<A: AtomCore + KeyLookup, F: Fn(&Rational) -> T + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &AHashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
    ) where
        T: Real + for<'c> From<&'c Rational>,
    {
        for (_, n) in &mut self.graph.nodes {
            n.evaluate_real(coeff_map, const_map, function_map);
        }
    }

    #[cfg(feature = "shadowing")]
    pub fn evaluate_complex<A: AtomCore + KeyLookup, F: Fn(&Rational) -> SymComplex<T> + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &AHashMap<A, SymComplex<T>>,
        function_map: &HashMap<Symbol, EvaluationFn<A, SymComplex<T>>>,
    ) where
        T: Real + for<'c> From<&'c Rational>,
        SymComplex<T>: Real + for<'c> From<&'c Rational>,
    {
        for (_, n) in &mut self.graph.nodes {
            n.evaluate_complex(coeff_map, const_map, function_map);
        }
    }

    pub fn to_fully_parametric(self) -> TensorNetwork<ParamTensor<S>, SerializableAtom>
    where
        T: TrySmallestUpgrade<Atom, LCM = Atom>,
        Complex<T>: TrySmallestUpgrade<Atom, LCM = Atom>,
    {
        let mut tensors = vec![];

        for n in self.graph.nodes.values() {
            tensors.push(match n {
                MixedTensor::Concrete(RealOrComplexTensor::Real(t)) => {
                    ParamTensor::composite(t.try_upgrade::<Atom>().unwrap().into_owned())
                }
                MixedTensor::Concrete(RealOrComplexTensor::Complex(t)) => {
                    ParamTensor::composite(t.try_upgrade::<Atom>().unwrap().into_owned())
                }
                MixedTensor::Param(t) => t.clone(),
            });
        }

        TensorNetwork {
            graph: TensorNetwork::<ParamTensor<S>, Atom>::generate_network_graph(tensors),
            // params: AHashSet::new(),
            scalar: self.scalar,
        }
    }
}

#[cfg(feature = "shadowing")]
use std::hash::Hash;

#[cfg(feature = "shadowing")]
impl<P: StorageTensor<Data = Atom>> TensorAtomMaps for TensorNetwork<P, Atom>
where
    P::Structure: Clone + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    type ContainerData<T> = TensorNetwork<P::ContainerData<T>, T>;

    fn replace_all<R: symbolica::id::BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self {
        let rhs = rhs.borrow();
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.replace_all(pattern, rhs, conditions, settings));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.replace_all(pattern, rhs, conditions, settings)),
        }
    }

    fn apart(&self, x: Symbol) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.apart(x));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.apart(x)),
        }
    }

    fn cancel(&self) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.cancel());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.cancel()),
        }
    }

    fn expand(&self) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.expand());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.expand()),
        }
    }

    fn factor(&self) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.factor());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.factor()),
        }
    }

    fn nsolve<N: SingleFloat + Real + PartialOrd + Clone>(
        &self,
        x: Symbol,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> std::result::Result<Self::ContainerData<N>, std::string::String> {
        let graph = self.graph.map_nodes_ref_result(|(_, a)| {
            a.nsolve(x, init.clone(), prec.clone(), max_iterations)
        })?;
        Ok(TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.nsolve(x, init.clone(), prec.clone(), max_iterations))
                .transpose()?,
        })
    }

    fn series<T: AtomCore>(
        &self,
        x: Symbol,
        expansion_point: T,
        depth: Rational,
        depth_is_absolute: bool,
    ) -> std::result::Result<
        Self::ContainerData<symbolica::poly::series::Series<symbolica::domains::atom::AtomField>>,
        &'static str,
    > {
        let graph = self.graph.map_nodes_ref_result(|(_, a)| {
            a.series(
                x,
                expansion_point.as_atom_view(),
                depth.clone(),
                depth_is_absolute,
            )
        })?;
        Ok(TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| {
                    a.series(
                        x,
                        expansion_point.as_atom_view(),
                        depth.clone(),
                        depth_is_absolute,
                    )
                })
                .transpose()?,
        })
    }

    fn evaluate<A: AtomCore + KeyLookup, T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
        // cache: &mut HashMap<AtomView<'b>, T>,
    ) -> std::result::Result<Self::ContainerData<T>, std::string::String> {
        let graph = self
            .graph
            .map_nodes_ref_result(|(_, a)| a.evaluate(coeff_map, const_map, function_map))?;
        Ok(TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.evaluate(coeff_map, const_map, function_map))
                .transpose()?,
        })
    }

    fn together(&self) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.together());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.together()),
        }
    }

    fn expand_in<T: AtomCore>(&self, var: T) -> Self {
        let var = var.as_atom_view();
        let graph = self.graph.map_nodes_ref(|(_, a)| a.expand_in(var));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.expand_in(var)),
        }
    }

    fn map_terms(
        &self,
        f: impl Fn(AtomView) -> Atom + Send + Sync + Clone,
        n_cores: usize,
    ) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.map_terms(f.clone(), n_cores));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.map_terms(f.clone(), n_cores)),
        }
    }

    fn zero_test(
        &self,
        iterations: usize,
        tolerance: f64,
    ) -> Self::ContainerData<symbolica::id::ConditionResult> {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.zero_test(iterations, tolerance));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.zero_test(iterations, tolerance)),
        }
    }

    fn derivative(&self, x: Symbol) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.derivative(x));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.derivative(x)),
        }
    }

    fn expand_num(&self) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.expand_num());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.expand_num()),
        }
    }

    fn to_pattern(&self) -> Self::ContainerData<Pattern> {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.to_pattern());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.to_pattern()),
        }
    }

    fn coefficient<T: AtomCore>(&self, x: T) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.coefficient(x.as_atom_view()));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.coefficient(x.as_atom_view())),
        }
    }

    fn collect_num(&self) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.collect_num());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.collect_num()),
        }
    }

    fn replace_map<F: Fn(AtomView, &symbolica::id::Context, &mut Atom) -> bool>(
        &self,
        m: &F,
    ) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.replace_map(m));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.replace_map(m)),
        }
    }

    fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: symbolica::poly::Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<MultivariatePolynomial<R, E>> {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.to_polynomial(field, var_map.clone()));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.to_polynomial(field, var_map.clone())),
        }
    }

    fn expand_via_poly<E: symbolica::poly::Exponent, T: AtomCore>(&self, var: Option<T>) -> Self {
        let var = var.as_ref().map(|a| a.as_atom_view());
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.expand_via_poly::<E, AtomView>(var));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.expand_via_poly::<E, AtomView>(var)),
        }
    }

    fn expand_in_symbol(&self, var: Symbol) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.expand_in_symbol(var));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.expand_in_symbol(var)),
        }
    }

    fn map_coefficient<
        F: Fn(symbolica::coefficient::CoefficientView) -> symbolica::coefficient::Coefficient + Copy,
    >(
        &self,
        f: F,
    ) -> Self {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.map_coefficient(f));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.map_coefficient(f)),
        }
    }

    fn replace_all_mut<R: symbolica::id::BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        let rhs = rhs.borrow();
        self.graph
            .map_nodes_mut(|(_, a)| a.replace_all_mut(pattern, rhs, conditions, settings));
        if let Some(a) = &mut self.scalar {
            a.replace_all_mut(pattern, rhs, conditions, settings)
        }
    }

    fn replace_map_mut<F: Fn(AtomView, &symbolica::id::Context, &mut Atom) -> bool>(
        &mut self,
        m: &F,
    ) {
        self.graph.map_nodes_mut(|(_, a)| a.replace_map_mut(m));
        if let Some(a) = self.scalar.as_mut() {
            a.replace_map_mut(m);
        }
    }

    fn replace_all_repeat<R: symbolica::id::BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self {
        let rhs = rhs.borrow();
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.replace_all_repeat(pattern, rhs, conditions, settings));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.replace_all_repeat(pattern, rhs, conditions, settings)),
        }
    }

    fn to_canonical_string(&self) -> Self::ContainerData<std::string::String> {
        let graph = self.graph.map_nodes_ref(|(_, a)| a.to_canonical_string());
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.to_canonical_string()),
        }
    }

    fn replace_all_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.replace_all_multiple(replacements));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.replace_all_multiple(replacements)),
        }
    }

    fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.set_coefficient_ring(vars));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.set_coefficient_ring(vars)),
        }
    }

    fn coefficients_to_float(&self, f: u32) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.coefficients_to_float(f));
        TensorNetwork {
            graph,
            scalar: self.scalar.as_ref().map(|a| a.coefficients_to_float(f)),
        }
    }

    fn map_terms_single_core(&self, f: impl Fn(AtomView) -> Atom + Clone) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.map_terms_single_core(f.clone()));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.map_terms_single_core(f.clone())),
        }
    }

    fn to_polynomial_in_vars<E: symbolica::poly::Exponent>(
        &self,
        var_map: &Arc<Vec<Variable>>,
    ) -> Self::ContainerData<MultivariatePolynomial<symbolica::domains::atom::AtomField, E>> {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.to_polynomial_in_vars(var_map));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.to_polynomial_in_vars(var_map)),
        }
    }

    fn replace_all_repeat_mut<R: symbolica::id::BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        let rhs = rhs.borrow();
        self.graph
            .map_nodes_mut(|(_, a)| a.replace_all_repeat_mut(pattern, rhs, conditions, settings));

        if let Some(a) = self.scalar.as_mut() {
            a.replace_all_repeat_mut(pattern, rhs, conditions, settings);
        }
    }

    fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<RationalPolynomial<RO, E>>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.to_rational_polynomial(field, out_field, var_map.clone()));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.to_rational_polynomial(field, out_field, var_map.clone())),
        }
    }

    fn rationalize_coefficients(&self, relative_error: &Rational) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.rationalize_coefficients(relative_error));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.rationalize_coefficients(relative_error)),
        }
    }

    fn replace_all_multiple_mut<T: BorrowReplacement>(&mut self, replacements: &[T]) {
        self.graph
            .map_nodes_mut(|(_, a)| a.replace_all_multiple_mut(replacements));
        if let Some(a) = self.scalar.as_mut() {
            a.replace_all_multiple_mut(replacements);
        }
    }

    fn replace_all_multiple_repeat<T: BorrowReplacement>(&self, replacements: &[T]) -> Self {
        let graph = self
            .graph
            .map_nodes_ref(|(_, a)| a.replace_all_multiple_repeat(replacements));
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.replace_all_multiple_repeat(replacements)),
        }
    }

    fn replace_all_multiple_repeat_mut<T: BorrowReplacement>(&mut self, replacements: &[T]) {
        self.graph
            .map_nodes_mut(|(_, a)| a.replace_all_multiple_repeat_mut(replacements));
        if let Some(a) = self.scalar.as_mut() {
            a.replace_all_multiple_repeat_mut(replacements);
        }
    }

    fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<FactorizedRationalPolynomial<RO, E>>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        let graph = self.graph.map_nodes_ref(|(_, a)| {
            a.to_factorized_rational_polynomial(field, out_field, var_map.clone())
        });
        TensorNetwork {
            graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| a.to_factorized_rational_polynomial(field, out_field, var_map.clone())),
        }
    }
}

// use crate::parametric::atomcore::PatternReplacement;
// fn replace_all(
//     &self,
//     pattern: &Pattern,
//     rhs: &PatternOrMap,
//     conditions: Option<&Condition<PatternRestriction>>,
//     settings: Option<&MatchSettings>,
// ) -> Self {
//     let graph = self
//         .graph
//         .map_nodes_ref(|(_, a)| a.replace_all(pattern, rhs, conditions, settings));
//     TensorNetwork {
//         graph,
//         scalar: self
//             .scalar
//             .as_ref()
//             .map(|a| SerializableAtom(a.0.replace_all(pattern, rhs, conditions, settings))),
//     }
// }

// fn replace_all_mut(
//     &mut self,
//     pattern: &Pattern,
//     rhs: &PatternOrMap,
//     conditions: Option<&Condition<PatternRestriction>>,
//     settings: Option<&MatchSettings>,
// ) {
//     self.graph
//         .map_nodes_mut(|(_, a)| a.replace_all_mut(pattern, rhs, conditions, settings));

//     if let Some(a) = self.scalar.as_mut() {
//         a.0 = a.0.replace_all(pattern, rhs, conditions, settings);
//     }
// }

// fn replace_all_multiple<R: BorrowReplacement>(&self, replacements: &[R]) -> Self {
//     let graph = self
//         .graph
//         .map_nodes_ref(|(_, a)| a.replace_all_multiple(replacements));
//     TensorNetwork {
//         graph,
//         scalar: self
//             .scalar
//             .as_ref()
//             .map(|a| SerializableAtom(a.0.replace_all_multiple(replacements))),
//     }
// }

// fn replace_all_multiple_mut(&mut self, replacements: &[Replacement<'_>]) {
//     self.graph
//         .map_nodes_mut(|(_, a)| a.replace_all_multiple_mut(replacements));
//     if let Some(a) = self.scalar.as_mut() {
//         a.0 = a.0.replace_all_multiple(replacements);
//     }
// }

// fn replace_all_multiple_repeat_mut(&mut self, replacements: &[Replacement<'_>]) {
//     self.graph
//         .map_nodes_mut(|(_, a)| a.replace_all_multiple_repeat_mut(replacements));
//     if let Some(a) = self.scalar.as_mut() {
//         Self::replace_repeat_multiple_atom(&mut a.0, replacements);
//     }
// }

#[cfg(feature = "shadowing")]
impl<S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone>
    TensorNetwork<ParamTensor<S>, SerializableAtom>
{
    pub fn eval_tree(
        &self,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<TensorNetwork<EvalTreeTensor<Rational, S>, EvalTree<Rational>>, String>
    where
        S: TensorStructure,
    {
        let mut evaluate_net = TensorNetwork::new();
        for (_, t) in &self.graph.nodes {
            let evaluated_tensor = t.to_evaluation_tree(fn_map, params)?;
            evaluate_net.push(evaluated_tensor);
        }

        evaluate_net.scalar = self
            .scalar
            .as_ref()
            .map(|a| a.0.as_view().to_evaluation_tree(fn_map, params))
            .transpose()?;

        Ok(evaluate_net)
    }

    pub fn evaluate<A: AtomCore + KeyLookup, D, F: Fn(&Rational) -> D + Copy>(
        &self,
        coeff_map: F,
        const_map: &AHashMap<A, D>,
        function_map: &HashMap<Symbol, EvaluationFn<A, D>>,
    ) -> Result<TensorNetwork<DataTensor<D, S>, D>, String>
    where
        D: Clone
            + symbolica::domains::float::Real
            + for<'c> std::convert::From<&'c symbolica::domains::rational::Rational>,
    {
        let mut evaluated_net = TensorNetwork::new();
        for (_, t) in &self.graph.nodes {
            let evaluated_tensor = t.evaluate(coeff_map, const_map, function_map)?;
            evaluated_net.push(evaluated_tensor);
        }

        evaluated_net.scalar = if let Some(s) = &self.scalar {
            Some(s.0.evaluate(coeff_map, const_map, function_map)?)
        } else {
            None
        };

        Ok(evaluated_net)
    }
}

#[cfg(feature = "shadowing")]
impl<T, S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>>
    TensorNetwork<EvalTreeTensor<T, S>, EvalTree<T>>
{
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(
        &self,
        f: &F,
    ) -> TensorNetwork<EvalTreeTensor<T2, S>, EvalTree<T2>>
    where
        T: Clone + PartialEq,
        S: Clone,
    {
        let new_graph = self.graph.map_nodes_ref(|(_, x)| x.map_coeff(f));
        TensorNetwork {
            graph: new_graph,
            scalar: self.scalar.as_ref().map(|a| a.map_coeff(f)),
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    pub fn linearize(
        self,
        cpe_rounds: Option<usize>,
    ) -> TensorNetwork<EvalTensor<ExpressionEvaluator<T>, S>, ExpressionEvaluator<T>>
    where
        T: Clone + Default + PartialEq,
        S: Clone,
    {
        let new_graph = self.graph.map_nodes(|(_, x)| x.linearize(cpe_rounds));
        TensorNetwork {
            graph: new_graph,
            scalar: self.scalar.map(|a| a.linearize(cpe_rounds)),
        }
    }

    pub fn common_subexpression_elimination(&mut self)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
        S: Clone,
    {
        self.graph
            .map_nodes_mut(|(_, x)| x.common_subexpression_elimination());
        if let Some(a) = self.scalar.as_mut() {
            a.common_subexpression_elimination()
        }
    }

    pub fn evaluate(&mut self, params: &[T]) -> TensorNetwork<DataTensor<T, S>, T>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let new_graph = self.graph.map_nodes_ref_mut(|(_, x)| x.evaluate(params));
        TensorNetwork {
            graph: new_graph,
            scalar: self.scalar.as_mut().map(|a| {
                let mut out = [zero];
                a.evaluate(params, &mut out);
                let [o] = out;
                o
            }),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T, S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>>
    TensorNetwork<EvalTensor<ExpressionEvaluator<T>, S>, ExpressionEvaluator<T>>
{
    pub fn evaluate(&mut self, params: &[T]) -> TensorNetwork<DataTensor<T, S>, T>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let new_graph = self.graph.map_nodes_ref_mut(|(_, x)| x.evaluate(params));
        TensorNetwork {
            graph: new_graph,
            scalar: self.scalar.as_mut().map(|a| {
                let mut out = [zero];
                a.evaluate(params, &mut out);
                let [o] = out;
                o
            }),
        }
    }
    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
        inline_asm: InlineASM,
    ) -> Result<
        TensorNetwork<EvalTensor<SerializableExportedCode, S>, SerializableExportedCode>,
        TensorNetworkError,
    >
    where
        T: NumericalFloatLike,
        S: Clone,
    {
        // TODO @Lucien with the new export_cpp you are now able to put these different functions in the same file!
        let new_graph = self.graph.map_nodes_ref_result(|(n, x)| {
            let function_name = format!("{function_name}_{}", n.data().as_ffi());
            let filename = format!("{filename}_{}.cpp", n.data().as_ffi());
            x.export_cpp(&filename, &function_name, include_header, inline_asm)
        })?;
        let function_name = format!("{function_name}_scalar");
        let filename = format!("{filename}_scalar.cpp");

        let exported_scalar = if let Some(ref s) = self.scalar {
            Some(SerializableExportedCode::export_cpp(
                s,
                &filename,
                &function_name,
                include_header,
                inline_asm,
            )?)
        } else {
            None
        };

        Ok(TensorNetwork {
            graph: new_graph,
            scalar: exported_scalar,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>>
    TensorNetwork<EvalTensor<SerializableExportedCode, S>, SerializableExportedCode>
{
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<
        TensorNetwork<EvalTensor<SerializableCompiledCode, S>, SerializableCompiledCode>,
        TensorNetworkError,
    >
    where
        S: Clone,
    {
        let new_graph = self
            .graph
            .map_nodes_ref_result(|(_, x)| x.compile(out, options.clone()))?;
        let exported_scalar = if let Some(ref s) = self.scalar {
            Some(s.compile(out, options.clone())?)
        } else {
            None
        };
        Ok(TensorNetwork {
            graph: new_graph,
            scalar: exported_scalar,
        })
    }

    pub fn compile_and_load(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<
        TensorNetwork<EvalTensor<SerializableCompiledEvaluator, S>, SerializableCompiledEvaluator>,
        TensorNetworkError,
    >
    where
        S: Clone,
    {
        let new_graph = self.graph.map_nodes_ref_result(|(_, x)| {
            x.compile(out, options.clone())?
                .load()
                .map_err(|s| TensorNetworkError::Other(anyhow!(s)))
        })?;
        let exported_scalar = if let Some(ref s) = self.scalar {
            Some(
                s.compile(out, options.clone())?
                    .load()
                    .map_err(|s| TensorNetworkError::Other(anyhow!(s)))?,
            )
        } else {
            None
        };
        Ok(TensorNetwork {
            graph: new_graph,
            scalar: exported_scalar,
        })
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>>
    TensorNetwork<CompiledEvalTensor<S>, SerializableCompiledEvaluator>
{
    pub fn evaluate_float(&mut self, params: &[f64]) -> TensorNetwork<DataTensor<f64, S>, f64>
    where
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let new_graph = self
            .graph
            .map_nodes_ref_mut(|(_, x)| x.evaluate_float(params));
        TensorNetwork {
            graph: new_graph,
            scalar: self.scalar.as_mut().map(|a| {
                let mut out = [zero];
                a.evaluate(params, &mut out);
                let [o] = out;
                o
            }),
        }
    }

    pub fn evaluate_complex(
        &mut self,
        params: &[SymComplex<f64>],
    ) -> TensorNetwork<DataTensor<SymComplex<f64>, S>, SymComplex<f64>>
    where
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        let new_graph = self
            .graph
            .map_nodes_ref_mut(|(_, x)| x.evaluate_complex(params));
        TensorNetwork {
            graph: new_graph,
            scalar: self.scalar.as_mut().map(|a| {
                let mut out = [zero];
                a.evaluate_complex(params, &mut out);
                let [o] = out;
                o
            }),
        }
    }

    pub fn evaluate<T: symbolica::evaluate::CompiledEvaluatorFloat + Default + Clone>(
        &mut self,
        params: &[T],
    ) -> TensorNetwork<DataTensor<T, S>, T>
    where
        S: TensorStructure + Clone,
    {
        let zero = T::default();
        let new_graph = self.graph.map_nodes_ref_mut(|(_, x)| x.evaluate(params));
        TensorNetwork {
            graph: new_graph,
            scalar: self.scalar.as_mut().map(|a| {
                let mut out = [zero];
                a.evaluate(params, &mut out);
                let [o] = out;
                o
            }),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<S: Clone + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>>
    TensorNetwork<EvalTreeTensor<Rational, S>, EvalTree<Rational>>
{
    pub fn horner_scheme(&mut self) {
        self.graph.map_nodes_mut(|(_, x)| x.horner_scheme());
        if let Some(a) = self.scalar.as_mut() {
            a.horner_scheme()
        }
    }
}

impl<T, S> From<Vec<T>> for TensorNetwork<T, S>
where
    T: HasStructure,
    T::Structure: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    fn from(tensors: Vec<T>) -> Self {
        TensorNetwork {
            graph: Self::generate_network_graph(tensors),
            // params: AHashSet::new(),
            scalar: None,
        }
    }
}

impl<T> Default for TensorNetwork<T, T::Scalar>
where
    T: HasStructure,
    T::Structure: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, S> TensorNetwork<T, S>
where
    T: HasStructure,
    T::Structure: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    pub fn new() -> Self {
        TensorNetwork {
            graph: HalfEdgeGraph::new(),
            // params: AHashSet::new(),
            scalar: None,
        }
    }

    pub fn push(&mut self, tensor: T) -> NodeId {
        let slots = tensor.external_structure().to_vec();
        self.graph
            .add_node_with_edges_fn(tensor, &slots, |s, so| (s.matches(so), Direction::None))
    }

    fn generate_network_graph(tensors: Vec<T>) -> HalfEdgeGraph<T, <T as TensorStructure>::Slot> {
        let mut graph = HalfEdgeGraph::<T, _>::new();

        for tensor in tensors {
            let slots = tensor.external_structure().to_vec();
            graph.add_node_with_edges_fn(tensor, &slots, |s, so| (s.matches(so), Direction::None));
        }

        graph
    }

    pub fn edge_to_min_degree_node_with_depth(&self, depth: usize) -> Option<HedgeId>
    where
        T: TracksCount,
    {
        let mut neighs: SecondaryMap<NodeId, HedgeId> = self
            .graph
            .reverse_nodemap
            .clone()
            .into_iter()
            .filter(|(n, _e)| self.graph.nodes[*n].contractions_num() < depth)
            .collect();
        if neighs.is_empty() {
            return None;
        }

        loop {
            let mut all_ext = true;
            for (node, initial) in &mut neighs {
                *initial = self.graph.neighbors[*initial];
                let start = self.graph.reverse_nodemap[node];

                if self.graph.involution[start].data != start
                    && self.graph.nodes[self.graph.nodemap[self.graph.involution[start].data]]
                        .contractions_num()
                        < depth
                {
                    all_ext = false;
                    if *initial == start {
                        return Some(start);
                    }
                }
            }
            if all_ext {
                return None;
            }
        }
    }
}
impl<T: HasTensorData + GetTensorData<GetDataOwned = T::Data>> TensorNetwork<T, T::Data>
where
    T: Clone + Contract<LCM = T> + Trace,
    T::Structure: TensorStructure<Slot: Serialize + for<'d> Deserialize<'d>>,
    T::Data: Clone,
{
    pub fn result_scalar(&self) -> Result<T::Data, TensorNetworkError> {
        match self.graph.nodes.len() {
            0 => self
                .scalar
                .clone()
                .ok_or(TensorNetworkError::ScalarFieldEmpty),
            1 => {
                let t = self.result()?.0;
                if t.is_scalar() {
                    Ok(t.get_owned_linear(0.into()).unwrap())
                } else {
                    Err(TensorNetworkError::NotScalarOutput)
                }
            }
            _ => Err(TensorNetworkError::MoreThanOneNode),
        }
    }
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorNetworkError {
    #[error("internal edge still present, contract it first")]
    InternalEdgePresent,
    #[error("Cannot contract edge")]
    CannotContractEdge,
    #[error("no nodes in the graph")]
    NoNodes,
    #[error("no scalar present")]
    NoScalar,
    #[error("more than one node in the graph")]
    MoreThanOneNode,
    #[error("is not scalar output")]
    NotScalarOutput,
    #[error("failed scalar multiplication")]
    FailedScalarMul,
    #[error("scalar field is empty")]
    ScalarFieldEmpty,
    #[error("not all scalars")]
    NotAllScalars,
    #[error("failed to contract")]
    FailedContract(ContractionError),
    #[error("negative exponent not yet supported")]
    NegativeExponent,
    #[error("failed to contract: {0}")]
    FailedContractMsg(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error("Io error")]
    InOut(#[from] std::io::Error),
}

impl<T, S> TensorNetwork<T, S>
where
    S: Clone,
    T: Clone
        + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>
        + Contract<LCM = T>
        + Trace,
{
    pub fn result(&self) -> Result<(T, Option<S>), TensorNetworkError> {
        if self.graph.involution.iter().any(|(ni, i)| ni != i.data) {
            Err(TensorNetworkError::InternalEdgePresent)
        } else {
            let mut iter = self.graph.nodes.iter();

            if let Some((_, t)) = iter.next() {
                let mut res = t.internal_contract();
                for (_, t) in iter {
                    res = res
                        .contract(&t.internal_contract())
                        .map_err(TensorNetworkError::FailedContract)?;
                }
                Ok((res, self.scalar.clone()))
            } else {
                Err(TensorNetworkError::NoNodes)
            }
        }
    }
}

impl<T, S: Clone> TensorNetwork<T, S>
where
    T: Clone
        + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>
        + HasStructure<Scalar: From<S>>
        + ScalarTensor
        + Contract<LCM = T>
        + Trace
        + ScalarMul<S, Output = T>,
{
    pub fn result_tensor_smart(&self) -> Result<T, TensorNetworkError> {
        match self.result() {
            Err(TensorNetworkError::NoNodes) => {
                let s = self
                    .scalar
                    .as_ref()
                    .ok_or(TensorNetworkError::NoScalar)?
                    .clone();
                Ok(T::new_scalar(s.into()))
            }
            Ok((t, s)) => {
                if let Some(s) = s {
                    Ok(t.scalar_mul(&s).unwrap())
                } else {
                    Ok(t)
                }
            }
            Err(e) => Err(e),
        }
    }
}

impl<T, S> TensorNetwork<T, S>
where
    T: Clone + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    pub fn result_tensor_ref(&self) -> Result<&T, TensorNetworkError> {
        match self.graph.nodes.len() {
            0 => Err(TensorNetworkError::NoNodes),
            1 => Ok(self.graph.nodes.iter().next().unwrap().1),
            _ => Err(TensorNetworkError::MoreThanOneNode),
        }
    }

    pub fn rich_graph(
        &self,
    ) -> HalfEdgeGraph<std::string::String, DisplayOption<<T as TensorStructure>::Slot>>
    where
        T: HasName<Name: Display>,
    {
        let mut rich = HalfEdgeGraph::new();
        let mut node_links = AHashMap::new();

        for (i, n) in &self.graph.edges {
            let node = self.graph.nodemap[i];
            let name = if let Some(n) = self.graph.nodes[node].name() {
                n.to_string()
            } else {
                format!("node{}", i.data().as_ffi())
            };

            let id =
                rich.add_node_with_edges_fn(name, &[DisplayOption::Some(*n)], |s, so| {
                    match (s, so) {
                        (DisplayOption::Some(s), DisplayOption::Some(so)) => {
                            (s.matches(so), Direction::None)
                        }
                        _ => (false, Direction::None),
                    }
                });
            node_links.entry(node).or_insert(Vec::new()).push(id);
        }

        for v in node_links.values() {
            for w in v.as_slice().windows(2) {
                if let [source, sink] = w {
                    rich.add_edge(*source, *sink, DisplayOption::None, Direction::Source);
                }
            }
        }
        rich
    }
}

impl<T: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a> + Display>, S>
    TensorNetwork<T, S>
{
    pub fn dot(&self) -> std::string::String {
        self.graph.dot()
    }
}

#[cfg(feature = "shadowing")]
impl<
        T: HasName<Name: IntoSymbol>
            + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a> + Display>,
        S,
    > TensorNetwork<T, S>
{
    pub fn dot_nodes(&self) -> std::string::String {
        let mut out = "graph {\n".to_string();
        out.push_str("  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\";");

        for (i, n) in &self.graph.nodes {
            out.push_str(&format!(
                "\n {} [label=\"{}\"] ",
                i.data().as_ffi(),
                n.name()
                    .map(|x| x.ref_into_symbol().to_string())
                    .unwrap_or("".into())
            ));
        }
        for (i, _) in &self.graph.neighbors {
            match i.cmp(&self.graph.involution[i].data) {
                std::cmp::Ordering::Greater => {
                    out.push_str(&format!(
                        "\n {} -- {} [label=\" {} \"];",
                        self.graph.nodemap[i].data().as_ffi(),
                        self.graph.nodemap[self.graph.involution[i].data]
                            .data()
                            .as_ffi(),
                        self.graph.edges[i]
                    ));
                }
                std::cmp::Ordering::Equal => {
                    out.push_str(&format!(
                        " \n ext{} [shape=none, label=\"\"];",
                        i.data().as_ffi()
                    ));
                    out.push_str(&format!(
                        "\n {} -- ext{} [label =\" {}\"];",
                        self.graph.nodemap[i].data().as_ffi(),
                        i.data().as_ffi(),
                        self.graph.edges[i]
                    ));
                }
                _ => {}
            }
        }

        out += "}";
        out
    }
}

#[cfg(feature = "shadowing")]
impl<T, S> TensorNetwork<MixedTensor<T, S>, SerializableAtom>
where
    S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone,
    T: Clone,
{
    pub fn replace_all<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self
    where
        S: Clone,
    {
        let rhs = rhs.borrow();
        let new_graph = self
            .graph
            .map_nodes_ref(|(_, t)| t.replace_all(pattern, rhs, conditions, settings));
        TensorNetwork {
            graph: new_graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| SerializableAtom(a.0.replace_all(pattern, rhs, conditions, settings))),
        }
    }

    pub fn replace_all_multiple<R: BorrowReplacement>(&self, replacements: &[R]) -> Self {
        let new_graph = self
            .graph
            .map_nodes_ref(|(_, t)| t.replace_all_multiple(replacements));
        TensorNetwork {
            graph: new_graph,
            scalar: self
                .scalar
                .as_ref()
                .map(|a| SerializableAtom(a.0.replace_all_multiple(replacements))),
        }
    }

    pub fn generate_params(&mut self) -> AHashSet<Atom> {
        let mut params = AHashSet::new();
        for (_, n) in self.graph.nodes.iter().filter(|(_, n)| n.is_parametric()) {
            for (_, a) in n.iter_flat() {
                if let AtomViewOrConcrete::Atom(a) = a {
                    params.insert(a.to_owned());
                }
            }
        }
        params
    }
}

// use log::trace;

#[cfg(feature = "shadowing")]
impl<'a, S, Sc> TryFrom<MulView<'a>> for TensorNetwork<MixedTensor<f64, S>, Sc>
where
    Sc: for<'r> TryFrom<AtomView<'r>>
        + FallibleMul<Output = Sc>
        + Clone
        + FallibleAdd<Sc, Output = Sc>,
    TensorNetworkError: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
    S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
    S::Slot: Serialize + for<'de> Deserialize<'de>,
    Rep: From<<S::Slot as IsAbstractSlot>::R>,
    MixedTensor<f64, S>: Contract<MixedTensor<f64, S>, LCM = MixedTensor<f64, S>>
        + Trace
        + ScalarMul<Sc, Output = MixedTensor<f64, S>>,
{
    type Error = TensorNetworkError;
    fn try_from(value: MulView<'a>) -> Result<Self, Self::Error> {
        let mut network: Self = TensorNetwork::new();

        let one = Atom::new_num(1);
        let mut scalars = Sc::try_from(one.as_view())?;
        let mut has_scalar = false;

        for arg in value.iter() {
            let mut net = Self::try_from(arg)?;
            // trace!("mul net: {}", net.dot_nodes());
            //
            if net.contract().is_err() {
                return Err(TensorNetworkError::FailedContractMsg(
                    format!("Mul failed: {}", arg).into(),
                ));
            }

            if let Some(ref s) = net.scalar {
                has_scalar = true;
                scalars = scalars.mul_fallible(s).unwrap();
            }
            match net.result() {
                Ok((t, _s)) => {
                    network.push(t);
                }
                Err(TensorNetworkError::NoNodes) => {}
                Err(e) => return Err(e),
            }
        }
        if has_scalar {
            // trace!("scalar mul : {}", scalars);
            network.scalar_mul(scalars);
        }
        Ok(network)
    }
}

#[cfg(feature = "shadowing")]
impl<'a, S, Sc> TryFrom<AtomView<'a>> for TensorNetwork<MixedTensor<f64, S>, Sc>
where
    Sc: for<'r> TryFrom<AtomView<'r>>
        + FallibleMul<Output = Sc>
        + Clone
        + FallibleAdd<Sc, Output = Sc>,
    TensorNetworkError: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
    S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
    S::Slot: Serialize + for<'de> Deserialize<'de>,
    Rep: From<<S::Slot as IsAbstractSlot>::R>,
    MixedTensor<f64, S>: Contract<MixedTensor<f64, S>, LCM = MixedTensor<f64, S>>
        + Trace
        + ScalarMul<Sc, Output = MixedTensor<f64, S>>,
{
    type Error = TensorNetworkError;
    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        match value {
            AtomView::Mul(m) => m.try_into(),
            AtomView::Fun(f) => f.try_into(),
            AtomView::Add(a) => a.try_into(),
            AtomView::Pow(p) => p.try_into(),
            a => {
                let mut network: Self = TensorNetwork::new();
                // let a = a.to_owned();

                // trace!("scalar atomview not a: {}", a);
                network.scalar = Some(a.try_into()?);
                Ok(network)
            }
        }
    }
}

#[cfg(feature = "shadowing")]
impl<'a, S, Sc> TryFrom<PowView<'a>> for TensorNetwork<MixedTensor<f64, S>, Sc>
where
    Sc: for<'r> TryFrom<AtomView<'r>>
        + FallibleMul<Output = Sc>
        + Clone
        + FallibleAdd<Sc, Output = Sc>,
    TensorNetworkError: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
    S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
    S::Slot: Serialize + for<'de> Deserialize<'de>,
    Rep: From<<S::Slot as IsAbstractSlot>::R>,
    MixedTensor<f64, S>: Contract<MixedTensor<f64, S>, LCM = MixedTensor<f64, S>>
        + Trace
        + ScalarMul<Sc, Output = MixedTensor<f64, S>>,
{
    type Error = TensorNetworkError;

    fn try_from(value: PowView<'a>) -> std::result::Result<Self, Self::Error> {
        let mut new: Self = TensorNetwork::new();

        let (base, exp) = value.get_base_exp();

        if let Ok(mut n) = i64::try_from(exp) {
            if n < 0 {
                new.scalar = Some(value.as_view().try_into()?);
            }
            if n == 0 {
                let one = Atom::new_num(1);
                new.scalar = Some(one.as_view().try_into()?);
                return Ok(new);
            } else if n == 1 {
                return base.try_into();
            }
            let mut net = Self::try_from(base)?;

            if net.contract().is_err() {
                return Err(TensorNetworkError::FailedContractMsg(
                    format!("Pow failed: {}", base).into(),
                ));
            }

            match net.result() {
                Ok((res, _s)) => {
                    new.push(res.clone());
                    while n > 1 {
                        if n % 2 == 0 {
                            new.push(res.clone().dual());
                        } else {
                            new.push(res.clone());
                        }

                        if new.contract().is_err() {
                            return Err(TensorNetworkError::FailedContractMsg(
                                value.as_view().to_string().into(),
                            ));
                        }

                        n -= 1;
                    }
                }
                Err(TensorNetworkError::NoNodes) => {
                    new.scalar = Some(value.as_view().try_into()?);
                }
                Err(e) => return Err(e),
            }
        } else {
            new.scalar = Some(value.as_view().try_into()?);
        }

        Ok(new)
    }
}

#[cfg(feature = "shadowing")]
impl<'a, S, Sc> TryFrom<FunView<'a>> for TensorNetwork<MixedTensor<f64, S>, Sc>
where
    Sc: for<'r> TryFrom<AtomView<'r>>
        + FallibleMul<Output = Sc>
        + Clone
        + FallibleAdd<Sc, Output = Sc>,
    TensorNetworkError: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
    S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
    S::Slot: Serialize + for<'de> Deserialize<'de>,
    Rep: From<<S::Slot as IsAbstractSlot>::R>,
    MixedTensor<f64, S>: Contract<MixedTensor<f64, S>, LCM = MixedTensor<f64, S>>
        + Trace
        + ScalarMul<Sc, Output = MixedTensor<f64, S>>,
{
    type Error = TensorNetworkError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        let mut network: Self = TensorNetwork::new();
        let s: Result<S, _> = value.try_into();

        let mut scalar = None;
        if let Ok(s) = s {
            let t = s
                .to_shell()
                .to_explicit()
                .ok_or(anyhow!("Cannot shadow"))?
                .internal_contract();
            network.push(t);
        } else {
            scalar = Some(value.as_view().try_into().map_err(Into::into)?);
        }

        network.scalar = scalar;
        Ok(network)
    }
}
// use log::trace;
#[cfg(feature = "shadowing")]
impl<'a, S, Sc> TryFrom<AddView<'a>> for TensorNetwork<MixedTensor<f64, S>, Sc>
where
    Sc: for<'r> TryFrom<AtomView<'r>>
        + FallibleMul<Output = Sc>
        + Clone
        + FallibleAdd<Sc, Output = Sc>,
    TensorNetworkError: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
    S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
    S::Slot: Serialize + for<'de> Deserialize<'de>,
    Rep: From<<S::Slot as IsAbstractSlot>::R>,
    MixedTensor<f64, S>: Contract<MixedTensor<f64, S>, LCM = MixedTensor<f64, S>>
        + Trace
        + ScalarMul<Sc, Output = MixedTensor<f64, S>>,
{
    type Error = TensorNetworkError;
    fn try_from(value: AddView<'a>) -> Result<Self, Self::Error> {
        // trace!("AddView: {}", value.as_view());
        let mut tensors = vec![];
        let zero = Atom::new_num(0);
        let mut scalars: Sc = zero.as_view().try_into()?;
        let mut is_scalar = false;
        for summand in value.iter() {
            // trace!("summand: {}", summand);
            let mut net = Self::try_from(summand)?;

            if net.contract().is_err() {
                return Err(TensorNetworkError::FailedContractMsg(
                    format!("Sum failed: {}", summand).into(),
                ));
            }

            match net.result() {
                Ok((mut t, s)) => {
                    if let Some(s) = s {
                        t = t.scalar_mul(&s).unwrap();
                    }
                    tensors.push(t);
                }
                Err(TensorNetworkError::NoNodes) => {
                    is_scalar = true;
                    if let Some(ref s) = net.scalar {
                        scalars = scalars.add_fallible(s).unwrap();
                    } else {
                        return Err(TensorNetworkError::ScalarFieldEmpty);
                    }

                    // println!("{:?}", net);
                }
                Err(e) => return Err(e),
            }
        }

        let net: TensorNetwork<_, _> = if let Some(sum) = tensors
            .into_iter()
            .reduce(|a, b| a.add_fallible(&b).unwrap())
        {
            if is_scalar {
                return Err(TensorNetworkError::NotAllScalars);
            }
            TensorNetwork::from(vec![sum])
        } else if !is_scalar {
            return Err(TensorNetworkError::NotAllScalars);
        } else {
            let mut net = TensorNetwork::new();
            // trace!("scalars sum: {}", scalars);
            net.scalar = Some(scalars);
            net
        };

        Ok(net)
    }
}

#[cfg(feature = "shadowing")]
impl<T, S> TensorNetwork<T, S>
where
    T: Shadowable + HasName<Name = Symbol, Args: IntoArgs>,
    T::Structure: Clone + ToSymbolic + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    pub fn sym_shadow(&mut self, name: &str) -> TensorNetwork<ParamTensor<T::Structure>, S> {
        {
            for (i, n) in &mut self.graph.nodes {
                n.set_name(Symbol::new(format!("{}{}", name, i.data().as_ffi())));
            }
        }

        let edges = self.graph.edges.clone();
        let involution = self.graph.involution.clone();
        let neighbors = self.graph.neighbors.clone();

        let mut nodes = DenseSlotMap::with_key();
        let mut nodemap = SecondaryMap::new();
        let mut reverse_nodemap = SecondaryMap::new();
        let mut params = AHashSet::new();

        for (i, n) in &self.graph.nodes {
            let node = n.expanded_shadow().unwrap();

            let nid = nodes.insert(ParamTensor::<T::Structure>::param(node.clone().into()));

            for (_, a) in node.flat_iter() {
                params.insert(a.clone());
            }
            let mut first = true;
            for e in self.graph.edges_incident(i) {
                if first {
                    reverse_nodemap.insert(nid, e);
                    first = false;
                }
                nodemap.insert(e, nid);
            }
        }

        let g = HalfEdgeGraph {
            edges,
            involution,
            reverse_nodemap,
            neighbors,
            nodes,
            nodemap,
        };

        let scalar = self.scalar.take();

        TensorNetwork {
            graph: g,
            // params,
            scalar,
        }
    }
}

impl<T: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>, S> TensorNetwork<T, S> {
    pub fn cast<U>(self) -> TensorNetwork<U, S>
    where
        T: CastStructure<U> + HasStructure,
        T::Structure: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
        U: HasStructure,
        U::Structure: From<T::Structure> + TensorStructure<Slot = T::Slot>,
    {
        TensorNetwork {
            graph: self.graph.map_nodes(|(_, x)| x.cast_structure()),
            scalar: self.scalar,
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T, S> TensorNetwork<T, S>
where
    T: HasName<Name: IntoSymbol, Args: IntoArgs>
        + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    pub fn append_map<U>(&self, fn_map: &mut FunctionMap<U>)
    where
        T: ShadowMapping<U>,
        T::Structure: Clone + ToSymbolic,
        S: Clone,
    {
        for (_, n) in &self.graph.nodes {
            n.expanded_append_map(fn_map)
        }
    }

    pub fn shadow(&self) -> TensorNetwork<ParamTensor<T::Structure>, S>
    where
        T: Shadowable,
        T::Structure: Clone + ToSymbolic + TensorStructure<Slot = T::Slot>,
        S: Clone,
    {
        let edges = self.graph.edges.clone();
        let involution = self.graph.involution.clone();
        let neighbors = self.graph.neighbors.clone();

        let mut nodes = DenseSlotMap::with_key();
        let mut nodemap = SecondaryMap::new();
        let mut reverse_nodemap = SecondaryMap::new();

        for (i, n) in &self.graph.nodes {
            let node = n.expanded_shadow().unwrap();

            let nid = nodes.insert(ParamTensor::<T::Structure>::param(node.clone().into()));

            let mut first = true;
            for e in self.graph.edges_incident(i) {
                if first {
                    reverse_nodemap.insert(nid, e);
                    first = false;
                }
                nodemap.insert(e, nid);
            }
        }

        let g = HalfEdgeGraph {
            edges,
            involution,
            reverse_nodemap,
            neighbors,
            nodes,
            nodemap,
        };

        let scalar = self.scalar.clone();

        TensorNetwork {
            graph: g,
            // params,
            scalar,
        }
    }
}

impl<T, S> TensorNetwork<T, S>
where
    T: HasName + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    pub fn name(&mut self, name: T::Name)
    where
        T::Name: From<std::string::String> + Display,
    {
        for (id, n) in &mut self.graph.nodes {
            n.set_name(format!("{}{}", name, id.data().as_ffi()).into());
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T, S> TensorNetwork<T, S>
where
    T: HasName<Name: IntoSymbol> + TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    pub fn namesym(&mut self, name: &str) {
        for (id, n) in &mut self.graph.nodes {
            n.set_name(IntoSymbol::from_str(&format!(
                "{}{}",
                name,
                id.data().as_ffi()
            )));
        }
    }
}

impl<T, S> TensorNetwork<T, S>
where
    T: Contract<T, LCM = T> + HasStructure,
    T::Structure: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>>,
{
    pub fn contract_algo(
        &mut self,
        edge_choice: impl Fn(&Self) -> Option<HedgeId>,
    ) -> Result<(), ContractionError> {
        if let Some(e) = edge_choice(self) {
            self.contract_edge(e)?;

            // println!("{}", self.dot());
            self.contract_algo(edge_choice)?;
        }
        Ok(())
    }
    fn contract_edge(&mut self, edge_idx: HedgeId) -> Result<(), ContractionError> {
        let a = self.graph.nodemap[edge_idx];
        let b = self.graph.nodemap[self.graph.involution[edge_idx].data];

        let ai = self.graph.nodes.get(a).unwrap();
        let bi = self.graph.nodes.get(b).unwrap();

        let f = ai.contract(bi)?;

        self.graph.merge_nodes(a, b, f);
        Ok(())
    }

    pub fn contract(&mut self) -> std::result::Result<(), ContractionError> {
        self.contract_algo(Self::edge_to_min_degree_node)
    }
}

#[cfg(feature = "shadowing")]
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Levels<
    T: HasStructure<Structure = S>,
    S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + HasName + Clone,
> {
    pub levels: Vec<TensorNetwork<ParamTensor<S>, SerializableAtom>>,
    pub initial: TensorNetwork<T, SerializableAtom>,
    // fn_map: FunctionMap<'static, Complex<T>>,
    params: Vec<SerializableAtom>,
}

#[cfg(feature = "shadowing")]
impl<T, S> From<TensorNetwork<T, SerializableAtom>> for Levels<T, S>
where
    T: HasStructure<Structure = S>,
    S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + HasName + Clone,
{
    fn from(t: TensorNetwork<T, SerializableAtom>) -> Self {
        Levels {
            initial: t,
            levels: vec![],
            // fn_map: FunctionMap::new(),
            params: vec![],
        }
    }
}

#[cfg(feature = "shadowing")]
impl<
        T: Clone + RefZero + Display,
        S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone + TracksCount + Display,
    > Levels<MixedTensor<T, S>, S>
where
    MixedTensor<T, S>: Contract<LCM = MixedTensor<T, S>> + Trace,

    S: HasName<Name: IntoSymbol, Args: IntoArgs> + ToSymbolic + StructureContract,
{
    fn contract_levels(
        &mut self,
        depth: usize,
        // fn_map: &mut FunctionMap<'a, Complex<T>>,
    ) {
        let mut not_done = true;
        let level = self.levels.len();

        if let Some(current_level) = self.levels.last_mut() {
            current_level.namesym(&format!("L{level}"))
        } else {
            not_done = false;
        }

        let nl = if let Some(current_level) = self.levels.last() {
            let mut new_level = current_level.shadow();
            new_level.contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(depth));
            if new_level.graph.nodes.len() == 1 {
                not_done = false;
            }
            Some(new_level)
        } else {
            None
        };

        if let Some(nl) = nl {
            self.levels.push(nl);
        }

        if not_done {
            self.contract_levels(depth)
        }
    }

    pub fn contract<R>(&mut self, depth: usize, fn_map: &mut FunctionMap<R>) -> ParamTensor<S>
    where
        R: From<T>,
    {
        self.initial
            .contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(depth));

        self.initial.namesym("L0");
        if self.initial.graph.nodes.len() > 1 {
            let mut new_level = self.initial.shadow();
            new_level.contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(depth));
            self.levels.push(new_level);

            self.contract_levels(depth);
            // println!("levels {}", self.levels.len());
            self.generate_fn_map(fn_map);
            self.levels.last().unwrap().result().unwrap().0
        } else {
            self.initial
                .result_tensor_ref()
                .unwrap()
                .expanded_shadow_with_map(fn_map)
                .unwrap()
        }
    }

    fn generate_fn_map<R>(&self, fn_map: &mut FunctionMap<R>)
    where
        R: From<T>,
    {
        self.initial.append_map(fn_map);
        for l in &self.levels {
            l.append_map(fn_map);
        }
    }
}

#[cfg(feature = "shadowing")]
impl<
        S: TensorStructure<Slot: Serialize + for<'a> Deserialize<'a>> + Clone + TracksCount + Display,
    > Levels<ParamTensor<S>, S>
where
    ParamTensor<S>: Contract<LCM = ParamTensor<S>>,

    S: HasName<Name = Symbol, Args: IntoArgs> + ToSymbolic + StructureContract,
{
    fn contract_levels(
        &mut self,
        depth: usize,
        // fn_map: &mut FunctionMap<'a, Complex<T>>,
    ) {
        let mut not_done = true;
        let level = self.levels.len();

        if let Some(current_level) = self.levels.last_mut() {
            current_level.namesym(&format!("L{level}"))
        } else {
            not_done = false;
        }

        let nl = if let Some(current_level) = self.levels.last() {
            let mut new_level = current_level.shadow();
            new_level.contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(depth));
            if new_level.graph.nodes.len() == 1 {
                not_done = false;
            }
            Some(new_level)
        } else {
            None
        };

        if let Some(nl) = nl {
            self.levels.push(nl);
        }

        if not_done {
            self.contract_levels(depth)
        }
    }

    pub fn contract<R>(&mut self, depth: usize, fn_map: &mut FunctionMap<R>) -> ParamTensor<S> {
        self.initial
            .contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(depth));

        self.initial.namesym("L0");
        if self.initial.graph.nodes.len() > 1 {
            let mut new_level = self.initial.shadow();
            new_level.contract_algo(|tn| tn.edge_to_min_degree_node_with_depth(depth));
            self.levels.push(new_level);

            self.contract_levels(depth);
            // println!("levels {}", self.levels.len());
            self.generate_fn_map(fn_map);
            self.levels.last().unwrap().result().unwrap().0
        } else {
            self.initial
                .result_tensor_ref()
                .unwrap()
                .expanded_shadow_with_map(fn_map)
                .unwrap()
        }
    }

    fn generate_fn_map<R>(&self, fn_map: &mut FunctionMap<R>) {
        self.initial.append_map(fn_map);
        for l in &self.levels {
            l.append_map(fn_map);
        }
    }
}

#[cfg(feature = "shadowing")]
#[cfg(test)]
mod shadowing_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod test {}

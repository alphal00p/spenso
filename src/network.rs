#[cfg(feature = "shadowing")]
use ahash::{AHashSet, HashMap};

use serde::{Deserialize, Serialize};
use slotmap::{new_key_type, DenseSlotMap, Key, SecondaryMap};

use crate::{FallibleMul, GetTensorData, HasTensorData, TensorStructure};

#[cfg(feature = "shadowing")]
use crate::{
    AtomViewOrConcrete, Complex, DataIterator, DataTensor, FallibleAdd, IntoArgs, IteratableTensor,
    MixedTensor, NamedStructure, ParamTensor, RealOrComplexTensor, Shadowable, ToSymbolic,
    TrySmallestUpgrade,
};

 use symbolica::{
    atom::{representation::FunView, AddView, AtomOrView, MulView},
    atom::{Atom, AtomView, Symbol},
    domains::float::Complex as SymComplex,
    domains::float::Real,
    domains::rational::Rational,
    evaluate::EvaluationFn,
    state::State,
};


#[cfg(feature = "shadowing")]
use ahash::AHashMap;

use super::{Contract, HasName, HasStructure, Slot, TracksCount};
use smartstring::alias::String;
use std::fmt::{Debug, Display};

#[cfg(feature = "shadowing")]
use anyhow::anyhow;

new_key_type! {
    pub struct NodeId;
    pub struct HedgeId;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalfEdgeGraph<N, E> {
    pub edges: DenseSlotMap<HedgeId, E>,
    pub involution: SecondaryMap<HedgeId, HedgeId>,
    pub neighbors: SecondaryMap<HedgeId, HedgeId>,
    pub nodes: DenseSlotMap<NodeId, N>,
    pub nodemap: SecondaryMap<HedgeId, NodeId>,
    pub reverse_nodemap: SecondaryMap<NodeId, HedgeId>,
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

    fn dot(&self) -> std::string::String
    where
        E: Display,
    {
        let mut out = "graph {\n".to_string();
        out.push_str("  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\";");

        // for (i, n) in &self.nodes {
        //     out.push_str(&format!("\n {}", i.data().as_ffi()));
        // }
        for (i, _) in &self.neighbors {
            match i.cmp(&self.involution[i]) {
                std::cmp::Ordering::Greater => {
                    out.push_str(&format!(
                        "\n {} -- {} [label=\" {} \"];",
                        self.nodemap[i].data().as_ffi(),
                        self.nodemap[self.involution[i]].data().as_ffi(),
                        self.edges[i]
                    ));
                }
                std::cmp::Ordering::Equal => {
                    out.push_str(&format!(
                        " \n ext{} [shape=none, label=\"\"];",
                        i.data().as_ffi()
                    ));
                    out.push_str(&format!(
                        "\n {} -- ext{} [label =\" {}\"];",
                        self.nodemap[i].data().as_ffi(),
                        i.data().as_ffi(),
                        self.edges[i]
                    ));
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
        let idx = self.add_node(data);
        for e in edges {
            let mut found_match = false;
            for (i, other_e) in &self.edges {
                if *e == *other_e && self.involution[i] == i {
                    found_match = true;
                    let eid = self.edges.insert(e.clone());
                    self.involution.insert(eid, i);
                    self.involution.insert(i, eid);
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
                self.involution.insert(eid, eid);
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

    #[allow(clippy::too_many_lines)]
    fn merge_nodes(&mut self, a: NodeId, b: NodeId, data: N) -> NodeId {
        let c = self.nodes.insert(data);

        // New initial edge for reverse_nodemap, that does not link to b
        // if none is found, all incident edges are link to b and must be removed from the neighbors list
        let mut new_initial_a = self
            .edges_incident(a)
            .find(|x| self.nodemap[self.involution[*x]] != b && self.involution[*x] != *x);

        if new_initial_a.is_none() {
            new_initial_a = self
                .edges_incident(a)
                .find(|x| self.nodemap[self.involution[*x]] != b);
        }

        let mut a_edge = new_initial_a;

        if a_edge.is_none() {
            // all edges link to b, and must be removed
            let initial = self.reverse_nodemap[a];
            let mut current = Some(initial);
            loop {
                if current.is_none() {
                    break;
                }
                let next = self.neighbors.remove(current.unwrap());

                if next == Some(initial) {
                    current = None;
                } else {
                    current = next;
                }
            }
        } else {
            loop {
                let mut next = self.neighbors[a_edge.unwrap()];

                while self.nodemap[self.involution[next]] == b {
                    next = self.neighbors.remove(next).unwrap();
                }

                self.nodemap.insert(a_edge.unwrap(), c);
                self.neighbors.insert(a_edge.unwrap(), next);

                if new_initial_a == Some(next) {
                    break;
                }

                a_edge = Some(next);
            }
        }

        let mut new_initial_b = self
            .edges_incident(b)
            .find(|x| self.nodemap[self.involution[*x]] != a && self.involution[*x] != *x);

        if new_initial_b.is_none() {
            new_initial_b = self
                .edges_incident(b)
                .find(|x| self.nodemap[self.involution[*x]] != a);
        }
        let mut b_edge = new_initial_b;

        if b_edge.is_none() {
            let initial = self.reverse_nodemap[b];
            let mut current = Some(initial);
            loop {
                if current.is_none() {
                    break;
                }
                let next = self.neighbors.remove(current.unwrap());

                if next == Some(initial) {
                    current = None;
                } else {
                    current = next;
                }
            }
        } else {
            loop {
                let mut next = self.neighbors[b_edge.unwrap()];

                while self.nodemap[self.involution[next]] == a {
                    next = self.neighbors.remove(next).unwrap();
                }

                self.nodemap.insert(b_edge.unwrap(), c);
                self.neighbors.insert(b_edge.unwrap(), next);

                if new_initial_b == Some(next) {
                    break;
                }

                b_edge = Some(next);
            }
        }

        match (new_initial_a, new_initial_b) {
            (Some(new_edge_a), Some(new_edge_b)) => {
                self.reverse_nodemap.insert(c, new_edge_a);
                self.reverse_nodemap.remove(a);
                self.reverse_nodemap.remove(b);
                let old_neig = self.neighbors.insert(new_edge_a, new_edge_b).unwrap();
                self.neighbors.insert(b_edge.unwrap(), old_neig);
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
        self.nodes.remove(a);
        self.nodes.remove(b);
        c
    }

    /// Add an internal edge between two nodes.
    fn add_edge(&mut self, a: NodeId, b: NodeId, data: E) -> HedgeId
    where
        E: Clone,
    {
        let hedge_id_a = self.edges.insert(data.clone());
        let hedge_id_b = self.edges.insert(data);
        self.involution.insert(hedge_id_a, hedge_id_b);
        self.involution.insert(hedge_id_b, hedge_id_a);
        self.nodemap.insert(hedge_id_a, a);
        if let Some(prev_eid) = self.reverse_nodemap.insert(a, hedge_id_a) {
            let next_eid = self.neighbors.insert(prev_eid, hedge_id_a).unwrap();
            self.neighbors.insert(hedge_id_a, next_eid).unwrap();
        } else {
            self.neighbors.insert(hedge_id_a, hedge_id_a);
        }
        self.nodemap.insert(hedge_id_b, b);
        if let Some(prev_eid) = self.reverse_nodemap.insert(b, hedge_id_b) {
            let next_eid = self.neighbors.insert(prev_eid, hedge_id_b).unwrap();
            self.neighbors.insert(hedge_id_b, next_eid).unwrap();
        } else {
            self.neighbors.insert(hedge_id_b, hedge_id_b);
        }
        hedge_id_a
    }

    /// Add external, as a fixed point involution half edge.
    fn add_external(&mut self, a: NodeId, data: E) -> HedgeId {
        let id = self.edges.insert(data);
        self.involution.insert(id, id);
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
            .filter(move |&i| self.nodemap[self.involution[i]] == b)
    }

    fn internal_edges_incident(&self, node: NodeId) -> impl Iterator<Item = HedgeId> + '_ {
        self.edges_incident(node)
            .filter(move |&i| self.nodemap[self.involution[i]] != node)
    }

    fn external_edges_incident(&self, node: NodeId) -> impl Iterator<Item = HedgeId> + '_ {
        self.edges_incident(node)
            .filter(move |&i| self.nodemap[self.involution[i]] == node)
    }

    fn degree(&self, node: NodeId) -> usize {
        self.edges_incident(node).collect::<Vec<_>>().len()
    }

    fn neighbors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.edges_incident(node)
            .map(move |i| self.nodemap[self.involution[i]])
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
    let a = graph.add_node_with_edges(1, &[1, 2, 3, 4, 5]);
    let b = graph.add_node_with_edges(2, &[1, 2, 6, 7, 8]);
    let c = graph.add_node_with_edges(4, &[4, 6, 9, 10, 11]);

    println!("{}", graph.dot());
    println!("{}", graph.degree(a));
    println!("{}", graph.degree(b));

    for (i, n) in &graph.neighbors {
        println!("{} {}", graph.edges[i], graph.edges[*n]);
    }

    let d = graph.merge_nodes(a, b, 3);

    // for (i, n) in &graph.neighbors {
    //     println!("{} {}", graph.edges[i], graph.edges[*n]);
    // }

    println!("{}", graph.dot());
    println!("{}", graph.degree(c));
    println!("{}", graph.neighbors.len());

    let e = graph.merge_nodes(c, d, 5);

    println!("{}", graph.dot());
    println!("{}", graph.degree(e));
    println!("{}", graph.neighbors.len());

    let mut graph = HalfEdgeGraph::new();
    let a = graph.add_node_with_edges("a", &[10, 2, 3]);
    let b = graph.add_node_with_edges("b", &[20, 3, 4]);
    let c = graph.add_node_with_edges("c", &[30, 4, 2]);
    let d = graph.add_node_with_edges("d", &[20]);
    let e = graph.add_node_with_edges("e", &[30]);

    println!("Test {}", graph.dot());
    println!("{}", graph.degree(a));
    println!("{}", graph.degree(b));

    for (i, n) in &graph.neighbors {
        println!("{} {}", graph.edges[i], graph.edges[*n]);
    }

    let d = graph.merge_nodes(d, b, "bd");

    // for (i, n) in &graph.neighbors {
    //     println!("{} {}", graph.edges[i], graph.edges[*n]);
    // }

    println!("{}", graph.degree(c));
    println!("{}", graph.neighbors.len());

    println!("{}", graph.dot());

    let e = graph.merge_nodes(c, e, "ce");

    if graph.validate_neighbors() {
        println!("valid");
    } else {
        println!("invalid");
    }

    println!("{}", graph.dot());
    let f = graph.merge_nodes(d, e, "de");

    if graph.validate_neighbors() {
        println!("valid");
    } else {
        println!("invalid");
    }

    println!("{}", graph.dot());
    println!("{}", graph.node_labels());
    println!("{}", graph.degree(a));
    println!("{}", graph.neighbors.len());

    let g = graph.merge_nodes(a, f, "af");

    if graph.validate_neighbors() {
        println!("valid");
    } else {
        println!("invalid");
    }

    println!("{}", graph.dot());
    println!("{}", graph.neighbors.len());
    println!("{}", graph.degree(g));

    // println!("{}", graph.degree(b));
}

#[derive(Debug, Clone)]
pub struct TensorNetwork<T, S> {
    pub graph: HalfEdgeGraph<T, Slot>,
    // pub params: AHashSet<Atom>,
    pub scalar: Option<S>,
}

impl<T, S> TensorNetwork<T, S> {
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

    fn edge_to_min_degree_node(&self) -> Option<HedgeId> {
        let mut neighs = self.graph.reverse_nodemap.clone();
        if neighs.is_empty() {
            return None;
        }

        loop {
            let mut all_ext = true;
            for (node, initial) in &mut neighs {
                *initial = self.graph.neighbors[*initial];
                let start = self.graph.reverse_nodemap[node];

                if self.graph.involution[start] != start {
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

    pub fn to_vec(&self) -> Vec<&T> {
        self.graph.nodes.values().collect()
    }
}

#[cfg(feature = "shadowing")]
impl<T, S> TensorNetwork<MixedTensor<T, S>, Atom>
where
    S: Clone + TensorStructure + Debug,
    T: Clone,
{
    #[cfg(feature = "shadowing")]
    pub fn to_symbolic_tensor_vec(mut self) -> Vec<ParamTensor<S>> {
        self.graph
            .nodes
            .drain()
            .into_iter()
            .map(|(_, n)| n.try_into_parametric()) //filters out all parametric tensors
            .flatten()
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
    pub fn evaluate_real<'a, F: Fn(&Rational) -> T + Copy>(
        &'a mut self,
        coeff_map: F,
        const_map: &AHashMap<AtomView<'a>, T>,
    ) where
        T: Real + for<'c> From<&'c Rational>,
    {
        for (_, n) in &mut self.graph.nodes {
            n.evaluate_real(coeff_map, const_map);
        }
    }

    #[cfg(feature = "shadowing")]
    pub fn evaluate_complex<'a, F: Fn(&Rational) -> SymComplex<T> + Copy>(
        &'a mut self,
        coeff_map: F,
        const_map: &AHashMap<AtomView<'a>, SymComplex<T>>,
    ) where
        T: Real + for<'c> From<&'c Rational>,
        SymComplex<T>: Real + for<'c> From<&'c Rational>,
    {
        for (_, n) in &mut self.graph.nodes {
            n.evaluate_complex(coeff_map, const_map);
        }
    }

    pub fn to_fully_parametric(self) -> TensorNetwork<ParamTensor<S>, Atom>
    where
        T: TrySmallestUpgrade<Atom, LCM = Atom>,
        Complex<T>: TrySmallestUpgrade<Atom, LCM = Atom>,
    {
        let mut tensors = vec![];

        for n in self.graph.nodes.values() {
            tensors.push(match n {
                MixedTensor::Concrete(RealOrComplexTensor::Real(t)) => {
                    ParamTensor::Composite(t.try_upgrade::<Atom>().unwrap().into_owned())
                }
                MixedTensor::Concrete(RealOrComplexTensor::Complex(t)) => {
                    ParamTensor::Composite(t.try_upgrade::<Atom>().unwrap().into_owned())
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
impl<S: TensorStructure + Clone> TensorNetwork<ParamTensor<S>, Atom> {
    pub fn evaluate<'a, D, F: Fn(&Rational) -> D + Copy>(
        &'a self,
        coeff_map: F,
        const_map: &AHashMap<AtomView<'a>, D>,
    ) -> TensorNetwork<DataTensor<D, S>, D>
    where
        D: Clone
            + symbolica::domains::float::Real
            + for<'c> std::convert::From<&'c symbolica::domains::rational::Rational>,
    {
        let mut evaluated_net = TensorNetwork::new();
        for (_, t) in &self.graph.nodes {
            let evaluated_tensor = t.evaluate(coeff_map, const_map);
            evaluated_net.push(evaluated_tensor);
        }
        let fn_map: HashMap<_, EvaluationFn<_>> = HashMap::default();
        let mut cache = HashMap::default();

        evaluated_net.scalar = self
            .scalar
            .as_ref()
            .map(|x| x.evaluate(coeff_map, const_map, &fn_map, &mut cache));
        // println!("{:?}", evaluated_net.scalar);
        evaluated_net
    }
}
impl<T> From<Vec<T>> for TensorNetwork<T, T::Scalar>
where
    T: HasStructure,
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
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, S> TensorNetwork<T, S>
where
    T: HasStructure,
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
        self.graph.add_node_with_edges(tensor, &slots)
    }

    fn generate_network_graph(tensors: Vec<T>) -> HalfEdgeGraph<T, Slot> {
        let mut graph = HalfEdgeGraph::<T, Slot>::new();

        for tensor in tensors {
            let slots = tensor.external_structure().to_vec();
            graph.add_node_with_edges(tensor, &slots);
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

                if self.graph.involution[start] != start
                    && self.graph.nodes[self.graph.nodemap[self.graph.involution[start]]]
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
impl<T: HasTensorData + GetTensorData<GetData = T::Data>> TensorNetwork<T, T::Data>
where
    T: Clone,
{
    pub fn result(&self) -> Result<T::Data, TensorNetworkError> {
        match self.graph.nodes.len() {
            0 => self
                .scalar
                .clone()
                .ok_or(TensorNetworkError::ScalarFieldEmpty),
            1 => {
                let t = self.result_tensor()?;
                if t.is_scalar() {
                    Ok(t.get_linear(0.into()).unwrap().clone())
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
    #[error("Cannot contract edge")]
    CannotContractEdge,
    #[error("no nodes in the graph")]
    NoNodes,
    #[error("more than one node in the graph")]
    MoreThanOneNode,
    #[error("is not scalar output")]
    NotScalarOutput,
    #[error("scalar field is empty")]
    ScalarFieldEmpty,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl<T, S> TensorNetwork<T, S>
where
    T: Clone,
{
    pub fn result_tensor(&self) -> Result<T, TensorNetworkError> {
        match self.graph.nodes.len() {
            0 => Err(TensorNetworkError::NoNodes),
            1 => Ok(self.graph.nodes.iter().next().unwrap().1.clone()),
            _ => Err(TensorNetworkError::MoreThanOneNode),
        }
    }
}

impl<T, S> TensorNetwork<T, S> {
    pub fn dot(&self) -> std::string::String {
        self.graph.dot()
    }
}

#[cfg(feature = "shadowing")]
impl<T, S> TensorNetwork<MixedTensor<T, S>, Atom>
where
    S: TensorStructure + Clone,
    T: Clone,
{
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

#[cfg(feature = "shadowing")]
impl<'a> TryFrom<MulView<'a>>
    for TensorNetwork<MixedTensor<f64, NamedStructure<Symbol, Vec<Atom>>>, Atom>
{
    type Error = TensorNetworkError;
    fn try_from(value: MulView<'a>) -> Result<Self, Self::Error> {
        let mut network: Self = TensorNetwork::new();
        let mut scalars = Atom::new_num(1);
        for arg in value.iter() {
            let mut net = Self::try_from(arg)?;
            net.contract();
            if let Some(ref s) = net.scalar {
                scalars = &scalars * s;
            }
            match net.result_tensor() {
                Ok(t) => {
                    network.push(t);
                }
                Err(TensorNetworkError::NoNodes) => {}
                Err(e) => return Err(e),
            }
        }
        network.scalar_mul(scalars);
        Ok(network)
    }
}

#[cfg(feature = "shadowing")]
impl<'a> TryFrom<AtomView<'a>>
    for TensorNetwork<MixedTensor<f64, NamedStructure<Symbol, Vec<Atom>>>, Atom>
{
    type Error = TensorNetworkError;
    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        match value {
            AtomView::Mul(m) => m.try_into(),
            AtomView::Fun(f) => f.try_into(),
            AtomView::Add(a) => a.try_into(),
            a => {
                let mut network: Self = TensorNetwork::new();
                let a = a.to_owned();
                network.scalar = Some(a);
                Ok(network)
            }
        }
    }
}

#[cfg(feature = "shadowing")]
impl<'a> TryFrom<FunView<'a>>
    for TensorNetwork<MixedTensor<f64, NamedStructure<Symbol, Vec<Atom>>>, Atom>
{
    type Error = TensorNetworkError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        let mut network: Self = TensorNetwork::new();
        let s: Result<NamedStructure<_, _>, _> = value.try_into();

        let mut scalar = None;
        if let Ok(s) = s {
            let t = s
                .to_shell()
                .smart_shadow()
                .ok_or(anyhow!("Cannot shadow"))?;
            network.push(t);
        } else {
            scalar = Some(value.as_view().to_owned());
        }
        network.scalar = scalar;
        Ok(network)
    }
}

#[cfg(feature = "shadowing")]
impl<'a> TryFrom<AddView<'a>>
    for TensorNetwork<MixedTensor<f64, NamedStructure<Symbol, Vec<Atom>>>, Atom>
{
    type Error = TensorNetworkError;
    fn try_from(value: AddView<'a>) -> Result<Self, Self::Error> {
        let mut tensors = vec![];
        let mut scalars = Atom::new_num(0);
        for summand in value.iter() {
            let mut net = Self::try_from(summand)?;
            net.contract();
            if let Some(ref s) = net.scalar {
                scalars = &scalars + s;
            }
            match net.result_tensor() {
                Ok(t) => {
                    tensors.push(t);
                }
                Err(TensorNetworkError::NoNodes) => {}
                Err(e) => return Err(e),
            }
        }
        let mut net = TensorNetwork::from(vec![tensors
            .into_iter()
            .reduce(|a, b| a.add_fallible(&b).unwrap())
            .unwrap()]);
        net.scalar = Some(scalars);
        Ok(net)
    }
}

#[cfg(feature = "shadowing")]
impl<T, S, I> TensorNetwork<T, S>
where
    T: HasStructure<Structure = I> + Clone + HasName<Name = Symbol>,
    I: TensorStructure + Clone + ToSymbolic,
    T::Args: IntoArgs,
{
    pub fn symbolic_shadow(&mut self, name: &str) -> TensorNetwork<MixedTensor<f64, I>, S> {
        {
            for (i, n) in &mut self.graph.nodes {
                n.set_name(State::get_symbol(format!("{}{}", name, i.data().as_ffi())));
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
            let node = n.shadow().unwrap();

            let nid = nodes.insert(MixedTensor::<f64, I>::param(node.clone().into()));

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

impl<T, S> TensorNetwork<T, S>
where
    T: HasName,
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
    T: HasName<Name = Symbol>,
{
    pub fn namesym(&mut self, name: &str) {
        for (id, n) in &mut self.graph.nodes {
            n.set_name(State::get_symbol(format!("{}{}", name, id.data().as_ffi())));
        }
    }
}

impl<T, S> TensorNetwork<T, S>
where
    T: Contract<T, LCM = T> + HasStructure,
{
    pub fn contract_algo(&mut self, edge_choice: fn(&Self) -> Option<HedgeId>) {
        if let Some(e) = edge_choice(self) {
            self.contract_edge(e);
            // println!("{}", self.dot());
            self.contract_algo(edge_choice);
        }
    }
    fn contract_edge(&mut self, edge_idx: HedgeId) {
        let a = self.graph.nodemap[edge_idx];
        let b = self.graph.nodemap[self.graph.involution[edge_idx]];

        let ai = self.graph.nodes.get(a).unwrap();
        let bi = self.graph.nodes.get(b).unwrap();

        let f = ai.contract(bi).unwrap();

        self.graph.merge_nodes(a, b, f);
    }

    pub fn contract(&mut self) {
        self.contract_algo(Self::edge_to_min_degree_node)
    }
}

#[cfg(feature = "shadowing")]
#[allow(dead_code)]
pub struct Levels<T, S: TensorStructure> {
    pub levels: Vec<TensorNetwork<DataTensor<Atom, S>, Atom>>,
    const_map: AHashMap<AtomOrView<'static>, T>,
    params: Vec<Atom>,
}

#[cfg(feature = "shadowing")]
impl<T, S> From<TensorNetwork<DataTensor<Atom, S>, Atom>> for Levels<T, S>
where
    T: Clone,
    S: TensorStructure,
{
    fn from(t: TensorNetwork<DataTensor<Atom, S>, Atom>) -> Self {
        Levels {
            levels: vec![t],
            const_map: AHashMap::new(),
            params: vec![],
        }
    }
}

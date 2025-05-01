use std::{
    fmt::Display,
    ops::{Add, Mul, Neg},
};

use bincode::{Decode, Encode};
use bitvec::vec::BitVec;
use linnet::{
    half_edge::{
        builder::HedgeGraphBuilder,
        involution::{EdgeData, Flow, Hedge},
        subgraph::{ModifySubgraph, SubGraph},
        tree::SimpleTraversalTree,
        HedgeGraph, HedgeGraphError, NodeIndex,
    },
    tree::{child_pointer::ParentChildStore, child_vec::ChildVecStore, Forest},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::structure::{
    abstract_index::AbstractIndex,
    representation::{LibrarySlot, RepName},
    slot::{DualSlotTo, IsAbstractSlot},
    TensorStructure,
};

use super::TensorNetworkError;

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
#[cfg_attr(
    feature = "shadowing",
    bincode(decode_context = "symbolica::state::StateMap")
)]
pub struct NetworkGraph<K> {
    pub graph: HedgeGraph<NetworkEdge, NetworkNode<K>, Forest<NetworkNode<K>, ChildVecStore<()>>>,
    #[bincode(with_serde)]
    uncontracted: BitVec,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Encode, Decode)]
#[cfg_attr(
    feature = "shadowing",
    bincode(decode_context = "symbolica::state::StateMap")
)]
pub enum NetworkEdge {
    // Port,
    Head,
    Slot(LibrarySlot),
}

impl NetworkEdge {
    pub fn is_head(&self) -> bool {
        matches!(self, NetworkEdge::Head)
    }

    pub fn is_slot(&self) -> bool {
        matches!(self, NetworkEdge::Slot(_))
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize)]
pub enum NetworkNode<LibKey> {
    Leaf(NetworkLeaf<LibKey>),
    Op(NetworkOp),
    // Port,
}

#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize)]
pub enum NetworkOp {
    Sum,
    Neg,
    Product,
}

#[derive(Debug, Clone, Copy, Encode, Decode, Serialize, Deserialize)]
pub enum NetworkLeaf<K> {
    LocalTensor(usize),
    LibraryKey(K),
    Scalar(usize),
}

#[derive(Debug, Clone)]
pub enum NetworkLeafWithInds<K> {
    LocalTensor(usize),
    LibraryKey { key: K, inds: Vec<AbstractIndex> },
    Scalar(usize),
}

impl<K> From<HedgeGraphBuilder<NetworkEdge, NetworkNode<K>>> for NetworkGraph<K> {
    fn from(builder: HedgeGraphBuilder<NetworkEdge, NetworkNode<K>>) -> Self {
        let graph = builder.build();
        let uncontracted = graph.empty_subgraph();
        Self {
            graph,
            uncontracted,
        }
    }
}
#[derive(Clone, Debug, Error)]
pub enum NetworkGraphError {
    #[error("Not head node")]
    NotHeadNode,
}

impl<K> NetworkGraph<K> {
    pub fn splice_descendents_of(&mut self, replacement: Self)
    where
        K: Clone,
    {
        // let tt: SimpleTraversalTree<ParentChildStore<()>> = self.expr_tree().cast();

        // let descendent_subgraph = tt.iter_preorder_tree_nodes(&self.graph, node_id).fold(
        //     self.graph.empty_subgraph::<BitVec>(),
        //     |mut a, b| {
        //         for h in self.graph.hairs_from_id(b).hairs.included_iter() {
        //             a.add(h);
        //         }
        //         a
        //     },
        // );

        // replacement.shift_scalars(self.);

        self.graph
            .join_mut(
                replacement.graph,
                |sf, sd, of, od| sf == -of && sd == od,
                |sf, sd, _, _| (sf, sd),
            )
            .unwrap();

        // self
    }

    pub fn find_all_ready_ops(&mut self) -> Vec<(Self, NetworkOp, Vec<NetworkLeaf<K>>)>
    where
        K: Clone,
    {
        let tt: SimpleTraversalTree<ParentChildStore<()>> = self.expr_tree().cast();
        let head = self.head();
        let mut out = vec![];
        let root_node = self.graph.node_id(head);
        let involution = self.graph.as_ref().clone();

        for nid in tt.iter_preorder_tree_nodes(&involution, root_node) {
            if let NetworkNode::Op(op) = &self.graph[nid] {
                let mut leaves = Vec::new();
                let mut subgraph: BitVec = self.graph.empty_subgraph();
                let ok = tt.iter_children(nid, &self.graph).all(|child| {
                    for h in self.graph.iter_crown(child) {
                        subgraph.add(h);
                    }
                    if let NetworkNode::Leaf(l) = &self.graph[child] {
                        leaves.push(l.clone());
                        true
                    } else {
                        false
                    }
                });
                if ok {
                    let op = *op;
                    let extracted = self.graph.extract(
                        &subgraph,
                        |a| a.map(Clone::clone),
                        |a| a,
                        |a| a.clone(),
                        |a| a,
                    );
                    out.push((
                        Self {
                            uncontracted: extracted.empty_subgraph(),
                            graph: extracted,
                        },
                        op,
                        leaves,
                    ))
                }
            }
        }
        out
    }
    pub fn extract_next_ready_op(
        &mut self,
    ) -> Option<(Self, NetworkOp, Vec<NetworkLeafWithInds<K>>)>
    where
        K: Clone,
    {
        // build a traversal over *all* internal edges
        let tt: SimpleTraversalTree<ParentChildStore<()>> = self.expr_tree().cast();
        let head = self.head();
        let root_node = self.graph.node_id(head);
        let mut subgraph: BitVec = self.graph.empty_subgraph();

        // look for the first op node whose children are all leaves
        for nid in tt.iter_preorder_tree_nodes(&self.graph, root_node) {
            if let NetworkNode::Op(op) = &self.graph[nid] {
                let mut leaves = Vec::new();
                let mut all_leaves = true;
                for child in tt.iter_children(nid, &self.graph) {
                    match &self.graph[child] {
                        NetworkNode::Leaf(a) => {
                            let mut slots = vec![];

                            for h in self.graph.iter_crown(child) {
                                if let NetworkEdge::Slot(s) = self.graph[[&h]] {
                                    slots.push(s.aind)
                                }
                            }

                            match a {
                                NetworkLeaf::LibraryKey(l) => {
                                    leaves.push(NetworkLeafWithInds::LibraryKey {
                                        key: l.clone(),
                                        inds: slots,
                                    });
                                }
                                NetworkLeaf::Scalar(a) => {
                                    leaves.push(NetworkLeafWithInds::Scalar(*a));
                                }
                                NetworkLeaf::LocalTensor(t) => {
                                    leaves.push(NetworkLeafWithInds::LocalTensor(*t));
                                }
                            }
                        }
                        _ => {
                            all_leaves = false;
                            break;
                        }
                    }
                    for h in self.graph.iter_crown(child) {
                        subgraph.add(h);
                    }
                }
                if all_leaves {
                    let op = *op;
                    let extracted = self.graph.extract(
                        &subgraph,
                        |a| a.map(Clone::clone),
                        |a| a,
                        |a| a.clone(),
                        |a| a,
                    );

                    return Some((
                        Self {
                            uncontracted: extracted.empty_subgraph(),
                            graph: extracted,
                        },
                        op,
                        leaves,
                    ));
                }
            }
        }
        None
    }

    pub fn sub_expression(&self, nid: NodeIndex) -> Result<SimpleTraversalTree, NetworkGraphError> {
        let include_hedge = self
            .graph
            .iter_crown(nid)
            .find(|i| !matches!(self.graph[[i]], NetworkEdge::Slot(_)));

        let headgraph: BitVec = self.graph.from_filter(|a| matches!(a, NetworkEdge::Head));

        if include_hedge.is_some() {
            Ok(SimpleTraversalTree::depth_first_traverse(
                &self.graph,
                &headgraph,
                &nid,
                include_hedge,
            )
            .unwrap())
        } else {
            Err(NetworkGraphError::NotHeadNode)
        }
    }
    pub fn one() -> Self {
        Self::mul(0)
    }

    pub fn shift_scalars(&mut self, shift: usize) {
        self.graph.iter_nodes_mut().for_each(|(_, _, d)| {
            if let NetworkNode::Leaf(NetworkLeaf::Scalar(s)) = d {
                *s += shift;
            }
        });
    }

    pub fn shift_tensors(&mut self, shift: usize) {
        self.graph.iter_nodes_mut().for_each(|(_, _, d)| {
            if let NetworkNode::Leaf(NetworkLeaf::LocalTensor(s)) = d {
                *s += shift;
            }
        });
    }

    pub fn dot(&self) -> String {
        self.graph.base_dot()
    }

    pub fn zero() -> Self {
        Self::add(0, &[])
    }

    fn head_builder(
        node: NetworkNode<K>,
    ) -> (HedgeGraphBuilder<NetworkEdge, NetworkNode<K>>, NodeIndex) {
        let mut graph = HedgeGraphBuilder::new();
        let head = graph.add_node(node);
        graph.add_external_edge(head, NetworkEdge::Head, false, Flow::Source);
        (graph, head)
    }

    pub fn neg() -> Self {
        let (mut graph, head) = Self::head_builder(NetworkNode::Op(NetworkOp::Neg));
        graph.add_external_edge(head, NetworkEdge::Head, false, Flow::Sink);
        graph.into()
    }

    pub fn mul(n: usize) -> Self {
        let (mut graph, head) = Self::head_builder(NetworkNode::Op(NetworkOp::Product));
        for _ in 0..n {
            graph.add_external_edge(head, NetworkEdge::Head, false, Flow::Sink);
        }

        graph.into()
    }

    pub fn add(n: usize, slots: &[LibrarySlot]) -> Self {
        let (mut graph, mut head) = Self::head_builder(NetworkNode::Op(NetworkOp::Sum));
        for _ in 0..n {
            graph.add_external_edge(head, NetworkEdge::Head, false, Flow::Sink);
        }

        for s in slots {
            let orientation = s.rep_name().orientation();
            graph.add_external_edge(
                head,
                NetworkEdge::Slot(s.clone()),
                orientation,
                Flow::Source,
            );
            for _ in 0..n {
                graph.add_external_edge(
                    head,
                    NetworkEdge::Slot(s.clone()),
                    orientation,
                    Flow::Sink,
                );
            }
        }

        graph.into()
    }

    pub fn scalar(pos: usize) -> Self {
        let mut graph = HedgeGraphBuilder::new();

        let head = graph.add_node(NetworkNode::Leaf(NetworkLeaf::Scalar(pos)));
        graph.add_external_edge(head, NetworkEdge::Head, false, Flow::Source);

        graph.into()
    }

    pub fn key(key: K) -> Self
    where
        K: TensorStructure,
    {
        let slots = key
            .external_structure_iter()
            .map(|a| a.to_lib())
            .collect::<Vec<_>>();

        let (mut graph, head) = Self::head_builder(NetworkNode::Leaf(NetworkLeaf::LibraryKey(key)));

        for lib in slots {
            let orientation = lib.rep_name().orientation();
            graph.add_external_edge(head, NetworkEdge::Slot(lib), orientation, Flow::Source);
        }
        graph.into()
    }

    pub fn tensor<T: TensorStructure>(tensor: &T, node: NetworkLeaf<K>) -> NetworkGraph<K> {
        let (mut graph, mut head) = Self::head_builder(NetworkNode::Leaf(node));

        for s in tensor.external_structure_iter() {
            let lib = s.to_lib();

            let orientation = lib.rep_name().orientation();
            graph.add_external_edge(head, NetworkEdge::Slot(lib), orientation, Flow::Source);
        }
        graph.into()
    }

    fn match_heads(
        self_flow: Flow,
        self_data: EdgeData<&NetworkEdge>,
        other_flow: Flow,
        other_data: EdgeData<&NetworkEdge>,
    ) -> bool {
        if let (NetworkEdge::Head, NetworkEdge::Head) = (self_data.data, other_data.data) {
            self_flow == -other_flow
        } else {
            false
        }
    }

    fn match_indices(
        _sf: Flow,
        self_data: EdgeData<&NetworkEdge>,
        _of: Flow,
        other_data: EdgeData<&NetworkEdge>,
    ) -> bool {
        if let (NetworkEdge::Slot(s), NetworkEdge::Slot(o)) = (self_data.data, other_data.data) {
            s.matches(o)
        } else {
            false
        }
    }

    fn prod_match(
        self_flow: Flow,
        self_data: EdgeData<&NetworkEdge>,
        other_flow: Flow,
        other_data: EdgeData<&NetworkEdge>,
    ) -> bool {
        match (self_data.data, other_data.data) {
            (NetworkEdge::Head, NetworkEdge::Head) => self_flow == -other_flow,
            (NetworkEdge::Slot(s), NetworkEdge::Slot(o)) => s.matches(o),
            _ => false,
        }
    }

    fn add_match(
        sf: Flow,
        sd: EdgeData<&NetworkEdge>,
        of: Flow,
        od: EdgeData<&NetworkEdge>,
    ) -> bool {
        match (sd.data, od.data) {
            (NetworkEdge::Head, NetworkEdge::Head) => sf == -of,
            (NetworkEdge::Slot(s), NetworkEdge::Slot(o)) => s == o && sf == -of,
            _ => false,
        }
    }

    pub fn head(&self) -> Hedge {
        let exts = self.graph.external_filter();
        let head = exts
            .included_iter()
            .find(|i| self.graph[[i]] == NetworkEdge::Head);
        head.unwrap()
    }

    pub fn dangling_indices(&self) -> Vec<LibrarySlot> {
        let exts = self.graph.external_filter();
        exts.included_iter()
            .filter_map(|i| {
                if let NetworkEdge::Slot(s) = self.graph[[&i]] {
                    Some(s)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn n_dangling(&self) -> usize {
        self.graph
            .external_filter()
            .included_iter()
            .filter(|i| matches!(self.graph[[i]], NetworkEdge::Slot(_)))
            .count()
    }

    pub fn n_nodes(&self) -> usize {
        self.graph.n_nodes()
    }

    pub fn result(&self) -> Result<(&NetworkNode<K>, Vec<LibrarySlot>), TensorNetworkError<K>>
    where
        K: Display,
    {
        let mut n_heads = 0;
        let headgraph: BitVec = self.graph.from_filter(|a| match a {
            NetworkEdge::Head => {
                n_heads += 1;
                true
            }
            // NetworkEdge::Port => true,
            _ => false,
        });

        match n_heads {
            0 => Err(TensorNetworkError::NoNodes),
            1 => {
                let head = self.head();
                let root_node = self.graph.node_id(head);

                let mut slots = vec![];
                for h in self.graph.iter_crown(root_node) {
                    if let NetworkEdge::Slot(s) = self.graph[[&h]] {
                        slots.push(s);
                    }
                }

                Ok((&self.graph[root_node], slots))
            }
            _ => Err(TensorNetworkError::MoreThanOneNode),
        }
    }

    fn join_heads(
        self_flow: Flow,
        self_data: EdgeData<NetworkEdge>,
        _other_flow: Flow,
        _other_data: EdgeData<NetworkEdge>,
    ) -> (Flow, EdgeData<NetworkEdge>) {
        (self_flow, self_data)
    }

    pub fn expr_tree(&self) -> SimpleTraversalTree {
        let headgraph: BitVec = self
            .graph
            .from_filter(|a| !matches!(a, NetworkEdge::Slot(_)));

        let head = self.head();
        let root_node = self.graph.node_id(head);
        SimpleTraversalTree::depth_first_traverse(&self.graph, &headgraph, &root_node, Some(head))
            .unwrap()
    }

    fn join_mut(
        &mut self,
        other: Self,
        matching_fn: impl Fn(Flow, EdgeData<&NetworkEdge>, Flow, EdgeData<&NetworkEdge>) -> bool,
        merge_fn: impl Fn(
            Flow,
            EdgeData<NetworkEdge>,
            Flow,
            EdgeData<NetworkEdge>,
        ) -> (Flow, EdgeData<NetworkEdge>),
    ) -> Result<(), HedgeGraphError> {
        self.graph.join_mut(other.graph, matching_fn, merge_fn)?;
        self.uncontracted.join_mut(other.uncontracted);
        Ok(())
    }
}

pub trait NMul<Rhs = Self> {
    type Output;

    fn n_mul<I: IntoIterator<Item = Rhs>>(self, iter: I) -> Self::Output;
}

impl<K> NMul for NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn n_mul<I: IntoIterator<Item = Self>>(self, iter: I) -> Self::Output {
        let all = iter.into_iter().collect::<Vec<_>>();
        let mut mul = Self::mul(all.len() + 1);

        mul.join_mut(
            self,
            NetworkGraph::<K>::match_heads,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        for rhs in all {
            mul.join_mut(
                rhs,
                NetworkGraph::<K>::prod_match,
                NetworkGraph::<K>::join_heads,
            )
            .unwrap();
        }

        mul
    }
}
// impl<S: NMul<R>, R> Mul<R> for S {grrr
//     type Output = S::Output;
//     fn mul(self, rhs: R) -> Self::Output {
//         self.n_mul([rhs])
//     }
// }

pub trait NAdd<Rhs = Self> {
    type Output;

    fn n_add<I: IntoIterator<Item = Rhs>>(self, iter: I) -> Self::Output;
}

impl<K> NAdd for NetworkGraph<K> {
    type Output = NetworkGraph<K>;

    fn n_add<I: IntoIterator<Item = Self>>(self, iter: I) -> Self::Output {
        let all = iter.into_iter().collect::<Vec<_>>();
        let slots = self.dangling_indices();

        let mut add = Self::add(all.len() + 1, &slots);

        add.join_mut(
            self,
            NetworkGraph::<K>::add_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        for rhs in all {
            debug_assert!(slots.len() == rhs.n_dangling());

            add.join_mut(
                rhs,
                NetworkGraph::<K>::add_match,
                NetworkGraph::<K>::join_heads,
            )
            .unwrap();
        }
        debug_assert!(slots.len() == add.n_dangling());

        add
    }
}

impl<K> Mul for NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut mul = Self::mul(2);

        mul.join_mut(
            self,
            NetworkGraph::<K>::match_heads,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        mul.join_mut(
            rhs,
            NetworkGraph::<K>::prod_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        mul
    }
}

impl<K> Add for NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn add(self, rhs: Self) -> Self::Output {
        let slots = self.dangling_indices();
        debug_assert!(slots.len() == rhs.n_dangling());

        let mut add = Self::add(2, &slots);

        add.join_mut(
            self,
            NetworkGraph::<K>::add_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        add.join_mut(
            rhs,
            NetworkGraph::<K>::add_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        debug_assert!(slots.len() == add.n_dangling());

        add
    }
}

impl<K> Neg for NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn neg(self) -> Self::Output {
        let mut neg = Self::neg();

        neg.join_mut(
            self,
            NetworkGraph::<K>::match_heads,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        neg
    }
}

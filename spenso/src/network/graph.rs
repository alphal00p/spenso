use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bincode::{Decode, Encode};
use bitvec::vec::BitVec;
use linnet::{
    half_edge::{
        builder::HedgeGraphBuilder,
        involution::{EdgeData, Flow, Hedge},
        nodestore::NodeStorageOps,
        subgraph::{ModifySubgraph, SubGraph, SubGraphOps},
        tree::{SimpleTraversalTree, TTRoot},
        HedgeGraph, HedgeGraphError, NodeIndex,
    },
    permutation::Permutation,
    tree::{child_pointer::ParentChildStore, child_vec::ChildVecStore},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::structure::{
    abstract_index::AbstractIndex,
    permuted::{Perm, PermuteTensor},
    representation::{LibrarySlot, RepName},
    slot::{DualSlotTo, IsAbstractSlot},
    PermutedStructure, TensorStructure,
};

use super::{
    library::{Library, LibraryTensor},
    TensorNetworkError,
};

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    bincode_trait_derive::Encode,
    bincode_trait_derive::Decode,
    // bincode_trait_derive::BorrowDecodeFromDecode,
)]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
pub struct NetworkGraph<K> {
    pub graph: HedgeGraph<NetworkEdge, NetworkNode<K>>, //, Forest<NetworkNode<K>, ChildVecStore<()>>>,
    pub slot_order: Vec<u8>,
    // #[bincode(with_serde)]
    // uncontracted: BitVec,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    Encode,
    bincode_trait_derive::Decode,
    // bincode_trait_derive::BorrowDecodeFromDecode,
)]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
pub enum NetworkEdge {
    // Port,
    Head,
    Slot(LibrarySlot),
}

impl Display for NetworkEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkEdge::Head => write!(f, "Head"),
            NetworkEdge::Slot(slot) => write!(f, "Slot({})", slot),
        }
    }
}

impl NetworkEdge {
    pub fn is_head(&self) -> bool {
        matches!(self, NetworkEdge::Head)
    }

    pub fn is_slot(&self) -> bool {
        matches!(self, NetworkEdge::Slot(_))
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Encode, bincode_trait_derive::Decode, Serialize, Deserialize,
)]
pub enum NetworkNode<LibKey> {
    Leaf(NetworkLeaf<LibKey>),
    Op(NetworkOp),
    // Port,
}

impl<K> NetworkNode<K> {
    pub fn is_leaf(&self) -> bool {
        matches!(self, NetworkNode::Leaf(_))
    }

    pub fn is_op(&self) -> bool {
        matches!(self, NetworkNode::Op(_))
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, NetworkNode::Leaf(NetworkLeaf::Scalar(_)))
    }

    pub fn is_tensor(&self) -> bool {
        self.is_leaf() && !self.is_scalar()
    }
}

impl<K: Display> Display for NetworkNode<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkNode::Leaf(l) => write!(f, "{}", l),
            NetworkNode::Op(o) => write!(f, "{o}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Serialize, Deserialize)]
pub enum NetworkOp {
    Sum,
    Neg,
    Product,
}

impl Display for NetworkOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkOp::Neg => write!(f, "⊖"),
            NetworkOp::Product => write!(f, "∏"),
            NetworkOp::Sum => write!(f, "∑"),
        }
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Encode, bincode_trait_derive::Decode, Serialize, Deserialize,
)]
pub enum NetworkLeaf<K> {
    LocalTensor(usize),
    LibraryKey(PermutedStructure<K>),
    Scalar(usize),
}

impl<K> NetworkLeaf<K> {
    pub fn is_scalar(&self) -> bool {
        matches!(self, NetworkLeaf::Scalar(_))
    }
}

impl<K: Display> Display for NetworkLeaf<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkLeaf::LibraryKey(k) => write!(f, "Key:{k}"),
            NetworkLeaf::LocalTensor(l) => write!(f, "Tensor:{l}"),
            NetworkLeaf::Scalar(s) => write!(f, "Scalar:{s}"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NetworkLeafWithInds<K> {
    LocalTensor(usize),
    LibraryKey { key: K, inds: Vec<AbstractIndex> },
    Scalar(usize),
}

impl<K> From<HedgeGraphBuilder<NetworkEdge, NetworkNode<K>>> for NetworkGraph<K> {
    fn from(builder: HedgeGraphBuilder<NetworkEdge, NetworkNode<K>>) -> Self {
        let graph: HedgeGraph<NetworkEdge, NetworkNode<K>> = builder.build();
        // graph.hed
        let mut slot_order = vec![0; graph.n_hedges()];

        for (_, n, _) in graph.iter_nodes() {
            let mut ord = 1;
            for c in n {
                slot_order[c.0] = ord;
                ord += 1;
            }
        }
        // let uncontracted = graph.empty_subgraph();
        Self {
            slot_order,
            graph,
            // uncontracted,
        }
    }
}
#[derive(Clone, Debug, Error)]
pub enum NetworkGraphError {
    #[error("Not head node")]
    NotHeadNode,
}

impl<K> NetworkGraph<K> {
    pub fn slots(&self, nodeid: NodeIndex) -> Vec<LibrarySlot> {
        let mut slots = Vec::new();
        let mut ord = Vec::new();
        if let NetworkNode::Leaf(_) = &self.graph[nodeid] {
            for n in self.graph.iter_crown(nodeid) {
                if let NetworkEdge::Slot(s) = self.graph[[&n]] {
                    slots.push(s);
                    ord.push(self.slot_order[n.0]);
                }
            }
        }

        let perm = Permutation::sort(ord);
        perm.apply_slice_in_place(&mut slots);
        slots
    }

    pub fn sync_order(&mut self) {
        for (v, n, _) in self.graph.iter_nodes() {
            let mut slots = Vec::new();
            let mut init_ord = Vec::new();
            let mut new_ord = Vec::new();
            let mut o = 0;
            let mut nc = n.clone();
            for h in nc {
                new_ord.push(o);
                o += 1;
                slots.push(self.graph[[&h]]);
                init_ord.push(self.slot_order[h.0]);
            }
            let perm = Permutation::sort(slots);
            let perm2 = Permutation::sort(init_ord);

            if perm != perm2 {
                // println!("FU");
                perm.apply_slice_in_place_inv(&mut new_ord);
                for (i, h) in n.enumerate() {
                    self.slot_order[h.0] = new_ord[i];
                }
            }
        }
    }

    pub fn inds(&self, nodeid: NodeIndex) -> Vec<AbstractIndex> {
        let mut slots = Vec::new();
        let mut ord = Vec::new();
        if let NetworkNode::Leaf(_) = &self.graph[nodeid] {
            for n in self.graph.iter_crown(nodeid) {
                if let NetworkEdge::Slot(s) = self.graph[[&n]] {
                    slots.push(s.aind);
                    ord.push(self.slot_order[n.0]);
                }
            }
        }

        let perm = Permutation::sort(ord);

        perm.apply_slice_in_place(&mut slots);
        // println!("Inds:{:?}", slots);
        slots
    }

    pub fn get_lib_data<
        S,
        LT: LibraryTensor + Clone,
        L: Library<S, Key = K, Value = PermutedStructure<LT>>,
    >(
        &self,
        lib: &L,
        nodeid: NodeIndex,
    ) -> Option<LT::WithIndices>
    where
        K: Display + Debug,
        LT::WithIndices: PermuteTensor<Permuted = LT::WithIndices>,
    {
        let mut inds = self.inds(nodeid);

        if let NetworkNode::Leaf(NetworkLeaf::LibraryKey(k)) = &self.graph[nodeid] {
            let libt = lib.get(&k.structure).unwrap();
            let mappingperm = &k.index_permutation;

            mappingperm.apply_slice_in_place(&mut inds);
            Some(libt.structure.with_indices(&inds).unwrap().permute())
        } else {
            None
        }
    }

    pub fn splice_descendents_of(&mut self, replacement: Self)
    where
        K: Clone,
    {
        // println!(
        //     "Joining \n{} with\n {}",
        //     self.dot_simple(),
        //     replacement.dot_simple()
        // );
        //
        self.slot_order.extend(replacement.slot_order);
        self.graph
            .join_mut(
                replacement.graph,
                |sf, sd, of, od| {
                    let flow_match = sf == -of;
                    let desc_match = sd == od;
                    // if desc_match{

                    // println!("looking at {sd} vs {od}, flow_match: {flow_match},sf  {sf:?},of {of:?}, desc_match: {desc_match}");
                    // }
                    // if flow_match && desc_match {
                    // sd.data
                    // println!("Splicing");
                    // }
                    flow_match && desc_match
                },
                |sf, sd, _, _| (sf, sd),
            )
            .unwrap();
    }

    pub fn extract<S: SubGraph<Base = BitVec>>(&mut self, subgraph: &S) -> Self
    where
        K: Clone,
    {
        let mut left = Hedge(0);
        let mut extracted = Hedge(self.graph.n_hedges());
        while left < extracted {
            if !subgraph.includes(&left) {
                //left is in the right place
                left.0 += 1;
            } else {
                //left needs to be swapped
                extracted.0 -= 1;
                if !subgraph.includes(&extracted) {
                    //only with an extracted that is in the wrong spot
                    self.slot_order.swap(left.0, extracted.0);
                    left.0 += 1;
                }
            }
        }

        let extracted = self.graph.extract(
            subgraph,
            |a| a.map(Clone::clone),
            |a| a,
            |a| a.clone(),
            |a| a,
        );

        let slot_order = self.slot_order.split_off(left.0);

        Self {
            slot_order,
            graph: extracted,
        }
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
                    let extracted = self.extract(&subgraph);

                    out.push((extracted, op, leaves))
                }
            }
        }
        out
    }
    pub fn extract_next_ready_op(&mut self) -> Option<(Self, NetworkOp)>
    where
        K: Clone,
    {
        // build a traversal over *all* internal edges
        let tt: SimpleTraversalTree<ChildVecStore<()>> = self.expr_tree().cast();
        let head = self.head();
        let root_node = self.graph.node_id(head);

        // println!("//tree:\n{}", self.graph.dot(&tt.tree_subgraph));
        // println!(
        // "//tree:\n{}",
        // self.graph.dot(&tt.tree_subgraph(self.graph.as_ref()))
        // );
        // println!("tree{:#?}", tt);
        // println!(
        //     "tree{}",
        //     tt.debug_draw(|a| match a {
        //         TTRoot::Child(a) => a.to_string(),
        //         TTRoot::Root => "Root".into(),
        //         _ => "".into(),
        //     })
        // );
        // look for the first op node whose children are all leaves
        for nid in tt.iter_preorder_tree_nodes(&self.graph, root_node) {
            if let NetworkNode::Op(op) = &self.graph[nid] {
                let mut subgraph: BitVec = self.graph.empty_subgraph();

                let mut all_leaves = true;
                let mut has_children = false;

                for h in self.graph.iter_crown(nid) {
                    subgraph.add(h);
                }

                for child in tt.iter_children(nid, &self.graph) {
                    has_children = true;
                    match &self.graph[child] {
                        NetworkNode::Leaf(a) => {
                            let mut slots = vec![];

                            for h in self.graph.iter_crown(child) {
                                if let NetworkEdge::Slot(s) = self.graph[[&h]] {
                                    slots.push(s.aind)
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
                if all_leaves && has_children {
                    let op = *op;

                    // println!(
                    //     "Extracting: {}",
                    //     self.graph.dot_impl(
                    //         &subgraph,
                    //         "",
                    //         &|a| if let NetworkEdge::Slot(s) = a {
                    //             Some(format!("label=\"{s}\""))
                    //         } else {
                    //             None
                    //         },
                    //         &|a| None
                    //     )
                    // );

                    let extracted = self.extract(&subgraph);

                    return Some((extracted, op));
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

    pub fn delete<S: SubGraph<Base = BitVec>>(&mut self, subgraph: &S) {
        let mut left = Hedge(0);
        let mut extracted = Hedge(self.graph.n_hedges());
        while left < extracted {
            if !subgraph.includes(&left) {
                //left is in the right place
                left.0 += 1;
            } else {
                //left needs to be swapped
                extracted.0 -= 1;
                if !subgraph.includes(&extracted) {
                    // println!("{extracted}<=>{left}");
                    //only with an extracted that is in the wrong spot
                    self.slot_order.swap(left.0, extracted.0);
                    left.0 += 1;
                }
            }
        }
        self.graph.delete_hedges(subgraph);
    }
    pub fn identify_nodes_without_self_edges(
        &mut self,
        nodes: &[NodeIndex],
        node_data: NetworkNode<K>,
    ) -> NodeIndex {
        let (n, sub) = self
            .graph
            .identify_nodes_without_self_edges::<BitVec>(nodes, node_data);

        self.graph.forget_identification_history();
        self.graph.node_store.check_and_set_nodes().unwrap();
        self.delete(&sub);

        self.graph.node_store.check_and_set_nodes().unwrap();
        n
    }

    pub fn identify_nodes_without_self_edges_merge_heads(
        &mut self,
        nodes: &[NodeIndex],
        node_data: NetworkNode<K>,
    ) -> NodeIndex {
        let (n, mut sub) = self
            .graph
            .identify_nodes_without_self_edges::<BitVec>(nodes, node_data);
        let mut first = false;
        for h in self.graph.iter_crown(n) {
            // println!("{h}");
            if self.graph[[&h]].is_head() {
                if first {
                    sub.add(h);
                    sub.add(self.graph.inv(h));
                }
                first = true;
            }
        }
        self.graph.forget_identification_history();
        self.graph.node_store.check_and_set_nodes().unwrap();
        self.delete(&sub);
        self.graph.node_store.check_and_set_nodes().unwrap();

        n
    }

    pub fn dot(&self) -> String
    where
        K: Display,
    {
        self.graph.dot_impl(
            &self.graph.full_filter(),
            "",
            &|e| {
                if let NetworkEdge::Slot(s) = e {
                    Some(format!("label=\"{s}\""))
                } else {
                    None
                }
            },
            &|n| match n {
                NetworkNode::Leaf(l) => match l {
                    NetworkLeaf::LibraryKey(l) => Some(format!("label= \"L{l}\"")),
                    NetworkLeaf::LocalTensor(l) => Some(format!("label = \"T{l}\"")),
                    NetworkLeaf::Scalar(s) => Some(format!("label = \"S{s}\"")),
                },
                NetworkNode::Op(o) => Some(format!("label = \"{o}\"")),
            },
        )
    }

    pub fn dot_impl(
        &self,
        scalar_disp: impl Fn(usize) -> String,
        library_disp: impl Fn(&K) -> String,
        tensor_disp: impl Fn(usize) -> String,
    ) -> String
// where
        // K: Display,
    {
        self.graph.dot_impl(
            &self.graph.full_filter(),
            "",
            &|e| {
                if let NetworkEdge::Slot(s) = e {
                    Some(format!("label=\"{s}\""))
                } else {
                    None
                }
            },
            &|n| match n {
                NetworkNode::Leaf(l) => match l {
                    NetworkLeaf::LibraryKey(l) => {
                        Some(format!("label= \"L:{}\"", library_disp(&l.structure)))
                    }
                    NetworkLeaf::LocalTensor(l) => {
                        Some(format!("label = \"T:{}\"", tensor_disp(*l)))
                    }
                    NetworkLeaf::Scalar(s) => Some(format!("label = \"S:{}\"", scalar_disp(*s))),
                },
                NetworkNode::Op(o) => Some(format!("label = \"{o}\"")),
            },
        )
    }

    pub fn dot_simple(&self) -> String
// where
        // K: Display,
    {
        self.dot_impl(|i| i.to_string(), |_| "".into(), |i| i.to_string())
    }

    pub fn zero() -> Self {
        Self::add(0, &[])
    }

    fn head_builder(
        node: NetworkNode<K>,
    ) -> (HedgeGraphBuilder<NetworkEdge, NetworkNode<K>>, NodeIndex) {
        let mut graph = HedgeGraphBuilder::new();
        let head = graph.add_node(node);
        graph.add_external_edge(head, NetworkEdge::Head, true, Flow::Source);
        (graph, head)
    }

    pub fn neg() -> Self {
        let (mut graph, head) = Self::head_builder(NetworkNode::Op(NetworkOp::Neg));
        graph.add_external_edge(head, NetworkEdge::Head, true, Flow::Sink);
        graph.into()
    }

    pub fn mul(n: usize) -> Self {
        let (mut graph, head) = Self::head_builder(NetworkNode::Op(NetworkOp::Product));
        for _ in 0..n {
            graph.add_external_edge(head, NetworkEdge::Head, true, Flow::Sink);
        }

        graph.into()
    }

    pub fn add(n: usize, slots: &[LibrarySlot]) -> Self {
        let (mut graph, head) = Self::head_builder(NetworkNode::Op(NetworkOp::Sum));
        for _ in 0..n {
            graph.add_external_edge(head, NetworkEdge::Head, true, Flow::Sink);
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
        graph.add_external_edge(head, NetworkEdge::Head, true, Flow::Source);

        graph.into()
    }

    pub fn key(key: PermutedStructure<K>) -> Self
    where
        K: TensorStructure,
    {
        let slots = key
            .structure
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
        let (mut graph, head) = Self::head_builder(NetworkNode::Leaf(node));

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

    // fn match_indices(
    //     _sf: Flow,
    //     self_data: EdgeData<&NetworkEdge>,
    //     _of: Flow,
    //     other_data: EdgeData<&NetworkEdge>,
    // ) -> bool {
    //     if let (NetworkEdge::Slot(s), NetworkEdge::Slot(o)) = (self_data.data, other_data.data) {
    //         s.matches(o)
    //     } else {
    //         false
    //     }
    // }

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

    pub fn result(
        &self,
    ) -> Result<(&NetworkNode<K>, NodeIndex, Vec<LibrarySlot>), TensorNetworkError<K>>
    where
        K: Display,
    {
        let mut n_heads = 0;
        let _headgraph: BitVec = self.graph.from_filter(|a| match a {
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

                Ok((&self.graph[root_node], root_node, slots))
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
        SimpleTraversalTree::depth_first_traverse(&self.graph, &headgraph, &root_node, None)
            .unwrap()
    }

    pub fn merge_ops(&mut self)
    where
        K: Clone,
    {
        // build a traversal over *all* internal edges
        let tt: SimpleTraversalTree<ParentChildStore<()>> = self.expr_tree().cast();
        let head = self.head();
        let root_node = self.graph.node_id(head);

        let mut sums: BitVec = self.graph.empty_subgraph();
        let mut prods: BitVec = self.graph.empty_subgraph();

        // look for repeated ops nodes along a chain
        for nid in tt.iter_preorder_tree_nodes(&self.graph, root_node) {
            if let NetworkNode::Op(op) = &self.graph[nid] {
                match op {
                    NetworkOp::Product => {
                        for h in self.graph.iter_crown(nid) {
                            if self.graph[[&h]].is_head() {
                                prods.add(h);
                            }
                        }
                    }
                    NetworkOp::Sum => {
                        for h in self.graph.iter_crown(nid) {
                            if self.graph[[&h]].is_head() {
                                sums.add(h);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut to_del: BitVec = self.graph.empty_subgraph();

        for sum in self.graph.connected_components(&sums) {
            let nodes: Vec<_> = self.graph.iter_nodes_of(&sum).map(|(a, _, _)| a).collect();

            if nodes.len() > 1 {
                let (_, sub) = self.graph.identify_nodes_without_self_edges::<BitVec>(
                    &nodes,
                    NetworkNode::Op(NetworkOp::Sum),
                );
                to_del.union_with(&sub);
            }
        }
        for prod in self.graph.connected_components(&prods) {
            let nodes: Vec<_> = self.graph.iter_nodes_of(&prod).map(|(a, _, _)| a).collect();
            if nodes.len() > 1 {
                let (_, sub) = self.graph.identify_nodes_without_self_edges::<BitVec>(
                    &nodes,
                    NetworkNode::Op(NetworkOp::Product),
                );
                to_del.union_with(&sub);
            };
        }

        self.graph.delete_hedges(&to_del);
        self.graph.forget_identification_history();
    }
    pub fn simplify_identity_ops(&mut self) {}

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
        self.slot_order.extend(other.slot_order);
        // self.uncontracted.join_mut(other.uncontracted);
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
            debug_assert!(
                slots.len() == rhs.n_dangling(),
                "Mismatched dangling edges in sum, Trying to add {} to {}",
                add.dot_simple(),
                rhs.dot_simple()
            );

            add.join_mut(
                rhs,
                NetworkGraph::<K>::add_match,
                NetworkGraph::<K>::join_heads,
            )
            .unwrap();
        }
        debug_assert!(
            slots.len() == add.n_dangling(),
            "addition with non matching dangling: \n{}",
            add.dot_impl(|_| "".to_string(), |_| "".to_string(), |_| "".to_string())
        );

        add
    }
}

impl<K> MulAssign for NetworkGraph<K> {
    fn mul_assign(&mut self, rhs: Self) {
        let mul = Self::mul(2);

        self.join_mut(
            mul,
            NetworkGraph::<K>::match_heads,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        self.join_mut(
            rhs,
            NetworkGraph::<K>::prod_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();
    }
}

impl<K: Clone> MulAssign<&NetworkGraph<K>> for NetworkGraph<K> {
    fn mul_assign(&mut self, rhs: &Self) {
        let mul = NetworkGraph::mul(2);

        self.join_mut(
            mul,
            NetworkGraph::<K>::match_heads,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        self.join_mut(
            rhs.clone(),
            NetworkGraph::<K>::prod_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();
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
impl<K: Clone> Mul<&NetworkGraph<K>> for NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn mul(self, rhs: &Self) -> Self::Output {
        self * rhs.clone()
    }
}

impl<'a, 'b, K: Clone> Mul<&'b NetworkGraph<K>> for &'a NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn mul(self, rhs: &'b NetworkGraph<K>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'a, K: Clone> Mul<NetworkGraph<K>> for &'a NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn mul(self, rhs: NetworkGraph<K>) -> Self::Output {
        rhs * self
    }
}

impl<K> AddAssign for NetworkGraph<K> {
    fn add_assign(&mut self, rhs: Self) {
        let slots = self.dangling_indices();
        debug_assert!(slots.len() == rhs.n_dangling());

        let add = Self::add(2, &slots);

        self.join_mut(
            add,
            NetworkGraph::<K>::add_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        self.join_mut(
            rhs,
            NetworkGraph::<K>::add_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();
    }
}

impl<K: Clone> AddAssign<&NetworkGraph<K>> for NetworkGraph<K> {
    fn add_assign(&mut self, rhs: &Self) {
        let slots = self.dangling_indices();
        debug_assert!(slots.len() == rhs.n_dangling());

        let add = Self::add(2, &slots);

        self.join_mut(
            add,
            NetworkGraph::<K>::add_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        self.join_mut(
            rhs.clone(),
            NetworkGraph::<K>::add_match,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();
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

impl<K: Clone> Add<&NetworkGraph<K>> for NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn add(self, rhs: &Self) -> Self::Output {
        self + rhs.clone()
    }
}

impl<'a, 'b, K: Clone> Add<&'b NetworkGraph<K>> for &'a NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn add(self, rhs: &'b NetworkGraph<K>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'a, K: Clone> Add<NetworkGraph<K>> for &'a NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn add(self, rhs: NetworkGraph<K>) -> Self::Output {
        rhs + self
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

impl<K: Clone> Neg for &NetworkGraph<K> {
    type Output = NetworkGraph<K>;
    fn neg(self) -> Self::Output {
        let mut neg = NetworkGraph::neg();

        neg.join_mut(
            self.clone(),
            NetworkGraph::<K>::match_heads,
            NetworkGraph::<K>::join_heads,
        )
        .unwrap();

        neg
    }
}

impl<K> SubAssign for NetworkGraph<K> {
    fn sub_assign(&mut self, rhs: NetworkGraph<K>) {
        *self += -rhs;
    }
}

impl<K: Clone> SubAssign<&NetworkGraph<K>> for NetworkGraph<K> {
    fn sub_assign(&mut self, rhs: &NetworkGraph<K>) {
        *self += -rhs;
    }
}

impl<K> Sub for NetworkGraph<K> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl<K: Clone> Sub<&NetworkGraph<K>> for NetworkGraph<K> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self + -rhs
    }
}
impl<'a, 'b, K: Clone> Sub<&'b NetworkGraph<K>> for &'a NetworkGraph<K> {
    type Output = NetworkGraph<K>;

    fn sub(self, rhs: &'b NetworkGraph<K>) -> Self::Output {
        -rhs + self
    }
}
impl<'a, K: Clone> Sub<NetworkGraph<K>> for &'a NetworkGraph<K> {
    type Output = NetworkGraph<K>;

    fn sub(self, rhs: NetworkGraph<K>) -> Self::Output {
        -rhs + self
    }
}

// pub trait TestContext {
//     fn get_statemap(&mut self) -> &mut StateMap;

//     fn get_model(&mut self) -> &mut Model;
// }

// pub struct Model {}

// pub struct Expr {}

// impl<T: TestContext> Decode<T> for Expr {
//     fn decode<D: bincode::de::Decoder<Context = T>>(
//         decoder: &mut D,
//     ) -> Result<Self, bincode::error::DecodeError> {
//         let context = decoder.context().get_statemap();
//         Ok(Expr {})
//     }
// }

// pub struct Test {
//     atom: Expr,
// }

// impl<T: TestContext> Decode<T> for Test {
//     fn decode<D: bincode::de::Decoder<Context = T>>(
//         decoder: &mut D,
//     ) -> Result<Self, bincode::error::DecodeError> {
//         let atom = Expr::decode(decoder)?;
//         Ok(Test { atom })
//     }
// }

#[cfg(test)]
pub mod test {

    use std::ops::Deref;

    use crate::{
        network::graph::NetworkLeaf,
        structure::{
            representation::{Euclidean, LibraryRep, Lorentz, Minkowski, RepName},
            slot::IsAbstractSlot,
            OrderedStructure,
        },
    };

    use super::NetworkGraph;

    #[test]
    fn addition() {
        let one = NetworkGraph::<i8>::one();
        let zero = NetworkGraph::zero();

        let s = NetworkGraph::scalar(2);

        let t = NetworkGraph::<i8>::tensor(
            &OrderedStructure::<LibraryRep>::from_iter([
                Minkowski {}.new_slot(1, 2),
                Minkowski {}.new_slot(2, 2),
            ])
            .structure,
            NetworkLeaf::LocalTensor(1),
        );

        let t2 = NetworkGraph::<i8>::tensor(
            &OrderedStructure::<LibraryRep>::from_iter([
                Minkowski {}.new_slot(1, 2),
                Minkowski {}.new_slot(2, 2),
            ])
            .structure,
            NetworkLeaf::LocalTensor(2),
        );

        let t3 = NetworkGraph::<i8>::tensor(
            &OrderedStructure::<LibraryRep>::from_iter([
                Lorentz {}.new_slot(1, 2).to_lib(),
                Euclidean {}.new_slot(2, 2).to_lib(),
            ])
            .structure,
            NetworkLeaf::LocalTensor(2),
        );

        let t3b = NetworkGraph::<i8>::tensor(
            &OrderedStructure::<LibraryRep>::from_iter([
                Lorentz {}.dual().new_slot(1, 2).to_lib(),
                Euclidean {}.new_slot(2, 1).to_lib(),
            ])
            .structure,
            NetworkLeaf::LocalTensor(2),
        );
        let s2 = NetworkGraph::scalar(3);

        let mut expr = t3b * (&t + &t2) * s2 * (&s + &zero) * (t3 * (t + t2) * s * (one + zero));

        // println!("{}", expr.dot());
        expr.merge_ops();
        println!("{}", expr.dot());

        // expr.extract_next_ready_op();
        if let Some((a, op)) = expr.extract_next_ready_op() {
            println!("{}", a.dot());
        }
    }
}

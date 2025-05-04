use bincode::{Decode, Encode};

use graph::{
    NAdd, NMul, NetworkEdge, NetworkGraph, NetworkLeaf, NetworkLeafWithInds, NetworkNode, NetworkOp,
};
use linnet::half_edge::NodeIndex;
use ref_ops::{RefAdd, RefMul, RefNeg};
use serde::{Deserialize, Serialize};

use tensor_library::{Library, LibraryError};

use crate::algebraic_traits::{One, Zero};
use crate::arithmetic::ScalarMul;
use crate::contraction::Contract;
use crate::network::tensor_library::LibraryTensor;
// use crate::shadowing::Concretize;
use crate::structure::representation::LibrarySlot;
use crate::structure::{StructureError, TensorShell};
use std::borrow::Cow;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};
use store::{NetworkStore, TensorScalarStore, TensorScalarStoreMapping};
use thiserror::Error;
// use log::trace;

#[cfg(feature = "shadowing")]
use symbolica::atom::{representation::FunView, AddView, Atom, AtomView, MulView, PowView};

use crate::{
    contraction::ContractionError,
    structure::{CastStructure, HasStructure, ScalarTensor, TensorStructure},
};
#[cfg(feature = "shadowing")]
use crate::{
    parametric::ParamTensor,
    shadowing::Concretize,
    structure::representation::LibraryRep,
    structure::slot::IsAbstractSlot,
    structure::HasName,
    symbolica_utils::{IntoArgs, IntoSymbol},
};

// use anyhow::Result;

use std::{convert::Infallible, fmt::Debug};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
#[cfg_attr(
    feature = "shadowing",
    bincode(decode_context = "symbolica::state::StateMap")
)]
pub struct Network<S, LibKey> {
    pub graph: NetworkGraph<LibKey>,
    pub store: S,
}

// pub type TensorNetwork<T, S, Str: TensorScalarStore<Tensor = T, Scalar = S>, K> = Network<Str, K>;

// pub struct TensorNetwork<
//     T,
//     S,
//     K,
//     Str: TensorScalarStore<Tensor = T, Scalar = S> = NetworkStore<T, S>,
// > {
//     net: Network<Str, K>,
// }

pub mod graph;
pub mod set;
pub mod store;
pub mod tensor_library;

impl<S: TensorScalarStoreMapping, K: Clone> TensorScalarStoreMapping for Network<S, K> {
    type Store<U, V> = Network<S::Store<U, V>, K>;
    type Scalar = S::Scalar;
    type Tensor = S::Tensor;

    fn iter_scalars(&self) -> impl Iterator<Item = &Self::Scalar> {
        self.store.iter_scalars()
    }

    fn iter_tensors(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.store.iter_tensors()
    }

    fn iter_scalars_mut(&mut self) -> impl Iterator<Item = &mut Self::Scalar> {
        self.store.iter_scalars_mut()
    }
    fn iter_tensors_mut(&mut self) -> impl Iterator<Item = &mut Self::Tensor> {
        self.store.iter_tensors_mut()
    }

    fn map<U, V>(
        self,
        scalar_map: impl FnMut(Self::Scalar) -> U,
        tensor_map: impl FnMut(Self::Tensor) -> V,
    ) -> Self::Store<V, U> {
        Network {
            store: self.store.map(scalar_map, tensor_map),
            graph: self.graph,
        }
    }

    fn map_result<U, V, Er>(
        self,
        scalar_map: impl FnMut(Self::Scalar) -> Result<U, Er>,
        tensor_map: impl FnMut(Self::Tensor) -> Result<V, Er>,
    ) -> Result<Self::Store<V, U>, Er> {
        Ok(Network {
            store: self.store.map_result(scalar_map, tensor_map)?,
            graph: self.graph,
        })
    }

    fn map_ref<'a, U, V>(
        &'a self,
        scalar_map: impl FnMut(&'a Self::Scalar) -> U,
        tensor_map: impl FnMut(&'a Self::Tensor) -> V,
    ) -> Self::Store<V, U> {
        Network {
            store: self.store.map_ref(scalar_map, tensor_map),
            graph: self.graph.clone(),
        }
    }

    fn map_ref_result<U, V, Er>(
        &self,
        scalar_map: impl FnMut(&Self::Scalar) -> Result<U, Er>,
        tensor_map: impl FnMut(&Self::Tensor) -> Result<V, Er>,
    ) -> Result<Self::Store<V, U>, Er> {
        Ok(Network {
            store: self.store.map_ref_result(scalar_map, tensor_map)?,
            graph: self.graph.clone(),
        })
    }

    fn map_ref_enumerate<U, V>(
        &self,
        scalar_map: impl FnMut((usize, &Self::Scalar)) -> U,
        tensor_map: impl FnMut((usize, &Self::Tensor)) -> V,
    ) -> Self::Store<V, U> {
        Network {
            store: self.store.map_ref_enumerate(scalar_map, tensor_map),
            graph: self.graph.clone(),
        }
    }

    fn map_ref_result_enumerate<U, V, Er>(
        &self,
        scalar_map: impl FnMut((usize, &Self::Scalar)) -> Result<U, Er>,
        tensor_map: impl FnMut((usize, &Self::Tensor)) -> Result<V, Er>,
    ) -> Result<Self::Store<V, U>, Er> {
        Ok(Network {
            store: self
                .store
                .map_ref_result_enumerate(scalar_map, tensor_map)?,
            graph: self.graph.clone(),
        })
    }

    fn map_ref_mut<U, V>(
        &mut self,
        scalar_map: impl FnMut(&mut Self::Scalar) -> U,
        tensor_map: impl FnMut(&mut Self::Tensor) -> V,
    ) -> Self::Store<V, U> {
        Network {
            store: self.store.map_ref_mut(scalar_map, tensor_map),
            graph: self.graph.clone(),
        }
    }

    fn map_ref_mut_result<U, V, Er>(
        &mut self,
        scalar_map: impl FnMut(&mut Self::Scalar) -> Result<U, Er>,
        tensor_map: impl FnMut(&mut Self::Tensor) -> Result<V, Er>,
    ) -> Result<Self::Store<V, U>, Er> {
        Ok(Network {
            store: self.store.map_ref_mut_result(scalar_map, tensor_map)?,
            graph: self.graph.clone(),
        })
    }

    fn map_ref_mut_enumerate<U, V>(
        &mut self,
        scalar_map: impl FnMut((usize, &mut Self::Scalar)) -> U,
        tensor_map: impl FnMut((usize, &mut Self::Tensor)) -> V,
    ) -> Self::Store<V, U> {
        Network {
            store: self.store.map_ref_mut_enumerate(scalar_map, tensor_map),
            graph: self.graph.clone(),
        }
    }

    fn map_ref_mut_result_enumerate<U, V, Er>(
        &mut self,
        scalar_map: impl FnMut((usize, &mut Self::Scalar)) -> Result<U, Er>,
        tensor_map: impl FnMut((usize, &mut Self::Tensor)) -> Result<V, Er>,
    ) -> Result<Self::Store<V, U>, Er> {
        Ok(Network {
            store: self
                .store
                .map_ref_mut_result_enumerate(scalar_map, tensor_map)?,
            graph: self.graph.clone(),
        })
    }
}

impl<S: TensorScalarStore, K> Default for Network<S, K> {
    fn default() -> Self {
        Self::one()
    }
}

impl<S: TensorScalarStore, K> NMul for Network<S, K> {
    type Output = Self;
    fn n_mul<I: IntoIterator<Item = Self>>(self, iter: I) -> Self::Output {
        let mut store = self.store;
        let items = iter.into_iter().map(|mut a| {
            a.graph.shift_scalars(store.n_scalars());
            a.graph.shift_tensors(store.n_tensors());
            store.extend(a.store);
            a.graph
        });

        Network {
            graph: self.graph.n_mul(items),
            store,
        }
    }
}

impl<S: TensorScalarStore, K> Mul for Network<S, K> {
    type Output = Self;
    fn mul(self, mut other: Self) -> Self::Output {
        let mut store = self.store;

        other.graph.shift_scalars(store.n_scalars());
        other.graph.shift_tensors(store.n_tensors());
        store.extend(other.store);

        Network {
            graph: self.graph * other.graph,
            store,
        }
    }
}

impl<S: TensorScalarStore, K> Add for Network<S, K> {
    type Output = Self;
    fn add(self, mut other: Self) -> Self::Output {
        let mut store = self.store;

        other.graph.shift_scalars(store.n_scalars());
        other.graph.shift_tensors(store.n_tensors());
        store.extend(other.store);

        Network {
            graph: self.graph + other.graph,
            store,
        }
    }
}

impl<S: TensorScalarStore, K> NAdd for Network<S, K> {
    type Output = Self;
    fn n_add<I: IntoIterator<Item = Self>>(self, iter: I) -> Self::Output {
        let mut store = self.store;

        let items = iter.into_iter().map(|mut a| {
            a.graph.shift_scalars(store.n_scalars());
            a.graph.shift_tensors(store.n_tensors());
            store.extend(a.store);
            a.graph
        });

        Network {
            graph: self.graph.n_add(items),
            store,
        }
    }
}

impl<S: TensorScalarStore, K> Network<S, K> {
    pub fn scalar(scalar: S::Scalar) -> Self {
        let mut store = S::default();
        let id = store.add_scalar(scalar);
        Network {
            graph: NetworkGraph::scalar(id),
            store,
        }
    }

    pub fn local_tensor(tensor: S::Tensor) -> Self
    where
        S::Tensor: TensorStructure,
    {
        let mut store = S::default();
        let id = store.add_tensor(tensor);
        Network {
            graph: NetworkGraph::tensor(store.get_tensor(id), NetworkLeaf::LocalTensor(id)),
            store,
        }
    }

    pub fn library_tensor<T>(tensor: &T, key: K) -> Self
    where
        T: TensorStructure,
    {
        Network {
            graph: NetworkGraph::tensor(tensor, NetworkLeaf::LibraryKey(key)),
            store: S::default(),
        }
    }

    pub fn one() -> Self {
        Network {
            graph: NetworkGraph::one(),
            store: S::default(),
        }
    }

    pub fn zero() -> Self {
        Network {
            graph: NetworkGraph::zero(),
            store: S::default(),
        }
    }
}

#[derive(Error, Debug)]
pub enum TensorNetworkError<K: Display> {
    #[error("Contraction Error:{0}")]
    ContractionError(#[from] ContractionError),
    #[error("Scalar connected by a slot edge")]
    ScalarSlotEdge,
    #[error("Structure Error:{0}")]
    StructErr(#[from] StructureError),
    #[error("LibraryError:{0}")]
    LibErr(#[from] LibraryError<K>),
    #[error("Non tensor node still present")]
    NonTensorNodePresent,
    #[error("invalid resulting node(0)")]
    InvalidResultNode(NetworkNode<()>),
    #[error("internal edge still present, contract it first")]
    InternalEdgePresent,
    #[error("uncontracted scalar")]
    UncontractedScalar,
    #[error("Cannot contract edge between {0} and {1}")]
    CannotContractEdgeBetween(NetworkNode<K>, NetworkNode<K>),
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
    #[error("not all scalars: {0}")]
    NotAllScalars(String),
    #[error("try to sum scalar with library tensor: {0}")]
    ScalarLibSum(String),
    #[error("try to sum scalar with a tensor: {0}")]
    SumScalarTensor(String),
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
    #[error("Infallible")]
    Infallible,
}

impl<K: Display> From<Infallible> for TensorNetworkError<K> {
    fn from(_: Infallible) -> Self {
        TensorNetworkError::Infallible
    }
}

pub enum TensorOrScalarOrKey<T, S, K> {
    Tensor {
        tensor: T,
        graph_slots: Vec<LibrarySlot>,
    },
    Scalar(S),
    Key {
        key: K,
        graph_slots: Vec<LibrarySlot>,
    },
    One,
    Zero,
}

impl<T: TensorStructure, S, K: Display, Str: TensorScalarStore<Tensor = T, Scalar = S>>
    Network<Str, K>
{
    pub fn result(&self) -> Result<TensorOrScalarOrKey<&T, &S, &K>, TensorNetworkError<K>> {
        let (node, graph_slots) = self.graph.result()?;

        match node {
            NetworkNode::Leaf(l) => match l {
                NetworkLeaf::LibraryKey(k) => Ok(TensorOrScalarOrKey::Key {
                    key: k,
                    graph_slots,
                }),
                NetworkLeaf::LocalTensor(t) => Ok(TensorOrScalarOrKey::Tensor {
                    tensor: self.store.get_tensor(*t),
                    graph_slots,
                }),
                NetworkLeaf::Scalar(t) => {
                    Ok(TensorOrScalarOrKey::Scalar(self.store.get_scalar(*t)))
                }
            },
            NetworkNode::Op(o) => match o {
                NetworkOp::Neg => Err(TensorNetworkError::InvalidResultNode(NetworkNode::Op(
                    NetworkOp::Neg,
                ))),
                NetworkOp::Product => Ok(TensorOrScalarOrKey::One),
                NetworkOp::Sum => Ok(TensorOrScalarOrKey::Zero),
            },
        }
    }

    pub fn result_tensor<'a, L: Library<S, Key = K>>(
        &'a self,
        lib: &L,
    ) -> Result<Cow<'a, T>, TensorNetworkError<K>>
    where
        S: 'a,
        T: Clone + ScalarTensor,
        T::Scalar: One + Zero,
        K: Display,
        L::Value: TensorStructure<Indexed = T> + Clone,
        for<'b> &'b S: Into<T::Scalar>,
    {
        Ok(match self.result()? {
            TensorOrScalarOrKey::One => Cow::Owned(T::new_scalar(T::Scalar::one())),
            TensorOrScalarOrKey::Zero => Cow::Owned(T::new_scalar(T::Scalar::zero())),
            TensorOrScalarOrKey::Tensor { tensor, .. } => Cow::Borrowed(tensor),
            TensorOrScalarOrKey::Scalar(s) => Cow::Owned(T::new_scalar(s.into())),
            TensorOrScalarOrKey::Key { key, graph_slots } => {
                let inds: Vec<_> = graph_slots.iter().map(|a| a.aind).collect();
                let less = lib.get(key)?.into_owned().reindex(&inds)?;

                Cow::Owned(less)
            }
        })
    }

    pub fn result_scalar<'a>(&'a self) -> Result<Cow<'a, S>, TensorNetworkError<K>>
    where
        T: Clone + ScalarTensor + 'a,
        T::Scalar: Into<S>,
        K: Display,
        S: One + Zero + Clone,
    {
        Ok(match self.result()? {
            TensorOrScalarOrKey::One => Cow::Owned(S::one()),
            TensorOrScalarOrKey::Zero => Cow::Owned(S::zero()),
            TensorOrScalarOrKey::Tensor { tensor: t, .. } => Cow::Owned(
                t.clone()
                    .scalar()
                    .ok_or(TensorNetworkError::NoScalar)?
                    .into(),
            ),
            TensorOrScalarOrKey::Scalar(s) => Cow::Borrowed(s),
            TensorOrScalarOrKey::Key { .. } => return Err(TensorNetworkError::NoScalar),
        })
    }

    pub fn cast<U>(self) -> Network<Str::Store<U, S>, K>
    where
        K: Clone,
        T: CastStructure<U> + HasStructure,
        T::Structure: TensorStructure,
        U: HasStructure,
        U::Structure: From<T::Structure> + TensorStructure<Slot = T::Slot>,
    {
        self.map(|a| a, |t| t.cast_structure())
    }
}

impl<S, K: Display> Network<S, K> {
    pub fn dot(&self) -> std::string::String {
        self.graph.dot()
    }
}

// use log::trace;
#[cfg(feature = "shadowing")]
pub mod parsing;
// use log::trace;

pub trait ContractionStrategy<E, L, K>: Sized {
    fn contract(
        executor: &mut E,
        graph: NetworkGraph<K>,
        lib: &L,
    ) -> Result<NetworkGraph<K>, TensorNetworkError<K>>
    where
        K: Display;
}

pub trait ExecutionStrategy<E, L, K>
where
    E: ExecuteOp<L, K>,
{
    /// Run the entire contraction to one leaf.
    fn execute_all<C: ContractionStrategy<E, L, K>>(
        executor: &mut E,
        graph: &mut NetworkGraph<K>,
        lib: &L,
    ) -> Result<(), TensorNetworkError<K>>
    where
        K: Display;
}

pub struct Sequential;

impl<E, L, K> ExecutionStrategy<E, L, K> for Sequential
where
    E: ExecuteOp<L, K>,
    K: Clone,
{
    fn execute_all<C: ContractionStrategy<E, L, K>>(
        executor: &mut E,
        graph: &mut NetworkGraph<K>,
        lib: &L,
    ) -> Result<(), TensorNetworkError<K>>
    where
        K: Display,
    {
        while {
            // find the *one* ready op
            if let Some((extracted_graph, op, leaves)) = graph.extract_next_ready_op() {
                // execute + splice
                let replacement = executor.execute::<C>(extracted_graph, lib, op, &leaves)?;
                graph.splice_descendents_of(replacement);
                true
            } else {
                false
            }
        } {}

        Ok(())
    }
}

/// 2b) Parallel: batch‐execute all ready ops, then splice serially.
// pub struct Parallel;

// impl<E, K> ExecutionStrategy<E, K> for Parallel
// where
//     E: ExecuteOp<K> + Clone + Send + Sync,
//     K: Clone + Send + Sync,
// {
//     fn contract_all<L: Library<Key = K> + Sync>(
//         &self,
//         executor: &mut E,
//         graph: &mut NetworkGraph<K>,
//         lib: &L,
//     ) {
//         loop {
//             // 1) collect *all* ready ops this round
//             let ready = graph.find_all_ready_ops();

//             if ready.is_empty() {
//                 break;
//             }

//             // 2) execute them in parallel
//             let results: Vec<(NodeIndex, NetworkGraph<K>)> = ready
//                 .into_par_iter()
//                 .map(|(nid, op, leaves)| {
//                     let mut local = executor.clone();
//                     let replacement = local.execute(lib, op, &leaves);
//                     (nid, replacement)
//                 })
//                 .collect();

//             // 3) splice back sequentially
//             for (nid, replacement) in results {
//                 graph.splice_descendents_of(nid, replacement);
//             }
//         }
//     }
// }

pub trait ExecuteOp<L, K>: Sized {
    // type LibStruct;
    fn execute<C: ContractionStrategy<Self, L, K>>(
        &mut self,
        graph: NetworkGraph<K>,
        lib: &L,
        op: NetworkOp,
        targets: &[NetworkLeafWithInds<K>],
    ) -> Result<NetworkGraph<K>, TensorNetworkError<K>>
    where
        K: Display;
}

impl<S, Store: TensorScalarStore, K> Network<Store, K>
where
    Store::Tensor: HasStructure<Structure = S>,
{
    pub fn execute<Strat: ExecutionStrategy<Store, L, K>, C: ContractionStrategy<Store, L, K>, L>(
        &mut self,
        lib: &L,
    ) -> Result<(), TensorNetworkError<K>>
    where
        K: Display,
        L: Library<S, Key = K, Value: LibraryTensor<WithIndices = Store::Tensor>> + Sync,
        Store: ExecuteOp<L, K>,
    {
        Strat::execute_all::<C>(&mut self.store, &mut self.graph, lib)
    }
}

impl<
        T: HasStructure
            + TensorStructure
            + Neg<Output = T>
            + Clone
            + for<'a> AddAssign<&'a T>
            + for<'a> AddAssign<<L::Value as LibraryTensor>::WithIndices>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: Neg<Output = Sc>
            + for<'a> AddAssign<&'a Sc>
            + Clone
            + for<'a> AddAssign<T::ScalarRef<'a>>
            + From<T::Scalar>,
        K: Display + Debug,
    > ExecuteOp<L, K> for NetworkStore<T, Sc>
{
    fn execute<C: ContractionStrategy<Self, L, K>>(
        &mut self,
        graph: NetworkGraph<K>,
        lib: &L,
        op: NetworkOp,
        targets: &[NetworkLeafWithInds<K>],
    ) -> Result<NetworkGraph<K>, TensorNetworkError<K>> {
        match op {
            NetworkOp::Neg => {
                debug_assert_eq!(targets.len(), 1);
                let target = &targets[0];

                match target {
                    NetworkLeafWithInds::Scalar(s) => {
                        let s = self.scalar[*s].clone().neg();
                        let pos = self.scalar.len();
                        self.scalar.push(s);
                        Ok(NetworkGraph::scalar(pos))
                    }
                    NetworkLeafWithInds::LibraryKey { key, inds } => {
                        let t = T::from(lib.get(key).unwrap().with_indices(inds).unwrap()).neg();
                        let pos = self.tensors.len();
                        let node = NetworkLeaf::LocalTensor(pos);
                        self.tensors.push(t);
                        Ok(NetworkGraph::tensor(&self.tensors[pos], node))
                    }
                    NetworkLeafWithInds::LocalTensor(t) => {
                        let t = self.tensors[*t].clone().neg();
                        let pos = self.tensors.len();
                        let node = NetworkLeaf::LocalTensor(pos);
                        self.tensors.push(t);
                        Ok(NetworkGraph::tensor(&self.tensors[pos], node))
                    }
                }
            }
            NetworkOp::Product => C::contract(self, graph, lib),
            NetworkOp::Sum => {
                let first = &targets[0];

                match first {
                    NetworkLeafWithInds::Scalar(s) => {
                        let mut accumulator = self.scalar[*s].clone();

                        for t in &targets[1..] {
                            match t {
                                NetworkLeafWithInds::Scalar(s) => {
                                    accumulator += &self.scalar[*s];
                                }
                                NetworkLeafWithInds::LocalTensor(t) => {
                                    if let Some(s) = self.tensors[*t].scalar_ref() {
                                        accumulator += s;
                                    } else {
                                        return Err(TensorNetworkError::NotAllScalars(
                                            "".to_string(),
                                        ));
                                    }
                                }
                                NetworkLeafWithInds::LibraryKey { .. } => {
                                    return Err(TensorNetworkError::ScalarLibSum("".to_string()));
                                }
                            }
                        }

                        let pos = self.scalar.len();
                        self.scalar.push(accumulator);

                        Ok(NetworkGraph::scalar(pos))
                    }
                    NetworkLeafWithInds::LocalTensor(t) => {
                        let mut accumulator = self.tensors[*t].clone();
                        if accumulator.is_scalar() {
                            let mut accumulator = Sc::from(accumulator.scalar().unwrap());

                            for t in &targets[1..] {
                                match t {
                                    NetworkLeafWithInds::Scalar(s) => {
                                        accumulator += &self.scalar[*s];
                                    }
                                    NetworkLeafWithInds::LocalTensor(t) => {
                                        if let Some(s) = self.tensors[*t].scalar_ref() {
                                            accumulator += s;
                                        } else {
                                            return Err(TensorNetworkError::NotAllScalars(
                                                "".to_string(),
                                            ));
                                        }
                                    }
                                    NetworkLeafWithInds::LibraryKey { .. } => {
                                        return Err(TensorNetworkError::ScalarLibSum(
                                            "".to_string(),
                                        ));
                                    }
                                }
                            }

                            let pos = self.scalar.len();
                            self.scalar.push(accumulator);

                            Ok(NetworkGraph::scalar(pos))
                        } else {
                            for t in &targets[1..] {
                                match t {
                                    NetworkLeafWithInds::Scalar(_) => {
                                        return Err(TensorNetworkError::SumScalarTensor(
                                            "".to_string(),
                                        ))
                                    }
                                    NetworkLeafWithInds::LocalTensor(t) => {
                                        accumulator += &self.tensors[*t];
                                    }
                                    NetworkLeafWithInds::LibraryKey { key, inds } => {
                                        let with_index = lib.get(key)?.with_indices(inds)?;
                                        accumulator += with_index;
                                    }
                                }
                            }

                            let pos = self.tensors.len();
                            self.tensors.push(accumulator);

                            Ok(NetworkGraph::tensor(
                                &self.tensors[pos],
                                NetworkLeaf::LocalTensor(pos),
                            ))
                        }
                    }
                    NetworkLeafWithInds::LibraryKey { key, inds } => {
                        let mut accumulator = T::from(lib.get(key)?.with_indices(inds)?);
                        for t in &targets[1..] {
                            match t {
                                NetworkLeafWithInds::Scalar(_) => {
                                    return Err(TensorNetworkError::SumScalarTensor("".to_string()))
                                }
                                NetworkLeafWithInds::LocalTensor(t) => {
                                    accumulator += &self.tensors[*t];
                                }
                                NetworkLeafWithInds::LibraryKey { key, inds } => {
                                    let with_index = lib.get(key)?.with_indices(inds)?;
                                    accumulator += with_index;
                                }
                            }
                        }

                        let pos = self.tensors.len();
                        self.tensors.push(accumulator);

                        Ok(NetworkGraph::tensor(
                            &self.tensors[pos],
                            NetworkLeaf::LocalTensor(pos),
                        ))
                    }
                }
            }
        }
    }
}

pub struct SmallestDegree;

impl<
        T: HasStructure
            + TensorStructure
            + Clone
            + Contract<LCM = T>
            + ScalarMul<Sc, Output = T>
            + Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: for<'a> MulAssign<&'a Sc> + Clone + for<'a> MulAssign<T::ScalarRef<'a>> + From<T::Scalar>,
        K: Display + Debug + Clone,
    > ContractionStrategy<NetworkStore<T, Sc>, L, K> for SmallestDegree
where
    <L::Value as LibraryTensor>::WithIndices:
        Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>,
{
    fn contract(
        executor: &mut NetworkStore<T, Sc>,
        mut graph: NetworkGraph<K>,
        lib: &L,
    ) -> Result<NetworkGraph<K>, TensorNetworkError<K>>
    where
        K: Display,
    {
        // First do all scalar products and then store the outcome of it in the head
        //
        //
        let head = graph.graph.node_id(graph.head());

        let (mut scalars, mut scalar_nodes): (Vec<_>, Vec<_>) = graph
            .graph
            .iter_nodes()
            .filter_map(|(_, nid, c)| {
                if let NetworkNode::Leaf(NetworkLeaf::Scalar(i)) = c {
                    Some((*i, nid))
                } else {
                    None
                }
            })
            .collect();

        scalar_nodes.push(head);

        if let Some(f) = scalars.pop() {
            let mut acc = executor.scalar[f].clone();

            for si in scalars {
                acc *= &executor.scalar[si];
            }

            let pos = executor.scalar.len();
            executor.scalar.push(acc);

            graph
                .graph
                .identify_nodes(&scalar_nodes, NetworkNode::Leaf(NetworkLeaf::Scalar(pos)));
        }

        SmallestDegree::contract_impl(executor, graph, lib, head)
    }
}

impl SmallestDegree {
    fn contract_impl<
        T: HasStructure
            + TensorStructure
            + Clone
            + Contract<LCM = T>
            + ScalarMul<Sc, Output = T>
            + Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: for<'a> MulAssign<&'a Sc> + Clone + for<'a> MulAssign<T::ScalarRef<'a>> + From<T::Scalar>,
        K: Display + Debug + Clone,
    >(
        executor: &mut NetworkStore<T, Sc>,
        mut graph: NetworkGraph<K>,
        lib: &L,
        head: NodeIndex,
    ) -> Result<NetworkGraph<K>, TensorNetworkError<K>>
    where
        K: Display,
        <L::Value as LibraryTensor>::WithIndices:
            Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>,
    {
        // First do all scalar products and then store the outcome of it in the head
        //
        //

        let edge_to_contract = graph
            .graph
            .iter_nodes()
            .filter(|(_, nid, _)| *nid != head)
            .filter_map(|(a, nid1, n1)| {
                let mut degree = 0;
                let mut first = None;
                for h in a {
                    if first.is_none() {
                        first = Some(h);
                    }
                    if graph.graph[[&h]].is_slot() && graph.graph.inv(h) != h {
                        first = Some(h); //override contracting hedge if it is a slot hedge
                    }
                    degree += 1
                }

                // let n1 = &graph.graph[graph.graph.node_id(first?)];
                let nid2 = graph.graph.involved_node_id(first?)?;
                let n2 = &graph.graph[nid2];

                Some((degree, nid1, n1, nid2, n2))
            })
            .min_by_key(|(degree, _, _, _, _)| *degree);

        if let Some((_, nid1, n1, nid2, n2)) = edge_to_contract {
            match (n1, n2) {
                (
                    NetworkNode::Leaf(NetworkLeaf::Scalar(_)),
                    NetworkNode::Op(NetworkOp::Product),
                )
                | (
                    NetworkNode::Op(NetworkOp::Product),
                    NetworkNode::Leaf(NetworkLeaf::Scalar(_)),
                ) => {
                    // This should have already been handled
                    return Err(TensorNetworkError::UncontractedScalar);
                }

                (
                    NetworkNode::Leaf(NetworkLeaf::LocalTensor(l)),
                    NetworkNode::Op(NetworkOp::Product),
                )
                | (
                    NetworkNode::Op(NetworkOp::Product),
                    NetworkNode::Leaf(NetworkLeaf::LocalTensor(l)),
                ) => {
                    graph.identify_nodes_without_self_edges(
                        &[nid1, nid2],
                        NetworkNode::Leaf(NetworkLeaf::LocalTensor(*l)),
                    );
                    Self::contract_impl(executor, graph, lib, head)
                }

                (
                    NetworkNode::Leaf(NetworkLeaf::LibraryKey(k)),
                    NetworkNode::Op(NetworkOp::Product),
                )
                | (
                    NetworkNode::Op(NetworkOp::Product),
                    NetworkNode::Leaf(NetworkLeaf::LibraryKey(k)),
                ) => {
                    graph.identify_nodes_without_self_edges(
                        &[nid1, nid2],
                        NetworkNode::Leaf(NetworkLeaf::LibraryKey(k.clone())),
                    );
                    Self::contract_impl(executor, graph, lib, head)
                }

                (NetworkNode::Leaf(l1), NetworkNode::Leaf(l2)) => match (l1, l2) {
                    (NetworkLeaf::Scalar(_), _) | (_, NetworkLeaf::Scalar(_)) => {
                        return Err(TensorNetworkError::ScalarSlotEdge);
                    }
                    (NetworkLeaf::LocalTensor(l1), NetworkLeaf::LocalTensor(l2)) => {
                        let contracted = executor.tensors[*l1].contract(&executor.tensors[*l2])?;
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);

                        graph.identify_nodes_without_self_edges(
                            &[nid1, nid2],
                            NetworkNode::Leaf(NetworkLeaf::LocalTensor(pos)),
                        );

                        Self::contract_impl(executor, graph, lib, head)
                    }
                    (NetworkLeaf::LibraryKey(l1), NetworkLeaf::LocalTensor(l2)) => {
                        let l1_inds: Vec<_> = graph
                            .graph
                            .iter_crown(nid1)
                            .filter_map(|i| match graph.graph[[&i]] {
                                NetworkEdge::Head => None,
                                NetworkEdge::Slot(s) => Some(s.aind),
                            })
                            .collect();
                        let contracted = executor.tensors[*l2]
                            .contract(&lib.get(l1)?.with_indices(&l1_inds)?)?;
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);

                        graph.identify_nodes_without_self_edges(
                            &[nid1, nid2],
                            NetworkNode::Leaf(NetworkLeaf::LocalTensor(pos)),
                        );

                        Self::contract_impl(executor, graph, lib, head)
                    }

                    (NetworkLeaf::LocalTensor(l2), NetworkLeaf::LibraryKey(l1)) => {
                        let l1_inds: Vec<_> = graph
                            .graph
                            .iter_crown(nid1)
                            .filter_map(|i| match graph.graph[[&i]] {
                                NetworkEdge::Head => None,
                                NetworkEdge::Slot(s) => Some(s.aind),
                            })
                            .collect();
                        let contracted = executor.tensors[*l2]
                            .contract(&lib.get(l1)?.with_indices(&l1_inds)?)?;
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);

                        graph.identify_nodes_without_self_edges(
                            &[nid1, nid2],
                            NetworkNode::Leaf(NetworkLeaf::LocalTensor(pos)),
                        );

                        Self::contract_impl(executor, graph, lib, head)
                    }
                    (NetworkLeaf::LibraryKey(l1), NetworkLeaf::LibraryKey(l2)) => {
                        let l1_inds: Vec<_> = graph
                            .graph
                            .iter_crown(nid1)
                            .filter_map(|i| match graph.graph[[&i]] {
                                NetworkEdge::Head => None,
                                NetworkEdge::Slot(s) => Some(s.aind),
                            })
                            .collect();

                        let l2_inds: Vec<_> = graph
                            .graph
                            .iter_crown(nid2)
                            .filter_map(|i| match graph.graph[[&i]] {
                                NetworkEdge::Head => None,
                                NetworkEdge::Slot(s) => Some(s.aind),
                            })
                            .collect();

                        let contracted = lib
                            .get(l1)?
                            .with_indices(&l1_inds)?
                            .contract(&lib.get(l2)?.with_indices(&l2_inds)?)?;
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);

                        graph.identify_nodes_without_self_edges(
                            &[nid1, nid2],
                            NetworkNode::Leaf(NetworkLeaf::LocalTensor(pos)),
                        );

                        Self::contract_impl(executor, graph, lib, head)
                    }
                },
                (a, b) => Err(TensorNetworkError::CannotContractEdgeBetween(
                    a.clone(),
                    b.clone(),
                )),
            }
        } else {
            Ok(graph)
        }
    }
}

#[cfg(feature = "shadowing")]
// pub mod levels;
#[cfg(feature = "shadowing")]
pub mod symbolica_interop;

#[cfg(feature = "shadowing")]
#[cfg(test)]
// mod shadowing_tests;
#[cfg(test)]
mod tests;

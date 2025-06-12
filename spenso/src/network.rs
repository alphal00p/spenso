use bincode::{Decode, Encode};

use graph::{
    NAdd, NMul, NetworkEdge, NetworkGraph, NetworkLeaf, NetworkLeafWithInds, NetworkNode, NetworkOp,
};
use linnet::half_edge::NodeIndex;
use serde::{Deserialize, Serialize};

use library::{Library, LibraryError};

use crate::algebra::algebraic_traits::{One, Zero};
use crate::algebra::ScalarMul;
use crate::contraction::Contract;
use crate::network::library::LibraryTensor;
use crate::structure::permuted::PermuteTensor;
// use crate::shadowing::Concretize;
use crate::structure::representation::LibrarySlot;
use crate::structure::StructureError;
use std::borrow::Cow;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::usize;
use store::{NetworkStore, TensorScalarStore, TensorScalarStoreMapping};
use thiserror::Error;
// use log::trace;

use crate::{
    contraction::ContractionError,
    structure::{CastStructure, HasStructure, ScalarTensor, TensorStructure},
};

// use anyhow::Result;

use std::{convert::Infallible, fmt::Debug};

#[derive(
    Debug, Clone, Serialize, Deserialize, bincode_trait_derive::Encode, bincode_trait_derive::Decode,
)]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
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
pub mod library;
pub mod set;
pub mod store;

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

impl<S: TensorScalarStore, K> MulAssign for Network<S, K> {
    fn mul_assign(&mut self, mut rhs: Self) {
        rhs.graph.shift_scalars(self.store.n_scalars());
        rhs.graph.shift_tensors(self.store.n_tensors());
        self.store.extend(rhs.store);

        self.graph *= rhs.graph;
    }
}

impl<T: TensorStructure, S, K> MulAssign<T> for Network<NetworkStore<T, S>, K> {
    fn mul_assign(&mut self, rhs: T) {
        *self *= Network::from_tensor(rhs);
    }
}

impl<T: TensorStructure, S, K> Mul<T> for Network<NetworkStore<T, S>, K> {
    type Output = Self;
    fn mul(self, other: T) -> Self::Output {
        let mut store = self.store;

        let mut other = Network::from_tensor(other);

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

impl<S: TensorScalarStore, K> AddAssign for Network<S, K> {
    fn add_assign(&mut self, mut rhs: Self) {
        rhs.graph.shift_scalars(self.store.n_scalars());
        rhs.graph.shift_tensors(self.store.n_tensors());
        self.store.extend(rhs.store);

        self.graph += rhs.graph;
    }
}

impl<T: TensorStructure, S, K> AddAssign<T> for Network<NetworkStore<T, S>, K> {
    fn add_assign(&mut self, rhs: T) {
        *self += Network::from_tensor(rhs);
    }
}

impl<T: TensorStructure, S, K> Add<T> for Network<NetworkStore<T, S>, K> {
    type Output = Self;
    fn add(mut self, other: T) -> Self::Output {
        self += other;
        self
    }
}

impl<T: TensorStructure, K> Add<i8> for Network<NetworkStore<T, i8>, K> {
    type Output = Self;
    fn add(mut self, other: i8) -> Self::Output {
        let mut other = Network::from_scalar(other);
        other.graph.shift_tensors(self.store.n_tensors());
        other.graph.shift_tensors(self.store.n_tensors());

        self.store.extend(other.store);
        Network {
            graph: self.graph + other.graph,
            store: self.store,
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

impl<S: TensorScalarStore, K> Neg for Network<S, K> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            store: self.store,
            graph: self.graph.neg(),
        }
    }
}

impl<S: TensorScalarStore, K> Sub for Network<S, K> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<S: TensorScalarStore, K> SubAssign for Network<S, K> {
    fn sub_assign(&mut self, mut rhs: Self) {
        rhs.graph.shift_scalars(self.store.n_scalars());
        rhs.graph.shift_tensors(self.store.n_tensors());
        self.store.extend(rhs.store);

        self.graph -= rhs.graph
    }
}

impl<T: TensorStructure, S, K> SubAssign<T> for Network<NetworkStore<T, S>, K> {
    fn sub_assign(&mut self, rhs: T) {
        *self -= Network::from_tensor(rhs)
    }
}

impl<T: TensorStructure, S, K> Sub<T> for Network<NetworkStore<T, S>, K> {
    type Output = Self;
    fn sub(mut self, other: T) -> Self::Output {
        self -= other;
        self
    }
}

impl<S: TensorScalarStore, K> Network<S, K> {
    pub fn from_scalar(scalar: S::Scalar) -> Self {
        let mut store = S::default();
        let id = store.add_scalar(scalar);
        Network {
            graph: NetworkGraph::scalar(id),
            store,
        }
    }

    pub fn merge_ops(&mut self)
    where
        K: Clone,
    {
        self.graph.merge_ops();
    }

    pub fn from_tensor(tensor: S::Tensor) -> Self
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
    #[error("Slot edge to prod node")]
    SlotEdgeToProdNode,
    #[error("Slot edge to scalar node")]
    SlotEdgeToScalarNode,
    #[error("More than one neg")]
    MoreThanOneNeg,
    #[error("Childless neg")]
    ChildlessNeg,
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
}

pub enum ExecutionResult<T> {
    One,
    Zero,
    Val(T),
}

impl<T: TensorStructure, S, K: Display, Str: TensorScalarStore<Tensor = T, Scalar = S>>
    Network<Str, K>
{
    pub fn result(
        &self,
    ) -> Result<ExecutionResult<TensorOrScalarOrKey<&T, &S, &K>>, TensorNetworkError<K>> {
        let (node, graph_slots) = self.graph.result()?;

        match node {
            NetworkNode::Leaf(l) => match l {
                NetworkLeaf::LibraryKey(k) => Ok(ExecutionResult::Val(TensorOrScalarOrKey::Key {
                    key: k,
                    graph_slots,
                })),
                NetworkLeaf::LocalTensor(t) => {
                    Ok(ExecutionResult::Val(TensorOrScalarOrKey::Tensor {
                        tensor: self.store.get_tensor(*t),
                        graph_slots,
                    }))
                }
                NetworkLeaf::Scalar(t) => Ok(ExecutionResult::Val(TensorOrScalarOrKey::Scalar(
                    self.store.get_scalar(*t),
                ))),
            },
            NetworkNode::Op(o) => match o {
                NetworkOp::Neg => Err(TensorNetworkError::InvalidResultNode(NetworkNode::Op(
                    NetworkOp::Neg,
                ))),
                NetworkOp::Product => Ok(ExecutionResult::One),
                NetworkOp::Sum => Ok(ExecutionResult::Zero),
            },
        }
    }

    pub fn result_tensor<'a, L: Library<T::Structure, Key = K>>(
        &'a self,
        lib: &L,
    ) -> Result<ExecutionResult<Cow<'a, T>>, TensorNetworkError<K>>
    where
        S: 'a,
        T: Clone + ScalarTensor + HasStructure,
        K: Display,
        L::Value: TensorStructure<Indexed = T> + Clone,
        for<'b> &'b S: Into<T::Scalar>,
    {
        Ok(match self.result()? {
            ExecutionResult::One => ExecutionResult::One,
            ExecutionResult::Zero => ExecutionResult::Zero,
            ExecutionResult::Val(v) => ExecutionResult::Val(match v {
                TensorOrScalarOrKey::Tensor { tensor, .. } => Cow::Borrowed(tensor),
                TensorOrScalarOrKey::Scalar(s) => Cow::Owned(T::new_scalar(s.into())),
                TensorOrScalarOrKey::Key { key, graph_slots } => {
                    let inds: Vec<_> = graph_slots.iter().map(|a| a.aind).collect();
                    let less = lib.get(key)?.into_owned().reindex(&inds)?.structure;

                    Cow::Owned(less)
                }
            }),
        })
    }

    pub fn result_scalar<'a>(&'a self) -> Result<ExecutionResult<Cow<'a, S>>, TensorNetworkError<K>>
    where
        T: Clone + ScalarTensor + 'a,
        T::Scalar: Into<S>,
        K: Display,
        S: Clone,
    {
        Ok(match self.result()? {
            ExecutionResult::One => ExecutionResult::One,
            ExecutionResult::Zero => ExecutionResult::Zero,
            ExecutionResult::Val(v) => ExecutionResult::Val(match v {
                TensorOrScalarOrKey::Tensor { tensor: t, .. } => Cow::Owned(
                    t.clone()
                        .scalar()
                        .ok_or(TensorNetworkError::NoScalar)?
                        .into(),
                ),
                TensorOrScalarOrKey::Scalar(s) => Cow::Borrowed(s),
                TensorOrScalarOrKey::Key { .. } => return Err(TensorNetworkError::NoScalar),
            }),
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

impl<T, S, K> Network<NetworkStore<T, S>, K> {
    pub fn dot_display_impl(
        &self,
        scalar_disp: impl Fn(&S) -> String,
        library_disp: impl Fn(&K) -> Option<String>,
        tensor_disp: impl Fn(&T) -> String,
    ) -> std::string::String {
        self.graph.graph.dot_impl(
            &self.graph.graph.full_filter(),
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
                        // if let Ok(v) = lib.get(l) {
                        Some(format!("label = \"L:{}\"", library_disp(&l)?))
                        // } else {
                        // None
                        // }
                    }
                    NetworkLeaf::LocalTensor(l) => Some(format!(
                        "label = \"T:{}\"",
                        tensor_disp(self.store.get_tensor(*l))
                    )),
                    NetworkLeaf::Scalar(s) => Some(format!(
                        "label = \"S:{}\"",
                        scalar_disp(self.store.get_scalar(*s))
                    )),
                },
                NetworkNode::Op(o) => Some(format!("label = \"{o}\"")),
            },
        )
        // self.graph.dot()
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
    ) -> Result<(NetworkGraph<K>, bool), TensorNetworkError<K>>
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

pub struct Nsteps {
    steps: usize,
}

pub struct Steps<const N: usize> {}
pub struct StepsDebug<const N: usize> {}

impl<const N: usize, E, L, K> ExecutionStrategy<E, L, K> for StepsDebug<N>
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
        for _ in 0..N {
            // find the *one* ready op
            if let Some((extracted_graph, op)) = graph.extract_next_ready_op() {
                println!(
                    "Extracted_graph: {}",
                    extracted_graph.dot_impl(
                        |s| s.to_string(),
                        |s| "".to_string(),
                        |s| s.to_string()
                    )
                );
                println!(
                    "Graph: {}",
                    graph.dot_impl(|s| s.to_string(), |s| "".to_string(), |s| s.to_string())
                );
                // execute + splice
                let replacement = executor.execute::<C>(extracted_graph, lib, op)?;
                graph.splice_descendents_of(replacement);
            }
        }

        Ok(())
    }
}

impl<const N: usize, E, L, K> ExecutionStrategy<E, L, K> for Steps<N>
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
        for _ in 0..N {
            // find the *one* ready op
            if let Some((extracted_graph, op)) = graph.extract_next_ready_op() {
                // execute + splice
                let replacement = executor.execute::<C>(extracted_graph, lib, op)?;
                graph.splice_descendents_of(replacement);
            }
        }

        Ok(())
    }
}

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
            if let Some((extracted_graph, op)) = graph.extract_next_ready_op() {
                // execute + splice
                let replacement = executor.execute::<C>(extracted_graph, lib, op)?;
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
        K: Display + Clone,
        L: Library<S, Key = K, Value: LibraryTensor<WithIndices = Store::Tensor>> + Sync,
        Store: ExecuteOp<L, K>,
    {
        self.merge_ops();
        // println!("Hi");
        // Ok(())
        Strat::execute_all::<C>(&mut self.store, &mut self.graph, lib)
    }
}

impl<
        T: HasStructure
            + TensorStructure
            + Neg<Output = T>
            + Clone
            + Ref
            + for<'a> AddAssign<T::Ref<'a>>
            + for<'a> AddAssign<<L::Value as LibraryTensor>::WithIndices>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: Neg<Output = Sc>
            + for<'a> AddAssign<Sc::Ref<'a>>
            + Clone
            + for<'a> AddAssign<T::ScalarRef<'a>>
            + From<T::Scalar>
            + Ref,
        K: Display + Debug,
    > ExecuteOp<L, K> for NetworkStore<T, Sc>
where
    <L::Value as LibraryTensor>::WithIndices:
        PermuteTensor<Permuted = <L::Value as LibraryTensor>::WithIndices>,
{
    fn execute<C: ContractionStrategy<Self, L, K>>(
        &mut self,
        mut graph: NetworkGraph<K>,
        lib: &L,
        op: NetworkOp,
    ) -> Result<NetworkGraph<K>, TensorNetworkError<K>> {
        graph.sync_order();
        match op {
            NetworkOp::Neg => {
                let mut ops = graph.graph.iter_nodes().find(|(nid, n, d)| {
                    if let NetworkNode::Op(NetworkOp::Neg) = d {
                        true
                    } else {
                        false
                    }
                });

                let (opid, children, _) = ops.unwrap();

                let mut child = None;
                for c in children {
                    if let Some(id) = graph.graph.involved_node_id(c) {
                        if let NetworkNode::Leaf(l) = &graph.graph[id] {
                            if let Some(_) = child {
                                return Err(TensorNetworkError::MoreThanOneNeg);
                            } else {
                                child = Some((id, l));
                            }
                        }
                    }
                }
                if let Some((child_id, leaf)) = child {
                    let new_node = match leaf {
                        NetworkLeaf::Scalar(s) => {
                            let s = self.scalar[*s].clone().neg();
                            let pos = self.scalar.len();
                            self.scalar.push(s);

                            NetworkLeaf::Scalar(pos)
                        }
                        NetworkLeaf::LibraryKey(key) => {
                            let inds = graph.inds(child_id);
                            let t = T::from(
                                lib.get(key).unwrap().with_indices(&inds).unwrap().permute(),
                            )
                            .neg();
                            let pos = self.tensors.len();
                            self.tensors.push(t);
                            NetworkLeaf::LocalTensor(pos)
                        }
                        NetworkLeaf::LocalTensor(t) => {
                            let t = self.tensors[*t].clone().neg();
                            let pos = self.tensors.len();
                            self.tensors.push(t);
                            NetworkLeaf::LocalTensor(pos)
                        }
                    };
                    graph.identify_nodes_without_self_edges(
                        &[child_id, opid],
                        NetworkNode::Leaf(new_node),
                    );
                    Ok(graph)
                } else {
                    return Err(TensorNetworkError::ChildlessNeg);
                }
            }
            NetworkOp::Product => {
                let (graph, _) = C::contract(self, graph, lib)?;
                Ok(graph)
            }
            NetworkOp::Sum => {
                let mut op = None;
                let mut targets = Vec::new();
                let mut all_nodes = Vec::new();
                for (n, c, v) in graph.graph.iter_nodes() {
                    all_nodes.push(n);
                    if let NetworkNode::Op(NetworkOp::Sum) = v {
                        op = Some((n, c))
                    } else if let NetworkNode::Leaf(l) = &v {
                        targets.push((n, l));
                    }
                }

                let (nf, first) = &targets[0];

                let new_node = match first {
                    NetworkLeaf::Scalar(s) => {
                        let mut accumulator = self.scalar[*s].clone();

                        for (nid, t) in &targets[1..] {
                            match t {
                                NetworkLeaf::Scalar(s) => {
                                    accumulator += self.scalar[*s].refer();
                                }
                                NetworkLeaf::LocalTensor(t) => {
                                    if let Some(s) = self.tensors[*t].scalar_ref() {
                                        accumulator += s;
                                    } else {
                                        return Err(TensorNetworkError::NotAllScalars(
                                            "".to_string(),
                                        ));
                                    }
                                }
                                NetworkLeaf::LibraryKey { .. } => {
                                    return Err(TensorNetworkError::ScalarLibSum("".to_string()));
                                }
                            }
                        }

                        let pos = self.scalar.len();
                        self.scalar.push(accumulator);
                        NetworkLeaf::Scalar(pos)
                    }
                    NetworkLeaf::LocalTensor(t) => {
                        let mut accumulator = self.tensors[*t].clone();
                        if accumulator.is_scalar() {
                            let mut accumulator = Sc::from(accumulator.scalar().unwrap());

                            for (_, t) in &targets[1..] {
                                match t {
                                    NetworkLeaf::Scalar(s) => {
                                        accumulator += self.scalar[*s].refer();
                                    }
                                    NetworkLeaf::LocalTensor(t) => {
                                        if let Some(s) = self.tensors[*t].scalar_ref() {
                                            accumulator += s;
                                        } else {
                                            return Err(TensorNetworkError::NotAllScalars(
                                                "".to_string(),
                                            ));
                                        }
                                    }
                                    NetworkLeaf::LibraryKey { .. } => {
                                        return Err(TensorNetworkError::ScalarLibSum(
                                            "".to_string(),
                                        ));
                                    }
                                }
                            }

                            let pos = self.scalar.len();
                            self.scalar.push(accumulator);
                            NetworkLeaf::Scalar(pos)
                        } else {
                            for (nid, t) in &targets[1..] {
                                match t {
                                    NetworkLeaf::Scalar(_) => {
                                        return Err(TensorNetworkError::SumScalarTensor(
                                            "".to_string(),
                                        ))
                                    }
                                    NetworkLeaf::LocalTensor(t) => {
                                        accumulator += self.tensors[*t].refer();
                                    }
                                    NetworkLeaf::LibraryKey(key) => {
                                        let inds = graph.inds(*nid);
                                        let with_index =
                                            lib.get(key)?.with_indices(&inds)?.permute();
                                        accumulator += with_index;
                                    }
                                }
                            }

                            let pos = self.tensors.len();
                            self.tensors.push(accumulator);

                            NetworkLeaf::LocalTensor(pos)
                        }
                    }
                    NetworkLeaf::LibraryKey(key) => {
                        let inds = graph.inds(*nf);
                        let mut accumulator = T::from(lib.get(key)?.with_indices(&inds)?.permute());
                        for (nid, t) in &targets[1..] {
                            match t {
                                NetworkLeaf::Scalar(_) => {
                                    return Err(TensorNetworkError::SumScalarTensor("".to_string()))
                                }
                                NetworkLeaf::LocalTensor(t) => {
                                    accumulator += self.tensors[*t].refer();
                                }
                                NetworkLeaf::LibraryKey(key) => {
                                    let inds = graph.inds(*nid);
                                    let with_index = lib.get(key)?.with_indices(&inds)?.permute();
                                    accumulator += with_index;
                                }
                            }
                        }

                        let pos = self.tensors.len();
                        self.tensors.push(accumulator);

                        NetworkLeaf::LocalTensor(pos)
                    }
                };

                graph.identify_nodes_without_self_edges(&all_nodes, NetworkNode::Leaf(new_node));
                Ok(graph)
            }
        }
    }
}

pub struct SmallestDegree;

pub struct SmallestDegreeIter<const N: usize>;

pub struct ContractScalars;

pub struct SingleSmallestDegree<const D: bool>;
pub trait Ref {
    type Ref<'a>
    where
        Self: 'a;
    fn refer<'a>(&'a self) -> Self::Ref<'a>;
}

impl Ref for f64 {
    type Ref<'a>
        = &'a f64
    where
        Self: 'a;

    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        self
    }
}

impl<
        T: HasStructure
            + TensorStructure
            + Clone
            + Contract<LCM = T>
            + ScalarMul<Sc, Output = T>
            + Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: for<'a> MulAssign<Sc::Ref<'a>>
            + Clone
            + for<'a> MulAssign<T::ScalarRef<'a>>
            + From<T::Scalar>
            + Ref,
        K: Display + Debug + Clone,
    > ContractionStrategy<NetworkStore<T, Sc>, L, K> for ContractScalars
where
    <L::Value as LibraryTensor>::WithIndices: Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
        + ScalarMul<Sc, Output = T>
        + PermuteTensor<Permuted = <L::Value as LibraryTensor>::WithIndices>,
{
    fn contract(
        executor: &mut NetworkStore<T, Sc>,
        mut graph: NetworkGraph<K>,
        lib: &L,
    ) -> Result<(NetworkGraph<K>, bool), TensorNetworkError<K>>
    where
        K: Display,
    {
        graph.sync_order();
        let mut other = None;
        let mut include_head = true;
        let mut head = None;
        let (mut scalars, mut scalar_nodes): (Vec<_>, Vec<_>) = graph
            .graph
            .iter_nodes()
            .filter_map(|(nid, _, c)| {
                if let NetworkNode::Leaf(l) = c {
                    match l {
                        NetworkLeaf::Scalar(i) => Some((*i, nid)),
                        _ => {
                            if other.is_none() {
                                other = Some(nid);
                            } else {
                                include_head = false;
                            }
                            None
                        }
                    }
                } else {
                    if let NetworkNode::Op(NetworkOp::Product) = c {
                        if head.is_some() {
                            panic!("multiple heads")
                        }
                        head = Some(nid);
                    }
                    None
                }
            })
            .collect();

        if let Some(f) = scalars.pop() {
            let mut acc = executor.scalar[f].clone();

            for si in scalars {
                acc *= executor.scalar[si].refer();
            }

            let new_node = if include_head {
                if let Some(head) = head {
                    scalar_nodes.push(head);
                }
                if let Some(other) = other {
                    scalar_nodes.push(other);
                    if let NetworkNode::Leaf(l) = &graph.graph[other] {
                        match l {
                            NetworkLeaf::LocalTensor(l) => {
                                let a = executor.tensors[*l].scalar_mul(&acc).unwrap();
                                let pos = executor.tensors.len();
                                executor.tensors.push(a);
                                NetworkLeaf::LocalTensor(pos)
                            }
                            NetworkLeaf::LibraryKey(l) => {
                                let l_inds: Vec<_> = graph.inds(other);
                                let a =
                                    lib.get(l)?.with_indices(&l_inds)?.scalar_mul(&acc).unwrap();

                                let pos = executor.tensors.len();
                                executor.tensors.push(a);
                                NetworkLeaf::LocalTensor(pos)
                            }
                            _ => {
                                unreachable!("aa")
                            }
                        }
                    } else {
                        unreachable!("aa")
                    }
                } else {
                    let pos = executor.scalar.len();
                    executor.scalar.push(acc);
                    NetworkLeaf::Scalar(pos)
                }
            } else {
                let pos = executor.scalar.len();
                executor.scalar.push(acc);
                NetworkLeaf::Scalar(pos)
            };

            if !include_head {
                graph.identify_nodes_without_self_edges_merge_heads(
                    &scalar_nodes,
                    NetworkNode::Leaf(new_node),
                );
            } else {
                graph.identify_nodes_without_self_edges(&scalar_nodes, NetworkNode::Leaf(new_node));
            }
            Ok((graph, true))
        } else {
            let mut didsmth = false;
            if include_head {
                if let Some(other) = other {
                    if let Some(head) = head {
                        let v = graph.graph[other].clone();
                        graph.identify_nodes_without_self_edges(&[head, other], v);
                        didsmth = true;
                    }
                }
            }
            Ok((graph, didsmth))
        }
    }
}

impl<
        T: HasStructure
            + TensorStructure
            + Clone
            + Contract<LCM = T>
            + ScalarMul<Sc, Output = T>
            + Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: for<'a> MulAssign<Sc::Ref<'a>>
            + Clone
            + for<'a> MulAssign<T::ScalarRef<'a>>
            + From<T::Scalar>
            + Ref,
        K: Display + Debug + Clone,
    > ContractionStrategy<NetworkStore<T, Sc>, L, K> for SmallestDegree
where
    <L::Value as LibraryTensor>::WithIndices: Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
        + ScalarMul<Sc, Output = T>
        + PermuteTensor<Permuted = <L::Value as LibraryTensor>::WithIndices>,
    <<L::Value as LibraryTensor>::WithIndices as HasStructure>::Structure: Display,
    T::Structure: Display,
{
    fn contract(
        executor: &mut NetworkStore<T, Sc>,
        graph: NetworkGraph<K>,
        lib: &L,
    ) -> Result<(NetworkGraph<K>, bool), TensorNetworkError<K>>
    where
        K: Display,
    {
        let (mut graph, mut didsmth) = ContractScalars::contract(executor, graph, lib)?;

        while {
            let (newgraph, smth) = SingleSmallestDegree::<false>::contract(executor, graph, lib)?;
            graph = newgraph;
            smth
        } {
            didsmth |= true
        }

        let (graph, _) = ContractScalars::contract(executor, graph, lib)?;

        Ok((graph, didsmth))
    }
}

impl<
        T: HasStructure
            + TensorStructure
            + Clone
            + Contract<LCM = T>
            + ScalarMul<Sc, Output = T>
            + Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: for<'a> MulAssign<Sc::Ref<'a>>
            + Clone
            + for<'a> MulAssign<T::ScalarRef<'a>>
            + From<T::Scalar>
            + Ref,
        K: Display + Debug + Clone,
        const N: usize,
    > ContractionStrategy<NetworkStore<T, Sc>, L, K> for SmallestDegreeIter<N>
where
    <L::Value as LibraryTensor>::WithIndices: Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
        + ScalarMul<Sc, Output = T>
        + PermuteTensor<Permuted = <L::Value as LibraryTensor>::WithIndices>,
    <<L::Value as LibraryTensor>::WithIndices as HasStructure>::Structure: Display,
    T::Structure: Display,
{
    fn contract(
        executor: &mut NetworkStore<T, Sc>,
        graph: NetworkGraph<K>,
        lib: &L,
    ) -> Result<(NetworkGraph<K>, bool), TensorNetworkError<K>>
    where
        K: Display,
    {
        let (mut graph, mut didsmth) = ContractScalars::contract(executor, graph, lib)?;

        for _ in 0..N {
            let (newgraph, smth) = SingleSmallestDegree::<false>::contract(executor, graph, lib)?;
            graph = newgraph;
            didsmth |= smth;
        }

        let (graph, _) = ContractScalars::contract(executor, graph, lib)?;

        Ok((graph, didsmth))
    }
}

impl<
        T: HasStructure
            + TensorStructure
            + Clone
            + Contract<LCM = T>
            + ScalarMul<Sc, Output = T>
            + Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
            + From<<L::Value as LibraryTensor>::WithIndices>,
        L: Library<T::Structure, Key = K, Value: LibraryTensor>,
        Sc: for<'a> MulAssign<Sc::Ref<'a>>
            + Clone
            + for<'a> MulAssign<T::ScalarRef<'a>>
            + From<T::Scalar>
            + Ref,
        K: Display + Debug + Clone,
        const D: bool,
    > ContractionStrategy<NetworkStore<T, Sc>, L, K> for SingleSmallestDegree<D>
where
    <L::Value as LibraryTensor>::WithIndices: Contract<<L::Value as LibraryTensor>::WithIndices, LCM = T>
        + ScalarMul<Sc, Output = T>
        + PermuteTensor<Permuted = <L::Value as LibraryTensor>::WithIndices>,
    <<L::Value as LibraryTensor>::WithIndices as HasStructure>::Structure: Display,
    T::Structure: Display,
{
    fn contract(
        executor: &mut NetworkStore<T, Sc>,
        mut graph: NetworkGraph<K>,
        lib: &L,
    ) -> Result<(NetworkGraph<K>, bool), TensorNetworkError<K>>
    where
        K: Display,
    {
        graph.sync_order();
        if D {
            println!("Contracting {}", graph.dot());
        }

        let mut last_tensor = None;
        let edge_to_contract = graph
            .graph
            .iter_nodes()
            .filter(|(_, _, d)| d.is_tensor())
            .filter_map(|(nid1, a, n1)| {
                let mut degree = 0;
                let mut first = None;
                for h in a {
                    if graph.graph[[&h]].is_slot() && graph.graph.inv(h) != h {
                        first = Some(h); //only contract slot hedges
                        degree += 1
                    }
                }

                let nid2 = if degree == 0 {
                    //no internal slots to contract
                    // contract with last tensor (give max  weight  so only happens when there are no internal slots)
                    degree = i32::MAX;
                    if let Some(last_tensor) = last_tensor {
                        last_tensor
                    } else {
                        last_tensor = Some(nid1);
                        return None;
                    }
                } else {
                    graph.graph.involved_node_id(first?)?
                };

                let n2 = &graph.graph[nid2];

                last_tensor = Some(nid1);

                Some((degree, nid1, n1, nid2, n2))
            })
            .min_by_key(|(degree, _, _, _, _)| *degree);

        if let Some((_, nid1, n1, nid2, n2)) = edge_to_contract {
            if D {
                println!("Contracting {} with {}", nid1, nid2);
            }
            let new_node = match (n1, n2) {
                (NetworkNode::Leaf(_), NetworkNode::Op(NetworkOp::Product))
                | (NetworkNode::Op(NetworkOp::Product), NetworkNode::Leaf(_)) => {
                    return Err(TensorNetworkError::SlotEdgeToProdNode)
                }
                (NetworkNode::Leaf(l1), NetworkNode::Leaf(l2)) => match (l1, l2) {
                    (NetworkLeaf::Scalar(_), _) | (_, NetworkLeaf::Scalar(_)) => {
                        return Err(TensorNetworkError::SlotEdgeToScalarNode)
                    }

                    (NetworkLeaf::LocalTensor(l1), NetworkLeaf::LocalTensor(l2)) => {
                        if D {
                            let st1 = executor.tensors[*l1].structure();
                            let st2 = executor.tensors[*l2].structure();

                            println!("Contracting {} with {}", st1, st2);
                        }

                        let contracted = executor.tensors[*l1].contract(&executor.tensors[*l2])?;

                        if D {
                            println!("Obtained {}", contracted.structure());
                        }
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);

                        NetworkLeaf::LocalTensor(pos)
                    }
                    (NetworkLeaf::LibraryKey(l1), NetworkLeaf::LocalTensor(l2)) => {
                        let l1_inds: Vec<_> = graph.inds(nid1);

                        let l1 = lib.get(l1)?.with_indices(&l1_inds)?.permute();
                        if D {
                            let st1 = l1.structure();
                            let st2 = executor.tensors[*l2].structure();
                            println!("Contracting {} with {}", st1, st2);
                        }

                        let contracted = executor.tensors[*l2].contract(&l1)?;
                        if D {
                            println!("Obtained {}", contracted.structure());
                        }
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);
                        NetworkLeaf::LocalTensor(pos)
                    }

                    (NetworkLeaf::LocalTensor(l2), NetworkLeaf::LibraryKey(l1)) => {
                        let l1_inds: Vec<_> = graph.inds(nid2);
                        let l1 = lib.get(l1)?.with_indices(&l1_inds)?.permute();
                        if D {
                            let st1 = l1.structure();
                            let st2 = executor.tensors[*l2].structure();
                            println!("Contracting {} with {}", st2, st1);
                        }

                        let contracted = executor.tensors[*l2].contract(&l1)?;
                        if D {
                            println!("Obtained {}", contracted.structure());
                        }
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);

                        NetworkLeaf::LocalTensor(pos)
                    }
                    (NetworkLeaf::LibraryKey(l1), NetworkLeaf::LibraryKey(l2)) => {
                        let l1_inds: Vec<_> = graph.inds(nid1);
                        let l1 = lib.get(l1)?.with_indices(&l1_inds)?;

                        let l2_inds: Vec<_> = graph.inds(nid2);
                        let l2 = lib.get(l2)?.with_indices(&l2_inds)?;
                        if D {
                            let st1 = l1.structure();
                            let st2 = l2.structure();
                            println!("Contracting {} with {}", st2, st1);
                        }

                        let contracted = l1.contract(&l2)?;
                        if D {
                            println!("Obtained {}", contracted.structure());
                        }
                        let pos = executor.tensors.len();
                        executor.tensors.push(contracted);

                        NetworkLeaf::LocalTensor(pos)
                    }
                },
                (a, b) => {
                    return Err(TensorNetworkError::CannotContractEdgeBetween(
                        a.clone(),
                        b.clone(),
                    ))
                }
            };
            graph.identify_nodes_without_self_edges_merge_heads(
                &[nid1, nid2],
                NetworkNode::Leaf(new_node),
            );
            Ok((graph, true))
        } else {
            Ok((graph, false))
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

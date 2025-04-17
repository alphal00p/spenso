use bincode::{Decode, Encode};

use graph::{NAdd, NMul, NetworkGraph, NetworkLeaf, NetworkLeafWithInds, NetworkNode, NetworkOp};
use serde::{Deserialize, Serialize};

use tensor_library::{Library, LibraryError};

use crate::algebraic_traits::{One, Zero};
use crate::network::tensor_library::LibraryTensor;
use crate::structure::representation::LibrarySlot;
use crate::structure::StructureError;
use std::borrow::Cow;
use std::fmt::Display;
use std::ops::{Add, Mul, Neg};
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
    shadowing::Shadowable,
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

    pub fn tensor(tensor: S::Tensor) -> Self
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
    #[error("not all scalars: {0}")]
    NotAllScalars(String),
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
            NetworkNode::Port => Err(TensorNetworkError::InvalidResultNode(NetworkNode::Port)),
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

impl<S, K> Network<S, K> {
    pub fn dot(&self) -> std::string::String {
        self.graph.dot()
    }
}

// use log::trace;

#[cfg(feature = "shadowing")]
impl<
        'a,
        Sc,
        S,
        T: HasStructure<Structure = S> + TensorStructure,
        K: Clone + Display,
        Str: TensorScalarStore<Tensor = T, Scalar = Sc> + Clone,
    > Network<Str, K>
where
    Sc: for<'r> TryFrom<AtomView<'r>> + Clone,
    TensorNetworkError<K>: for<'r> From<<Sc as TryFrom<AtomView<'r>>>::Error>,
    S: TryFrom<FunView<'a>> + TensorStructure + Clone + HasName,
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
    LibraryRep: From<<S::Slot as IsAbstractSlot>::R>,
    T: Clone + From<ParamTensor<S>>,
{
    pub fn try_from_view<Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: AtomView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>> {
        match value {
            AtomView::Mul(m) => Self::try_from_mul(m, library),
            AtomView::Fun(f) => Self::try_from_fun(f, library),
            AtomView::Add(a) => Self::try_from_add(a, library),
            AtomView::Pow(p) => Self::try_from_pow(p, library),
            a => Ok(Network::scalar(a.try_into()?)),
        }
    }

    fn try_from_mul<Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: MulView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>> {
        let mut iter = value.iter();

        let first = Self::try_from_view(iter.next().unwrap(), library)?;

        let rest: Result<Vec<_>, _> = iter.map(|a| Self::try_from_view(a, library)).collect();

        Ok(first.n_mul(rest?))
    }

    fn try_from_fun<Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: FunView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>> {
        let s: Result<S, _> = value.try_into();

        if let Ok(s) = s {
            let shell = s.clone().to_shell();

            let inds = s.external_indices();
            let key = library.key_for_structure(s.into());
            let explicit = if let Some(k) = key {
                library.get(&k).ok().map(|t| t.with_indices(&inds).unwrap())
            } else {
                None
            };

            Ok(Self::tensor(if let Some(e) = explicit {
                e
            } else {
                ParamTensor::param(shell.expanded_shadow()?.into()).into()
            }))
        } else {
            Ok(Self::scalar(
                value.as_view().try_into().map_err(Into::into)?,
            ))
        }
    }

    fn try_from_pow<Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: PowView<'a>,
        library: &Lib,
    ) -> std::result::Result<Self, TensorNetworkError<K>> {
        let (base, exp) = value.get_base_exp();

        if let Ok(n) = i64::try_from(exp) {
            if n < 0 {
                return Ok(Self::scalar(value.as_view().try_into()?));
            }
            if n == 0 {
                let one = Atom::new_num(1);
                return Ok(Self::scalar(one.as_view().try_into()?));
            } else if n == 1 {
                return Self::try_from_view(base, library);
            }
            let net = Self::try_from_view(base, library)?;
            let cloned_net = net.clone();

            Ok(net.n_mul((0..n).map(|_| cloned_net.clone())))
        } else {
            Ok(Self::scalar(value.as_view().try_into()?))
        }
    }

    fn try_from_add<Lib: Library<S, Key = K, Value: LibraryTensor<WithIndices = T>>>(
        value: AddView<'a>,
        library: &Lib,
    ) -> Result<Self, TensorNetworkError<K>> {
        let mut iter = value.iter();

        let first = Self::try_from_view(iter.next().unwrap(), library)?;

        let rest: Result<Vec<_>, _> = iter.map(|a| Self::try_from_view(a, library)).collect();

        Ok(first.n_add(rest?))
    }
}
// use log::trace;

pub trait ContractionStrategy<E, L, K>: Sized {
    fn contract(executor: &mut E, graph: NetworkGraph<K>, lib: &L) -> NetworkGraph<K>;
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
    );
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
    ) {
        while {
            // find the *one* ready op
            if let Some((extracted_graph, op, leaves)) = graph.extract_next_ready_op() {
                // execute + splice
                let replacement = executor.execute::<C>(extracted_graph, lib, op, &leaves);
                graph.splice_descendents_of(replacement);
                true
            } else {
                false
            }
        } {}
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
    ) -> NetworkGraph<K>;
}

impl<S, Store: TensorScalarStore, K> Network<Store, K>
where
    Store::Tensor: HasStructure<Structure = S>,
{
    pub fn execute<Strat: ExecutionStrategy<Store, L, K>, C: ContractionStrategy<Store, L, K>, L>(
        &mut self,
        lib: &L,
    ) where
        L: Library<S, Key = K, Value: LibraryTensor<WithIndices = Store::Tensor>> + Sync,
        Store: ExecuteOp<L, K>,
    {
        Strat::execute_all::<C>(&mut self.store, &mut self.graph, lib);
    }
}

impl<
        T: HasStructure + TensorStructure + Neg<Output = T> + Clone,
        L: Library<T::Structure, Key = K, Value: LibraryTensor<WithIndices = T>>,
        Sc: Neg<Output = Sc>
            + Add<Output = Sc>
            + Mul<Output = Sc>
            + Clone
            + Add<T::Scalar, Output = Sc>
            + Mul<T::Scalar, Output = Sc>,
        K: Display + Debug,
    > ExecuteOp<L, K> for NetworkStore<T, Sc>
{
    fn execute<C: ContractionStrategy<Self, L, K>>(
        &mut self,
        graph: NetworkGraph<K>,
        lib: &L,
        op: NetworkOp,
        targets: &[NetworkLeafWithInds<K>],
    ) -> NetworkGraph<K> {
        match op {
            NetworkOp::Neg => {
                debug_assert_eq!(targets.len(), 1);
                let target = &targets[0];

                match target {
                    NetworkLeafWithInds::Scalar(s) => {
                        let s = self.scalar[*s].clone().neg();
                        let pos = self.scalar.len();
                        self.scalar.push(s);
                        NetworkGraph::scalar(pos)
                    }
                    NetworkLeafWithInds::LibraryKey { key, inds } => {
                        let t = lib.get(key).unwrap().with_indices(inds).unwrap().neg();
                        let pos = self.tensors.len();
                        let node = NetworkLeaf::LocalTensor(pos);
                        self.tensors.push(t);
                        NetworkGraph::tensor(&self.tensors[pos], node)
                    }
                    NetworkLeafWithInds::LocalTensor(t) => {
                        let t = self.tensors[*t].clone().neg();
                        let pos = self.tensors.len();
                        let node = NetworkLeaf::LocalTensor(pos);
                        self.tensors.push(t);
                        NetworkGraph::tensor(&self.tensors[pos], node)
                    }
                }
            }
            NetworkOp::Product => C::contract(self, graph, lib),
            NetworkOp::Sum => {
                let first = &targets[0];

                match first {
                    NetworkLeafWithInds::Scalar(s) => {}
                    NetworkLeafWithInds::LocalTensor(t) => {}
                    NetworkLeafWithInds::LibraryKey { key, inds } => {}
                }

                graph
            }
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

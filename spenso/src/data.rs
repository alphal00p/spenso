use crate::{
    complex::Complex,
    contraction::IsZero,
    iterators::{DenseTensorLinearIterator, IteratableTensor, SparseTensorLinearIterator},
    structure::{
        concrete_index::{ConcreteIndex, ExpandedIndex, FlatIndex},
        CastStructure, HasName, HasStructure, ScalarStructure, ScalarTensor, StructureContract,
        TensorStructure, TracksCount, VecStructure,
    },
    upgrading_arithmetic::{TryFromUpgrade, TrySmallestUpgrade},
};

use crate::structure::abstract_index::AbstractIndex;
use crate::structure::dimension::Dimension;
use crate::structure::representation::Representation;
use crate::structure::slot::IsAbstractSlot;
use crate::structure::StructureError;
use delegate::delegate;

#[cfg(feature = "shadowing")]
use crate::{
    parametric::{ExpandedCoefficent, FlatCoefficent, TensorCoefficient},
    shadowing::{ShadowMapping, Shadowable},
    symbolica_utils::{atomic_expanded_label_id, IntoArgs, IntoSymbol},
};

use bincode::{Decode, Encode};
use derive_more::From;
use enum_try_as_inner::EnumTryAsInner;
use indexmap::IndexMap;
use num::Zero;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fmt::{Display, LowerExp},
    hash::Hash,
    ops::{Index, IndexMut},
};

#[cfg(feature = "shadowing")]
use symbolica::{atom::Atom, atom::Symbol};
pub trait DataIterator<T> {
    type FlatIter<'a>: Iterator<Item = (FlatIndex, &'a T)>
    where
        Self: 'a,
        T: 'a;

    fn flat_iter(&self) -> Self::FlatIter<'_>;
}

impl<T, I> DataIterator<T> for SparseTensor<T, I> {
    type FlatIter<'a>
        = SparseTensorLinearIterator<'a, T>
    where
        I: 'a,
        T: 'a;

    fn flat_iter(&self) -> Self::FlatIter<'_> {
        SparseTensorLinearIterator::new(self)
    }
}

impl<T, I: TensorStructure> DataIterator<T> for DenseTensor<T, I> {
    type FlatIter<'a>
        = DenseTensorLinearIterator<'a, T, I>
    where
        I: 'a,
        T: 'a;

    fn flat_iter(&self) -> Self::FlatIter<'_> {
        DenseTensorLinearIterator::new(self)
    }
}

#[allow(dead_code)]
trait Settable {
    type SetData;
    fn set(&mut self, index: usize, data: Self::SetData);
}

impl<T> Settable for Vec<T> {
    type SetData = T;
    fn set(&mut self, index: usize, data: T) {
        self[index] = data;
    }
}

impl<T> Settable for HashMap<usize, T> {
    type SetData = T;
    fn set(&mut self, index: usize, data: T) {
        self.insert(index, data);
    }
}

/// Trait for getting the data of a tensor
pub trait HasTensorData: HasStructure {
    type Data: Clone;
    // type Storage: Settable<SetData = Self::Data>;
    /// Returns all the data in the tensor, withouth any structure
    fn data(&self) -> Vec<Self::Data>;

    /// Returns all the indices of the tensor, the order of the indices is the same as the order of the data
    fn indices(&self) -> Vec<ExpandedIndex>;

    /// Returns a hashmap of the data, with the (expanded) indices as keys
    fn hashmap(&self) -> IndexMap<ExpandedIndex, Self::Data>;

    /// Returns a hashmap of the data, with the the shadowed indices as keys
    #[cfg(feature = "shadowing")]
    fn symhashmap(&self, name: Symbol, args: &[Atom]) -> HashMap<Atom, Self::Data>;

    // fn map(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self;
}

/// Trait for setting the data of a tensor
pub trait SetTensorData {
    type SetData;
    /// Set the data at the given indices, returns an error if the indices are out of bounds
    ///
    /// # Errors
    ///
    /// Forwards the error from [`TensorStructure::verify_indices`]
    ///
    fn set(&mut self, indices: &[ConcreteIndex], value: Self::SetData) -> Result<()>;

    fn set_flat(&mut self, index: FlatIndex, value: Self::SetData) -> Result<()>;
}

/// Trait for getting the data of a tensor
pub trait GetTensorData {
    type GetDataRef<'a>
    where
        Self: 'a;

    type GetDataRefMut<'a>
    where
        Self: 'a;

    type GetDataOwned;

    fn get_ref<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataRef<'_>>;

    fn get_ref_linear(&self, index: FlatIndex) -> Option<Self::GetDataRef<'_>>;

    fn get_mut_linear(&mut self, index: FlatIndex) -> Option<Self::GetDataRefMut<'_>>;

    fn get_owned<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone;

    fn get_owned_linear(&self, index: FlatIndex) -> Option<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone;
}

/// Sparse data tensor, generic on storage type `T`, and structure type `I`.
///
/// Stores data in a hashmap of usize, using ahash's hashmap.
/// The usize key is the flattened index of the corresponding position in the dense tensor
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Encode, Decode)]
pub struct SparseTensor<T, I = VecStructure> {
    // #[bincode(with_serde)]
    pub elements: std::collections::HashMap<FlatIndex, T>,
    pub structure: I,
}

impl<T, S> crate::network::Ref for SparseTensor<T, S> {
    type Ref<'a>
        = &'a SparseTensor<T, S>
    where
        Self: 'a;

    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        self
    }
}

impl<T, S> TensorStructure for SparseTensor<T, S>
where
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = SparseTensor<T, S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        self.map_structure_result(|s| s.reindex(indices))
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tensor<Store, Structure> {
    pub elements: Store,
    pub structure: Structure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SparseStore<T> {
    pub elements: HashMap<FlatIndex, T>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseStore<T> {
    pub elements: Vec<T>,
}

impl<T: Hash, I: Hash> Hash for SparseTensor<T, I> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut vecel: Vec<_> = self.elements.iter().collect();

        vecel.sort_by(|(i, _), (j, _)| i.cmp(j));

        vecel.hash(state);
        self.structure.hash(state);
    }
}

pub trait CastData<O: HasTensorData>: HasStructure<Structure = O::Structure> {
    fn cast_data(self) -> O;
}

impl<T, U: From<T> + Clone, I: TensorStructure + Clone> CastData<SparseTensor<U, I>>
    for SparseTensor<T, I>
{
    fn cast_data(self) -> SparseTensor<U, I> {
        SparseTensor {
            elements: self
                .elements
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            structure: self.structure,
        }
    }
}

impl<T, S: TensorStructure, O: From<S> + TensorStructure> CastStructure<SparseTensor<T, O>>
    for SparseTensor<T, S>
{
    fn cast_structure(self) -> SparseTensor<T, O> {
        SparseTensor {
            elements: self.elements,
            structure: self.structure.into(),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure> Shadowable for SparseTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure, R> ShadowMapping<R> for SparseTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
    R: From<T>,
{
    // fn shadow_with_map<'a, C>(
    //     &self,
    //     fn_map: &mut symbolica::evaluate::FunctionMap<'a, R>,
    //     index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> C,
    // ) -> Option<ParamTensor<Self::Structure>>
    // where
    //     C: TensorCoefficient,
    // {
    //     let mut data = vec![];
    //     for (i, d) in self.flat_iter() {
    //         let labeled_coef = index_to_atom(self.structure(), i).to_atom().unwrap();
    //         fn_map.add_constant(labeled_coef.clone().into(), d.clone().into());
    //         data.push(labeled_coef);
    //     }

    //     let param = DenseTensor {
    //         data,
    //         structure: self.structure.clone(),
    //     };

    //     Some(ParamTensor::Param(param.into()))
    // }

    fn append_map<U>(
        &self,
        fn_map: &mut symbolica::evaluate::FunctionMap<R>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> U,
    ) where
        U: TensorCoefficient,
    {
        for (i, d) in self.flat_iter() {
            let labeled_coef = index_to_atom(self.structure(), i).to_atom().unwrap();
            fn_map.add_constant(labeled_coef.clone(), d.clone().into());
        }
    }
}

impl<T, S> HasName for SparseTensor<T, S>
where
    S: HasName + TensorStructure,
{
    type Args = S::Args;
    type Name = S::Name;
    fn name(&self) -> Option<Self::Name> {
        self.structure.name()
    }

    fn args(&self) -> Option<Self::Args> {
        self.structure.args()
    }

    fn set_name(&mut self, name: Self::Name) {
        self.structure.set_name(name);
    }

    #[cfg(feature = "shadowing")]
    fn expanded_coef(&self, id: FlatIndex) -> ExpandedCoefficent<Self::Args>
    where
        Self: TensorStructure,
        Self::Name: IntoSymbol,
        Self::Args: IntoArgs,
    {
        self.structure.expanded_coef(id)
    }

    #[cfg(feature = "shadowing")]
    fn flat_coef(&self, id: FlatIndex) -> FlatCoefficent<Self::Args>
    where
        Self: TensorStructure,
        Self::Name: IntoSymbol,
        Self::Args: IntoArgs,
    {
        self.structure.flat_coef(id)
    }
}

impl<T, I> HasTensorData for SparseTensor<T, I>
where
    T: Clone,
    I: TensorStructure + Clone,
{
    type Data = T;
    // type Storage = AHashMap<usize, T>;

    fn data(&self) -> Vec<T> {
        let mut d: Vec<(FlatIndex, T)> = self.iter_flat().map(|(i, v)| (i, v.clone())).collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.into_iter().map(|(_, v)| v).collect()
    }

    fn indices(&self) -> Vec<ExpandedIndex> {
        self.elements
            .keys()
            .map(|k| self.expanded_index(*k).unwrap())
            .collect()
    }

    fn hashmap(&self) -> IndexMap<ExpandedIndex, T> {
        let mut hashmap = IndexMap::new();
        for (k, v) in self.iter_expanded() {
            hashmap.insert(k.clone(), v.clone());
        }
        hashmap
    }
    #[cfg(feature = "shadowing")]
    fn symhashmap(&self, name: Symbol, args: &[Atom]) -> HashMap<Atom, T> {
        let mut hashmap = HashMap::new();

        for (k, v) in &self.elements {
            hashmap.insert(
                atomic_expanded_label_id(&self.expanded_index(*k).unwrap(), name, args),
                v.clone(),
            );
        }
        hashmap
    }
}

impl<T, S: TensorStructure> SparseTensor<T, S> {
    pub fn map_structure<S2>(self, f: impl Fn(S) -> S2) -> SparseTensor<T, S2>
    where
        S2: TensorStructure,
    {
        SparseTensor {
            elements: self.elements,
            structure: f(self.structure),
        }
    }

    pub fn map_structure_fallible<S2, E>(
        self,
        f: impl Fn(S) -> Result<S2, E>,
    ) -> Result<SparseTensor<T, S2>, E>
    where
        S2: TensorStructure,
    {
        Ok(SparseTensor {
            elements: self.elements,
            structure: f(self.structure)?,
        })
    }

    pub fn map_data_ref<U>(&self, f: impl Fn(&T) -> U) -> SparseTensor<U, S>
    where
        // T: Clone,
        // U: Clone,
        S: Clone,
    {
        let elements = self.flat_iter().map(|(k, v)| (k, f(v))).collect();
        SparseTensor {
            elements,
            structure: self.structure.clone(),
        }
    }

    pub fn map_data_ref_result<U, E>(
        &self,
        f: impl Fn(&T) -> Result<U, E>,
    ) -> Result<SparseTensor<U, S>, E>
    where
        // T: Clone,
        // U: Clone,
        S: Clone,
    {
        let elements: Result<HashMap<FlatIndex, _>, E> = self
            .flat_iter()
            .map(|(k, v)| f(v).map(|v| (k, v)))
            .collect();
        Ok(SparseTensor {
            elements: elements?,
            structure: self.structure.clone(),
        })
    }

    pub fn map_data<U>(self, f: impl Fn(T) -> U) -> SparseTensor<U, S> {
        let elements = self.elements.into_iter().map(|(k, v)| (k, f(v))).collect();
        SparseTensor {
            elements,
            structure: self.structure,
        }
    }

    pub fn map_data_ref_mut<U>(&mut self, mut f: impl FnMut(&mut T) -> U) -> SparseTensor<U, S>
    where
        // T: Clone,
        // U: Clone,
        S: Clone,
    {
        let elements = self.elements.iter_mut().map(|(k, v)| (*k, f(v))).collect();
        SparseTensor {
            elements,
            structure: self.structure.clone(),
        }
    }

    pub fn map_data_mut(&mut self, f: impl FnMut(&mut T))
    where
        // T: Clone,
        // U: Clone,
        S: Clone,
    {
        self.elements.values_mut().for_each(f);
    }
}

impl<T, S> Display for SparseTensor<T, S>
where
    T: Display,
    S: TensorStructure,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for (i, v) in self.iter_expanded() {
            s.push_str(&format!("{}: {}\n", i, v));
        }
        write!(f, "{}", s)
    }
}

impl<T, U, S> TryFromUpgrade<SparseTensor<U, S>> for SparseTensor<T, S>
where
    U: TrySmallestUpgrade<T, LCM = T>,
    S: TensorStructure + Clone,
    T: Clone,
{
    fn try_from_upgrade(data: &SparseTensor<U, S>) -> Option<SparseTensor<T, S>> {
        data.try_upgrade().map(Cow::into_owned)
    }
}

// #[derive(Error, Debug)]
// pub enum DataTensorError {
//     #[error("Data length does not match shape")]
//     DataLengthMismatch,
// }
use anyhow::{anyhow, Result};

impl<T, I> SetTensorData for SparseTensor<T, I>
where
    I: TensorStructure,
{
    type SetData = T;
    /// falible set method, returns an error if the indices are out of bounds.
    /// Does not check if the inserted value is zero.
    fn set(&mut self, indices: &[ConcreteIndex], value: T) -> Result<()> {
        self.verify_indices(indices)?;
        self.elements
            .insert(self.flat_index(indices).unwrap(), value);
        Ok(())
    }

    /// falible set given a flat index, returns an error if the indices are out of bounds.
    fn set_flat(&mut self, index: FlatIndex, value: T) -> Result<()> {
        if index >= self.size()?.into() {
            return Err(anyhow!("Index out of bounds"));
        }
        self.elements.insert(index, value);
        Ok(())
    }
}
impl<T, I> GetTensorData for SparseTensor<T, I>
where
    I: TensorStructure,
{
    type GetDataRef<'a>
        = &'a T
    where
        Self: 'a;
    type GetDataRefMut<'a>
        = &'a mut T
    where
        Self: 'a;

    type GetDataOwned = T;
    fn get_ref<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<&T> {
        if let Ok(idx) = self.flat_index(&indices) {
            self.elements
                .get(&idx)
                .ok_or(anyhow!("No elements at that spot"))
        } else if self.structure.is_scalar() && indices.as_ref().is_empty() {
            self.elements
                .iter()
                .next()
                .map(|(_, v)| v)
                .ok_or(anyhow!("err"))
        } else {
            Err(anyhow!("Index out of bounds"))
        }
    }

    fn get_ref_linear(&self, index: FlatIndex) -> Option<&T> {
        self.elements.get(&index)
    }

    fn get_mut_linear(&mut self, index: FlatIndex) -> Option<&mut T> {
        self.elements.get_mut(&index)
    }

    fn get_owned<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataOwned>
    where
        T: Clone,
    {
        self.get_ref(indices).cloned()
    }

    fn get_owned_linear(&self, index: FlatIndex) -> Option<Self::GetDataOwned>
    where
        T: Clone,
    {
        self.elements.get(&index).cloned()
    }
}

impl<T, I> ScalarTensor for SparseTensor<T, I>
where
    I: TensorStructure + ScalarStructure,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        let mut elements = HashMap::new();
        elements.insert(0.into(), scalar);
        SparseTensor {
            elements,
            structure: I::scalar_structure(),
        }
    }
}

impl<T, I> HasStructure for SparseTensor<T, I>
where
    I: TensorStructure,
{
    type Scalar = T;
    type ScalarRef<'a>
        = &'a T
    where
        Self: 'a;
    type Structure = I;
    type Store<S>
        = SparseTensor<T, S>
    where
        S: TensorStructure;

    fn map_structure<O: TensorStructure>(
        self,
        f: impl FnOnce(Self::Structure) -> O,
    ) -> Self::Store<O> {
        SparseTensor {
            structure: f(self.structure),
            elements: self.elements,
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl FnOnce(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        Ok(SparseTensor {
            structure: f(self.structure)?,
            elements: self.elements,
        })
    }
    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        SparseTensor {
            elements: self.elements,
            structure: f(self.structure),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }

    fn scalar(mut self) -> Option<Self::Scalar> {
        if self.structure.is_scalar() {
            self.elements.drain().next().map(|(_, v)| v)
        } else {
            None
        }
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        if self.structure.is_scalar() {
            self.elements.values().next()
        } else {
            None
        }
    }
}

impl<T, I> TracksCount for SparseTensor<T, I>
where
    I: TracksCount + TensorStructure,
{
    fn contractions_num(&self) -> usize {
        self.structure.contractions_num()
    }
}

impl<U, I> SparseTensor<U, I>
where
    I: TensorStructure + Clone,
{
    fn try_upgrade<T>(&self) -> Option<Cow<SparseTensor<U::LCM, I>>>
    where
        U: TrySmallestUpgrade<T>,
        U::LCM: Clone,
    {
        let structure = self.structure.clone();
        let elements: Option<HashMap<FlatIndex, U::LCM>> = self
            .elements
            .iter()
            .map(|(k, v)| match v.try_upgrade() {
                Some(Cow::Owned(u)) => Some((*k, u)),
                Some(Cow::Borrowed(u)) => Some((*k, u.clone())),
                None => None,
            })
            .collect();
        Some(Cow::Owned(SparseTensor {
            elements: elements?,
            structure,
        }))
    }
}

impl<T, I> SparseTensor<T, I>
where
    I: TensorStructure,
{
    /// Create a new empty sparse tensor with the given structure
    pub fn empty(structure: I) -> Self {
        SparseTensor {
            elements: HashMap::default(),
            structure,
        }
    }

    /// Checks if there is a value at the given indices
    pub fn is_empty_at_expanded(&self, indices: &[ConcreteIndex]) -> bool {
        !self
            .elements
            .contains_key(&self.flat_index(indices).unwrap())
    }

    pub fn is_empty_at_flat(&self, index: FlatIndex) -> bool {
        !self.elements.contains_key(&index)
    }
    /// Calulates how dense the tensor is, i.e. the ratio of non-zero elements to total elements
    pub fn density(&self) -> f64 {
        f64::from(self.elements.len() as u32) / f64::from(self.size().unwrap() as u32)
    }

    /// Converts the sparse tensor to a dense tensor, with the same structure
    pub fn to_dense(&self) -> DenseTensor<T, I>
    where
        T: Clone + Default,
        I: Clone,
    {
        self.to_dense_with(&T::default())
    }

    pub fn to_dense_with(&self, zero: &T) -> DenseTensor<T, I>
    where
        T: Clone,
        I: Clone,
    {
        let mut dense = DenseTensor::fill(self.structure.clone(), zero.clone());
        for (indices, value) in self.elements.iter() {
            let _ = dense.set_flat(*indices, value.clone());
        }
        dense
    }

    /// fallible smart set method, returns an error if the indices are out of bounds.
    /// If the value is zero, it removes the element at the given indices.
    pub fn smart_set(&mut self, indices: &[ConcreteIndex], value: T) -> Result<()>
    where
        T: IsZero,
    {
        self.verify_indices(indices)?;
        if value.is_zero() {
            _ = self.elements.remove(&self.flat_index(indices).unwrap());
            return Ok(());
        }
        self.elements
            .insert(self.flat_index(indices).unwrap(), value);
        Ok(())
    }

    /// Generates a new sparse tensor from the given data and structure
    pub fn from_data(
        data: impl IntoIterator<Item = (Vec<ConcreteIndex>, T)>,
        structure: I,
    ) -> Result<Self> {
        let mut elements = HashMap::default();
        for (index, value) in data {
            if index.len() != structure.order() {
                return Err(anyhow!("Mismatched order"));
            }
            elements.insert(structure.flat_index(&index).unwrap(), value);
        }

        Ok(SparseTensor {
            elements,
            structure,
        })
    }

    /// fallible smart get method, returns an error if the indices are out of bounds.
    /// If the index is in the bTree return the value, else return zero.
    pub fn smart_get(&self, indices: &[ConcreteIndex]) -> Result<Cow<T>>
    where
        T: Default + Clone,
    {
        self.verify_indices(indices)?;
        // if the index is in the bTree return the value, else return default, lazily allocating the default
        Ok(
            match self.elements.get(&self.flat_index(indices).unwrap()) {
                Some(value) => Cow::Borrowed(value),
                None => Cow::Owned(T::default()),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Hash, Eq, Encode, Decode)]
pub struct DenseTensor<T, S = VecStructure> {
    pub data: Vec<T>,
    pub structure: S,
}

impl<T, S> crate::network::Ref for DenseTensor<T, S> {
    type Ref<'a>
        = &'a DenseTensor<T, S>
    where
        Self: 'a;

    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        self
    }
}

impl<T, S> TensorStructure for DenseTensor<T, S>
where
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = DenseTensor<T, S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        self.map_structure_result(|s| s.reindex(indices))
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl<T, U: From<T> + Clone, I: TensorStructure + Clone> CastData<DenseTensor<U, I>>
    for DenseTensor<T, I>
{
    fn cast_data(self) -> DenseTensor<U, I> {
        DenseTensor {
            data: self.data.into_iter().map(|v| v.into()).collect(),
            structure: self.structure,
        }
    }
}

impl<T, S: TensorStructure, O: From<S> + TensorStructure> CastStructure<DenseTensor<T, O>>
    for DenseTensor<T, S>
{
    fn cast_structure(self) -> DenseTensor<T, O> {
        DenseTensor {
            data: self.data,
            structure: self.structure.into(),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure> Shadowable for DenseTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure, R> ShadowMapping<R> for DenseTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
    R: From<T>,
{
    // fn shadow_with_map<'a, U>(
    //     &self,
    //     fn_map: &mut symbolica::evaluate::FunctionMap<'a, R>,
    //     index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> U,
    // ) -> Option<ParamTensor<Self::Structure>>
    // where
    //     U: TensorCoefficient,
    //     R: From<T>,
    // {
    //     let mut data = vec![];
    //     for (i, d) in self.flat_iter() {
    //         let labeled_coef = index_to_atom(self.structure(), i).to_atom().unwrap();
    //         fn_map.add_constant(labeled_coef.clone().into(), d.clone().into());
    //         data.push(labeled_coef);
    //     }

    //     let param = DenseTensor {
    //         data,
    //         structure: self.structure.clone(),
    //     };

    //     Some(ParamTensor::Param(param.into()))
    // }

    fn append_map<U>(
        &self,
        fn_map: &mut symbolica::evaluate::FunctionMap<R>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> U,
    ) where
        U: TensorCoefficient,
    {
        for (i, d) in self.flat_iter() {
            let labeled_coef = index_to_atom(self.structure(), i).to_atom().unwrap();
            fn_map.add_constant(labeled_coef.clone(), d.clone().into());
        }
    }
}

impl<T: Display, I: TensorStructure> std::fmt::Display for DenseTensor<T, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for (i, v) in self.iter_expanded() {
            s.push_str(&format!("{}: {}\n", i, v));
        }
        write!(f, "{}", s)
    }
}

impl<T: LowerExp, I: TensorStructure> std::fmt::LowerExp for DenseTensor<T, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for (i, v) in self.iter_expanded() {
            s.push_str(&format!("{}: {:+e}\n", i, v));
        }
        write!(f, "{}", s)
    }
}

impl<T, I> Index<FlatIndex> for DenseTensor<T, I> {
    type Output = T;

    fn index(&self, index: FlatIndex) -> &Self::Output {
        let i: usize = index.into();
        &self.data[i]
    }
}

impl<T, I> IndexMut<FlatIndex> for DenseTensor<T, I> {
    fn index_mut(&mut self, index: FlatIndex) -> &mut Self::Output {
        let i: usize = index.into();
        &mut self.data[i]
    }
}

impl<T, I> HasStructure for DenseTensor<T, I>
where
    I: TensorStructure,
{
    type Scalar = T;
    type ScalarRef<'a>
        = &'a T
    where
        Self: 'a;
    type Structure = I;
    type Store<S>
        = DenseTensor<T, S>
    where
        S: TensorStructure;

    fn map_structure<O: TensorStructure>(
        self,
        f: impl FnOnce(Self::Structure) -> O,
    ) -> Self::Store<O> {
        DenseTensor {
            structure: f(self.structure),
            data: self.data,
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl FnOnce(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        Ok(DenseTensor {
            structure: f(self.structure)?,
            data: self.data,
        })
    }

    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        DenseTensor {
            data: self.data,
            structure: f(self.structure),
        }
    }
    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }
    fn scalar(mut self) -> Option<Self::Scalar> {
        if self.is_scalar() {
            self.data.drain(0..).next()
        } else {
            None
        }
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        if self.is_scalar() {
            self.data.first()
        } else {
            None
        }
    }
}

impl<T, I> ScalarTensor for DenseTensor<T, I>
where
    I: TensorStructure + ScalarStructure,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        DenseTensor {
            data: vec![scalar],
            structure: I::scalar_structure(),
        }
    }
}

impl<T, I> TracksCount for DenseTensor<T, I>
where
    I: TracksCount + TensorStructure,
{
    fn contractions_num(&self) -> usize {
        self.structure.contractions_num()
    }
}

impl<T, S> HasName for DenseTensor<T, S>
where
    S: HasName + TensorStructure,
{
    type Args = S::Args;
    type Name = S::Name;
    fn name(&self) -> Option<Self::Name> {
        self.structure.name()
    }

    fn args(&self) -> Option<Self::Args> {
        self.structure.args()
    }

    fn set_name(&mut self, name: Self::Name) {
        self.structure.set_name(name);
    }
}

impl<T, I> DenseTensor<T, I>
where
    I: TensorStructure,
{
    pub fn default(structure: I) -> Self
    where
        T: Default + Clone,
    {
        Self::fill(structure, T::default())
    }

    pub fn fill(structure: I, fill: T) -> Self
    where
        T: Clone,
    {
        let length = if structure.is_scalar() {
            1
        } else {
            structure.size().unwrap()
        };
        DenseTensor {
            data: vec![fill; length],
            structure,
        }
    }
}

impl<T, S: TensorStructure> DenseTensor<T, S> {
    pub fn repeat(structure: S, r: T) -> Self
    where
        T: Clone,
    {
        let length = if structure.is_scalar() {
            1
        } else {
            structure.size().unwrap()
        };
        DenseTensor {
            data: vec![r; length],
            structure,
        }
    }
}

impl<T: Zero + Clone, I> DenseTensor<T, I>
where
    I: TensorStructure,
{
    pub fn zero(structure: I) -> Self {
        let length = if structure.is_scalar() {
            1
        } else {
            structure.size().unwrap()
        };
        DenseTensor {
            data: vec![T::zero(); length],
            structure,
        }
    }
}

// impl<T,S:TensorStructure> DenseTensor<T,S>{}

impl<U, I> DenseTensor<U, I>
where
    I: TensorStructure + Clone,
{
    pub fn try_upgrade<T>(&self) -> Option<Cow<DenseTensor<U::LCM, I>>>
    where
        U: TrySmallestUpgrade<T>,
        U::LCM: Clone,
    {
        let structure = self.structure.clone();
        let data: Option<Vec<U::LCM>> = self
            .data
            .iter()
            .map(|v| match v.try_upgrade() {
                Some(Cow::Owned(u)) => Some(u),
                Some(Cow::Borrowed(u)) => Some(u.clone()),
                None => None,
            })
            .collect();
        Some(Cow::Owned(DenseTensor {
            data: data?,
            structure,
        }))
    }
}

impl<T: Clone, I> DenseTensor<T, I>
where
    I: TensorStructure,
{
    /// Generates a new dense tensor from the given data and structure
    pub fn from_data(data: Vec<T>, structure: I) -> Result<Self> {
        if data.len() != structure.size()? && !(data.len() == 1 && structure.is_scalar()) {
            return Err(anyhow!("Data length does not match shape"));
        }
        Ok(DenseTensor { data, structure })
    }

    pub fn cast<U>(&self) -> DenseTensor<U, I>
    where
        U: Clone + From<T>,
        I: Clone,
    {
        let data = self.data.iter().map(|x| x.clone().into()).collect();
        DenseTensor {
            data,
            structure: self.structure.clone(),
        }
    }

    /// Generates a new dense tensor from the given data and structure, truncating the data if it is too long with respect to the structure
    pub fn from_data_coerced(data: &[T], structure: I) -> Result<Self> {
        if data.len() < structure.size()? {
            return Err(anyhow!("Data length is too small"));
        }
        let mut data = data.to_vec();
        if structure.is_scalar() {
            data.truncate(1);
        } else {
            data.truncate(structure.size()?);
        }
        Ok(DenseTensor { data, structure })
    }
}

impl<T, U, S> TryFromUpgrade<DenseTensor<U, S>> for DenseTensor<T, S>
where
    U: TrySmallestUpgrade<T, LCM = T>,
    S: TensorStructure + Clone,
    T: Clone,
{
    fn try_from_upgrade(data: &DenseTensor<U, S>) -> Option<DenseTensor<T, S>> {
        data.try_upgrade().map(Cow::into_owned)
    }
}

impl<T, I> DenseTensor<T, I>
where
    I: TensorStructure + Clone,
{
    /// converts the dense tensor to a sparse tensor, with the same structure
    pub fn to_sparse(&self) -> SparseTensor<T, I>
    where
        T: Clone + Default + PartialEq,
    {
        let mut sparse = SparseTensor::empty(self.structure.clone());
        for (i, value) in self.iter_expanded() {
            if *value != T::default() {
                let _ = sparse.set(&i, value.clone());
            }
        }
        sparse
    }
}
// why no specialization? :(
// impl<T, U> From<DenseTensor<U>> for DenseTensor<T>
// where
//     U: Into<T>,
// {
//     fn from(other: DenseTensor<U>) -> Self {
//         let data = other.data.into_iter().map(|x| x.into()).collect();
//         DenseTensor {
//             data,
//             structure: other.structure,
//         }
//     }
// }

impl<T, I> DenseTensor<T, I>
where
    I: Clone + TensorStructure,
{
    pub fn convert_to<U>(&self) -> DenseTensor<U, I>
    where
        U: for<'a> From<&'a T>,
    {
        let data = self.data.iter().map(|x| x.into()).collect();
        DenseTensor {
            data,
            structure: self.structure.clone(),
        }
    }
}

impl<T, I> SparseTensor<T, I>
where
    I: Clone + TensorStructure,
{
    pub fn convert_to<U>(&self) -> SparseTensor<U, I>
    where
        U: for<'a> From<&'a T>,
    {
        let elements = self.elements.iter().map(|(k, v)| (*k, v.into())).collect();
        SparseTensor {
            elements,
            structure: self.structure.clone(),
        }
    }
}

impl<T, I> HasTensorData for DenseTensor<T, I>
where
    T: Clone,
    I: TensorStructure + Clone,
{
    type Data = T;
    fn data(&self) -> Vec<T> {
        self.data.clone()
    }

    fn indices(&self) -> Vec<ExpandedIndex> {
        let mut indices = Vec::new();
        for i in 0..self.size().unwrap() {
            indices.push(self.expanded_index(i.into()).unwrap());
        }
        indices
    }

    fn hashmap(&self) -> IndexMap<ExpandedIndex, T> {
        let mut hashmap = IndexMap::new();
        for (k, v) in self.iter_expanded() {
            hashmap.insert(k.clone(), v.clone());
        }
        hashmap
    }
    #[cfg(feature = "shadowing")]
    fn symhashmap(&self, name: Symbol, args: &[Atom]) -> HashMap<Atom, T> {
        let mut hashmap = HashMap::new();

        for (k, v) in self.iter_expanded() {
            hashmap.insert(atomic_expanded_label_id(&k, name, args), v.clone());
        }
        hashmap
    }
}

impl<T, I> SetTensorData for DenseTensor<T, I>
where
    I: TensorStructure,
{
    type SetData = T;
    fn set(&mut self, indices: &[ConcreteIndex], value: T) -> Result<()> {
        self.verify_indices(indices)?;
        let idx = self.flat_index(indices);
        if let Ok(i) = idx {
            self[i] = value;
        }
        Ok(())
    }

    fn set_flat(&mut self, index: FlatIndex, value: T) -> Result<()> {
        if index < self.size()?.into() {
            self[index] = value;
        } else {
            return Err(anyhow!("Index out of bounds"));
        }
        Ok(())
    }
}

impl<T, I> GetTensorData for DenseTensor<T, I>
where
    I: TensorStructure,
{
    type GetDataRef<'a>
        = &'a T
    where
        Self: 'a;
    type GetDataRefMut<'a>
        = &'a mut T
    where
        Self: 'a;

    type GetDataOwned = T;
    fn get_ref_linear(&self, index: FlatIndex) -> Option<&T> {
        let i: usize = index.into();
        self.data.get(i)
    }

    fn get_ref<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<&T> {
        if let Ok(idx) = self.flat_index(&indices) {
            Ok(&self[idx])
        } else if self.structure.is_scalar() && indices.as_ref().is_empty() {
            Ok(&self.data[0])
        } else {
            Err(anyhow!("Index out of bounds"))
        }
    }

    fn get_mut_linear(&mut self, index: FlatIndex) -> Option<&mut T> {
        let i: usize = index.into();
        self.data.get_mut(i)
    }

    fn get_owned<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        self.get_ref(indices).cloned()
    }

    fn get_owned_linear(&self, index: FlatIndex) -> Option<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        self.get_ref_linear(index).cloned()
    }
}

/// Enum for storing either a dense or a sparse tensor, with the same structure
#[derive(
    Debug, Clone, EnumTryAsInner, Serialize, Deserialize, From, Hash, PartialEq, Eq, Encode, Decode,
)]
#[derive_err(Debug)]
pub enum DataTensor<T, I = VecStructure> {
    Dense(DenseTensor<T, I>),
    Sparse(SparseTensor<T, I>),
}

impl<T, S> crate::network::Ref for DataTensor<T, S> {
    type Ref<'a>
        = &'a DataTensor<T, S>
    where
        Self: 'a;

    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        self
    }
}

pub trait SparseOrDense {
    fn to_sparse(self) -> Self;
    fn to_dense(self) -> Self;
}

impl<T: Clone, U: From<T> + Clone, I: TensorStructure + Clone> CastData<DataTensor<U, I>>
    for DataTensor<T, I>
{
    fn cast_data(self) -> DataTensor<U, I> {
        match self {
            Self::Dense(d) => DataTensor::Dense(d.cast_data()),

            Self::Sparse(d) => DataTensor::Sparse(d.cast_data()),
        }
    }
}

impl<T: Clone, S: TensorStructure, O: From<S> + TensorStructure> CastStructure<DataTensor<T, O>>
    for DataTensor<T, S>
{
    fn cast_structure(self) -> DataTensor<T, O> {
        match self {
            DataTensor::Dense(d) => DataTensor::Dense(d.cast_structure()),
            DataTensor::Sparse(d) => DataTensor::Sparse(d.cast_structure()),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure> Shadowable for DataTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure, R> ShadowMapping<R> for DataTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
    R: From<T>,
{
    // fn shadow_with_map<'a, U>(
    //     &'a self,
    //     fn_map: &mut symbolica::evaluate::FunctionMap<'a, R>,
    //     index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> U,
    // ) -> Option<ParamTensor<Self::Structure>>
    // where
    //     U: TensorCoefficient,
    // {
    //     match self {
    //         DataTensor::Dense(d) => d.shadow_with_map(fn_map, index_to_atom),
    //         DataTensor::Sparse(s) => s.shadow_with_map(fn_map, index_to_atom),
    //     }
    // }

    fn append_map<U>(
        &self,
        fn_map: &mut symbolica::evaluate::FunctionMap<R>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> U,
    ) where
        U: TensorCoefficient,
    {
        match self {
            DataTensor::Dense(d) => d.append_map(fn_map, index_to_atom),
            DataTensor::Sparse(s) => s.append_map(fn_map, index_to_atom),
        }
    }
}

impl<T, I> DataTensor<T, I>
where
    I: TensorStructure + Clone,
{
    pub fn actual_size(&self) -> usize {
        match self {
            DataTensor::Dense(d) => d.data.len(),
            DataTensor::Sparse(s) => s.elements.len(),
        }
    }
    pub fn to_bare_sparse(self) -> SparseTensor<T, I>
    where
        T: Clone + Default + PartialEq,
    {
        match self {
            DataTensor::Dense(d) => d.to_sparse(),
            DataTensor::Sparse(s) => s,
        }
    }

    pub fn to_bare_dense(self) -> DenseTensor<T, I>
    where
        T: Clone + Default + PartialEq,
    {
        match self {
            DataTensor::Dense(d) => d,
            DataTensor::Sparse(s) => s.to_dense(),
        }
    }
}

impl<T, I> SparseOrDense for DataTensor<T, I>
where
    I: TensorStructure + Clone,
    T: Clone + Default + PartialEq,
{
    fn to_dense(self) -> Self {
        DataTensor::Dense(self.to_bare_dense())
    }

    fn to_sparse(self) -> Self {
        DataTensor::Sparse(self.to_bare_sparse())
    }
}

impl<T, I> HasTensorData for DataTensor<T, I>
where
    I: TensorStructure + Clone,
    T: Clone,
{
    type Data = T;
    fn data(&self) -> Vec<T> {
        match self {
            DataTensor::Dense(d) => d.data(),
            DataTensor::Sparse(s) => s.data(),
        }
    }

    fn indices(&self) -> Vec<ExpandedIndex> {
        match self {
            DataTensor::Dense(d) => d.indices(),
            DataTensor::Sparse(s) => s.indices(),
        }
    }

    fn hashmap(&self) -> IndexMap<ExpandedIndex, T> {
        match self {
            DataTensor::Dense(d) => d.hashmap(),
            DataTensor::Sparse(s) => s.hashmap(),
        }
    }
    #[cfg(feature = "shadowing")]
    fn symhashmap(&self, name: Symbol, args: &[Atom]) -> HashMap<Atom, T> {
        match self {
            DataTensor::Dense(d) => d.symhashmap(name, args),
            DataTensor::Sparse(s) => s.symhashmap(name, args),
        }
    }
}

impl<T, S> TensorStructure for DataTensor<T, S>
where
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = DataTensor<T, S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        self.map_structure_result(|s| s.reindex(indices))
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl<T, I> HasStructure for DataTensor<T, I>
where
    I: TensorStructure,
{
    type Scalar = T;
    type ScalarRef<'a>
        = &'a T
    where
        Self: 'a;
    type Structure = I;

    type Store<S>
        = DataTensor<T, S>
    where
        S: TensorStructure;

    fn map_structure<O: TensorStructure>(self, f: impl Fn(Self::Structure) -> O) -> Self::Store<O> {
        match self {
            DataTensor::Dense(d) => DataTensor::Dense(d.map_structure(f)),
            DataTensor::Sparse(s) => DataTensor::Sparse(s.map_structure(f)),
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl Fn(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        Ok(match self {
            DataTensor::Dense(d) => match d.map_structure_result(f) {
                Ok(d) => DataTensor::Dense(d),
                Err(e) => return Err(e),
            },
            DataTensor::Sparse(s) => match s.map_structure_result(f) {
                Ok(s) => DataTensor::Sparse(s),
                Err(e) => return Err(e),
            },
        })
    }

    fn structure(&self) -> &Self::Structure {
        match self {
            DataTensor::Dense(d) => d.structure(),
            DataTensor::Sparse(s) => s.structure(),
        }
    }
    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            DataTensor::Dense(d) => d.mut_structure(),
            DataTensor::Sparse(s) => s.mut_structure(),
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        match self {
            DataTensor::Dense(d) => d.scalar(),
            DataTensor::Sparse(s) => s.scalar(),
        }
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        match self {
            DataTensor::Dense(d) => d.scalar_ref(),
            DataTensor::Sparse(s) => s.scalar_ref(),
        }
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        match self {
            DataTensor::Dense(d) => DataTensor::Dense(d.map_same_structure(f)),
            DataTensor::Sparse(s) => DataTensor::Sparse(s.map_same_structure(f)),
        }
    }
}

impl<T, I> ScalarTensor for DataTensor<T, I>
where
    I: TensorStructure + ScalarStructure,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        DataTensor::Dense(DenseTensor::new_scalar(scalar))
    }
}

impl<T: Display, S: TensorStructure> Display for DataTensor<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataTensor::Dense(d) => write!(f, "{}", d),
            DataTensor::Sparse(s) => write!(f, "{}", s),
        }
    }
}

impl<T, I> TracksCount for DataTensor<T, I>
where
    I: TracksCount,
    T: Clone,
    I: TensorStructure,
{
    fn contractions_num(&self) -> usize {
        match self {
            DataTensor::Dense(d) => d.contractions_num(),
            DataTensor::Sparse(s) => s.contractions_num(),
        }
    }
}

impl<T, S> HasName for DataTensor<T, S>
where
    S: HasName + TensorStructure,
{
    type Args = S::Args;
    type Name = S::Name;
    fn name(&self) -> Option<Self::Name> {
        match self {
            DataTensor::Dense(d) => d.name(),
            DataTensor::Sparse(s) => s.name(),
        }
    }

    fn args(&self) -> Option<Self::Args> {
        match self {
            DataTensor::Dense(d) => d.args(),
            DataTensor::Sparse(s) => s.args(),
        }
    }

    fn set_name(&mut self, name: Self::Name) {
        match self {
            DataTensor::Dense(d) => d.set_name(name),
            DataTensor::Sparse(s) => s.set_name(name),
        }
    }
}
impl<U, I> DataTensor<U, I>
where
    I: TensorStructure + Clone,
{
    pub fn try_upgrade<T>(&self) -> Option<Cow<DataTensor<U::LCM, I>>>
    where
        U: TrySmallestUpgrade<T>,
        U::LCM: Clone,
    {
        match self {
            DataTensor::Dense(d) => d
                .try_upgrade()
                .map(|x| Cow::Owned(DataTensor::Dense(x.into_owned()))),
            DataTensor::Sparse(s) => s
                .try_upgrade()
                .map(|x| Cow::Owned(DataTensor::Sparse(x.into_owned()))),
        }
    }
}

impl<T, U, S> TryFromUpgrade<DataTensor<U, S>> for DataTensor<T, S>
where
    U: TrySmallestUpgrade<T, LCM = T>,
    S: TensorStructure + Clone,
    T: Clone,
{
    fn try_from_upgrade(data: &DataTensor<U, S>) -> Option<DataTensor<T, S>> {
        data.try_upgrade().map(Cow::into_owned)
    }
}

impl<T, S> SetTensorData for DataTensor<T, S>
where
    S: TensorStructure,
{
    type SetData = T;

    fn set(&mut self, indices: &[ConcreteIndex], value: Self::SetData) -> Result<()> {
        match self {
            DataTensor::Dense(d) => d.set(indices, value),
            DataTensor::Sparse(s) => s.set(indices, value),
        }
    }

    fn set_flat(&mut self, index: FlatIndex, value: Self::SetData) -> Result<()> {
        match self {
            DataTensor::Dense(d) => d.set_flat(index, value),
            DataTensor::Sparse(s) => s.set_flat(index, value),
        }
    }
}

impl<T, S> GetTensorData for DataTensor<T, S>
where
    S: TensorStructure,
{
    type GetDataRef<'a>
        = &'a T
    where
        Self: 'a;
    type GetDataRefMut<'a>
        = &'a mut T
    where
        Self: 'a;

    type GetDataOwned = T;
    fn get_ref<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<&T> {
        match self {
            DataTensor::Dense(d) => d.get_ref(indices),
            DataTensor::Sparse(s) => s.get_ref(indices),
        }
    }

    fn get_ref_linear(&self, index: FlatIndex) -> Option<&T> {
        match self {
            DataTensor::Dense(d) => d.get_ref_linear(index),
            DataTensor::Sparse(s) => s.get_ref_linear(index),
        }
    }

    fn get_mut_linear(&mut self, index: FlatIndex) -> Option<&mut T> {
        match self {
            DataTensor::Dense(d) => d.get_mut_linear(index),
            DataTensor::Sparse(s) => s.get_mut_linear(index),
        }
    }

    fn get_owned<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        self.get_ref(indices).cloned()
    }

    fn get_owned_linear(&self, index: FlatIndex) -> Option<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        self.get_ref_linear(index).cloned()
    }
}

/// Enum for a datatensor with specific numeric data type, generic on the structure type `I`
#[derive(Debug, Clone, EnumTryAsInner, Serialize, Deserialize)]
#[derive_err(Debug)]
pub enum NumTensor<T: TensorStructure = VecStructure> {
    Float(DataTensor<f64, T>),
    Complex(DataTensor<Complex<f64>, T>),
}

impl<S> TensorStructure for NumTensor<S>
where
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = NumTensor<S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        self.map_structure_result(|s| s.reindex(indices))
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl<T> HasStructure for NumTensor<T>
where
    T: TensorStructure,
{
    type Scalar = Complex<f64>;
    type ScalarRef<'a>
        = Complex<&'a f64>
    where
        Self: 'a;
    type Structure = T;
    type Store<S>
        = NumTensor<S>
    where
        S: TensorStructure;

    fn map_structure<O: TensorStructure>(self, f: impl Fn(Self::Structure) -> O) -> Self::Store<O> {
        match self {
            NumTensor::Float(fl) => NumTensor::Float(fl.map_structure(f)),
            NumTensor::Complex(c) => NumTensor::Complex(c.map_structure(f)),
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl Fn(Self::Structure) -> Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        Ok(match self {
            NumTensor::Float(fl) => match fl.map_structure_result(f) {
                Ok(fl) => NumTensor::Float(fl),
                Err(er) => return Err(er),
            },
            NumTensor::Complex(c) => match c.map_structure_result(f) {
                Ok(c) => NumTensor::Complex(c),
                Err(er) => return Err(er),
            },
        })
    }

    fn structure(&self) -> &Self::Structure {
        match self {
            NumTensor::Float(f) => f.structure(),
            NumTensor::Complex(c) => c.structure(),
        }
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        match self {
            NumTensor::Float(v) => NumTensor::Float(v.map_same_structure(f)),
            NumTensor::Complex(c) => NumTensor::Complex(c.map_same_structure(f)),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            NumTensor::Float(f) => f.mut_structure(),
            NumTensor::Complex(c) => c.mut_structure(),
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        match self {
            NumTensor::Float(f) => f.scalar().map(|x| Complex { re: x, im: 0. }),
            NumTensor::Complex(c) => c.scalar(),
        }
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        match self {
            NumTensor::Float(f) => f.scalar_ref().map(|x| Complex { re: x, im: &0. }),
            NumTensor::Complex(c) => c.scalar_ref().map(|a| a.as_ref()),
        }
    }
}

impl<T> ScalarTensor for NumTensor<T>
where
    T: TensorStructure + ScalarStructure,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        NumTensor::Complex(DataTensor::new_scalar(scalar))
    }
}

impl<T> TracksCount for NumTensor<T>
where
    T: TracksCount + TensorStructure,
{
    fn contractions_num(&self) -> usize {
        match self {
            NumTensor::Float(f) => f.contractions_num(),
            NumTensor::Complex(c) => c.contractions_num(),
        }
    }
}

impl<T> From<DenseTensor<f64, T>> for NumTensor<T>
where
    T: TensorStructure,
{
    fn from(other: DenseTensor<f64, T>) -> Self {
        NumTensor::Float(DataTensor::Dense(other))
    }
}

impl<T> From<SparseTensor<f64, T>> for NumTensor<T>
where
    T: TensorStructure,
{
    fn from(other: SparseTensor<f64, T>) -> Self {
        NumTensor::Float(DataTensor::Sparse(other))
    }
}

impl<T> From<DenseTensor<Complex<f64>, T>> for NumTensor<T>
where
    T: TensorStructure,
{
    fn from(other: DenseTensor<Complex<f64>, T>) -> Self {
        NumTensor::Complex(DataTensor::Dense(other))
    }
}

impl<T> From<SparseTensor<Complex<f64>, T>> for NumTensor<T>
where
    T: TensorStructure,
{
    fn from(other: SparseTensor<Complex<f64>, T>) -> Self {
        NumTensor::Complex(DataTensor::Sparse(other))
    }
}

impl<T, S: TensorStructure + StructureContract> DenseTensor<DenseTensor<T, S>, S> {
    pub fn flatten(self) -> Result<DenseTensor<T, S>> {
        // Check that the data is not empty
        if self.data.is_empty() {
            return Err(anyhow!("Cannot flatten an empty tensor"));
        }

        // Verify that all inner tensors have the same structure
        let first_inner_structure = &self.data[0].structure;
        // for tensor in &self.data {
        //     if tensor.structure != *first_inner_structure {
        //         return Err(anyhow!("Inner tensors have different structures"));
        //     }
        // }

        // Concatenate the outer and inner structures
        let mut combined_structure = self.structure;

        combined_structure.concat(first_inner_structure);

        // Flatten the data by concatenating inner tensors' data
        let data = self
            .data
            .into_iter()
            .flat_map(|tensor| tensor.data.into_iter())
            .collect();

        // Create the new flattened tensor
        Ok(DenseTensor {
            data,
            structure: combined_structure,
        })
    }
}

impl<T: Clone, S: TensorStructure + StructureContract + Clone> DataTensor<DataTensor<T, S>, S> {
    pub fn flatten(self, fill: &T) -> Result<DataTensor<T, S>> {
        let densified = self.map_data(|a| match a {
            DataTensor::Dense(d) => d,
            DataTensor::Sparse(s) => s.to_dense_with(fill),
        });
        match densified {
            DataTensor::Dense(d) => d.flatten().map(DataTensor::Dense),
            DataTensor::Sparse(s) => {
                let dense_fill = DenseTensor::fill(s.structure().clone(), fill.clone());
                s.to_dense_with(&dense_fill)
                    .flatten()
                    .map(DataTensor::Dense)
            }
        }
    }
}

pub trait StorageTensor: Sized + HasStructure<Structure: Clone> {
    // type ContainerStructure<S: TensorStructure>: HasStructure<Structure = S>;
    type ContainerData<Data>: HasStructure<Structure = Self::Structure>;
    type Data;

    // fn map_structure<S>(self, f: impl Fn(Self::Structure) -> S) -> Self::ContainerStructure<S>
    // where
    //     S: TensorStructure;

    fn map_data_ref<U>(&self, f: impl Fn(&Self::Data) -> U) -> Self::ContainerData<U>;

    fn map_data_ref_self(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self;

    fn map_data_ref_result<U, E>(
        &self,
        f: impl Fn(&Self::Data) -> Result<U, E>,
    ) -> Result<Self::ContainerData<U>, E>;

    fn map_data_ref_result_self<E>(
        &self,
        f: impl Fn(&Self::Data) -> Result<Self::Data, E>,
    ) -> Result<Self, E>;

    fn map_data_ref_mut<U>(
        &mut self,
        f: impl FnMut(&mut Self::Data) -> U,
    ) -> Self::ContainerData<U>;

    fn map_data_ref_mut_result<U, E>(
        &mut self,
        f: impl FnMut(&mut Self::Data) -> Result<U, E>,
    ) -> Result<Self::ContainerData<U>, E>;

    fn map_data_ref_mut_self(&mut self, f: impl FnMut(&mut Self::Data) -> Self::Data) -> Self;

    fn map_data_mut(&mut self, f: impl FnMut(&mut Self::Data));

    fn map_data<U>(self, f: impl Fn(Self::Data) -> U) -> Self::ContainerData<U>;

    fn map_data_self(self, f: impl Fn(Self::Data) -> Self::Data) -> Self;
}

impl<S: TensorStructure + Clone, T> StorageTensor for DataTensor<T, S> {
    type Data = T;
    type ContainerData<Data> = DataTensor<Data, S>;

    fn map_data_self(self, f: impl Fn(Self::Data) -> Self::Data) -> Self {
        self.map_data(f)
    }

    fn map_data_ref_mut_result<U, E>(
        &mut self,
        f: impl FnMut(&mut Self::Data) -> Result<U, E>,
    ) -> Result<Self::ContainerData<U>, E> {
        match self {
            DataTensor::Dense(d) => Ok(DataTensor::Dense(d.map_data_ref_mut_result(f)?)),
            DataTensor::Sparse(s) => Ok(DataTensor::Sparse(s.map_data_ref_mut_result(f)?)),
        }
    }

    fn map_data_ref_self(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self {
        self.map_data_ref(f)
    }

    fn map_data_ref_mut_self(&mut self, f: impl FnMut(&mut Self::Data) -> Self::Data) -> Self {
        self.map_data_ref_mut(f)
    }

    fn map_data_ref_result_self<E>(
        &self,
        f: impl Fn(&Self::Data) -> Result<Self::Data, E>,
    ) -> Result<Self, E> {
        self.map_data_ref_result(f)
    }

    // fn map_structure<S2: TensorStructure>(self, f: impl Fn(S) -> S2) -> DataTensor<T, S2> {
    //     match self {
    //         DataTensor::Dense(d) => DataTensor::Dense(d.map_structure(f)),
    //         DataTensor::Sparse(s) => DataTensor::Sparse(s.map_structure(f)),
    //     }
    // }

    fn map_data_ref_result<U, E>(
        &self,
        f: impl Fn(&T) -> Result<U, E>,
    ) -> Result<DataTensor<U, S>, E> {
        match self {
            DataTensor::Dense(d) => Ok(DataTensor::Dense(d.map_data_ref_result(f)?)),
            DataTensor::Sparse(s) => Ok(DataTensor::Sparse(s.map_data_ref_result(f)?)),
        }
    }

    fn map_data_ref<U>(&self, f: impl Fn(&T) -> U) -> DataTensor<U, S> {
        match self {
            DataTensor::Dense(d) => DataTensor::Dense(d.map_data_ref(f)),
            DataTensor::Sparse(s) => DataTensor::Sparse(s.map_data_ref(f)),
        }
    }

    fn map_data<U>(self, f: impl Fn(T) -> U) -> DataTensor<U, S> {
        match self {
            DataTensor::Dense(d) => DataTensor::Dense(d.map_data(f)),
            DataTensor::Sparse(s) => DataTensor::Sparse(s.map_data(f)),
        }
    }

    fn map_data_mut(&mut self, f: impl FnMut(&mut T)) {
        match self {
            DataTensor::Dense(d) => d.map_data_mut(f),
            DataTensor::Sparse(s) => s.map_data_mut(f),
        }
    }

    fn map_data_ref_mut<U>(&mut self, f: impl FnMut(&mut T) -> U) -> DataTensor<U, S> {
        match self {
            DataTensor::Dense(d) => DataTensor::Dense(d.map_data_ref_mut(f)),
            DataTensor::Sparse(s) => DataTensor::Sparse(s.map_data_ref_mut(f)),
        }
    }
}

impl<S: TensorStructure + Clone, T> StorageTensor for SparseTensor<T, S> {
    type Data = T;
    type ContainerData<Data> = SparseTensor<Data, S>;

    fn map_data_self(self, f: impl Fn(Self::Data) -> Self::Data) -> Self {
        self.map_data(f)
    }

    fn map_data_ref_mut_result<U, E>(
        &mut self,
        mut f: impl FnMut(&mut Self::Data) -> Result<U, E>,
    ) -> Result<Self::ContainerData<U>, E> {
        let elements: Result<HashMap<FlatIndex, _>, E> = self
            .elements
            .iter_mut()
            .map(|(k, v)| f(v).map(|v| (*k, v)))
            .collect();
        Ok(SparseTensor {
            elements: elements?,
            structure: self.structure.clone(),
        })
    }

    fn map_data_ref_self(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self {
        self.map_data_ref(f)
    }

    fn map_data_ref_mut_self(&mut self, f: impl FnMut(&mut Self::Data) -> Self::Data) -> Self {
        self.map_data_ref_mut(f)
    }

    fn map_data_ref_result_self<E>(
        &self,
        f: impl Fn(&Self::Data) -> Result<Self::Data, E>,
    ) -> Result<Self, E> {
        self.map_data_ref_result(f)
    }

    fn map_data_ref<U>(&self, f: impl Fn(&T) -> U) -> SparseTensor<U, S> {
        let elements = self.flat_iter().map(|(k, v)| (k, f(v))).collect();
        SparseTensor {
            elements,
            structure: self.structure.clone(),
        }
    }

    fn map_data_ref_result<U, E>(
        &self,
        f: impl Fn(&T) -> Result<U, E>,
    ) -> Result<SparseTensor<U, S>, E> {
        let elements: Result<HashMap<FlatIndex, _>, E> = self
            .flat_iter()
            .map(|(k, v)| f(v).map(|v| (k, v)))
            .collect();
        Ok(SparseTensor {
            elements: elements?,
            structure: self.structure.clone(),
        })
    }

    fn map_data<U>(self, f: impl Fn(T) -> U) -> SparseTensor<U, S> {
        let elements = self.elements.into_iter().map(|(k, v)| (k, f(v))).collect();
        SparseTensor {
            elements,
            structure: self.structure,
        }
    }

    fn map_data_ref_mut<U>(&mut self, mut f: impl FnMut(&mut T) -> U) -> SparseTensor<U, S> {
        let elements = self.elements.iter_mut().map(|(k, v)| (*k, f(v))).collect();
        SparseTensor {
            elements,
            structure: self.structure.clone(),
        }
    }

    fn map_data_mut(&mut self, f: impl FnMut(&mut T)) {
        self.elements.values_mut().for_each(f);
    }
}
impl<S: TensorStructure + Clone, D> StorageTensor for DenseTensor<D, S> {
    type ContainerData<Data> = DenseTensor<Data, S>;

    type Data = D;

    fn map_data_self(self, f: impl Fn(Self::Data) -> Self::Data) -> Self {
        self.map_data(f)
    }

    fn map_data_ref_self(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self {
        self.map_data_ref(f)
    }

    fn map_data_ref_mut_self(&mut self, f: impl FnMut(&mut Self::Data) -> Self::Data) -> Self {
        self.map_data_ref_mut(f)
    }

    fn map_data_ref_result_self<E>(
        &self,
        f: impl Fn(&Self::Data) -> Result<Self::Data, E>,
    ) -> Result<Self, E> {
        self.map_data_ref_result(f)
    }

    fn map_data_ref<U>(&self, f: impl Fn(&D) -> U) -> DenseTensor<U, S> {
        let data = self.data.iter().map(f).collect();
        DenseTensor {
            data,
            structure: self.structure.clone(),
        }
    }

    fn map_data_ref_result<U, E>(
        &self,
        f: impl Fn(&D) -> Result<U, E>,
    ) -> Result<DenseTensor<U, S>, E> {
        let data: Result<Vec<U>, E> = self.data.iter().map(f).collect();
        Ok(DenseTensor {
            data: data?,
            structure: self.structure.clone(),
        })
    }

    fn map_data_ref_mut<U>(&mut self, f: impl FnMut(&mut D) -> U) -> DenseTensor<U, S> {
        let data = self.data.iter_mut().map(f).collect();
        DenseTensor {
            data,
            structure: self.structure.clone(),
        }
    }

    fn map_data_ref_mut_result<U, E>(
        &mut self,
        f: impl FnMut(&mut Self::Data) -> Result<U, E>,
    ) -> Result<Self::ContainerData<U>, E> {
        let data: Result<Vec<U>, E> = self.data.iter_mut().map(f).collect();
        Ok(DenseTensor {
            data: data?,
            structure: self.structure.clone(),
        })
    }

    fn map_data_mut(&mut self, f: impl FnMut(&mut D)) {
        self.data.iter_mut().for_each(f);
    }

    fn map_data<U>(self, f: impl Fn(D) -> U) -> DenseTensor<U, S> {
        let data = self.data.into_iter().map(f).collect();
        DenseTensor {
            data,
            structure: self.structure,
        }
    }
}

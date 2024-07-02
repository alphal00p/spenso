use crate::{
    Complex, ExpandedIndex, FlatIndex, HasName, IsZero, IteratableTensor, TensorStructure,
    TryFromUpgrade,
};

use super::{
    ConcreteIndex, DenseTensorLinearIterator, HasStructure, SparseTensorLinearIterator,
    TracksCount, TrySmallestUpgrade, VecStructure,
};
use ahash::AHashMap;
use derive_more::From;
use enum_try_as_inner::EnumTryAsInner;
use indexmap::IndexMap;
use num::Zero;

use serde::{Deserialize, Serialize};
use smartstring::alias::String;
use std::{
    borrow::Cow,
    fmt::Display,
    ops::{Index, IndexMut},
};

#[cfg(feature = "shadowing")]
use super::atomic_expanded_label_id;
#[cfg(feature = "shadowing")]
use std::collections::HashMap;
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
    type FlatIter<'a> = SparseTensorLinearIterator<'a, T> where  I:'a,T: 'a;

    fn flat_iter(&self) -> Self::FlatIter<'_> {
        SparseTensorLinearIterator::new(self)
    }
}

impl<T, I: TensorStructure> DataIterator<T> for DenseTensor<T, I> {
    type FlatIter<'a> = DenseTensorLinearIterator<'a,T,I> where  I:'a,T: 'a;

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

impl<T> Settable for AHashMap<usize, T> {
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

    fn map(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self;
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
    fn set(&mut self, indices: &[ConcreteIndex], value: Self::SetData) -> Result<(), String>;

    fn set_flat(&mut self, index: FlatIndex, value: Self::SetData) -> Result<(), String>;
}

/// Trait for getting the data of a tensor
pub trait GetTensorData {
    type GetData;

    fn get(&self, indices: &[ConcreteIndex]) -> Result<&Self::GetData, String>;

    fn get_linear(&self, index: FlatIndex) -> Option<&Self::GetData>;

    fn get_linear_mut(&mut self, index: FlatIndex) -> Option<&mut Self::GetData>;
}

/// Sparse data tensor, generic on storage type `T`, and structure type `I`.  
///
/// Stores data in a hashmap of usize, using ahash's hashmap.
/// The usize key is the flattened index of the corresponding position in the dense tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTensor<T, I = VecStructure> {
    pub elements: AHashMap<FlatIndex, T>,
    pub structure: I,
}

impl<T, S> HasName for SparseTensor<T, S>
where
    S: HasName,
{
    type Args = S::Args;
    type Name = S::Name;
    fn name(&self) -> Option<Self::Name> {
        self.structure.name()
    }

    fn id(&self) -> Option<Self::Args> {
        self.structure.id()
    }

    fn set_name(&mut self, name: Self::Name) {
        self.structure.set_name(name);
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
        self.elements.values().cloned().collect()
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

    fn map(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self
    where
        I: Clone,
    {
        let elements = self.elements.iter().map(|(k, v)| (*k, f(v))).collect();
        SparseTensor {
            elements,
            structure: self.structure.clone(),
        }
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

impl<T, I> SetTensorData for SparseTensor<T, I>
where
    I: TensorStructure,
{
    type SetData = T;
    /// falible set method, returns an error if the indices are out of bounds.
    /// Does not check if the inserted value is zero.
    fn set(&mut self, indices: &[ConcreteIndex], value: T) -> Result<(), String> {
        self.verify_indices(indices)?;
        self.elements
            .insert(self.flat_index(indices).unwrap(), value);
        Ok(())
    }

    /// falible set given a flat index, returns an error if the indices are out of bounds.
    fn set_flat(&mut self, index: FlatIndex, value: T) -> Result<(), String> {
        if index >= self.size().into() {
            return Err("Index out of bounds".into());
        }
        self.elements.insert(index, value);
        Ok(())
    }
}
impl<T, I> GetTensorData for SparseTensor<T, I>
where
    I: TensorStructure,
{
    type GetData = T;
    fn get(&self, indices: &[ConcreteIndex]) -> Result<&T, String> {
        if let Ok(idx) = self.flat_index(indices) {
            self.elements
                .get(&idx)
                .ok_or("No elements at that spot".into())
        } else if self.structure.is_scalar() && indices.is_empty() {
            self.elements
                .iter()
                .next()
                .map(|(_, v)| v)
                .ok_or("err".into())
        } else {
            Err("Index out of bounds".into())
        }
    }

    fn get_linear(&self, index: FlatIndex) -> Option<&T> {
        self.elements.get(&index)
    }

    fn get_linear_mut(&mut self, index: FlatIndex) -> Option<&mut T> {
        self.elements.get_mut(&index)
    }
}

impl<T, I> HasStructure for SparseTensor<T, I>
where
    I: TensorStructure,
{
    type Scalar = T;
    type Structure = I;
    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
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
        let elements: Option<AHashMap<FlatIndex, U::LCM>> = self
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
            elements: AHashMap::default(),
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
        f64::from(self.elements.len() as u32) / f64::from(self.size() as u32)
    }

    /// Converts the sparse tensor to a dense tensor, with the same structure
    pub fn to_dense(&self) -> DenseTensor<T, I>
    where
        T: Clone + Default,
        I: Clone,
    {
        let mut dense = DenseTensor::default(self.structure.clone());
        for (indices, value) in self.elements.iter() {
            let _ = dense.set_flat(*indices, value.clone());
        }
        dense
    }

    /// fallible smart set method, returns an error if the indices are out of bounds.
    /// If the value is zero, it removes the element at the given indices.
    pub fn smart_set(&mut self, indices: &[ConcreteIndex], value: T) -> Result<(), String>
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
    pub fn from_data(data: &[(Vec<ConcreteIndex>, T)], structure: I) -> Result<Self, String>
    where
        T: Clone,
    {
        let mut dimensions = vec![0; structure.order()];
        for (index, _) in data {
            if index.len() != structure.order() {
                return Err("Mismatched order".into());
            }
            for (i, &idx) in index.iter().enumerate() {
                if idx >= dimensions[i] {
                    dimensions[i] = idx + 1;
                }
            }
        }
        let mut elements = AHashMap::default();
        for (index, value) in data {
            elements.insert(structure.flat_index(index).unwrap(), value.clone());
        }

        Ok(SparseTensor {
            elements,
            structure,
        })
    }

    /// fallible smart get method, returns an error if the indices are out of bounds.
    /// If the index is in the bTree return the value, else return zero.
    pub fn smart_get(&self, indices: &[ConcreteIndex]) -> Result<Cow<T>, String>
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DenseTensor<T, I = VecStructure> {
    pub data: Vec<T>,
    pub structure: I,
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
    type Structure = I;
    fn structure(&self) -> &Self::Structure {
        &self.structure
    }
    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
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
    S: HasName,
{
    type Args = S::Args;
    type Name = S::Name;
    fn name(&self) -> Option<Self::Name> {
        self.structure.name()
    }

    fn id(&self) -> Option<Self::Args> {
        self.structure.id()
    }

    fn set_name(&mut self, name: Self::Name) {
        self.structure.set_name(name);
    }
}

impl<T: Default + Clone, I> DenseTensor<T, I>
where
    I: TensorStructure,
{
    pub fn default(structure: I) -> Self {
        let length = if structure.is_scalar() {
            1
        } else {
            structure.size()
        };
        DenseTensor {
            data: vec![T::default(); length],
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
            structure.size()
        };
        DenseTensor {
            data: vec![T::zero(); length],
            structure,
        }
    }
}

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
    pub fn from_data(data: &[T], structure: I) -> Result<Self, String> {
        if data.len() != structure.size() && !(data.len() == 1 && structure.is_scalar()) {
            return Err("Data length does not match shape".into());
        }
        Ok(DenseTensor {
            data: data.to_vec(),
            structure,
        })
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
    pub fn from_data_coerced(data: &[T], structure: I) -> Result<Self, String> {
        if data.len() < structure.size() {
            return Err("Data length is too small".into());
        }
        let mut data = data.to_vec();
        if structure.is_scalar() {
            data.truncate(1);
        } else {
            data.truncate(structure.size());
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
        for i in 0..self.size() {
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

    fn map(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self {
        let data = self.data.iter().map(f).collect();
        DenseTensor {
            data,
            structure: self.structure.clone(),
        }
    }
}

impl<T, I> SetTensorData for DenseTensor<T, I>
where
    I: TensorStructure,
{
    type SetData = T;
    fn set(&mut self, indices: &[ConcreteIndex], value: T) -> Result<(), String> {
        self.verify_indices(indices)?;
        let idx = self.flat_index(indices);
        if let Ok(i) = idx {
            self[i] = value;
        }
        Ok(())
    }

    fn set_flat(&mut self, index: FlatIndex, value: T) -> Result<(), String> {
        if index < self.size().into() {
            self[index] = value;
        } else {
            return Err("Index out of bounds".into());
        }
        Ok(())
    }
}

impl<T, I> GetTensorData for DenseTensor<T, I>
where
    I: TensorStructure,
{
    type GetData = T;
    fn get_linear(&self, index: FlatIndex) -> Option<&T> {
        let i: usize = index.into();
        self.data.get(i)
    }

    fn get(&self, indices: &[ConcreteIndex]) -> Result<&T, String> {
        if let Ok(idx) = self.flat_index(indices) {
            Ok(&self[idx])
        } else if self.structure.is_scalar() && indices.is_empty() {
            Ok(&self.data[0])
        } else {
            Err("Index out of bounds".into())
        }
    }

    fn get_linear_mut(&mut self, index: FlatIndex) -> Option<&mut Self::GetData> {
        let i: usize = index.into();
        self.data.get_mut(i)
    }
}

/// Enum for storing either a dense or a sparse tensor, with the same structure
#[derive(Debug, Clone, EnumTryAsInner, Serialize, Deserialize, From)]
#[derive_err(Debug)]
pub enum DataTensor<T, I: TensorStructure = VecStructure> {
    Dense(DenseTensor<T, I>),
    Sparse(SparseTensor<T, I>),
}

impl<T, I> DataTensor<T, I>
where
    I: TensorStructure + Clone,
{
    pub fn to_sparse(self) -> SparseTensor<T, I>
    where
        T: Clone + Default + PartialEq,
    {
        match self {
            DataTensor::Dense(d) => d.to_sparse(),
            DataTensor::Sparse(s) => s,
        }
    }

    pub fn to_dense(self) -> DenseTensor<T, I>
    where
        T: Clone + Default + PartialEq,
    {
        match self {
            DataTensor::Dense(d) => d,
            DataTensor::Sparse(s) => s.to_dense(),
        }
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

    fn map(&self, f: impl Fn(&Self::Data) -> Self::Data) -> Self {
        match self {
            DataTensor::Dense(d) => DataTensor::Dense(d.map(f)),
            DataTensor::Sparse(s) => DataTensor::Sparse(s.map(f)),
        }
    }
}

impl<T, I> HasStructure for DataTensor<T, I>
where
    I: TensorStructure,
    T: Clone,
{
    type Scalar = T;
    type Structure = I;
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

    fn id(&self) -> Option<Self::Args> {
        match self {
            DataTensor::Dense(d) => d.id(),
            DataTensor::Sparse(s) => s.id(),
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

    fn set(&mut self, indices: &[ConcreteIndex], value: Self::SetData) -> Result<(), String> {
        match self {
            DataTensor::Dense(d) => d.set(indices, value),
            DataTensor::Sparse(s) => s.set(indices, value),
        }
    }

    fn set_flat(&mut self, index: FlatIndex, value: Self::SetData) -> Result<(), String> {
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
    type GetData = T;

    fn get(&self, indices: &[ConcreteIndex]) -> Result<&Self::GetData, String> {
        match self {
            DataTensor::Dense(d) => d.get(indices),
            DataTensor::Sparse(s) => s.get(indices),
        }
    }

    fn get_linear(&self, index: FlatIndex) -> Option<&Self::GetData> {
        match self {
            DataTensor::Dense(d) => d.get_linear(index),
            DataTensor::Sparse(s) => s.get_linear(index),
        }
    }

    fn get_linear_mut(&mut self, index: FlatIndex) -> Option<&mut Self::GetData> {
        match self {
            DataTensor::Dense(d) => d.get_linear_mut(index),
            DataTensor::Sparse(s) => s.get_linear_mut(index),
        }
    }
}

/// Enum for a datatensor with specific numeric data type, generic on the structure type `I`
#[derive(Debug, Clone, EnumTryAsInner, Serialize, Deserialize)]
#[derive_err(Debug)]
pub enum NumTensor<T: TensorStructure = VecStructure> {
    Float(DataTensor<f64, T>),
    Complex(DataTensor<Complex<f64>, T>),
}

impl<T> HasStructure for NumTensor<T>
where
    T: TensorStructure,
{
    type Scalar = Complex<f64>;
    type Structure = T;
    fn structure(&self) -> &Self::Structure {
        match self {
            NumTensor::Float(f) => f.structure(),
            NumTensor::Complex(c) => c.structure(),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            NumTensor::Float(f) => f.mut_structure(),
            NumTensor::Complex(c) => c.mut_structure(),
        }
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

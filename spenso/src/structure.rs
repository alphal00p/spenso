use abstract_index::AbstractIndex;
use ahash::AHashMap;

use anyhow::{anyhow, Result};
use concrete_index::ConcreteIndex;
use concrete_index::DualConciousExpandedIndex;
use concrete_index::DualConciousIndex;
use concrete_index::ExpandedIndex;
use concrete_index::FlatIndex;
use delegate::delegate;
use dimension::Dimension;
use indexmap::IndexMap;

#[cfg(feature = "shadowing")]
use symbolica::symbol;
use thiserror::Error;

#[cfg(feature = "shadowing")]
use crate::{
    data::DenseTensor,
    parametric::{ExpandedCoefficent, FlatCoefficent, TensorCoefficient},
    structure::abstract_index::ABSTRACTIND,
    structure::slot::ConstructibleSlot,
    symbolica_utils::{IntoArgs, IntoSymbol, SerializableAtom, SerializableSymbol},
};
use linnet::permutation::Permutation;
use representation::{LibraryRep, RepName, Representation};
use serde::Deserialize;
use serde::Serialize;
use slot::DualSlotTo;
use slot::IsAbstractSlot;
use slot::Slot;
use slot::SlotError;
use std::fmt::Debug;

#[cfg(feature = "shadowing")]
use std::fmt::Display;
use std::hash::Hash;
use std::ops::Range;
#[cfg(feature = "shadowing")]
use symbolica::atom::{representation::FunView, Atom, AtomView, FunctionBuilder, MulView, Symbol};

use crate::iterators::TensorStructureIndexIterator;
use std::collections::HashMap;
use std::collections::HashSet;

// use smartstring::alias::String;
pub mod abstract_index;
pub mod concrete_index;
pub mod dimension;
pub mod representation;
pub mod slot;

pub trait ScalarTensor: HasStructure<Structure: ScalarStructure> {
    fn new_scalar(scalar: Self::Scalar) -> Self;
}
pub trait HasStructure {
    type Structure: TensorStructure;
    type Scalar;

    fn structure(&self) -> &Self::Structure;
    fn mut_structure(&mut self) -> &mut Self::Structure;
    fn scalar(self) -> Option<Self::Scalar>;
    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self;

    fn set_structure_name<N>(&mut self, name: N)
    where
        Self::Structure: HasName<Name = N>,
    {
        self.mut_structure().set_name(name);
    }
    fn structure_name(&self) -> Option<<Self::Structure as HasName>::Name>
    where
        Self::Structure: HasName,
    {
        self.structure().name()
    }
    fn structure_id(&self) -> Option<<Self::Structure as HasName>::Args>
    where
        Self::Structure: HasName,
    {
        self.structure().args()
    }
    // fn cast_structure<O, S>(self) -> O
    // where
    //     O: HasStructure<Structure = S, Scalar = Self::Scalar>,
    //     S: TensorStructure + From<Self::Structure>;
}

pub trait CastStructure<O: HasStructure<Structure: From<Self::Structure>>>: HasStructure {
    fn cast_structure(self) -> O;
}

pub struct Tensor<Store, Structure> {
    pub store: Store,
    pub structure: Structure,
}

#[allow(dead_code)]
impl<Store, Structure> Tensor<Store, Structure> {
    fn cast<NewStructure>(self) -> Tensor<Store, NewStructure>
    where
        NewStructure: TensorStructure + From<Structure>,
    {
        Tensor {
            store: self.store,
            structure: self.structure.into(),
        }
    }
}

impl<T> TensorStructure for T
where
    T: HasStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Slot = <T::Structure as TensorStructure>::Slot;
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

#[cfg(feature = "shadowing")]
impl<T: HasName> ToSymbolic for T
where
    T: TensorStructure,
    T::Name: IntoSymbol,
    T::Args: IntoArgs,
{
    fn concrete_atom(&self, id: FlatIndex) -> ExpandedCoefficent<()> {
        ExpandedCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: self.co_expanded_index(id).unwrap(),
            args: None,
        }
    }

    fn flat_atom(&self, id: FlatIndex) -> FlatCoefficent<()> {
        FlatCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: id,
            args: None,
        }
    }
}

#[cfg(feature = "shadowing")]
pub trait ToSymbolic: TensorStructure {
    fn concrete_atom(&self, id: FlatIndex) -> ExpandedCoefficent<()> {
        ExpandedCoefficent {
            name: None,
            index: self.co_expanded_index(id).unwrap(),
            args: None,
        }
    }

    fn flat_atom(&self, id: FlatIndex) -> FlatCoefficent<()> {
        FlatCoefficent {
            name: None,
            index: id,
            args: None,
        }
    }

    fn to_dense_expanded_labels(self) -> Result<DenseTensor<Atom, Self>>
    where
        Self: std::marker::Sized + Clone,
    {
        self.to_dense_labeled(Self::concrete_atom)
    }

    fn to_dense_labeled<T>(
        self,
        index_to_atom: impl Fn(&Self, FlatIndex) -> T,
    ) -> Result<DenseTensor<Atom, Self>>
    where
        Self: Sized,
        T: TensorCoefficient,
    {
        let mut data = vec![];
        for index in 0..self.size()? {
            data.push(index_to_atom(&self, index.into()).to_atom().unwrap());
        }

        Ok(DenseTensor {
            data,
            structure: self,
        })
    }

    fn to_dense_labeled_complex<T>(
        self,
        index_to_atom: impl Fn(&Self, FlatIndex) -> T,
    ) -> Result<DenseTensor<Atom, Self>>
    where
        Self: Sized,
        T: TensorCoefficient,
    {
        let mut data = vec![];
        for index in 0..self.size()? {
            let re = index_to_atom(&self, index.into()).to_atom_re().unwrap();
            let im = index_to_atom(&self, index.into()).to_atom_im().unwrap();
            let i = Atom::new_var(Atom::I);
            data.push(&re + i * &im);
        }

        Ok(DenseTensor {
            data,
            structure: self,
        })
    }

    fn to_dense_flat_labels(self) -> Result<DenseTensor<Atom, Self>>
    where
        Self: std::marker::Sized + Clone,
    {
        self.to_dense_labeled(Self::flat_atom)
    }

    fn to_symbolic(&self) -> Option<Atom>
    where
        Self: HasName<Name: IntoSymbol, Args: IntoArgs>,
    {
        let args = self.args().map(|s| s.args()).unwrap_or_default();

        Some(self.to_symbolic_with(self.name()?.ref_into_symbol(), &args))
    }

    fn to_symbolic_with(&self, name: Symbol, args: &[Atom]) -> Atom {
        let slots = self
            .external_structure_iter()
            .map(|slot| slot.to_atom())
            .collect::<Vec<_>>();

        let mut value_builder = FunctionBuilder::new(name.ref_into_symbol());

        for arg in args {
            value_builder = value_builder.add_arg(arg);
        }

        for s in slots {
            value_builder = value_builder.add_arg(&s);
        }
        value_builder.finish()
    }
}

pub trait ScalarStructure {
    fn scalar_structure() -> Self;
}
pub trait TensorStructure {
    type Slot: IsAbstractSlot + DualSlotTo<Dual = Self::Slot>;
    // type R: Rep;
    //
    fn dual(self) -> Self;

    fn string_rep(&self) -> String {
        self.external_structure_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("")
    }

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension>;
    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
    fn get_aind(&self, i: usize) -> Option<AbstractIndex>;
    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
    fn get_dim(&self, i: usize) -> Option<Dimension>;
    fn get_slot(&self, i: usize) -> Option<Self::Slot>;
    fn order(&self) -> usize;
    /// returns the list of slots that are the external indices of the tensor

    fn external_structure(&self) -> Vec<Self::Slot> {
        self.external_structure_iter().collect()
    }

    fn to_shell(self) -> TensorShell<Self>
    where
        Self: Sized,
    {
        TensorShell::new(self)
    }

    fn contains_matching(&self, slot: &Self::Slot) -> bool {
        self.external_structure_iter().any(|s| s.matches(slot))
    }

    fn external_reps(&self) -> Vec<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.external_reps_iter().collect()
    }

    fn external_indices(&self) -> Vec<AbstractIndex> {
        self.external_indices_iter().collect()
    }

    // fn iter_index_along_fiber(&self,fiber_position: &[bool]  )-> TensorStructureMultiFiberIterator where Self: Sized{
    //     TensorStructureMultiFiberIterator::new(self, fiber_position)
    // }

    // fn single_fiber_at(&self,fiber_pos:usize)->Fiber{
    //     let mut  f =Fiber::zeros(self.external_structure().len());
    //     f.free(fiber_pos);
    //     f.is_single();
    //     f
    // }

    /// checks if the tensor has the same exact structure as another tensor
    fn same_content(&self, other: &Self) -> bool {
        self.same_external(other)
    }

    /// Given two [`TensorStructure`]s, returns the index of the first matching slot in each external index list, along with a boolean indicating if there is a single match
    fn match_index(&self, other: &Self) -> Option<(bool, usize, usize)> {
        let posmap = self
            .external_structure_iter()
            .enumerate()
            .map(|(i, slot)| (slot, i))
            .collect::<AHashMap<_, _>>();

        let mut first_pair: Option<(usize, usize)> = None;

        for (j, slot) in other.external_structure_iter().enumerate() {
            if let Some(&i) = posmap.get(&slot.dual()) {
                if let Some((i, j)) = first_pair {
                    // Found a second match, return early with false indicating non-unique match
                    return Some((false, i, j));
                }
                first_pair = Some((i, j));
            }
        }

        first_pair.map(|(i, j)| (true, i, j)) // Maps the found pair to Some with true indicating a unique match, or None if no match was found
    }

    /// Given two [`TensorStructure`]s, returns the index of the first matching slot in each external index list
    fn match_indices(&self, other: &Self) -> Option<(Permutation, Vec<bool>, Vec<bool>)> {
        let mut self_matches = vec![false; self.order()];
        let mut perm = Vec::new();
        let mut other_matches = vec![false; other.order()];

        let posmap = self
            .external_structure_iter()
            .enumerate()
            .map(|(i, slot)| (slot, i))
            .collect::<AHashMap<_, _>>();

        for (j, slot_other) in other.external_structure_iter().enumerate() {
            if let Some(&i) = posmap.get(&slot_other.dual()) {
                self_matches[i] = true;
                other_matches[j] = true;
                perm.push(i);
            }
        }

        if perm.is_empty() {
            None
        } else {
            let p: Permutation = Permutation::sort(&perm);
            Some((p, self_matches, other_matches))
        }
    }
    /// Identify the repeated slots in the external index list
    fn traces(&self) -> Vec<[usize; 2]> {
        let mut positions: HashMap<<Self as TensorStructure>::Slot, Vec<usize>> = HashMap::new();

        // Track the positions of each element
        for (index, key) in self.external_structure_iter().enumerate() {
            if let Some(v) = positions.get_mut(&key.dual()) {
                v.push(index);
            } else {
                positions.insert(key, vec![index]);
            }
        }

        // Collect only the positions of repeated elements
        positions
            .into_iter()
            .filter_map(|(_, indices)| {
                if indices.len() == 2 {
                    Some([indices[0], indices[1]])
                } else {
                    None
                }
            })
            .collect()
    }

    /// yields the (outwards facing) shape of the tensor as a list of dimensions
    fn shape(&self) -> Vec<Dimension> {
        self.external_dims_iter().collect()
    }

    fn reps(&self) -> Vec<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.external_reps_iter().collect()
    }

    /// checks if externally, the two tensors are the same
    fn same_external(&self, other: &Self) -> bool {
        let set1: HashSet<_> = self.external_structure_iter().collect();
        let set2: HashSet<_> = other.external_structure_iter().collect();
        set1 == set2
    }

    /// find the permutation of the external indices that would make the two tensors the same. Applying the permutation to other should make it the same as self
    fn find_permutation(&self, other: &Self) -> Result<Permutation> {
        if self.order() != other.order() {
            return Err(anyhow!(
                "Mismatched order: {} vs {}",
                self.order(),
                other.order()
            ));
        }
        let other_structure = other.external_structure();
        let self_structure = self.external_structure();

        let other_sort = Permutation::sort(&other_structure);
        let self_sort = Permutation::sort(&self_structure);

        if other_sort.apply_slice(&other_structure) == self_sort.apply_slice(&self_structure) {
            Ok(other_sort.compose(&self_sort.inverse()))
        } else {
            Err(anyhow!("Mismatched structure"))
        }

        // let mut index_map = HashMap::new();
        // for (i, item) in other.external_structure_iter().enumerate() {
        //     index_map.entry(item).or_insert_with(Vec::new).push(i);
        // }

        // let mut permutation = Vec::with_capacity(self.order());
        // let mut used_indices = HashSet::new();
        // for item in self.external_structure_iter() {
        //     if let Some(indices) = index_map.get_mut(&item) {
        //         // Find an index that hasn't been used yet
        //         if let Some(&index) = indices.iter().find(|&&i| !used_indices.contains(&i)) {
        //             permutation.push(index);
        //             used_indices.insert(index);
        //         } else {
        //             // No available index for this item
        //             return Err(anyhow!("No available index for {:?}", item));
        //         }
        //     } else {
        //         // Item not found in other
        //         return Err(anyhow!("Item {:?} not found in other", item));
        //     }
        // }

        // Ok(permutation)
    }

    /// yields the strides of the tensor in column major order
    fn strides_column_major(&self) -> Result<Vec<usize>> {
        let mut strides: Vec<usize> = vec![1; self.order()];

        if self.order() == 0 {
            return Ok(strides);
        }

        for i in 0..self.order() - 1 {
            strides[i + 1] = strides[i] * usize::try_from(self.shape()[i])?;
        }

        Ok(strides)
    }

    /// yields the strides of the tensor in row major order
    fn strides_row_major(&self) -> Result<Vec<usize>> {
        let mut strides = vec![1; self.order()];
        if self.order() == 0 {
            return Ok(strides);
        }

        for i in (0..self.order() - 1).rev() {
            strides[i] = strides[i + 1] * usize::try_from(self.shape()[i + 1])?;
        }

        Ok(strides)
    }

    /// By default, the strides are row major
    fn strides(&self) -> Result<Vec<usize>> {
        self.strides_row_major()
    }

    /// Verifies that the list of indices provided are valid for the tensor
    ///
    /// # Errors
    ///
    /// `Mismatched order` = if the length of the indices is different from the order of the tensor,
    ///
    /// `Index out of bounds` = if the index is out of bounds for the dimension of that index
    ///
    fn verify_indices<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<()> {
        if indices.as_ref().len() != self.order() {
            return Err(anyhow!(
                "Mismatched order: {} indices, vs order {}",
                indices.as_ref().len(),
                self.order()
            ));
        }

        for (i, dim_len) in self
            .external_structure_iter()
            .map(|slot| slot.dim())
            .enumerate()
        {
            if indices.as_ref()[i] >= usize::try_from(dim_len)? {
                return Err(anyhow!(
                    "Index {} out of bounds for dimension {} of size {}",
                    indices.as_ref()[i],
                    i,
                    usize::try_from(dim_len)?
                ));
            }
        }
        Ok(())
    }

    /// yields the flat index of the tensor given a list of indices
    ///
    /// # Errors
    ///
    /// Same as [`Self::verify_indices`]
    fn flat_index<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<FlatIndex> {
        let strides = self.strides()?;
        self.verify_indices(&indices)?;

        let mut idx = 0;
        for (i, &index) in indices.as_ref().iter().enumerate() {
            idx += index * strides[i];
        }
        Ok(idx.into())
    }

    /// yields the expanded index of the tensor given a flat index
    ///
    /// # Errors
    ///
    /// `Index out of bounds` = if the flat index is out of bounds for the tensor
    fn expanded_index(&self, flat_index: FlatIndex) -> Result<ExpandedIndex> {
        let mut indices = vec![];
        let mut index: usize = flat_index.into();
        for &stride in &self.strides()? {
            indices.push(index / stride);
            index %= stride;
        }
        if usize::from(flat_index) < self.size()? {
            Ok(indices.into())
        } else {
            Err(anyhow!("Index {flat_index} out of bounds"))
        }
    }

    fn co_expanded_index(&self, flat_index: FlatIndex) -> Result<DualConciousExpandedIndex> {
        let mut indices = vec![];

        for (r, i) in self
            .external_reps_iter()
            .zip(self.expanded_index(flat_index)?.iter())
        {
            if r.rep.is_base() && r.rep.is_dual() {
                indices.push(DualConciousIndex::SelfDual(*i));
            } else if r.rep.is_base() {
                indices.push(DualConciousIndex::Up(*i));
            } else {
                indices.push(DualConciousIndex::Down(*i));
            }
        }
        Ok(indices.into())
    }

    /// yields an iterator over the indices of the tensor
    fn index_iter(&self) -> TensorStructureIndexIterator<Self>
    where
        Self: Sized,
    {
        TensorStructureIndexIterator::new(self)
    }

    /// if the tensor has no (external) indices, it is a scalar
    fn is_scalar(&self) -> bool {
        self.order() == 0
    }

    // /// get the metric along the i-th index
    // fn get_ith_metric(&self, i: usize) -> Result<Vec<bool>> {
    //     self.get_rep(i)
    //         .ok_or(anyhow!("out of bounds access"))?
    //         .negative()
    // }

    /// yields the size of the tensor, i.e. the product of the dimensions. This is the length of the vector of the data in a dense tensor
    fn size(&self) -> Result<usize> {
        if self.order() == 0 {
            return Ok(1);
        }
        let mut size = 1;
        for dim in self.shape() {
            size *= usize::try_from(dim)?;
        }
        Ok(size)
    }
}

// impl<'a> HasStructure for &'a [Slot] {
//     type Structure = &'a [Slot];
//     type Scalar = ();

//     fn order(&self) -> usize {
//         self.len()
//     }

//     fn structure(&self) -> &Self::Structure {
//         self
//     }

//     fn mut_structure(&mut self) -> &mut Self::Structure {
//         self
//     }
// }

impl<S: IsAbstractSlot<R: RepName> + DualSlotTo<Dual = S>> ScalarStructure for Vec<S> {
    fn scalar_structure() -> Self {
        vec![]
    }
}

impl<S: IsAbstractSlot<R: RepName> + DualSlotTo<Dual = S>> TensorStructure for Vec<S> {
    type Slot = S;
    // type R = S::R;

    fn dual(self) -> Self {
        self.into_iter().map(|s| s.dual()).collect()
    }
    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.iter().map(|s| s.rep())
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex> {
        self.iter().map(|s| s.aind())
    }

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot> {
        self.iter().cloned()
    }

    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension> {
        self.iter().map(|s| s.dim())
    }

    fn order(&self) -> usize {
        self.len()
    }

    fn get_slot(&self, i: usize) -> Option<S> {
        self.get(i).cloned()
    }

    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.get(i).map(|s| s.rep())
    }

    fn get_dim(&self, i: usize) -> Option<Dimension> {
        self.get(i).map(|s| s.dim())
    }

    fn get_aind(&self, i: usize) -> Option<AbstractIndex> {
        self.get(i).map(|s| s.aind())
    }
}

#[cfg(feature = "shadowing")]
impl<S: IsAbstractSlot<R: RepName> + ConstructibleSlot<S::R> + DualSlotTo<Dual = S>> ToSymbolic
    for Vec<S>
{
}

/// A trait for a structure that can be traced and merged, during a contraction.
pub trait StructureContract {
    fn trace(&mut self, i: usize, j: usize);

    fn trace_out(&mut self);

    fn merge(&mut self, other: &Self) -> Option<usize>;

    fn concat(&mut self, other: &Self);

    #[must_use]
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self;
}

impl<S: DualSlotTo<Dual = S, R: RepName>> StructureContract for Vec<S> {
    fn trace(&mut self, i: usize, j: usize) {
        if i < j {
            self.trace(j, i);
            return;
        }
        let a = self.remove(i);
        let b = self.remove(j);
        assert_eq!(a, b);
    }

    fn concat(&mut self, other: &Self) {
        self.extend(other.iter().cloned());
    }

    fn trace_out(&mut self) {
        let mut positions = IndexMap::new();

        // Track the positions of each element
        for (index, &value) in self.iter().enumerate() {
            positions.entry(value).or_insert_with(Vec::new).push(index);
        }
        // Collect only the positions of non- repeated elements

        *self = positions
            .into_iter()
            .filter_map(|(value, indices)| {
                if indices.len() == 1 {
                    Some(value)
                } else {
                    None
                }
            })
            .collect();
    }

    fn merge(&mut self, other: &Self) -> Option<usize> {
        let mut positions = IndexMap::new();
        let mut i = 0;

        self.retain(|x| {
            let e = positions.get(x);
            if e.is_some() {
                return false;
            }
            positions.insert(*x, (Some(i), None));
            i += 1;
            true
        });

        let mut first = true;
        let mut first_other = 0;

        for (index, &value) in self.iter().enumerate() {
            positions.entry(value).or_insert((Some(index), None));
        }

        for (index, &value) in other.iter().enumerate() {
            let e = positions.get(&value.dual());
            if let Some((Some(selfi), None)) = e {
                positions.insert(value.dual(), (Some(*selfi), Some(index)));
            } else {
                positions.insert(value, (None, Some(index)));
                self.push(value);
            }
        }

        let mut i = 0;

        self.retain(|x| {
            let pos = positions.get(x).unwrap();
            if pos.1.is_none() {
                i += 1;
                return true;
            }
            if pos.0.is_none() {
                if first {
                    first = false;
                    first_other = i;
                }
                return true;
            }
            false
        });

        if first {
            None
        } else {
            Some(first_other)
        }
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        let mut slots_b = other.clone();
        let mut slots_a = self.clone();

        slots_a.remove(positions.0);
        slots_b.remove(positions.1);

        slots_a.append(&mut slots_b);
        slots_a
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Default, Hash)]
pub struct IndexLess<T: RepName = LibraryRep> {
    pub structure: Vec<Representation<T>>,
}
impl<R: RepName> std::fmt::Display for IndexLess<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, item) in self.structure.iter().enumerate() {
            if index != 0 {
                // To avoid a newline at the start
                writeln!(f)?;
            }
            write!(
                f,
                "({})",
                // IDPRINTER
                //     .encode_string(usize::from(item.index) as u64)
                //     .unwrap(),
                item
            )?;
        }
        Ok(())
    }
}
impl<R: RepName> FromIterator<Representation<R>> for IndexLess<R> {
    fn from_iter<I: IntoIterator<Item = Representation<R>>>(iter: I) -> Self {
        IndexLess {
            structure: iter.into_iter().collect(),
        }
    }
}

impl<R: RepName> From<VecStructure<R>> for IndexLess<R> {
    fn from(structure: VecStructure<R>) -> Self {
        IndexLess {
            structure: structure.into_iter().map(|a| a.rep).collect(),
        }
    }
}

impl<R: RepName> From<ContractionCountStructure<R>> for IndexLess<R> {
    fn from(structure: ContractionCountStructure<R>) -> Self {
        structure.structure.into()
    }
}

impl<N, A, R: RepName> From<NamedStructure<N, A, R>> for IndexLess<R> {
    fn from(structure: NamedStructure<N, A, R>) -> Self {
        structure.structure.into()
    }
}

impl<N, A, R: RepName> From<IndexlessNamedStructure<N, A, R>> for IndexLess<R> {
    fn from(structure: IndexlessNamedStructure<N, A, R>) -> Self {
        structure.structure
    }
}

impl<N, A, R: RepName> From<SmartShadowStructure<N, A, R>> for IndexLess<R> {
    fn from(structure: SmartShadowStructure<N, A, R>) -> Self {
        structure.structure.into()
    }
}

impl<N, A, R: RepName> From<HistoryStructure<N, A, R>> for IndexLess<R> {
    fn from(structure: HistoryStructure<N, A, R>) -> Self {
        structure.external.into()
    }
}

impl<T: RepName> IndexLess<T> {
    pub fn new(structure: Vec<Representation<T>>) -> Self {
        Self { structure }
    }

    pub fn empty() -> Self {
        Self { structure: vec![] }
    }

    pub fn to_indexed(self, indices: &[AbstractIndex]) -> Vec<Slot<T>> {
        indices
            .iter()
            .cloned()
            .zip(self.structure.iter().cloned())
            .map(|(i, r)| Representation::slot(&r, i))
            .collect()
    }
}

impl<T: RepName<Dual = T>> ScalarStructure for IndexLess<T> {
    fn scalar_structure() -> Self {
        Self::empty()
    }
}

impl<T: RepName<Dual = T>> TensorStructure for IndexLess<T> {
    type Slot = Slot<T>;
    // type R = T;
    //
    fn dual(self) -> Self {
        self.structure.into_iter().map(|r| r.dual()).collect()
    }

    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.structure.iter().copied()
    }

    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension> {
        self.structure.iter().map(|s| s.dim)
    }

    fn get_aind(&self, _: usize) -> Option<AbstractIndex> {
        None
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex> {
        [].iter().cloned()
    }

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot> {
        [].iter().cloned()
    }

    fn order(&self) -> usize {
        self.structure.len()
    }

    fn get_slot(&self, _: usize) -> Option<Self::Slot> {
        None
    }

    fn find_permutation(&self, other: &Self) -> Result<Permutation> {
        if self.order() != other.order() {
            return Err(anyhow!(
                "Mismatched order: {} vs {}",
                self.order(),
                other.order()
            ));
        }
        let other_structure = &other.structure;
        let self_structure = &self.structure;

        let other_sort = Permutation::sort(other_structure);
        let self_sort = Permutation::sort(self_structure);

        if other_sort.apply_slice(other_structure) == self_sort.apply_slice(self_structure) {
            Ok(other_sort.compose(&self_sort.inverse()))
        } else {
            Err(anyhow!("Mismatched structure"))
        }

        // let mut index_map = HashMap::new();
        // for (i, item) in other.structure.iter().enumerate() {
        //     index_map.entry(item).or_insert_with(Vec::new).push(i);
        // }

        // let mut permutation = Vec::with_capacity(self.order());
        // let mut used_indices = HashSet::new();
        // for item in self.structure.iter() {
        //     if let Some(indices) = index_map.get_mut(&item) {
        //         // Find an index that hasn't been used yet
        //         if let Some(&index) = indices.iter().find(|&&i| !used_indices.contains(&i)) {
        //             permutation.push(index);
        //             used_indices.insert(index);
        //         } else {
        //             // No available index for this item
        //             return Err(anyhow!("No available index for {:?}", item));
        //         }
        //     } else {
        //         // Item not found in other
        //         return Err(anyhow!("Item {:?} not found in other", item));
        //     }
        // }

        // Ok(permutation)
    }

    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.structure.get(i).copied()
    }

    fn get_dim(&self, i: usize) -> Option<Dimension> {
        self.structure.get(i).map(|&r| r.into())
    }
}

#[cfg(feature = "shadowing")]
impl<T: RepName<Dual = T>> ToSymbolic for IndexLess<T> {}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Hash)]
pub struct VecStructure<R: RepName = LibraryRep> {
    pub structure: Vec<Slot<R>>,
}

impl<R: RepName> Default for VecStructure<R> {
    fn default() -> Self {
        Self { structure: vec![] }
    }
}

#[cfg(feature = "shadowing")]
impl<R: RepName> TryFrom<AtomView<'_>> for VecStructure<R> {
    type Error = SlotError;
    fn try_from(value: AtomView) -> Result<Self, Self::Error> {
        match value {
            AtomView::Mul(mul) => mul.try_into(),
            AtomView::Fun(fun) => fun.try_into(),
            AtomView::Pow(_) => {
                Ok(VecStructure::<R>::default()) // powers do not have a structure
            }
            _ => Err(anyhow!("Not a structure: {value}").into()), // could check if it
        }
    }
}

// impl From<Vec<PhysicalSlots>> for VecStructure {
//     fn from(value: Vec<PhysicalSlots>) -> Self {
//         VecStructure { structure: value }
//     }
// }

#[cfg(feature = "shadowing")]
impl<R: RepName> TryFrom<FunView<'_>> for VecStructure<R> {
    type Error = SlotError;
    fn try_from(value: FunView) -> Result<Self, Self::Error> {
        if value.get_symbol() == symbol!(ABSTRACTIND) {
            let mut structure: Vec<Slot<R>> = vec![];

            for arg in value.iter() {
                structure.push(arg.try_into()?);
            }

            Ok(VecStructure { structure })
        } else {
            let mut structure: Self = vec![].into();
            for arg in value.iter() {
                structure.extend(arg.try_into()?); // append all the structures found
            }
            Ok(structure)
        }
    }
}

#[cfg(feature = "shadowing")]
impl<R: RepName> TryFrom<MulView<'_>> for VecStructure<R> {
    type Error = SlotError;
    fn try_from(value: MulView) -> Result<Self, Self::Error> {
        let mut structure: Self = vec![].into();

        for arg in value.iter() {
            structure.extend(arg.try_into()?);
        }
        Ok(structure)
    }
}

impl<S: RepName, R: From<S> + RepName> FromIterator<Slot<S>> for VecStructure<R> {
    fn from_iter<T: IntoIterator<Item = Slot<S>>>(iter: T) -> Self {
        Self {
            structure: iter.into_iter().map(|a| a.cast()).collect(),
        }
    }
}

impl<R: RepName> From<Vec<Slot<R>>> for VecStructure<R> {
    fn from(structure: Vec<Slot<R>>) -> Self {
        Self { structure }
    }
}

impl<R: RepName> IntoIterator for VecStructure<R> {
    type Item = Slot<R>;
    type IntoIter = std::vec::IntoIter<Slot<R>>;
    fn into_iter(self) -> std::vec::IntoIter<Slot<R>> {
        self.structure.into_iter()
    }
}

impl<'a, R: RepName> IntoIterator for &'a VecStructure<R> {
    type Item = &'a Slot<R>;
    type IntoIter = std::slice::Iter<'a, Slot<R>>;
    fn into_iter(self) -> std::slice::Iter<'a, Slot<R>> {
        self.structure.iter()
    }
}

impl<'a, R: RepName> IntoIterator for &'a mut VecStructure<R> {
    type Item = &'a mut Slot<R>;
    type IntoIter = std::slice::IterMut<'a, Slot<R>>;
    fn into_iter(self) -> std::slice::IterMut<'a, Slot<R>> {
        self.structure.iter_mut()
    }
}

impl<R: RepName> VecStructure<R> {
    pub fn new(structure: Vec<Slot<R>>) -> Self {
        Self { structure }
    }

    pub fn push(&mut self, item: Slot<R>) {
        self.structure.push(item)
    }

    fn extend(&mut self, other: Self) {
        self.structure.extend(other.structure)
    }

    pub fn to_named<N, A>(self, name: N, args: Option<A>) -> NamedStructure<N, A, R> {
        NamedStructure::from_iter(self, name, args)
    }

    pub fn empty() -> Self {
        Self { structure: vec![] }
    }
}

impl<R: RepName> From<ContractionCountStructure<R>> for VecStructure<R> {
    fn from(structure: ContractionCountStructure<R>) -> Self {
        structure.structure
    }
}

impl<N, A, R: RepName> From<NamedStructure<N, A, R>> for VecStructure<R> {
    fn from(structure: NamedStructure<N, A, R>) -> Self {
        structure.structure
    }
}

impl<N, A, R: RepName> From<SmartShadowStructure<N, A, R>> for VecStructure<R> {
    fn from(structure: SmartShadowStructure<N, A, R>) -> Self {
        structure.structure
    }
}

impl<N, A, R: RepName> From<HistoryStructure<N, A, R>> for VecStructure<R> {
    fn from(structure: HistoryStructure<N, A, R>) -> Self {
        structure.external.into()
    }
}

impl<R: RepName> From<VecStructure<R>> for ContractionCountStructure<R> {
    fn from(structure: VecStructure<R>) -> Self {
        Self {
            structure,
            contractions: 0,
        }
    }
}

impl<R: RepName> From<VecStructure<R>> for Vec<Slot<R>> {
    fn from(structure: VecStructure<R>) -> Self {
        structure.structure
    }
}

// const IDPRINTER: Lazy<BlockId<char>> = Lazy::new(|| BlockId::new(Alphabet::alphanumeric(), 1, 1));

impl<R: RepName> std::fmt::Display for VecStructure<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, item) in self.structure.iter().enumerate() {
            if index != 0 {
                // To avoid a newline at the start
                writeln!(f)?;
            }
            write!(
                f,
                "{:<3} ({})",
                usize::from(item.aind()),
                // IDPRINTER
                //     .encode_string(usize::from(item.index) as u64)
                //     .unwrap(),
                item.rep()
            )?;
        }
        Ok(())
    }
}

impl<R: RepName> ScalarStructure for VecStructure<R> {
    fn scalar_structure() -> Self {
        VecStructure { structure: vec![] }
    }
}

impl<R: RepName<Dual = R>> TensorStructure for VecStructure<R> {
    type Slot = Slot<R>;
    // type R = PhysicalReps;
    //
    fn dual(self) -> Self {
        self.structure.dual().into()
    }

    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;

            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}
#[cfg(feature = "shadowing")]
impl<R: RepName<Dual = R>> ToSymbolic for VecStructure<R> {}

impl<R: RepName<Dual = R>> StructureContract for VecStructure<R> {
    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.structure.merge(&other.structure)
    }

    fn trace_out(&mut self) {
        self.structure.trace_out();
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        Self {
            structure: self.structure.merge_at(&other.structure, positions),
        }
    }

    fn trace(&mut self, i: usize, j: usize) {
        self.structure.trace(i, j);
    }

    fn concat(&mut self, other: &Self) {
        self.structure.extend(other.structure.clone())
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Default, Hash)]
pub struct IndexlessNamedStructure<Name = String, Args = usize, R: RepName = LibraryRep> {
    pub structure: IndexLess<R>,
    pub global_name: Option<Name>,
    pub additional_args: Option<Args>,
}
impl<Name, Args, R: RepName<Dual = R>> TensorStructure for IndexlessNamedStructure<Name, Args, R> {
    type Slot = Slot<R>;

    fn dual(self) -> Self {
        Self {
            structure: self.structure.dual(),
            global_name: self.global_name,
            additional_args: self.additional_args,
        }
    }

    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}

impl<Name, Args, R: RepName> IndexlessNamedStructure<Name, Args, R> {
    #[must_use]
    pub fn from_iter<I, T>(iter: T, name: Name, args: Option<Args>) -> Self
    where
        I: RepName,
        R: From<I>,
        T: IntoIterator<Item = Representation<I>>,
    {
        Self {
            structure: iter.into_iter().map(|a| a.cast()).collect(),
            global_name: Some(name),
            additional_args: args,
        }
    }

    pub fn to_indexed(self, indices: &[AbstractIndex]) -> NamedStructure<Name, Args, R> {
        NamedStructure {
            structure: VecStructure::from_iter(self.structure.to_indexed(indices)),
            global_name: self.global_name,
            additional_args: self.additional_args,
        }
    }
}

impl<N, A, R: RepName> HasName for IndexlessNamedStructure<N, A, R>
where
    N: Clone,
    A: Clone,
{
    type Name = N;
    type Args = A;

    fn name(&self) -> Option<Self::Name> {
        self.global_name.clone()
    }
    fn set_name(&mut self, name: Self::Name) {
        self.global_name = Some(name);
    }
    fn args(&self) -> Option<Self::Args> {
        self.additional_args.clone()
    }
}

impl<N, A, R: RepName> From<IndexLess<R>> for IndexlessNamedStructure<N, A, R> {
    fn from(value: IndexLess<R>) -> Self {
        IndexlessNamedStructure {
            structure: value,
            global_name: None,
            additional_args: None,
        }
    }
}

#[cfg(feature = "shadowing")]
impl<N: IntoSymbol, A: IntoArgs, R: RepName> Display for IndexlessNamedStructure<N, A, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref name) = self.global_name {
            write!(f, "{}", name.ref_into_symbol())?;
        }
        write!(f, "(")?;
        if let Some(ref args) = self.additional_args {
            let args: Vec<std::string::String> =
                args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }

        write!(f, "{})", self.structure)?;
        Result::Ok(())
    }
}

/// A named structure is a structure with a global name, and a list of slots
///
/// It is useful when you want to shadow tensors, to nest tensor network contraction operations.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Default, Hash)]
pub struct NamedStructure<Name = String, Args = usize, R: RepName = LibraryRep> {
    pub structure: VecStructure<R>,
    pub global_name: Option<Name>,
    pub additional_args: Option<Args>,
}

#[cfg(feature = "shadowing")]
pub type AtomStructure<R> = NamedStructure<SerializableSymbol, Vec<SerializableAtom>, R>;

impl<Name, Args, R: RepName> NamedStructure<Name, Args, R> {
    #[must_use]
    pub fn from_iter<I, T>(iter: T, name: Name, args: Option<Args>) -> Self
    where
        R: From<I>,
        I: RepName,
        T: IntoIterator<Item = Slot<I>>,
    {
        Self {
            structure: iter.into_iter().map(|a| a.cast()).collect(),
            global_name: Some(name),
            additional_args: args,
        }
    }
}

#[derive(Error, Debug)]
pub enum StructureError {
    #[error("SlotError: {0}")]
    SlotError(#[from] SlotError),
    #[error("empty structure {0}")]
    EmptyStructure(SlotError),
}

#[cfg(feature = "shadowing")]
impl<'a, R: RepName<Dual = R>> TryFrom<FunView<'a>>
    for SmartShadowStructure<SerializableSymbol, Vec<SerializableAtom>, R>
{
    type Error = StructureError;
    fn try_from(value: FunView<'a>) -> std::result::Result<Self, Self::Error> {
        AtomStructure::<R>::try_from(value).map(|x| x.into())
    }
}

#[cfg(feature = "shadowing")]
impl<'a, R: RepName> TryFrom<FunView<'a>> for AtomStructure<R> {
    type Error = StructureError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == symbol!(ABSTRACTIND) => {
                let mut structure: Vec<Slot<R>> = vec![];

                for arg in value.iter() {
                    structure.push(arg.try_into()?);
                }

                Ok(VecStructure::from(structure).into())
            }
            name => {
                let mut structure: AtomStructure<R> = VecStructure::default().into();
                structure.set_name(name.into());
                let mut args = vec![];
                let mut is_structure = Some(SlotError::EmptyStructure);

                for arg in value.iter() {
                    let slot: Result<Slot<R>, _> = arg.try_into();

                    match slot {
                        Ok(slot) => {
                            is_structure = None;
                            structure.structure.push(slot);
                        }
                        Err(e) => {
                            if let AtomView::Fun(f) = arg {
                                if f.get_symbol() == symbol!(ABSTRACTIND) {
                                    let internal_s = AtomStructure::try_from(f);

                                    if let Ok(s) = internal_s {
                                        structure.extend(s);
                                        is_structure = None;
                                        continue;
                                    }
                                }
                            }
                            is_structure = Some(e);
                            args.push(arg.to_owned().into());
                        }
                    }
                }

                if !args.is_empty() {
                    structure.additional_args = Some(args);
                }
                if let Some(e) = is_structure {
                    Err(StructureError::EmptyStructure(e))
                } else {
                    Ok(structure)
                }
            }
        }
    }
}

impl<N, A, R: RepName> NamedStructure<N, Vec<A>, R> {
    pub fn extend(&mut self, other: Self) {
        let result = match (self.additional_args.take(), other.additional_args) {
            (Some(mut v1), Some(v2)) => {
                v1.extend(v2);
                Some(v1)
            }
            (None, Some(v2)) => Some(v2),
            (Some(v1), None) => Some(v1),
            (None, None) => None,
        };
        self.additional_args = result;
        self.structure.extend(other.structure);
    }
}

impl<N, A, R: RepName> From<VecStructure<R>> for NamedStructure<N, A, R> {
    fn from(value: VecStructure<R>) -> Self {
        Self {
            structure: value,
            global_name: None,
            additional_args: None,
        }
    }
}

/// A trait for a structure that has a name

impl<N, A, R: RepName> HasName for NamedStructure<N, A, R>
where
    N: Clone,
    A: Clone,
{
    type Name = N;
    type Args = A;

    fn name(&self) -> Option<Self::Name> {
        self.global_name.clone()
    }
    fn set_name(&mut self, name: Self::Name) {
        self.global_name = Some(name);
    }
    fn args(&self) -> Option<Self::Args> {
        self.additional_args.clone()
    }
}

pub trait HasName {
    type Name: Clone;
    type Args: Clone;
    fn name(&self) -> Option<Self::Name>;
    fn args(&self) -> Option<Self::Args>;
    fn set_name(&mut self, name: Self::Name);

    #[cfg(feature = "shadowing")]
    fn expanded_coef(&self, id: FlatIndex) -> ExpandedCoefficent<Self::Args>
    where
        Self: TensorStructure,
        Self::Name: IntoSymbol,
        Self::Args: IntoArgs,
    {
        ExpandedCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: self.co_expanded_index(id).unwrap(),
            args: self.args(),
        }
    }

    #[cfg(feature = "shadowing")]
    fn flat_coef(&self, id: FlatIndex) -> FlatCoefficent<Self::Args>
    where
        Self: TensorStructure,
        Self::Name: IntoSymbol,
        Self::Args: IntoArgs,
    {
        FlatCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: id,
            args: self.args(),
        }
    }
}

impl<N, A, R: RepName> ScalarStructure for NamedStructure<N, A, R> {
    fn scalar_structure() -> Self {
        NamedStructure {
            structure: VecStructure::default(),
            global_name: None,
            additional_args: None,
        }
    }
}

impl<N, A, R: RepName<Dual = R>> TensorStructure for NamedStructure<N, A, R> {
    type Slot = Slot<R>;
    // type R = PhysicalReps;

    fn dual(self) -> Self {
        NamedStructure {
            structure: self.structure.dual(),
            global_name: self.global_name,
            additional_args: self.additional_args,
        }
    }
    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}

#[cfg(feature = "shadowing")]
impl<N: IntoSymbol, A: IntoArgs, R: RepName> Display for NamedStructure<N, A, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref name) = self.global_name {
            write!(f, "{}", name.ref_into_symbol())?
        }
        write!(f, "(")?;
        if let Some(ref args) = self.additional_args {
            let args: Vec<std::string::String> =
                args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }

        write!(f, "{})", self.structure)?;
        Result::Ok(())
    }
}
impl<N, A, R: RepName<Dual = R>> StructureContract for NamedStructure<N, A, R> {
    delegate! {
        to self.structure{
            fn trace_out(&mut self);
            fn trace(&mut self, i: usize, j: usize);

        }
    }

    fn concat(&mut self, other: &Self) {
        self.structure.concat(&other.structure)
    }

    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.structure.merge(&other.structure)
    }

    /// when merging two named structures, the global name is lost
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        Self {
            structure: self.structure.merge_at(&other.structure, positions),
            global_name: None,
            additional_args: None,
        }
    }
}

/// A contraction count structure
///
/// Useful for tensor network contraction algorithm.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ContractionCountStructure<R: RepName> {
    pub structure: VecStructure<R>,
    pub contractions: usize,
}

impl<R: RepName, I: Into<Slot<R>>> FromIterator<I> for ContractionCountStructure<R> {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        Self {
            structure: iter.into_iter().map(I::into).collect(),
            contractions: 0,
        }
    }
}

pub trait TracksCount {
    fn contractions_num(&self) -> usize;

    fn is_composite(&self) -> bool {
        self.contractions_num() > 0
    }
}

impl<R: RepName> TracksCount for ContractionCountStructure<R> {
    fn contractions_num(&self) -> usize {
        self.contractions
    }
}

impl<R: RepName> ScalarStructure for ContractionCountStructure<R> {
    fn scalar_structure() -> Self {
        ContractionCountStructure {
            structure: VecStructure::default(),
            contractions: 0,
        }
    }
}

impl<R: RepName<Dual = R>> TensorStructure for ContractionCountStructure<R> {
    type Slot = Slot<R>;
    // type R = PhysicalReps;
    //
    fn dual(self) -> Self {
        ContractionCountStructure {
            structure: self.structure.dual(),
            contractions: self.contractions,
        }
    }

    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}

#[cfg(feature = "shadowing")]
impl<R: RepName<Dual = R>> ToSymbolic for ContractionCountStructure<R> {}

impl<R: RepName<Dual = R>> StructureContract for ContractionCountStructure<R> {
    delegate! {
        to self.structure{
            fn trace_out(&mut self);
            fn trace(&mut self, i: usize, j: usize);
        }
    }

    fn concat(&mut self, other: &Self) {
        self.structure.concat(&other.structure)
    }
    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.contractions += other.contractions + 1;
        self.structure.merge(&other.structure)
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        Self {
            structure: self.structure.merge_at(&other.structure, positions),
            contractions: self.contractions + other.contractions + 1,
        }
    }
}

/// A structure to enable smart shadowing of tensors in a tensor network contraction algorithm.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Hash)]
pub struct SmartShadowStructure<Name = String, Args = usize, R: RepName = LibraryRep> {
    pub structure: VecStructure<R>,
    pub contractions: usize,
    pub global_name: Option<Name>,
    additional_args: Option<Args>,
}

impl<Name, Args, R: RepName> SmartShadowStructure<Name, Args, R> {
    /// Constructs a new [`SmartShadow`] from a list of tuples of indices and dimension (assumes they are all euclidean), along with a name
    #[must_use]
    pub fn from_iter<I, T>(iter: T, name: Option<Name>, args: Option<Args>) -> Self
    where
        I: Into<Slot<R>>,
        T: IntoIterator<Item = I>,
    {
        Self {
            structure: iter.into_iter().map(I::into).collect(),
            global_name: name,
            additional_args: args,
            contractions: 0,
        }
    }
}

impl<N, A, R: RepName> HasName for SmartShadowStructure<N, A, R>
where
    N: Clone,
    A: Clone,
{
    type Name = N;
    type Args = A;

    fn name(&self) -> Option<Self::Name> {
        self.global_name.clone()
    }
    fn set_name(&mut self, name: Self::Name) {
        self.global_name = Some(name);
    }
    fn args(&self) -> Option<Self::Args> {
        self.additional_args.clone()
    }
}

#[cfg(feature = "shadowing")]
impl<N: IntoSymbol, A: IntoArgs, R: RepName> Display for SmartShadowStructure<N, A, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref name) = self.global_name {
            write!(f, "{}", name.ref_into_symbol())?
        }
        write!(f, "(")?;
        if let Some(ref args) = self.additional_args {
            let args: Vec<std::string::String> =
                args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }

        write!(f, "{})", self.structure)?;
        Result::Ok(())
    }
}

impl<N, A, R: RepName> ScalarStructure for SmartShadowStructure<N, A, R> {
    fn scalar_structure() -> Self {
        SmartShadowStructure {
            structure: VecStructure::default(),
            contractions: 0,
            global_name: None,
            additional_args: None,
        }
    }
}

impl<N, A, R: RepName<Dual = R>> TensorStructure for SmartShadowStructure<N, A, R> {
    type Slot = Slot<R>;
    // type R = PhysicalReps;
    //
    fn dual(self) -> Self {
        SmartShadowStructure {
            structure: self.structure.dual(),
            contractions: self.contractions,
            global_name: self.global_name,
            additional_args: self.additional_args,
        }
    }

    delegate! {
        to self.structure{
           fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}

impl<N, A, R: RepName> TracksCount for SmartShadowStructure<N, A, R> {
    fn contractions_num(&self) -> usize {
        self.contractions
    }
}

impl<N, A, R: RepName<Dual = R>> StructureContract for SmartShadowStructure<N, A, R> {
    fn concat(&mut self, other: &Self) {
        self.structure.concat(&other.structure)
    }
    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.contractions += other.contractions;
        self.structure.merge(&other.structure)
    }

    delegate! {
        to self.structure{
            fn trace_out(&mut self);
            fn trace(&mut self, i: usize, j: usize);
        }
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        SmartShadowStructure {
            structure: self.structure.merge_at(&other.structure, positions),
            contractions: self.contractions + other.contractions,
            global_name: None,
            additional_args: None,
        }
    }
}

impl<N, A, R: RepName<Dual = R>> From<NamedStructure<N, A, R>> for SmartShadowStructure<N, A, R> {
    fn from(value: NamedStructure<N, A, R>) -> Self {
        Self {
            structure: value.structure,
            contractions: 0,
            global_name: value.global_name,
            additional_args: value.additional_args,
        }
    }
}

/// A tracking structure
///
/// It contains two vecs of [`Slot`]s, one for the internal structure, simply extended during each contraction, and one external, coresponding to all the free indices
///
/// It enables keeping track of the contraction history of the tensor, mostly for debugging and display purposes.
/// A [`SymbolicTensor`] can also be used in this way, however it needs a symbolica state and workspace during contraction.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct HistoryStructure<Name, Args = (), R: RepName = LibraryRep> {
    internal: VecStructure<R>,
    pub names: AHashMap<Range<usize>, Name>, //ideally this is a named partion.. maybe a btreemap<usize, N>, and the range is from previous to next
    external: NamedStructure<Name, Args, R>,
}

impl<N, A, R: RepName<Dual = R>> From<NamedStructure<N, A, R>> for HistoryStructure<N, A, R>
where
    N: Clone,
{
    fn from(external: NamedStructure<N, A, R>) -> Self {
        Self {
            internal: external.structure.clone(),
            names: AHashMap::from([(0..external.order(), external.global_name.clone().unwrap())]),
            external,
        }
    }
}

impl<N, A, R: RepName<Dual = R>> HistoryStructure<N, A, R> {
    /// make the indices in the internal index list of self independent from the indices in the internal index list of other
    /// This is done by shifting the indices in the internal index list of self by the the maximum index present.
    pub fn independentize_internal(&mut self, other: &Self) {
        let internal_set: HashSet<<Self as TensorStructure>::Slot> = self
            .internal
            .clone()
            .into_iter()
            .filter(|s| self.external.contains_matching(s))
            .collect();

        let other_set: HashSet<<Self as TensorStructure>::Slot> =
            other.internal.clone().into_iter().collect();

        let mut replacement_value = internal_set
            .union(&other_set)
            .map(|s| s.aind())
            .max()
            .unwrap_or(0.into())
            + 1.into();

        for item in &mut self.internal {
            if other_set.contains(item) {
                item.set_aind(replacement_value);
                replacement_value += 1.into();
            }
        }
    }
}

impl<N, A, R: RepName> HasName for HistoryStructure<N, A, R>
where
    N: Clone,
    A: Clone,
{
    type Name = N;
    type Args = A;
    delegate! {
        to self.external {
            fn name(&self) -> Option<Self::Name>;
            fn set_name(&mut self, name: Self::Name);
            fn args(&self) -> Option<Self::Args>;
        }
    }
}

impl<N, A, R: RepName> TracksCount for HistoryStructure<N, A, R> {
    /// Since each time we contract, we merge the name maps, the amount of contractions, is the size of the name map
    /// This function returns the number of contractions thus computed
    fn contractions_num(&self) -> usize {
        self.names.len()
    }
}

impl<N, A, R: RepName> ScalarStructure for HistoryStructure<N, A, R> {
    fn scalar_structure() -> Self {
        HistoryStructure {
            internal: VecStructure::default(),
            names: AHashMap::default(),
            external: NamedStructure::scalar_structure(),
        }
    }
}

impl<N, A, R: RepName<Dual = R>> TensorStructure for HistoryStructure<N, A, R> {
    type Slot = Slot<R>;
    // type R = PhysicalReps;
    //
    fn dual(self) -> Self {
        Self {
            internal: self.internal,
            names: self.names,
            external: self.external.dual(),
        }
    }

    delegate! {
        to self.external{
           fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
    /// checks if internally, the two tensors are the same. This implies that the external indices are the same
    fn same_content(&self, other: &Self) -> bool {
        let set1: HashSet<_> = (&self.internal).into_iter().collect();
        let set2: HashSet<_> = (&other.internal).into_iter().collect();
        set1 == set2
        // TODO: check names
    }
}

// impl TensorStructure for [Slot] {
//     type Structure = [Slot];

//     fn external_structure(&self) -> &[Slot] {
//         self
//     }
// }

impl<N, A, R: RepName<Dual = R>> StructureContract for HistoryStructure<N, A, R>
where
    N: Clone,
    A: Clone,
{
    fn concat(&mut self, other: &Self) {
        self.external.structure.concat(&other.external.structure)
    }
    /// remove the repeated indices in the external index list
    fn trace_out(&mut self) {
        let mut positions = IndexMap::new();

        // Track the positions of each element
        for (index, value) in (self.external).external_structure_iter().enumerate() {
            positions.entry(value).or_insert_with(Vec::new).push(index);
        }
        // Collect only the positions of non- repeated elements

        self.external.structure = positions
            .into_iter()
            .filter_map(|(value, indices)| {
                if indices.len() == 1 {
                    Some(value)
                } else {
                    None
                }
            })
            .collect();
    }

    /// remove the given indices from the external index list
    fn trace(&mut self, i: usize, j: usize) {
        if i < j {
            self.trace(j, i);
            return;
        }
        let a = self.external.structure.structure.remove(i);
        let b = self.external.structure.structure.remove(j);
        assert_eq!(a, b);
    }

    /// essentially contract.
    fn merge(&mut self, other: &Self) -> Option<usize> {
        let shift = self.internal.order();
        for (range, name) in &other.names {
            self.names
                .insert((range.start + shift)..(range.end + shift), name.clone());
        }
        self.trace_out();
        self.independentize_internal(other);
        self.internal
            .structure
            .append(&mut other.internal.structure.clone());
        self.external.merge(&other.external)
    }

    /// Merge two [`HistoryStructure`] at the given positions of the external index list. Ideally the internal index list should be independentized before merging
    /// This is essentially a contraction of only one index. The name maps are merged, and shifted accordingly. The global name is lost, since the resulting tensor is composite
    /// The global name can be set again with the [`Self::set_global_name`] function
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        let external = self.external.merge_at(&other.external, positions);

        let mut slots_self_int = self.internal.clone();
        let slots_other_int = other.internal.clone();
        slots_self_int.extend(slots_other_int);

        let mut names = self.names.clone();
        let shift = self.internal.order();
        for (range, name) in &other.names {
            names.insert((range.start + shift)..(range.end + shift), name.clone());
        }
        HistoryStructure {
            internal: slots_self_int,
            external,
            names,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct TensorShell<S: TensorStructure> {
    structure: S,
}

impl<S: TensorStructure + ScalarStructure> ScalarTensor for TensorShell<S> {
    fn new_scalar(_scalar: Self::Scalar) -> Self {
        TensorShell {
            structure: S::scalar_structure(),
        }
    }
}

impl<S: TensorStructure> HasStructure for TensorShell<S> {
    type Structure = S;
    type Scalar = ();
    fn structure(&self) -> &S {
        &self.structure
    }

    fn map_same_structure(mut self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        self.structure = f(self.structure);
        self
    }

    fn scalar(self) -> Option<Self::Scalar> {
        if self.structure.is_scalar() {
            Some(())
        } else {
            None
        }
    }
    fn mut_structure(&mut self) -> &mut S {
        &mut self.structure
    }
}

impl<S: TensorStructure> HasName for TensorShell<S>
where
    S: HasName,
{
    type Args = S::Args;
    type Name = S::Name;

    fn args(&self) -> Option<Self::Args> {
        self.structure.args()
    }

    fn name(&self) -> Option<Self::Name> {
        self.structure.name()
    }

    fn set_name(&mut self, name: Self::Name) {
        self.structure.set_name(name);
    }
}

// impl<I> HasName for I
// where
//     I: HasStructure,
//     I::Structure: HasName,
// {
//     type Name = <I::Structure as HasName>::Name;
//     fn name(&self) -> Option<Cow<Self::Name>> {
//         self.structure().name()
//     }
//     fn set_name(&mut self, name: &Self::Name) {
//         self.mut_structure().set_name(name);
//     }
// }

impl<S: TensorStructure> TensorShell<S> {
    pub fn new(structure: S) -> Self {
        Self { structure }
    }
}

impl<S: TensorStructure, O: From<S> + TensorStructure> CastStructure<TensorShell<O>>
    for TensorShell<S>
{
    fn cast_structure(self) -> TensorShell<O> {
        TensorShell {
            structure: self.structure.into(),
        }
    }
}

impl<S: TensorStructure> From<S> for TensorShell<S> {
    fn from(structure: S) -> Self {
        Self::new(structure)
    }
}

#[cfg(feature = "shadowing")]
impl<N, A> std::fmt::Display for HistoryStructure<N, A>
where
    N: Display + Clone + IntoSymbol,
    A: Display + Clone + IntoArgs,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut string = String::new();
        if let Some(global_name) = self.name() {
            string.push_str(&format!("{global_name}:"));
        }
        for (range, name) in self
            .names
            .iter()
            .filter(|(r, _)| *r != &(0..self.internal.order()) || !self.is_composite())
        {
            string.push_str(&format!("{name}("));
            for slot in &self.internal.structure[range.clone()] {
                string.push_str(&format!("{slot},"));
            }
            string.pop();
            string.push(')');
        }
        write!(f, "{string}")
    }
}

// pub struct Kroneker {
//     structure: VecStructure,
// }

// impl Kroneker {
//     pub fn new<T: Rep>(i: GenSlot<T>, j: GenSlot<T::Dual>) -> Self {}
// }

#[cfg(test)]
#[cfg(feature = "shadowing")]
mod shadowing_tests {
    use super::representation::Lorentz;
    use super::*;
    use symbolica::atom::AtomCore;
    use symbolica::parse;

    #[test]
    fn named_structure_from_atom() {
        let expr = parse!("p(1,mu,aind(lor(4,4)))").unwrap();

        if let AtomView::Fun(f) = expr.as_atom_view() {
            let named_structure = AtomStructure::<Lorentz>::try_from(f);
            match named_structure {
                Ok(named_structure) => println!("{}", named_structure),
                Err(e) => println!("{}", e),
            }
        }
    }
}

use indexmap::IndexMap;
use linnet::permutation::{Permutation, PermutationInvIter, PermutationInvIterMut};

use super::{
    abstract_index::AbstractIndex,
    dimension::Dimension,
    representation::{LibraryRep, RepName, Representation},
    slot::{ConstructibleSlot, DualSlotTo, IsAbstractSlot, Slot},
    IndexlessNamedStructure, NamedStructure, OrderedStructure, ScalarStructure,
    SmartShadowStructure, StructureContract, StructureError, TensorStructure,
};

use anyhow::{anyhow, Result};

#[cfg(feature = "shadowing")]
use super::ToSymbolic;

#[cfg(not(feature = "shadowing"))]
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
// #[cfg_attr(not(feature = "shadowing"), derive(Serialize, Deserialize))]
pub struct PermutedStructure<R: RepName = LibraryRep> {
    pub structure: Vec<Slot<R>>,
    pub permutation: Permutation,
}

// impl<R: RepName<Dual = R>> TensorStructure for OrderedStructure<R> {
//     type Slot = Slot<R>;
//     type Indexed = Self;

//     fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
//         Ok(Self {
//             structure: self.structure.reindex(indices)?,
//         })
//     }
//     fn dual(self) -> Self {
//         self.structure.dual().into()
//     }

//     delegate! {
//         to self.structure{
//             fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;

//             fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
//             fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
//             fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
//             fn order(&self) -> usize;
//             fn get_slot(&self, i: usize) -> Option<Self::Slot>;
//             fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
//             fn get_aind(&self,i:usize)->Option<AbstractIndex>;
//             fn get_dim(&self, i: usize) -> Option<Dimension>;
//         }
//     }
// }

impl<R: RepName> Default for PermutedStructure<R> {
    fn default() -> Self {
        Self {
            structure: vec![],
            permutation: Permutation::id(0),
        }
    }
}

impl<S: RepName, R: From<S> + RepName> FromIterator<Slot<S>> for PermutedStructure<R> {
    fn from_iter<T: IntoIterator<Item = Slot<S>>>(iter: T) -> Self {
        let mut structure = iter.into_iter().map(|a| a.cast()).collect();
        let permutation = Permutation::sort(&structure);
        permutation.apply_slice_in_place(&mut structure);

        Self {
            structure,
            permutation,
        }
    }
}

impl<R: RepName> From<Vec<Slot<R>>> for PermutedStructure<R> {
    fn from(mut structure: Vec<Slot<R>>) -> Self {
        let permutation = Permutation::sort(&structure);
        permutation.apply_slice_in_place(&mut structure);

        Self {
            structure,
            permutation,
        }
    }
}

impl<R: RepName> IntoIterator for PermutedStructure<R> {
    type Item = Slot<R>;
    type IntoIter = std::vec::IntoIter<Slot<R>>;
    fn into_iter(mut self) -> std::vec::IntoIter<Slot<R>> {
        self.permutation
            .apply_slice_in_place_inv(&mut self.structure);
        self.structure.into_iter()
    }
}

impl<'a, R: RepName> IntoIterator for &'a PermutedStructure<R> {
    type Item = &'a Slot<R>;
    type IntoIter = PermutationInvIter<'a, Slot<R>>;
    fn into_iter(self) -> PermutationInvIter<'a, Slot<R>> {
        self.permutation.iter_slice_inv(&self.structure)
    }
}

impl<'a, R: RepName> IntoIterator for &'a mut PermutedStructure<R> {
    type Item = &'a mut Slot<R>;
    type IntoIter = PermutationInvIterMut<'a, Slot<R>>;
    fn into_iter(self) -> PermutationInvIterMut<'a, Slot<R>> {
        self.permutation.iter_slice_inv_mut(&mut self.structure)
    }
}

impl<R: RepName> PermutedStructure<R> {
    pub fn new(structure: Vec<Slot<R>>) -> Self {
        structure.into()
    }

    fn extend(&mut self, mut other: Self) {
        self.permutation.iter_slice_inv(&mut self.structure);
        other.permutation.iter_slice_inv(&mut other.structure);

        self.structure.extend(other.structure);

        let new_perm = Permutation::sort(&self.structure);

        new_perm.apply_slice_in_place(&mut self.structure);
        self.permutation = new_perm;
    }

    pub fn to_named<N, A>(self, name: N, args: Option<A>) -> NamedStructure<N, A, R> {
        NamedStructure::from_iter(self, name, args)
    }

    pub fn empty() -> Self {
        Self::default()
    }
}

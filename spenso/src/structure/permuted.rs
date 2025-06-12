use std::{f64::consts::LN_2, ops::Deref};

use linnet::permutation::Permutation;

use super::{
    representation::RepName,
    slot::{IsAbstractSlot, Slot},
    OrderedStructure, TensorStructure,
};

pub struct PermutedStructure<S> {
    pub structure: S,
    pub permutation: Permutation,
}

impl<S> Deref for PermutedStructure<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.structure
    }
}

impl<S> PermutedStructure<S> {
    pub fn map_structure<U>(self, f: impl FnOnce(S) -> U) -> PermutedStructure<U> {
        PermutedStructure {
            structure: f(self.structure),
            permutation: self.permutation,
        }
    }

    pub fn permute(self) -> S::Permuted
    where
        S: PermuteTensor,
    {
        self.structure.permute(&self.permutation)
        // let mut structure = self.structure.external_structure();
        // self.permutation.apply_slice_in_place_inv(&mut structure);
        // for s in structure {
        //     let dummy = s.to_dummy();
        // }
    }
}

pub trait PermuteTensor {
    type Permuted;
    type Id;
    type IdSlot;

    fn permute(self, permutation: &Permutation) -> Self::Permuted;

    fn id(i: Self::IdSlot, j: Self::IdSlot) -> Self::Id;
}

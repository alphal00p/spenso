use std::{f64::consts::LN_2, ops::Deref};

use linnet::permutation::Permutation;
use num::one;

use super::{
    abstract_index::AbstractIndex,
    representation::RepName,
    slot::{IsAbstractSlot, Slot},
    OrderedStructure, StructureError, TensorStructure,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    pub fn reindex<I: IntoIterator<Item: Into<AbstractIndex>>>(
        self,
        indices: I,
    ) -> Result<PermutedStructure<PermutedStructure<S::Indexed>>, StructureError>
    where
        S: TensorStructure,
    {
        let indices = indices.into_iter().map(|i| i.into()).collect::<Vec<_>>();
        let structure = self.structure.reindex(&indices)?;
        println!("Og:{}", self.permutation);
        println!("New:{}", structure.permutation);
        Ok(PermutedStructure {
            structure: structure,
            permutation: self.permutation,
        })
    }
}
pub trait Perm {
    type Permuted;
    fn permute(self) -> Self::Permuted;
}

impl<S> Perm for PermutedStructure<PermutedStructure<S>>
where
    S: PermuteTensor,
{
    type Permuted = S::Permuted;
    fn permute(self) -> Self::Permuted {
        self.structure
            .structure
            .permute_reps(&self.permutation, &self.structure.permutation)
    }
}

impl<S> Perm for PermutedStructure<S>
where
    S: PermuteTensor,
{
    type Permuted = S::Permuted;
    fn permute(self) -> Self::Permuted {
        self.structure.permute(&self.permutation)
    }
}

pub trait PermuteTensor {
    type Permuted;
    type Id;
    type IdSlot;

    fn permute(self, permutation: &Permutation) -> Self::Permuted;

    fn permute_reps(self, ind_perm: &Permutation, rep_perm: &Permutation) -> Self::Permuted;
    fn id(i: Self::IdSlot, j: Self::IdSlot) -> Self::Id;
}

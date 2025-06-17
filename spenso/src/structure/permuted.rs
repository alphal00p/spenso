use std::{f64::consts::LN_2, fmt::Display, ops::Deref};

use bincode_trait_derive::{Decode, Encode};
use linnet::permutation::Permutation;
use num::one;
use serde::{Deserialize, Serialize};

use super::{
    abstract_index::AbstractIndex,
    representation::RepName,
    slot::{IsAbstractSlot, Slot},
    OrderedStructure, StructureError, TensorStructure,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, Encode, Decode)]
pub struct PermutedStructure<S> {
    pub structure: S,
    pub permutation: Permutation,
}

impl<S: Display> Display for PermutedStructure<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.permutation, self.structure)
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
    ) -> Result<PermutedStructure<S::Indexed>, StructureError>
    where
        S: TensorStructure,
    {
        let mut indices = indices.into_iter().map(|i| i.into()).collect::<Vec<_>>();

        self.permutation.apply_slice_in_place_inv(&mut indices);
        let structure = self.structure.reindex(&indices)?;
        println!("Rep:{}", self.permutation);
        println!("Ind:{}", structure.permutation);
        Ok(structure)
    }
}
pub trait Perm: Sized {
    type Permuted;
    type Wrapped<P>;
    fn permute(self) -> Self::Permuted;
    fn permute_wrapped(self) -> Self::Wrapped<Self::Permuted>;
}

impl<S> Perm for PermutedStructure<PermutedStructure<S>>
where
    S: PermuteTensor,
{
    type Permuted = S::Permuted;
    type Wrapped<P> = PermutedStructure<PermutedStructure<P>>;
    fn permute(self) -> Self::Permuted {
        self.structure
            .structure
            .permute_reps(&self.structure.permutation, &self.permutation)
    }
    fn permute_wrapped(self) -> Self::Wrapped<Self::Permuted> {
        PermutedStructure {
            structure: PermutedStructure {
                structure: self
                    .structure
                    .structure
                    .permute_reps(&self.structure.permutation, &self.permutation),
                permutation: self.structure.permutation,
            },
            permutation: self.permutation,
        }
    }
}

impl<S> Perm for PermutedStructure<S>
where
    S: PermuteTensor,
{
    type Wrapped<P> = PermutedStructure<P>;
    type Permuted = S::Permuted;
    fn permute(self) -> Self::Permuted {
        self.structure.permute(&self.permutation)
    }
    fn permute_wrapped(self) -> Self::Wrapped<Self::Permuted> {
        PermutedStructure {
            structure: self
                .structure
                .permute_reps(&self.permutation, &self.permutation),
            permutation: self.permutation,
        }
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

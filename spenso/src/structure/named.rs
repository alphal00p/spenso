use std::fmt::Display;

use indexmap::IndexMap;
use linnet::permutation::Permutation;

use super::{
    abstract_index::AbstractIndex,
    dimension::Dimension,
    representation::{LibraryRep, RepName, Representation},
    slot::{DualSlotTo, IsAbstractSlot, Slot, SlotError},
    HasName, IndexlessNamedStructure, MergeInfo, OrderedStructure, ScalarStructure,
    SmartShadowStructure, StructureContract, StructureError, TensorStructure,
};

use anyhow::{anyhow, Result};
use delegate::delegate;

#[cfg(feature = "shadowing")]
use symbolica::atom::{
    representation::{FunView, MulView},
    AtomView,
};

#[cfg(feature = "shadowing")]
use crate::{
    shadowing::symbolica_utils::{IntoArgs, IntoSymbol, SerializableAtom, SerializableSymbol},
    structure::abstract_index::AIND_SYMBOLS,
    structure::slot::ConstructibleSlot,
    tensors::data::DenseTensor,
    tensors::parametric::{ExpandedCoefficent, FlatCoefficent, TensorCoefficient},
};

#[cfg(not(feature = "shadowing"))]
use serde::{Deserialize, Serialize};
/// A named structure is a structure with a global name, and a list of slots
///
/// It is useful when you want to shadow tensors, to nest tensor network contraction operations.
#[derive(
    Clone,
    PartialEq,
    Eq,
    Debug,
    Default,
    Hash,
    bincode_trait_derive::Encode,
    bincode_trait_derive::Decode,
)]
#[cfg_attr(not(feature = "shadowing"), derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
pub struct NamedStructure<Name = String, Args = usize, R: RepName = LibraryRep> {
    pub structure: OrderedStructure<R>,
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
impl<'a, R: RepName<Dual = R>> TryFrom<FunView<'a>> for AtomStructure<R> {
    type Error = StructureError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == AIND_SYMBOLS.aind => {
                let mut structure: Vec<Slot<R>> = vec![];

                for arg in value.iter() {
                    structure.push(arg.try_into()?);
                }

                Ok(OrderedStructure::from(structure).into())
            }
            name => {
                let mut structure: AtomStructure<R> = OrderedStructure::default().into();
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
                                if f.get_symbol() == AIND_SYMBOLS.aind {
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

impl<N, A, R: RepName<Dual = R>> NamedStructure<N, Vec<A>, R> {
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
        self.structure.concat(other.structure);
    }
}

impl<N, A, R: RepName> From<OrderedStructure<R>> for NamedStructure<N, A, R> {
    fn from(value: OrderedStructure<R>) -> Self {
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

impl<N, A, R: RepName> ScalarStructure for NamedStructure<N, A, R> {
    fn scalar_structure() -> Self {
        NamedStructure {
            structure: OrderedStructure::default(),
            global_name: None,
            additional_args: None,
        }
    }
}

impl<N, A, R: RepName<Dual = R>> TensorStructure for NamedStructure<N, A, R> {
    type Slot = Slot<R>;
    // type R = PhysicalReps;
    type Indexed = Self;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        Ok(Self {
            global_name: self.global_name,
            additional_args: self.additional_args,
            structure: self.structure.reindex(indices)?,
        })
    }

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

    fn concat(&mut self, other: Self) {
        self.structure.concat(other.structure)
    }

    fn merge(
        &self,
        other: &Self,
    ) -> Result<(Self, Vec<usize>, Vec<usize>, MergeInfo), StructureError> {
        let (structure, pos_self, pos_other, mergeinfo) = self.structure.merge(&other.structure)?;
        Ok((
            Self {
                structure,
                global_name: None,
                additional_args: None,
            },
            pos_self,
            pos_other,
            mergeinfo,
        ))
    }
}

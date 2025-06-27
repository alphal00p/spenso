use std::fmt::Display;

use indexmap::IndexMap;
use linnet::permutation::Permutation;
use tabled::{builder::Builder, settings::Style};

use super::{
    abstract_index::AbstractIndex,
    dimension::Dimension,
    permuted::PermuteTensor,
    representation::{LibraryRep, RepName, Representation},
    slot::{DualSlotTo, IsAbstractSlot, Slot, SlotError},
    HasName, IndexlessNamedStructure, MergeInfo, OrderedStructure, PermutedStructure,
    ScalarStructure, SmartShadowStructure, StructureContract, StructureError, TensorStructure,
};
use bitvec::{order::Lsb0, vec::BitVec};

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
    Clone, PartialEq, Eq, Default, Hash, bincode_trait_derive::Encode, bincode_trait_derive::Decode,
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
    pub fn from_iter<I, T>(iter: T, name: Name, args: Option<Args>) -> PermutedStructure<Self>
    where
        R: From<I>,
        I: RepName,
        T: IntoIterator<Item = Slot<I>>,
    {
        iter.into_iter()
            .map(|a| a.cast())
            .collect::<PermutedStructure<_>>()
            .map_structure(move |structure| Self {
                structure,
                global_name: Some(name),
                additional_args: args,
            })
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

pub trait IdentityName {
    fn id() -> Self;
}
pub const ID_NAME: &'static str = "𝟙";
impl IdentityName for String {
    fn id() -> Self {
        ID_NAME.to_string()
    }
}

impl<N: IdentityName, A, R: RepName<Dual = R>> PermuteTensor for NamedStructure<N, A, R> {
    type Id = Self;
    type IdSlot = Slot<R>;
    type Permuted = (
        NamedStructure<N, A, LibraryRep>,
        Vec<NamedStructure<N, A, LibraryRep>>,
    );

    fn id(i: Self::IdSlot, j: Self::IdSlot) -> Self::Id {
        Self {
            structure: OrderedStructure::id(i, j),
            global_name: Some(N::id()),
            additional_args: None,
        }
    }

    fn permute(self, permutation: &Permutation) -> Self::Permuted {
        let mut dummy_structure = Vec::new();
        let mut ids = Vec::new();

        for s in permutation.iter_slice_inv(&self.structure.structure) {
            let d = s.to_dummy();
            let ogs = s.to_lib();
            dummy_structure.push(d);
            ids.push(NamedStructure::id(d, ogs));
        }
        let strct = OrderedStructure::new(dummy_structure);
        if !strct.index_permutation.is_identity() {
            panic!("should be identity")
        }

        (
            NamedStructure {
                global_name: self.global_name,
                additional_args: self.additional_args,
                structure: strct.structure,
            },
            ids,
        )
    }

    fn permute_reps(self, ind_perm: &Permutation, rep_perm: &Permutation) -> Self::Permuted {
        let mut dummy_structure = Vec::new();
        let mut og_reps = Vec::new();
        let mut ids = Vec::new();

        if rep_perm.is_identity() {
            return self.permute(ind_perm);
        }
        for s in rep_perm.iter_slice(&self.structure.structure) {
            og_reps.push(s.rep.to_lib());
            let d = s.to_dummy();
            dummy_structure.push(d);
        }

        for (i, s) in ind_perm.iter_slice(&self.structure.structure).enumerate() {
            let d = dummy_structure[i];
            let new_slot = og_reps[i].slot(s.aind);

            ids.push(NamedStructure::id(d, new_slot));
        }
        let strct = OrderedStructure::new(dummy_structure);
        if !strct.index_permutation.is_identity() {
            panic!("should be identity")
        }
        (
            NamedStructure {
                global_name: self.global_name,
                additional_args: self.additional_args,
                structure: strct.structure,
            },
            ids,
        )
    }
}

impl<N, A, R: RepName<Dual = R>> TensorStructure for NamedStructure<N, A, R> {
    type Slot = Slot<R>;
    // type R = PhysicalReps;
    type Indexed = Self;

    // fn id(i: Self::Slot, j: Self::Slot) -> Self {
    //     Self {
    //         structure: OrderedStructure::id(i, j),
    //         global_name: Some(N::id()),
    //         additional_args: None,
    //     }
    // }

    fn reindex(
        self,
        indices: &[AbstractIndex],
    ) -> Result<PermutedStructure<Self::Indexed>, StructureError> {
        let res = self.structure.reindex(indices)?;

        Ok(PermutedStructure {
            rep_permutation: res.rep_permutation,
            structure: Self {
                global_name: self.global_name,
                additional_args: self.additional_args,
                structure: res.structure,
            },
            index_permutation: res.index_permutation,
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

pub trait ArgDisplay {
    fn arg_display(&self) -> String;

    fn arg_debug(&self) -> String;
}

pub trait ArgDisplayMarker {}

impl<T: std::fmt::Display + ArgDisplayMarker + std::fmt::Debug> ArgDisplay for T {
    fn arg_display(&self) -> String {
        self.to_string()
    }

    fn arg_debug(&self) -> String {
        format!("{:?}", self)
    }
}

impl ArgDisplayMarker for () {}
impl ArgDisplayMarker for usize {}
impl ArgDisplayMarker for isize {}
impl ArgDisplayMarker for f64 {}
impl ArgDisplayMarker for f32 {}
impl ArgDisplayMarker for i8 {}
impl ArgDisplayMarker for i16 {}
impl ArgDisplayMarker for i32 {}
impl ArgDisplayMarker for i64 {}

impl ArgDisplayMarker for u8 {}
impl ArgDisplayMarker for u16 {}
impl ArgDisplayMarker for u32 {}
impl ArgDisplayMarker for u64 {}

#[cfg(feature = "shadowing")]
impl ArgDisplayMarker for symbolica::atom::Atom {}
#[cfg(feature = "shadowing")]
impl ArgDisplayMarker for symbolica::atom::Symbol {}

#[cfg(feature = "shadowing")]
impl ArgDisplay for Vec<symbolica::atom::Atom> {
    fn arg_display(&self) -> String {
        self.iter()
            .map(|a| a.arg_display())
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn arg_debug(&self) -> String {
        self.iter()
            .map(|a| a.arg_debug())
            .collect::<Vec<String>>()
            .join(", ")
    }
}

impl<N: std::fmt::Display, A: ArgDisplay, R: RepName> std::fmt::Display
    for NamedStructure<N, A, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut table = Builder::new();

        table.push_record(&[
            self.global_name
                .as_ref()
                .map(|a| format!("{a}"))
                .unwrap_or("NO NAME".to_string()),
            self.additional_args
                .as_ref()
                .map(|a| a.arg_display())
                .unwrap_or("".to_string()),
        ]);
        for (index, item) in self.structure.structure.iter().enumerate() {
            if item.rep.rep.is_self_dual() {
                table.push_record(&[item.rep.to_string(), format!("{}", item.aind)]);
            } else if item.rep.rep.is_base() {
                table.push_record(&[item.rep.to_string(), format!("{:+}", item.aind)]);
            } else {
                table.push_record(&[item.rep.to_string(), format!("{:-}", item.aind)]);
            }
        }
        writeln!(f)?;
        table.build().with(Style::rounded()).fmt(f)
    }
}
impl<N: std::fmt::Debug, A: ArgDisplay, R: RepName> std::fmt::Debug for NamedStructure<N, A, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut table = Builder::new();

        table.push_record(&[
            self.global_name
                .as_ref()
                .map(|a| format!("{:?}", a))
                .unwrap_or("NO NAME".to_string()),
            self.additional_args
                .as_ref()
                .map(|a| a.arg_debug())
                .unwrap_or("".to_string()),
        ]);
        for (index, item) in self.structure.structure.iter().enumerate() {
            table.push_record(&[
                index.to_string(),
                format!("{:?}", item.rep.rep),
                format!("{:?}", item.rep.dim),
                format!("{:?}", item.aind),
            ]);
        }
        writeln!(f)?;
        write!(f, "{}", format!("{}", table.build().with(Style::rounded())))
    }
}
impl<N, A, R: RepName<Dual = R>> StructureContract for NamedStructure<N, A, R> {
    delegate! {
        to self.structure{
            fn trace_out(&mut self);
            fn trace(&mut self, i: usize, j: usize);

        }
    }

    fn merge(&self, other: &Self) -> Result<(Self, BitVec, BitVec, MergeInfo), StructureError> {
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

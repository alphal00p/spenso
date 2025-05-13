use bitvec::vec::BitVec;
use indexmap::IndexMap;

use crate::utils::MergeOrdered;

#[cfg(feature = "shadowing")]
use super::{ToSymbolic, AIND_SYMBOLS};

use anyhow::anyhow;
use delegate::delegate;
#[cfg(feature = "shadowing")]
use symbolica::atom::{
    representation::{FunView, MulView},
    AtomView,
};

use super::{
    abstract_index::AbstractIndex,
    dimension::Dimension,
    representation::{LibraryRep, RepName, Representation},
    slot::{ConstructibleSlot, DualSlotTo, IsAbstractSlot, Slot, SlotError},
    MergeInfo, NamedStructure, ScalarStructure, SmartShadowStructure, StructureContract,
    StructureError, TensorStructure,
};
#[cfg(not(feature = "shadowing"))]
use serde::{Deserialize, Serialize};

#[derive(
    Clone, PartialEq, Eq, Debug, Hash, bincode_trait_derive::Encode, bincode_trait_derive::Decode,
)]
#[cfg_attr(not(feature = "shadowing"), derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
pub struct OrderedStructure<R: RepName = LibraryRep> {
    pub structure: Vec<Slot<R>>,
}

impl<R: RepName<Dual = R>> TensorStructure for OrderedStructure<R> {
    type Slot = Slot<R>;
    type Indexed = Self;
    // type R = PhysicalReps;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        if self.structure.len() != indices.len() {
            return Err(StructureError::WrongNumberOfArguments(
                self.structure.len(),
                indices.len(),
            ));
        }

        Ok(self
            .into_iter()
            .zip(indices)
            .map(|(s, index)| s.reindex(*index))
            .collect())
    }

    fn dual(self) -> Self {
        self.into_iter().map(|s| s.dual()).collect()
    }
    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.structure.iter().map(|s| s.rep())
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex> {
        self.structure.iter().map(|s| s.aind())
    }

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot> {
        self.structure.iter().cloned()
    }

    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension> {
        self.structure.iter().map(|s| s.dim())
    }

    fn order(&self) -> usize {
        self.structure.len()
    }

    fn get_slot(&self, i: usize) -> Option<Slot<R>> {
        self.structure.get(i).cloned()
    }

    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.structure.get(i).map(|s| s.rep())
    }

    fn get_dim(&self, i: usize) -> Option<Dimension> {
        self.structure.get(i).map(|s| s.dim())
    }

    fn get_aind(&self, i: usize) -> Option<AbstractIndex> {
        self.structure.get(i).map(|s| s.aind())
    }
}

impl<R: RepName> Default for OrderedStructure<R> {
    fn default() -> Self {
        Self { structure: vec![] }
    }
}

#[cfg(feature = "shadowing")]
impl<R: RepName> TryFrom<AtomView<'_>> for OrderedStructure<R> {
    type Error = SlotError;
    fn try_from(value: AtomView) -> Result<Self, Self::Error> {
        match value {
            AtomView::Mul(mul) => mul.try_into(),
            AtomView::Fun(fun) => fun.try_into(),
            AtomView::Pow(_) => {
                Ok(OrderedStructure::<R>::default()) // powers do not have a structure
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
impl<R: RepName> TryFrom<FunView<'_>> for OrderedStructure<R> {
    type Error = SlotError;
    fn try_from(value: FunView) -> Result<Self, Self::Error> {
        if value.get_symbol() == AIND_SYMBOLS.aind {
            let mut structure: Vec<Slot<R>> = vec![];

            for arg in value.iter() {
                structure.push(arg.try_into()?);
            }

            Ok(OrderedStructure { structure })
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
impl<R: RepName> TryFrom<MulView<'_>> for OrderedStructure<R> {
    type Error = SlotError;
    fn try_from(value: MulView) -> Result<Self, Self::Error> {
        let mut structure: Self = vec![].into();

        for arg in value.iter() {
            structure.extend(arg.try_into()?);
        }
        Ok(structure)
    }
}

impl<S: RepName, R: From<S> + RepName> FromIterator<Slot<S>> for OrderedStructure<R> {
    fn from_iter<T: IntoIterator<Item = Slot<S>>>(iter: T) -> Self {
        Self {
            structure: iter.into_iter().map(|a| a.cast()).collect(),
        }
    }
}

impl<R: RepName> From<Vec<Slot<R>>> for OrderedStructure<R> {
    fn from(structure: Vec<Slot<R>>) -> Self {
        Self { structure }
    }
}

impl<R: RepName> IntoIterator for OrderedStructure<R> {
    type Item = Slot<R>;
    type IntoIter = std::vec::IntoIter<Slot<R>>;
    fn into_iter(self) -> std::vec::IntoIter<Slot<R>> {
        self.structure.into_iter()
    }
}

impl<'a, R: RepName> IntoIterator for &'a OrderedStructure<R> {
    type Item = &'a Slot<R>;
    type IntoIter = std::slice::Iter<'a, Slot<R>>;
    fn into_iter(self) -> std::slice::Iter<'a, Slot<R>> {
        self.structure.iter()
    }
}

impl<'a, R: RepName> IntoIterator for &'a mut OrderedStructure<R> {
    type Item = &'a mut Slot<R>;
    type IntoIter = std::slice::IterMut<'a, Slot<R>>;
    fn into_iter(self) -> std::slice::IterMut<'a, Slot<R>> {
        self.structure.iter_mut()
    }
}

impl<R: RepName> OrderedStructure<R> {
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

impl<N, A, R: RepName> From<NamedStructure<N, A, R>> for OrderedStructure<R> {
    fn from(structure: NamedStructure<N, A, R>) -> Self {
        structure.structure
    }
}

impl<N, A, R: RepName> From<SmartShadowStructure<N, A, R>> for OrderedStructure<R> {
    fn from(structure: SmartShadowStructure<N, A, R>) -> Self {
        structure.structure
    }
}

impl<R: RepName> From<OrderedStructure<R>> for Vec<Slot<R>> {
    fn from(structure: OrderedStructure<R>) -> Self {
        structure.structure
    }
}

// const IDPRINTER: Lazy<BlockId<char>> = Lazy::new(|| BlockId::new(Alphabet::alphanumeric(), 1, 1));

impl<R: RepName> std::fmt::Display for OrderedStructure<R> {
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

impl<R: RepName> ScalarStructure for OrderedStructure<R> {
    fn scalar_structure() -> Self {
        OrderedStructure { structure: vec![] }
    }
}

#[cfg(feature = "shadowing")]
impl<R: RepName<Dual = R>> ToSymbolic for OrderedStructure<R> {}

impl<R: RepName<Dual = R>> StructureContract for OrderedStructure<R> {
    fn trace(&mut self, i: usize, j: usize) {
        if i < j {
            self.trace(j, i);
            return;
        }
        let a = self.structure.remove(i);
        let b = self.structure.remove(j);
        assert_eq!(a, b);
    }

    fn concat(&mut self, other: Self) {
        self.structure.extend(other.structure.iter().cloned());
        self.structure.sort();
    }

    fn trace_out(&mut self) {
        let mut positions = IndexMap::new();

        // Track the positions of each element
        for (index, &value) in self.structure.iter().enumerate() {
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

    fn merge(
        &self,
        other: &Self,
    ) -> Result<(Self, Vec<usize>, Vec<usize>, MergeInfo), StructureError> {
        let (structure, pos_self, pos_other, mergeinfo) = self
            .structure
            .merge_ordered_ref_with_common_indices(&other.structure)?;

        Ok((Self { structure }, pos_self, pos_other, mergeinfo))
    }

    // fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
    //     let mut slots_b = other.clone();
    //     let mut slots_a = self.clone();

    //     slots_a.structure.remove(positions.0);
    //     slots_b.structure.remove(positions.1);

    //     slots_a.structure.append(&mut slots_b.structure);
    //     slots_a
    // }
}

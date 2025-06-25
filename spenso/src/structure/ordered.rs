use std::ops::Deref;

use bitvec::vec::BitVec;
use indexmap::IndexMap;
use linnet::permutation::Permutation;

use crate::utils::MergeOrdered;

#[cfg(feature = "shadowing")]
use crate::{
    shadowing::symbolica_utils::IntoSymbol,
    structure::{ExpandedCoefficent, FlatIndex, ToSymbolic, AIND_SYMBOLS},
    tensors::{data::DenseTensor, parametric::TensorCoefficient},
};

#[cfg(feature = "shadowing")]
use anyhow::Result;

#[cfg(feature = "shadowing")]
use symbolica::atom::{
    representation::{FunView, MulView},
    Atom, AtomView, FunctionBuilder, Symbol,
};

use super::{
    abstract_index::AbstractIndex,
    dimension::Dimension,
    permuted::PermuteTensor,
    representation::{LibraryRep, RepName, Representation},
    slot::{ConstructibleSlot, DualSlotTo, IsAbstractSlot, Slot, SlotError},
    MergeInfo, NamedStructure, PermutedStructure, ScalarStructure, SmartShadowStructure,
    StructureContract, StructureError, TensorStructure,
};
use anyhow::anyhow;
use delegate::delegate;
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
    pub(crate) structure: Vec<Slot<R>>,
    // permutation: Option<Permutation>,
}

impl<R: RepName<Dual = R>> PermuteTensor for OrderedStructure<R> {
    type Id = Self;
    type Permuted = (
        OrderedStructure<LibraryRep>,
        Vec<OrderedStructure<LibraryRep>>,
    );
    type IdSlot = Slot<R>;

    fn permute(self, permutation: &Permutation) -> Self::Permuted {
        let mut dummy_structure = Vec::new();
        let mut ids = Vec::new();

        if permutation.is_identity() {
            return (
                OrderedStructure {
                    structure: self.structure.into_iter().map(|s| s.to_lib()).collect(),
                },
                ids,
            );
        }
        println!("{}", permutation);
        for s in permutation.iter_slice(&self.structure) {
            let d = s.to_dummy();
            let ogs = s.to_lib();
            println!("{ogs}");
            println!("{d}");
            dummy_structure.push(d);
            ids.push(OrderedStructure::id(d, ogs));
        }
        let strct = OrderedStructure::new(dummy_structure);
        if !strct.index_permutation.is_identity() {
            panic!("should be identity")
        }
        (strct.structure, ids)
    }

    fn permute_reps(self, ind_perm: &Permutation, rep_perm: &Permutation) -> Self::Permuted {
        let mut dummy_structure = Vec::new();
        let mut og_reps = Vec::new();
        let mut ids = Vec::new();

        if rep_perm.is_identity() {
            return self.permute(ind_perm);
        }
        println!("{rep_perm}");
        for s in rep_perm.iter_slice_inv(&self.structure) {
            og_reps.push(s.rep.to_lib());
            let d = s.to_dummy();
            println!("{d}");
            dummy_structure.push(d);
        }

        for (i, s) in ind_perm.iter_slice_inv(&self.structure).enumerate() {
            let d = dummy_structure[i];
            let new_slot = og_reps[i].slot(s.aind);
            println!("{new_slot}");
            println!("{d}");
            ids.push(OrderedStructure::id(d, new_slot));
        }
        let strct = OrderedStructure::new(dummy_structure);
        if !strct.index_permutation.is_identity() {
            panic!("should be identity")
        }
        (strct.structure, ids)
    }

    fn id(i: Slot<R>, j: Slot<R>) -> Self::Id {
        if i.dim() == j.dim() {
            OrderedStructure::new(vec![i, j]).structure
        } else {
            panic!("Not same dimension for ID")
        }
    }
}

impl<R: RepName<Dual = R>> TensorStructure for OrderedStructure<R> {
    type Slot = Slot<R>;
    type Indexed = Self;
    // type R = PhysicalReps;
    //
    // fn id(i: Self::Slot, j: Self::Slot) -> Self::Indexed {
    //     if i.dim() == j.dim() {
    //         OrderedStructure::new(vec![i, j]).structure
    //     } else {
    //         panic!("Not same dimension for ID")
    //     }
    // }

    fn reindex(
        self,
        indices: &[AbstractIndex],
    ) -> Result<PermutedStructure<Self::Indexed>, StructureError> {
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
        self.into_iter()
            .map(|s| s.dual())
            .collect::<PermutedStructure<_>>()
            .structure
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

// impl From<Vec<PhysicalSlots>> for VecStructure {
//     fn from(value: Vec<PhysicalSlots>) -> Self {
//         VecStructure { structure: value }
//     }
// }

impl<R: RepName> OrderedStructure<R> {
    pub fn from_iter<S: RepName, T: IntoIterator<Item = Slot<S>>>(
        iter: T,
    ) -> PermutedStructure<Self>
    where
        R: From<S>,
    {
        let structure: Vec<Slot<R>> = iter.into_iter().map(|a| a.cast()).collect();
        PermutedStructure::from(structure)
    }
}

impl<S: RepName, R: From<S> + RepName> FromIterator<Slot<S>>
    for PermutedStructure<OrderedStructure<R>>
{
    fn from_iter<T: IntoIterator<Item = Slot<S>>>(iter: T) -> Self {
        let structure: Vec<Slot<R>> = iter.into_iter().map(|a| a.cast()).collect();
        PermutedStructure::from(structure)
    }
}

impl<R: RepName> From<Vec<Slot<R>>> for PermutedStructure<OrderedStructure<R>> {
    fn from(mut structure: Vec<Slot<R>>) -> Self {
        let rep_permutation = Permutation::sort_by_key(&structure, |a| a.rep);
        rep_permutation.apply_slice_in_place(&mut structure);
        let index_permutation = Permutation::sort(&structure);
        index_permutation.apply_slice_in_place(&mut structure);

        PermutedStructure {
            structure: OrderedStructure { structure },
            rep_permutation,
            index_permutation,
        }
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
    /// Creates a new ordered structure from this unsorted list of slots.
    /// Returns a tuple struct of a the permutation that was used to sort the vector as well as the ordered structure itself
    pub fn new(structure: Vec<Slot<R>>) -> PermutedStructure<Self> {
        PermutedStructure::from(structure)
    }

    pub fn push(&mut self, item: Slot<R>) {
        self.structure.push(item)
    }

    pub fn sort(&mut self) -> Permutation {
        let permutation = Permutation::sort(&self.structure);
        permutation.apply_slice_in_place(&mut self.structure);
        permutation
    }

    fn extend(&mut self, other: Self) {
        self.structure.extend(other.structure)
    }

    pub fn to_named<N, A>(self, name: N, args: Option<A>) -> NamedStructure<N, A, R> {
        NamedStructure {
            structure: self,
            global_name: Some(name),
            additional_args: args,
        }
    }

    pub fn empty() -> Self {
        Self::default()
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
                item.aind(),
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
        Self::default()
    }
}

#[cfg(feature = "shadowing")]
impl<R: RepName<Dual = R>> ToSymbolic for OrderedStructure<R> {
    fn concrete_atom(&self, id: FlatIndex) -> ExpandedCoefficent<()> {
        ExpandedCoefficent {
            name: None,
            index: self.co_expanded_index(id).unwrap(),
            args: None,
        }
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
            let i = Atom::i();
            data.push(&re + i * &im);
        }

        Ok(DenseTensor {
            data,
            structure: self,
        })
    }

    fn to_symbolic_with(&self, name: Symbol, args: &[Atom], perm: Option<Permutation>) -> Atom {
        let mut slots = self
            .external_structure_iter()
            .map(|slot| slot.to_atom())
            .collect::<Vec<_>>();
        if let Some(p) = perm {
            p.apply_slice_in_place(&mut slots);
        }
        FunctionBuilder::new(name.ref_into_symbol())
            .add_args(args)
            .add_args(&slots)
            .finish()
    }
}

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
            .collect::<PermutedStructure<_>>()
            .structure;
    }

    fn merge(&self, other: &Self) -> Result<(Self, BitVec, BitVec, MergeInfo), StructureError> {
        // println!("self\n{}", self);
        // println!("other\n{}", other);
        let (structure, pos_self, pos_other, mergeinfo) = self
            .structure
            .merge_ordered_ref_with_comparison_and_matching(
                &other.structure,
                |a, b| a.cmp(b),
                |a, b| {
                    // println!("Does {a} match {b}?");
                    let a = a.matches(b);
                    // println!("{a}");
                    a
                },
            )?;
        // println!("{:?}", structure);

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
#[cfg(test)]
pub mod test {
    use crate::structure::{
        representation::{Euclidean, LibraryRep, Lorentz, Minkowski, RepName},
        slot::IsAbstractSlot,
        StructureContract, TensorStructure,
    };

    use super::OrderedStructure;

    #[test]
    fn orderedmerge() {
        let a: OrderedStructure<LibraryRep> =
            OrderedStructure::from_iter([Lorentz {}.new_slot(3, 3).to_lib()]).structure;
        let b = a.clone().dual();

        println!("{}", a.merge(&b).unwrap().0);
    }
}

use std::{cmp::Ordering, ops::Deref, usize};

use bitvec::vec::BitVec;
use indexmap::IndexMap;
use linnet::permutation::Permutation;
use tabled::{builder::Builder, settings::Style};

use crate::utils::{to_subscript, to_superscript, MergeOrdered};

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

#[derive(Clone, Hash, bincode_trait_derive::Encode, bincode_trait_derive::Decode)]
#[cfg_attr(not(feature = "shadowing"), derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
pub struct OrderedStructure<R: RepName = LibraryRep> {
    pub(crate) structure: Vec<Slot<R>>,
    dual_start: usize,
    base_start: usize,
    // permutation: Option<Permutation>,
}

impl<R: PartialEq + RepName> PartialEq for OrderedStructure<R> {
    fn eq(&self, other: &Self) -> bool {
        let len = self.structure.len();

        self.structure == other.structure
            && ((self.dual_start >= len && other.dual_start >= len)
                || (self.dual_start == other.dual_start))
            && ((self.base_start >= len && other.base_start >= len)
                || (self.base_start == other.base_start))
    }
}
impl<R: Eq + RepName> Eq for OrderedStructure<R> {}

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
                    dual_start: self.dual_start,
                    base_start: self.base_start,
                },
                ids,
            );
        }
        // println!("{}", permutation);
        for s in permutation.iter_slice(&self.structure) {
            let d = s.to_dummy();
            let ogs = s.to_lib();
            // println!("{ogs}");
            // println!("{d}");
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
        // println!("{rep_perm}");
        for s in rep_perm.iter_slice_inv(&self.structure) {
            og_reps.push(s.rep.to_lib());
            let d = s.to_dummy();
            // println!("{d}");
            dummy_structure.push(d);
        }

        for (i, s) in ind_perm.iter_slice_inv(&self.structure).enumerate() {
            let d = dummy_structure[i];
            let new_slot = og_reps[i].slot(s.aind);
            // println!("{new_slot}");
            // println!("{d}");
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
        Self {
            structure: vec![],
            base_start: usize::MAX,
            dual_start: usize::MAX,
        }
    }
}

// impl From<Vec<PhysicalSlots>> for VecStructure {
//     fn from(value: Vec<PhysicalSlots>) -> Self {
//         VecStructure { structure: value }
//     }
// }

impl<R: RepName> OrderedStructure<R> {
    pub fn dual_slice(&self) -> &[Slot<R>] {
        if self.dual_start >= self.structure.len() {
            &self.structure[0..0]
        } else {
            &self.structure[self.dual_start..]
        }
    }

    pub fn base_slice(&self) -> &[Slot<R>] {
        if self.base_start >= self.structure.len() {
            &self.structure[0..0]
        } else if self.dual_start >= self.structure.len() {
            &self.structure[self.base_start..]
        } else {
            &self.structure[self.base_start..self.dual_start]
        }
    }

    pub fn self_dual_slice(&self) -> &[Slot<R>] {
        if self.base_start >= self.structure.len() {
            &self.structure[..]
        } else {
            &self.structure[self.base_start..]
        }
    }

    pub fn n_self_dual(&self) -> usize {
        if self.base_start >= self.structure.len() {
            self.structure.len()
        } else {
            self.base_start
        }
    }

    pub fn n_base(&self) -> usize {
        if self.base_start >= self.structure.len() {
            0
        } else if self.dual_start >= self.structure.len() {
            self.structure.len() - self.base_start
        } else {
            self.dual_start - self.base_start
        }
    }

    pub fn n_dual(&self) -> usize {
        if self.dual_start >= self.structure.len() {
            0
        } else {
            self.structure.len() - self.dual_start
        }
    }

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

        let mut base_start =
            if structure[0].rep.rep.is_base() && !structure[0].rep.rep.is_self_dual() {
                0
            } else {
                usize::MAX
            };
        let mut dual_start =
            if structure[0].rep.rep.is_dual() && !structure[0].rep.rep.is_self_dual() {
                base_start = 0;
                0
            } else {
                usize::MAX
            };

        for i in 0..(structure.len() - 1) {
            if structure[i].rep.rep.is_self_dual() && !structure[i + 1].rep.rep.is_self_dual() {
                base_start = i + 1;
            }

            if structure[i].rep.rep.is_base()
                && !structure[i + 1].rep.rep.is_self_dual()
                && structure[i + 1].rep.rep.is_dual()
            {
                dual_start = i + 1;
            }
        }

        let index_permutation = Permutation::sort(&structure);
        index_permutation.apply_slice_in_place(&mut structure);

        PermutedStructure {
            structure: OrderedStructure {
                structure,
                base_start,
                dual_start,
            },
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
        let mut table = Builder::new();

        table.push_record(&["".to_string()]);
        for (index, item) in self.structure.iter().enumerate() {
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

impl<R: RepName> std::fmt::Debug for OrderedStructure<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut table = Builder::new();

        table.push_record(&[
            "OrderedStructure".to_string(),
            format!("base_start:{}", self.base_start),
            format!("dual_start:{}", self.dual_start),
        ]);
        for (index, item) in self.structure.iter().enumerate() {
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
        debug_assert!(
            self.self_dual_slice().windows(3).all(|w| w[0] <= w[1]
                && w[1] <= w[2]
                && !(w[0].matches(&w[1]) && w[1].matches(&w[2]))),
            "Input vector 'self' to merge must be sorted in self dual part: {:?}",
            self.self_dual_slice()
        );
        debug_assert!(
            self.dual_slice().windows(2).all(|w| w[0] < w[1]),
            "Input vector 'self' to merge must be sorted in dual part: {:?}",
            self.dual_slice()
        );
        debug_assert!(
            self.base_slice().windows(2).all(|w| w[0] < w[1]),
            "Input vector 'self' to merge must be sorted in base part: {:?}",
            self.base_slice()
        );

        debug_assert!(
            other.self_dual_slice().windows(3).all(|w| w[0] <= w[1]
                && w[1] <= w[2]
                && !(w[0].matches(&w[1]) && w[1].matches(&w[2]))),
            "Input vector 'other' to merge must be sorted in self dual part: {:?}",
            other.self_dual_slice()
        );
        debug_assert!(
            other.dual_slice().windows(2).all(|w| w[0] < w[1]),
            "Input vector 'other' to merge must be sorted in dual part: {:?}",
            other.dual_slice()
        );
        debug_assert!(
            other.base_slice().windows(2).all(|w| w[0] < w[1]),
            "Input vector 'other' to merge must be sorted in base part: {:?}",
            other.base_slice()
        );

        let mut common_indices_self = BitVec::with_capacity(self.order()); // BitVec for common slots in self
        common_indices_self.resize(self.order(), false);
        let mut common_indices_other = BitVec::with_capacity(other.order()); // BitVec for common slots in other
        common_indices_other.resize(other.order(), false);

        if self.is_scalar() || other.is_scalar() {
            // If either is empty, there are no common slots
            if self.is_scalar() {
                return Ok((
                    other.clone(),
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::SecondBeforeFirst,
                ));
            } else {
                return Ok((
                    self.clone(),
                    common_indices_self,
                    common_indices_other,
                    MergeInfo::FirstBeforeSecond,
                ));
            }
        }

        let mut partition = BitVec::new(); // For interleaved mergeinfo
        partition.reserve(self.order() + other.order());

        let mut i = 0;
        let mut j = 0;

        // Merge self dual parts:

        let mut resulting_structure = Vec::new();

        while i < self.n_self_dual() && j < other.n_self_dual() {
            match self.structure[i].cmp(&other.structure[j]) {
                std::cmp::Ordering::Less => {
                    resulting_structure.push(self.structure[i].clone());
                    partition.push(true);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    resulting_structure.push(other.structure[j].clone());
                    partition.push(false);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    if self.structure[i].matches(&other.structure[j]) {
                        // Common item
                        common_indices_self.set(i, true); // Set bit for index in first vector
                        common_indices_other.set(j, true); // Set bit for index in second vector
                        i += 1;
                        j += 1;
                        continue; // Skip transition check for common items since they're not added to result_non_common
                    } else {
                        panic!("Matching and equal items must be identical")
                    }
                }
            }
        }

        while i < self.n_self_dual() {
            resulting_structure.push(self.structure[i].clone());
            partition.push(true);
            i += 1;
        }

        while j < other.n_self_dual() {
            resulting_structure.push(other.structure[j].clone());
            partition.push(false);
            j += 1;
        }

        let base_start = partition.len();

        //Merge base with dual

        let mut ibase = 0;
        let mut idual = 0;
        let mut jbase = 0;
        let mut jdual = 0;

        let snbase = self.n_base();
        let onbase = other.n_base();

        let find_match_in_duals = |base_slot: &Slot<R>,
                                   base_slot_index: usize,
                                   common_indices_base: &mut BitVec,
                                   dual_structure: &Self,
                                   dual_cursor: &mut usize,
                                   common_indices_dual: &mut BitVec,
                                   dual_offset: usize|
         -> bool {
            let mut found_match = false;
            let dual_count = dual_structure.n_dual();

            // println!("dual_count: {}", dual_count);
            // println!("dual_cursor: {}", dual_cursor);
            while *dual_cursor < dual_count {
                let dual_slot_index = dual_offset + *dual_cursor;
                let dual_slot = &dual_structure.structure[dual_slot_index];
                // println!("{}vs{}", base_slot, dual_slot);
                match base_slot.match_cmp(dual_slot) {
                    std::cmp::Ordering::Less => {
                        // print!("less");
                        break;
                    }
                    std::cmp::Ordering::Greater => {
                        // print!("more");
                        *dual_cursor += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        if base_slot.matches(dual_slot) {
                            common_indices_base.set(base_slot_index, true);
                            common_indices_dual.set(dual_slot_index, true);
                            found_match = true;
                            *dual_cursor += 1;
                            break;
                        } else {
                            panic!("Matching and equal items must be identical");
                        }
                    }
                }
            }
            found_match
        };

        while ibase < snbase && jbase < onbase {
            match self.structure[i + ibase].cmp(&other.structure[j + jbase]) {
                std::cmp::Ordering::Less => {
                    let base_slot = &self.structure[i + ibase];
                    let found_match = find_match_in_duals(
                        base_slot,
                        i + ibase,
                        &mut common_indices_self,
                        other,
                        &mut jdual,
                        &mut common_indices_other,
                        j + onbase,
                    );
                    if !found_match {
                        resulting_structure.push(base_slot.clone());
                        partition.push(true);
                    }
                    ibase += 1;
                }
                std::cmp::Ordering::Greater => {
                    let base_slot = &other.structure[j + jbase];
                    let found_match = find_match_in_duals(
                        base_slot,
                        j + jbase,
                        &mut common_indices_other,
                        self,
                        &mut idual,
                        &mut common_indices_self,
                        i + snbase,
                    );
                    if !found_match {
                        resulting_structure.push(base_slot.clone());
                        partition.push(false);
                    }
                    jbase += 1;
                }
                std::cmp::Ordering::Equal => {
                    panic!("Cannot have equal bases")
                }
            }
        }

        while ibase < snbase {
            let base_slot = &self.structure[i + ibase];
            let found_match = find_match_in_duals(
                base_slot,
                i + ibase,
                &mut common_indices_self,
                other,
                &mut jdual,
                &mut common_indices_other,
                j + onbase,
            );
            if !found_match {
                resulting_structure.push(base_slot.clone());
                partition.push(true);
            }
            ibase += 1;
        }

        while jbase < onbase {
            let base_slot = &other.structure[j + jbase];
            let found_match = find_match_in_duals(
                base_slot,
                j + jbase,
                &mut common_indices_other,
                self,
                &mut idual,
                &mut common_indices_self,
                i + snbase,
            );
            if !found_match {
                resulting_structure.push(base_slot.clone());
                partition.push(false);
            }
            jbase += 1;
        }

        let dual_start = partition.len();

        let mut idual = 0;
        let mut jdual = 0;

        while idual < self.n_dual() && jdual < other.n_dual() {
            match self.structure[i + ibase + idual].cmp(&other.structure[j + jbase + jdual]) {
                Ordering::Less => {
                    if !common_indices_self[i + ibase + idual] {
                        resulting_structure.push(self.structure[i + ibase + idual].clone());
                        partition.push(false);
                    }
                    idual += 1;
                }
                Ordering::Greater => {
                    if !common_indices_other[j + jbase + jdual] {
                        resulting_structure.push(other.structure[j + jbase + jdual].clone());
                        partition.push(false);
                    }
                    jdual += 1;
                }
                Ordering::Equal => {
                    panic!("Cannot have equal duals")
                }
            }
        }

        while idual < self.n_dual() {
            if !common_indices_self[i + ibase + idual] {
                resulting_structure.push(self.structure[i + ibase + idual].clone());
                partition.push(false);
            }
            idual += 1;
        }

        while jdual < other.n_dual() {
            if !common_indices_other[j + jbase + jdual] {
                resulting_structure.push(other.structure[j + jbase + jdual].clone());
                partition.push(false);
            }
            jdual += 1;
        }

        assert!(
            Permutation::sort(&resulting_structure).is_identity(),
            "Permutation is not identity for  {}\n{:?}\nself:{:?}\n other:{:?}",
            Permutation::sort(&resulting_structure),
            OrderedStructure {
                structure: resulting_structure,
                dual_start,
                base_start,
            },
            self,
            other
        );

        Ok((
            OrderedStructure {
                structure: resulting_structure,
                dual_start,
                base_start,
            },
            common_indices_self,
            common_indices_other,
            partition.into(),
        ))
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
        slot::{DualSlotTo, IsAbstractSlot},
        StructureContract, TensorStructure,
    };

    use super::OrderedStructure;

    #[test]
    fn orderedmerge() {
        let a: OrderedStructure<LibraryRep> = OrderedStructure::from_iter([
            Lorentz {}.new_slot(3, 2).to_lib(),
            Minkowski {}.new_slot(4, 2).to_lib(),
            Lorentz {}.new_slot(7, 1).to_lib(),
            Euclidean {}.new_slot(4, 2).to_lib(),
            Euclidean {}.new_slot(2, 3).to_lib(),
        ])
        .structure;
        let b: OrderedStructure<LibraryRep> = OrderedStructure::from_iter([
            Euclidean {}.new_slot(4, 2).to_lib(),
            Minkowski {}.new_slot(4, 11).to_lib(),
            Lorentz {}.new_slot(7, 1).dual().to_lib(),
        ])
        .structure;

        println!("{:?}", a);
        assert_eq!(a.n_base(), 2);
        println!("{:?}", b);
        println!("{}", a.merge(&b).unwrap().0);
        println!("{}", b.merge(&a).unwrap().0);
    }
}

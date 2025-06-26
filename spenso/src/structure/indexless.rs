use linnet::permutation::Permutation;

use super::{
    abstract_index::AbstractIndex,
    dimension::Dimension,
    named::IdentityName,
    permuted::PermuteTensor,
    representation::{LibraryRep, RepName, Representation},
    slot::{IsAbstractSlot, Slot},
    HasName, NamedStructure, OrderedStructure, PermutedStructure, ScalarStructure,
    SmartShadowStructure, StructureError, TensorStructure,
};

use anyhow::{anyhow, Result};
use delegate::delegate;

#[cfg(feature = "shadowing")]
use crate::{
    shadowing::symbolica_utils::{IntoArgs, IntoSymbol},
    structure::ToSymbolic,
    tensors::parametric::{ExpandedCoefficent, TensorCoefficient},
};

#[cfg(feature = "shadowing")]
use crate::{structure::FlatIndex, tensors::data::DenseTensor};
#[cfg(feature = "shadowing")]
use std::fmt::Display;

#[cfg(not(feature = "shadowing"))]
use serde::{Deserialize, Serialize};
#[cfg(feature = "shadowing")]
use symbolica::atom::{Atom, FunctionBuilder, Symbol};

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
// #[cfg_attr(not(feature = "shadowing"), derive(bincode::Decode))]
pub struct IndexLess<T: RepName = LibraryRep> {
    pub structure: Vec<Representation<T>>,
}

impl<R: RepName<Dual = R>> PermuteTensor for IndexLess<R> {
    type Id = Self;
    type Permuted = (IndexLess<LibraryRep>, Vec<IndexLess<LibraryRep>>);
    type IdSlot = Slot<R>;

    fn permute(self, permutation: &Permutation) -> Self::Permuted {
        todo!()
    }

    fn permute_reps(self, ind_perm: &Permutation, rep_perm: &Permutation) -> Self::Permuted {
        todo!()
    }

    fn id(i: Slot<R>, j: Slot<R>) -> Self::Id {
        if i.dim() == j.dim() {
            IndexLess::new(vec![i.rep, j.rep])
        } else {
            panic!("Not same dimension for ID")
        }
    }
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
impl<R: RepName> FromIterator<Representation<R>> for PermutedStructure<IndexLess<R>> {
    fn from_iter<I: IntoIterator<Item = Representation<R>>>(iter: I) -> Self {
        let structure: Vec<_> = iter.into_iter().collect();
        structure.into()
    }
}

impl<R: RepName> From<Vec<Representation<R>>> for PermutedStructure<IndexLess<R>> {
    fn from(mut structure: Vec<Representation<R>>) -> Self {
        let permutation = Permutation::sort(&structure);
        permutation.apply_slice_in_place(&mut structure);

        PermutedStructure {
            rep_permutation: permutation,
            index_permutation: Permutation::id(structure.len()),
            structure: IndexLess { structure },
        }
    }
}

impl<R: RepName> From<OrderedStructure<R>> for IndexLess<R> {
    fn from(structure: OrderedStructure<R>) -> Self {
        IndexLess {
            structure: structure.into_iter().map(|a| a.rep).collect(),
        }
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

impl<T: RepName> IndexLess<T> {
    pub fn new(structure: Vec<Representation<T>>) -> Self {
        Self { structure }
    }

    pub fn empty() -> Self {
        Self { structure: vec![] }
    }

    pub fn to_indexed(self, indices: &[AbstractIndex]) -> Result<Vec<Slot<T>>, StructureError> {
        if self.structure.len() != indices.len() {
            return Err(StructureError::WrongNumberOfArguments(
                indices.len(),
                self.structure.len(),
            ));
        }

        Ok(indices
            .iter()
            .cloned()
            .zip(self.structure.iter().cloned())
            .map(|(i, r)| Representation::slot(&r, i))
            .collect())
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
    type Indexed = OrderedStructure<T>;

    fn reindex(
        self,
        indices: &[AbstractIndex],
    ) -> Result<PermutedStructure<OrderedStructure<T>>, StructureError> {
        if self.structure.len() != indices.len() {
            return Err(StructureError::WrongNumberOfArguments(
                indices.len(),
                self.structure.len(),
            ));
        }

        Ok(indices
            .iter()
            .cloned()
            .zip(self.structure.iter().cloned())
            .map(|(i, r)| Representation::slot(&r, i))
            .collect())
    }
    fn dual(self) -> Self {
        self.structure
            .into_iter()
            .map(|r| r.dual())
            .collect::<PermutedStructure<_>>()
            .structure
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
impl<T: RepName<Dual = T>> ToSymbolic for IndexLess<T> {
    fn concrete_atom(&self, id: FlatIndex) -> ExpandedCoefficent<()> {
        ExpandedCoefficent {
            name: None,
            index: self.co_expanded_index(id).unwrap(),
            args: None,
        }
    }

    fn to_dense_labeled<R>(
        self,
        index_to_atom: impl Fn(&Self, FlatIndex) -> R,
    ) -> Result<DenseTensor<Atom, Self>>
    where
        Self: Sized,
        R: TensorCoefficient,
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

    fn to_dense_labeled_complex<R>(
        self,
        index_to_atom: impl Fn(&Self, FlatIndex) -> R,
    ) -> Result<DenseTensor<Atom, Self>>
    where
        Self: Sized,
        R: TensorCoefficient,
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
pub struct IndexlessNamedStructure<Name = String, Args = usize, R: RepName = LibraryRep> {
    pub structure: IndexLess<R>,
    pub global_name: Option<Name>,
    pub additional_args: Option<Args>,
}

impl<N: IdentityName, A, R: RepName<Dual = R>> PermuteTensor for IndexlessNamedStructure<N, A, R> {
    type Id = Self;
    type IdSlot = Slot<R>;
    type Permuted = (
        IndexlessNamedStructure<N, A, LibraryRep>,
        Vec<IndexlessNamedStructure<N, A, LibraryRep>>,
    );

    fn id(i: Self::IdSlot, j: Self::IdSlot) -> Self::Id {
        Self {
            structure: IndexLess::id(i, j),
            global_name: Some(N::id()),
            additional_args: None,
        }
    }

    fn permute(self, permutation: &Permutation) -> Self::Permuted {
        todo!()
    }

    fn permute_reps(self, ind_perm: &Permutation, rep_perm: &Permutation) -> Self::Permuted {
        todo!()
    }
}
// impl<N: IdentityName, A, R: RepName<Dual = R>> PermuteTensor for IndexlessNamedStructure<N, A, R> {
//     type Id = NamedStructure<N, A, R>;
//     type IdSlot = Slot<R>;
//     type Permuted = (
//         NamedStructure<N, A, LibraryRep>,
//         Vec<NamedStructure<N, A, LibraryRep>>,
//     );

//     fn id(i: Self::IdSlot, j: Self::IdSlot) -> Self::Id {
//         NamedStructure {
//             structure: OrderedStructure::id(i, j),
//             global_name: Some(N::id()),
//             additional_args: None,
//         }
//     }

//     fn permute(self, permutation: &Permutation) -> Self::Permuted {
//         let mut dummy_structure = Vec::new();
//         let mut ids = Vec::new();

//         for s in permutation.iter_slice_inv(&self.structure.structure) {
//             let dind = AbstractIndex::new_dummy();
//             let d = s.to_dummy().to_lib().slot(dind);
//             let ogs = s.to_lib().slot(dind);
//             dummy_structure.push(d);
//             ids.push(NamedStructure::id(d, ogs));
//         }
//         let strct = OrderedStructure::new(dummy_structure);
//         if !strct.permutation.is_identity() {
//             panic!("should be identity")
//         }

//         (
//             NamedStructure {
//                 global_name: self.global_name,
//                 additional_args: self.additional_args,
//                 structure: strct.structure,
//             },
//             ids,
//         )
//     }
// }

impl<Name, Args, R: RepName<Dual = R>> TensorStructure for IndexlessNamedStructure<Name, Args, R> {
    type Slot = Slot<R>;
    type Indexed = NamedStructure<Name, Args, R>;

    fn reindex(
        self,
        indices: &[AbstractIndex],
    ) -> Result<PermutedStructure<NamedStructure<Name, Args, R>>, StructureError> {
        let res = self.structure.reindex(indices)?;

        Ok(PermutedStructure {
            rep_permutation: Permutation::id(res.structure.order()),
            structure: NamedStructure {
                global_name: self.global_name,
                additional_args: self.additional_args,
                structure: res.structure,
            },
            index_permutation: res.index_permutation,
        })
    }

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

impl<N, A, T: RepName<Dual = T>> ScalarStructure for IndexlessNamedStructure<N, A, T> {
    fn scalar_structure() -> Self {
        IndexlessNamedStructure {
            structure: IndexLess::scalar_structure(),
            global_name: None,
            additional_args: None,
        }
    }
}

impl<Name, Args, R: RepName> IndexlessNamedStructure<Name, Args, R> {
    #[must_use]
    pub fn from_iter<I, T>(iter: T, name: Name, args: Option<Args>) -> PermutedStructure<Self>
    where
        I: RepName,
        R: From<I>,
        T: IntoIterator<Item = Representation<I>>,
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

impl<N, A, R: RepName> From<NamedStructure<N, A, R>> for IndexlessNamedStructure<N, A, R> {
    fn from(value: NamedStructure<N, A, R>) -> Self {
        IndexlessNamedStructure {
            structure: value.structure.into(),
            global_name: value.global_name,
            additional_args: value.additional_args,
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

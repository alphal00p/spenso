use bitvec::vec::BitVec;
use std::ops::AddAssign;

use crate::{
    algebra::ScalarMul,
    contraction::{Contract, ContractionError},
    network::{
        library::{
            symbolic::{ExplicitKey, TensorLibrary},
            TensorLibraryData,
        },
        parsing::ShadowedStructure,
        store::NetworkStore,
        Network, Ref, TensorNetworkError,
    },
    shadowing::symbolica_utils::{IntoArgs, IntoSymbol},
    structure::{
        abstract_index::AIND_SYMBOLS,
        permuted::PermuteTensor,
        representation::{LibraryRep, LibrarySlot},
        slot::{AbsInd, DummyAind},
        HasName, HasStructure, MergeInfo, NamedStructure, OrderedStructure, PermutedStructure,
        StructureContract, TensorShell, TensorStructure, ToSymbolic,
    },
    tensors::parametric::MixedTensor,
};

use crate::structure::abstract_index::AbstractIndex;
use crate::structure::dimension::Dimension;
use crate::structure::representation::Representation;
use crate::structure::slot::IsAbstractSlot;
use crate::structure::StructureError;
use delegate::delegate;

use symbolica::atom::{Atom, AtomCore, AtomView, Symbol};

/// A fully symbolic tensor, with no concrete values.
///
/// This tensor is used to represent the structure of a tensor, and is used to perform symbolic contraction.
/// Currently contraction is just a multiplication of the atoms, but in the future this will ensure that internal indices are independent accross the contraction.
///
/// Additionally, this can also be used as a tensor structure, that tracks the history, much like [`HistoryStructure`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicTensor<Aind: AbsInd = AbstractIndex> {
    pub structure: OrderedStructure<LibraryRep, Aind>,
    pub expression: symbolica::atom::Atom,
}

impl<Aind: AbsInd> Ref for SymbolicTensor<Aind> {
    type Ref<'a>
        = &'a SymbolicTensor<Aind>
    where
        Self: 'a;

    fn refer(&self) -> Self::Ref<'_> {
        self
    }
}

impl<Aind: AbsInd + DummyAind> PermuteTensor for SymbolicTensor<Aind>
where
    Atom: From<Aind>,
{
    type Id = SymbolicTensor<Aind>;
    type IdSlot = LibrarySlot<Aind>;
    type Permuted = SymbolicTensor<Aind>;

    fn id(i: Self::IdSlot, j: Self::IdSlot) -> Self::Id {
        Self::from_named(&NamedStructure::<Symbol, (), LibraryRep, Aind>::id(i, j)).unwrap()
    }

    fn permute_inds(mut self, permutation: &linnet::permutation::Permutation) -> Self::Permuted {
        let (new_structure, idstructures) = self.structure.clone().permute_inds(permutation);

        for (o, n) in self
            .structure
            .structure
            .iter()
            .zip(new_structure.structure.iter())
        {
            self.expression = self.expression.replace(o.to_atom()).with(n.to_atom());
        }

        let mut ids = Atom::one();
        for s in idstructures.iter() {
            let o = s.external_structure();
            ids *= Self::id(o[0], o[1]).expression;
        }
        self.expression *= ids;

        self
    }

    fn permute_reps(mut self, rep_perm: &linnet::permutation::Permutation) -> Self::Permuted {
        let (new_structure, idstructures) = self.structure.clone().permute_reps(rep_perm);

        for (o, n) in self
            .structure
            .structure
            .iter()
            .zip(new_structure.structure.iter())
        {
            self.expression = self.expression.replace(o.to_atom()).with(n.to_atom());
        }

        let mut ids = Atom::one();
        for s in idstructures.iter() {
            let o = s.external_structure();
            ids *= Self::id(o[0], o[1]).expression;
        }

        self.expression *= ids;

        self
    }
}

impl<Aind: AbsInd> TensorStructure for SymbolicTensor<Aind> {
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = SymbolicTensor<Aind>;
    type Slot = LibrarySlot<Aind>;

    fn reindex(self, indices: &[Aind]) -> Result<PermutedStructure<Self::Indexed>, StructureError> {
        let res = self.structure.reindex(indices)?;
        Ok(PermutedStructure {
            structure: Self {
                structure: res.structure,
                expression: self.expression,
            },
            rep_permutation: res.rep_permutation,
            index_permutation: res.index_permutation,
        })
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = Aind>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<Aind>;
            fn order(&self)-> usize;
        }
    }
}

impl<Aind: AbsInd> HasStructure for SymbolicTensor<Aind> {
    type Structure = OrderedStructure<LibraryRep, Aind>;
    type Scalar = Atom;
    type ScalarRef<'a>
        = &'a Atom
    where
        Self: 'a;
    type Store<S>
        = TensorShell<S>
    where
        S: TensorStructure;

    fn map_structure<O: TensorStructure>(
        self,
        f: impl FnOnce(Self::Structure) -> O,
    ) -> Self::Store<O> {
        TensorShell {
            structure: f(self.structure),
        }
    }

    fn map_structure_result<O: TensorStructure, Er>(
        self,
        f: impl FnOnce(Self::Structure) -> anyhow::Result<O, Er>,
    ) -> std::result::Result<Self::Store<O>, Er> {
        Ok(TensorShell {
            structure: f(self.structure)?,
        })
    }

    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        SymbolicTensor {
            structure: f(self.structure),
            expression: self.expression,
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        if self.is_scalar() {
            Some(self.expression)
        } else {
            None
        }
    }

    fn scalar_ref(&self) -> Option<&Self::Scalar> {
        if self.is_scalar() {
            Some(&self.expression)
        } else {
            None
        }
    }
}

impl<Aind: AbsInd> StructureContract for SymbolicTensor<Aind> {
    fn merge(&self, other: &Self) -> Result<(Self, BitVec, BitVec, MergeInfo), StructureError> {
        let expression = &other.expression * &self.expression;
        let (structure, pos_self, pos_other, mergeinfo) = self.structure.merge(&other.structure)?;

        Ok((
            Self {
                structure,
                expression,
            },
            pos_self,
            pos_other,
            mergeinfo,
        ))
    }

    fn trace_out(&mut self) {
        self.structure.trace_out();
    }

    fn trace(&mut self, i: usize, j: usize) {
        self.structure.trace(i, j);
    }
}

// impl<Const> Shadowable<Const> for SymbolicTensor {}

#[allow(dead_code)]
impl<Aind: AbsInd> SymbolicTensor<Aind> {
    pub fn from_named<N>(structure: &N) -> Option<Self>
    where
        N: ToSymbolic + HasName + TensorStructure<Slot = LibrarySlot<Aind>>,
        N::Name: IntoSymbol + Clone,
        N::Args: IntoArgs,
    {
        let permuted_structure = PermutedStructure::from(structure.external_structure());
        Some(SymbolicTensor {
            expression: structure.to_symbolic(Some(permuted_structure.index_permutation))?,
            structure: permuted_structure.structure,
        })
    }

    pub fn from_permuted<N>(structure: &PermutedStructure<N>) -> Option<Self>
    where
        N: ToSymbolic + HasName + TensorStructure<Slot = LibrarySlot<Aind>>,
        N::Name: IntoSymbol + Clone,
        N::Args: IntoArgs,
    {
        let permuted_structure = PermutedStructure::from(structure.structure.external_structure());

        Some(SymbolicTensor {
            expression: structure
                .structure
                .to_symbolic(Some(structure.index_permutation.clone()))?,
            structure: permuted_structure.structure,
        })
    }

    pub fn to_named(&self) -> NamedStructure<Symbol, Vec<Atom>, LibraryRep, Aind> {
        NamedStructure {
            structure: self.structure.clone(),
            global_name: self.name(),
            additional_args: self.args(),
        }
    }

    pub fn empty(expression: Atom) -> Self {
        SymbolicTensor {
            structure: OrderedStructure::empty(),
            expression,
        }
    }

    // #[must_use]
    pub fn get_atom(&self) -> &Atom {
        &self.expression
    }

    // pub fn to_mixed(self) -> MixedTensor {
    //     self.to_named().to_shell().to_explicit().unwrap()
    // }
    #[allow(clippy::type_complexity, clippy::result_large_err)]
    pub fn to_network(
        &self,
        library: &TensorLibrary<MixedTensor<f64, ExplicitKey<AbstractIndex>>, AbstractIndex>,
    ) -> Result<
        Network<
            NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
            ExplicitKey<AbstractIndex>,
            AbstractIndex,
        >,
        TensorNetworkError<ExplicitKey<AbstractIndex>>,
    > {
        Network::<
            NetworkStore<MixedTensor<f64, ShadowedStructure<AbstractIndex>>, Atom>,
            ExplicitKey<AbstractIndex>,
            AbstractIndex,
        >::try_from_view(self.expression.as_view(), library)
    }
}

// impl TryFrom<Atom> for SymbolicTensor {
//     type Error = String;
//     fn try_from(value: Atom) -> Result<Self, Self::Error> {
//         let structure = value
//             .as_view()
//             .try_into()
//             .unwrap_or(OrderedStructure::empty());

//         Ok(SymbolicTensor {
//             structure,
//             expression: value,
//         })
//     }
// }

impl<Aind: AbsInd> HasName for SymbolicTensor<Aind> {
    type Name = Symbol;
    type Args = Vec<Atom>;

    fn name(&self) -> Option<Self::Name> {
        if let AtomView::Fun(f) = self.expression.as_view() {
            Some(f.get_symbol())
        } else {
            None
        }
    }

    fn set_name(&mut self, _name: Self::Name) {
        unimplemented!("Cannot set name of a symbolic tensor")
    }

    fn args(&self) -> Option<Self::Args> {
        let mut args = vec![];
        match self.expression.as_view() {
            AtomView::Fun(f) => {
                for arg in f.iter() {
                    if let AtomView::Fun(f) = arg {
                        if f.get_symbol() != AIND_SYMBOLS.aind {
                            args.push(arg.to_owned());
                        }
                    } else {
                        args.push(arg.to_owned());
                    }
                }
                Some(args)
            }
            _ => None,
        }
    }
}

/// Symbolic contraction of two symbolic tensors is just a multiplication of the atoms.
///
impl<Aind: AbsInd> Contract<SymbolicTensor<Aind>> for SymbolicTensor<Aind> {
    type LCM = SymbolicTensor<Aind>;
    fn contract(&self, other: &SymbolicTensor<Aind>) -> Result<Self::LCM, ContractionError> {
        let expression = &other.expression * &self.expression;
        let (structure, _, _, _) = self.structure.merge(&other.structure)?;
        Ok(SymbolicTensor {
            expression,
            structure,
        })
    }
}

impl<Aind: AbsInd> std::ops::Neg for SymbolicTensor<Aind> {
    type Output = SymbolicTensor<Aind>;

    fn neg(self) -> Self::Output {
        Self {
            expression: -self.expression,
            structure: self.structure,
        }
    }
}

impl<Aind: AbsInd> AddAssign<SymbolicTensor<Aind>> for SymbolicTensor<Aind> {
    fn add_assign(&mut self, rhs: SymbolicTensor<Aind>) {
        debug_assert_eq!(self.structure, rhs.structure);
        self.expression += rhs.expression;
    }
}

impl<Aind: AbsInd> AddAssign<&SymbolicTensor<Aind>> for SymbolicTensor<Aind> {
    fn add_assign(&mut self, rhs: &SymbolicTensor<Aind>) {
        debug_assert_eq!(self.structure, rhs.structure);
        self.expression += &rhs.expression;
    }
}

impl<Aind: AbsInd> ScalarMul<Atom> for SymbolicTensor<Aind> {
    type Output = SymbolicTensor<Aind>;

    fn scalar_mul(&self, rhs: &Atom) -> Option<Self::Output> {
        Some(Self {
            expression: rhs * &self.expression,
            structure: self.structure.clone(),
        })
    }
}

impl<Aind: AbsInd> std::fmt::Display for SymbolicTensor<Aind> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expression)
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::network::library::symbolic::ETS;
    use symbolica::parse;

    #[test]
    fn parse() {
        let _ = ETS.metric;
        let expr = parse!("g(mink(4,6),mink(4,7))");

        let structure = SymbolicTensor::from_permuted(
            &PermutedStructure::<ShadowedStructure<AbstractIndex>>::try_from(expr).unwrap(),
        )
        .unwrap();

        structure.contract(&structure).unwrap();
    }
}

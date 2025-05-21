use bitvec::{order::Lsb0, vec::BitVec};
use std::ops::AddAssign;

use crate::{
    algebra::ScalarMul,
    contraction::{Contract, ContractionError},
    network::{
        library::symbolic::{ExplicitKey, TensorLibrary},
        parsing::ShadowedStructure,
        store::NetworkStore,
        Network, Ref, TensorNetworkError,
    },
    shadowing::symbolica_utils::{IntoArgs, IntoSymbol},
    structure::{
        abstract_index::AIND_SYMBOLS, representation::LibrarySlot, HasName, HasStructure,
        MergeInfo, NamedStructure, OrderedStructure, StructureContract, TensorShell,
        TensorStructure, ToSymbolic,
    },
    tensors::parametric::MixedTensor,
};

use crate::structure::abstract_index::AbstractIndex;
use crate::structure::dimension::Dimension;
use crate::structure::representation::Representation;
use crate::structure::slot::IsAbstractSlot;
use crate::structure::StructureError;
use delegate::delegate;

use symbolica::atom::{Atom, AtomView, Symbol};

/// A fully symbolic tensor, with no concrete values.
///
/// This tensor is used to represent the structure of a tensor, and is used to perform symbolic contraction.
/// Currently contraction is just a multiplication of the atoms, but in the future this will ensure that internal indices are independent accross the contraction.
///
/// Additionally, this can also be used as a tensor structure, that tracks the history, much like [`HistoryStructure`].
#[derive(Debug, Clone)]
pub struct SymbolicTensor {
    pub structure: OrderedStructure,
    pub expression: symbolica::atom::Atom,
}

impl Ref for SymbolicTensor {
    type Ref<'a>
        = &'a SymbolicTensor
    where
        Self: 'a;

    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        self
    }
}

impl TensorStructure for SymbolicTensor {
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = SymbolicTensor;
    type Slot = LibrarySlot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        Ok(Self {
            structure: self.structure.reindex(indices)?,
            expression: self.expression,
        })
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl HasStructure for SymbolicTensor {
    type Structure = OrderedStructure;
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

impl StructureContract for SymbolicTensor {
    fn concat(&mut self, other: Self) {
        self.structure.concat(other.structure);
        self.expression = &other.expression * &self.expression;
    }

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
impl SymbolicTensor {
    pub fn from_named<N>(structure: &N) -> Option<Self>
    where
        N: ToSymbolic + HasName + TensorStructure<Slot = LibrarySlot>,
        N::Name: IntoSymbol + Clone,
        N::Args: IntoArgs,
    {
        Some(SymbolicTensor {
            expression: structure.to_symbolic()?,
            structure: structure.external_structure().into(),
        })
    }

    pub fn to_named(&self) -> NamedStructure<Symbol, Vec<Atom>> {
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

    #[must_use]
    pub fn get_atom(&self) -> &Atom {
        &self.expression
    }

    // pub fn to_mixed(self) -> MixedTensor {
    //     self.to_named().to_shell().to_explicit().unwrap()
    // }
    #[allow(clippy::type_complexity)]
    pub fn to_network(
        &self,
        library: &TensorLibrary<MixedTensor<f64, ExplicitKey>>,
    ) -> Result<
        Network<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, ExplicitKey>,
        TensorNetworkError<ExplicitKey>,
    > {
        Network::<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>,ExplicitKey>::try_from_view(
            self.expression.as_view(),
            library,
        )
    }
}

impl TryFrom<Atom> for SymbolicTensor {
    type Error = String;
    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        let structure = value
            .as_view()
            .try_into()
            .unwrap_or(OrderedStructure::empty());

        Ok(SymbolicTensor {
            structure,
            expression: value,
        })
    }
}

impl HasName for SymbolicTensor {
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
impl Contract<SymbolicTensor> for SymbolicTensor {
    type LCM = SymbolicTensor;
    fn contract(&self, other: &SymbolicTensor) -> Result<Self::LCM, ContractionError> {
        let expression = &other.expression * &self.expression;
        let (structure, _, _, _) = self.structure.merge(&other.structure)?;
        Ok(SymbolicTensor {
            expression,
            structure,
        })
    }
}

impl std::ops::Neg for SymbolicTensor {
    type Output = SymbolicTensor;

    fn neg(self) -> Self::Output {
        Self {
            expression: -self.expression,
            structure: self.structure,
        }
    }
}

impl AddAssign<SymbolicTensor> for SymbolicTensor {
    fn add_assign(&mut self, rhs: SymbolicTensor) {
        debug_assert_eq!(self.structure, rhs.structure);
        self.expression += rhs.expression;
    }
}

impl AddAssign<&SymbolicTensor> for SymbolicTensor {
    fn add_assign(&mut self, rhs: &SymbolicTensor) {
        debug_assert_eq!(self.structure, rhs.structure);
        self.expression += &rhs.expression;
    }
}

impl ScalarMul<Atom> for SymbolicTensor {
    type Output = SymbolicTensor;

    fn scalar_mul(&self, rhs: &Atom) -> Option<Self::Output> {
        Some(Self {
            expression: rhs * &self.expression,
            structure: self.structure.clone(),
        })
    }
}

impl std::fmt::Display for SymbolicTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expression)
    }
}

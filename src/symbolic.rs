use std::borrow::Cow;

use crate::{
    ContractionError, DataIterator, HasStructure, IntoArgs, NamedStructure, TensorNetworkError,
    TensorStructure, ABSTRACTIND,
};

use super::{
    Contract, FallibleAdd, HasName, IntoSymbol, MixedTensor, Shadowable, Slot, StructureContract,
    TensorNetwork, VecStructure,
};

use bitvec::vec;
use symbolica::{
    atom::{AddView, Atom, AtomView, MulView, Symbol},
    coefficient::CoefficientView,
    state::State,
};

/// A fully symbolic tensor, with no concrete values.
///
/// This tensor is used to represent the structure of a tensor, and is used to perform symbolic contraction.
/// Currently contraction is just a multiplication of the atoms, but in the future this will ensure that internal indices are independent accross the contraction.
///
/// Additionally, this can also be used as a tensor structure, that tracks the history, much like [`HistoryStructure`].
#[derive(Debug, Clone)]
pub struct SymbolicTensor {
    structure: VecStructure,
    expression: symbolica::atom::Atom,
}

impl HasStructure for SymbolicTensor {
    type Structure = VecStructure;
    type Scalar = Atom;

    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }
}

impl StructureContract for SymbolicTensor {
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        let structure = self.structure.merge_at(&other.structure, positions);
        // let mut out: Atom<Linear> = Atom::new();
        // other.expression.mul(state, ws, &self.expression, &mut out);

        SymbolicTensor {
            structure,
            expression: &other.expression * &self.expression,
        }
    }

    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.expression = &other.expression * &self.expression;
        self.structure.merge(&other.structure)
    }

    fn trace_out(&mut self) {
        self.structure.trace_out();
    }

    fn trace(&mut self, i: usize, j: usize) {
        self.structure.trace(i, j);
    }
}

#[allow(dead_code)]
impl SymbolicTensor {
    pub fn from_named<N>(structure: &N) -> Option<Self>
    where
        N: Shadowable + HasName,
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
            global_name: self.name().map(Cow::into_owned),
            additional_args: self.id().map(Cow::into_owned),
        }
    }

    pub fn empty(expression: Atom) -> Self {
        SymbolicTensor {
            structure: VecStructure::empty(),
            expression,
        }
    }

    #[must_use]
    pub fn get_atom(&self) -> &Atom {
        &self.expression
    }

    pub fn to_mixed(self) -> MixedTensor {
        self.smart_shadow().unwrap()
    }

    pub fn to_network(
        &self,
    ) -> Result<
        TensorNetwork<MixedTensor<f64, NamedStructure<Symbol, Vec<Atom>>>, Atom>,
        TensorNetworkError,
    > {
        self.expression.as_view().try_into()
    }
}

impl TryFrom<Atom> for SymbolicTensor {
    type Error = String;
    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        let structure = value.as_view().try_into().unwrap_or(VecStructure::empty());

        Ok(SymbolicTensor {
            structure,
            expression: value,
        })
    }
}

impl HasName for SymbolicTensor {
    type Name = Symbol;
    type Args = Vec<Atom>;

    fn name(&self) -> Option<Cow<Self::Name>> {
        if let AtomView::Fun(f) = self.expression.as_view() {
            Some(std::borrow::Cow::Owned(f.get_symbol()))
        } else {
            None
        }
    }

    fn set_name(&mut self, _name: Self::Name) {
        unimplemented!("Cannot set name of a symbolic tensor")
    }

    fn id(&self) -> Option<Cow<Self::Args>> {
        let mut args = vec![];
        match self.expression.as_view() {
            AtomView::Fun(f) => {
                for arg in f.iter() {
                    if let AtomView::Fun(f) = arg {
                        if f.get_symbol() != State::get_symbol(ABSTRACTIND) {
                            args.push(arg.to_owned());
                        }
                    } else {
                        args.push(arg.to_owned());
                    }
                }
                Some(Cow::Owned(args))
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
        let mut new_structure = self.structure.clone();

        let expression = &other.expression * &self.expression;
        new_structure.merge(&other.structure);
        Ok(SymbolicTensor {
            expression,
            structure: new_structure,
        })
    }
}

impl std::fmt::Display for SymbolicTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expression)
    }
}

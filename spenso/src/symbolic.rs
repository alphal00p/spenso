use crate::{
    contraction::{Contract, ContractionError},
    network::{TensorNetwork, TensorNetworkError},
    parametric::MixedTensor,
    structure::{
        abstract_index::AIND_SYMBOLS, representation::LibrarySlot, HasName, HasStructure,
        NamedStructure, StructureContract, TensorStructure, ToSymbolic, VecStructure,
    },
    symbolica_utils::{IntoArgs, IntoSymbol},
    tensor_library::{ExplicitKey, ShadowedStructure, TensorLibrary},
};

use symbolica::atom::{Atom, AtomView, Symbol};

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

    fn concat(&mut self, other: &Self) {
        self.structure.concat(&other.structure);
        self.expression = &other.expression * &self.expression;
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
            structure: VecStructure::empty(),
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
    ) -> Result<TensorNetwork<MixedTensor<f64, ShadowedStructure>, Atom>, TensorNetworkError> {
        TensorNetwork::<MixedTensor<f64, ShadowedStructure>, Atom>::try_from_view(
            self.expression.as_view(),
            library,
        )
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

use derive_more::Display;
use serde::{Deserialize, Serialize};
use symbolica::{atom::AtomView, coefficient::CoefficientView};
use thiserror::Error;

#[cfg(feature = "shadowing")]
use crate::symbolica_utils::SerializableSymbol;

#[cfg(feature = "shadowing")]
use symbolica::atom::{Atom, Symbol};

/// A Dimension
#[derive(
    Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Serialize, Deserialize, Display,
)]
pub enum Dimension {
    Concrete(usize),
    #[cfg(feature = "shadowing")]
    Symbolic(SerializableSymbol),
}

impl Dimension {
    pub fn new_concrete(value: usize) -> Self {
        Self::Concrete(value)
    }

    #[cfg(feature = "shadowing")]
    pub fn to_symbolic(&self) -> Atom {
        match self {
            Self::Concrete(c) => Atom::new_num(*c as i64),
            Self::Symbolic(s) => Atom::new_var((*s).into()),
        }
    }
}

#[derive(Error, Debug)]
pub enum DimensionError {
    #[error("Dimension is not concrete")]
    NotConcrete,
    #[error("Dimension too large")]
    TooLarge,
    #[error("Dimension negative")]
    Negative,
    #[error("Dimension is not natural")]
    NotNatural,
}

#[allow(unreachable_patterns)]
impl TryFrom<Dimension> for usize {
    type Error = DimensionError;

    fn try_from(value: Dimension) -> Result<Self, Self::Error> {
        match value {
            Dimension::Concrete(c) => Ok(c),
            _ => Err(DimensionError::NotConcrete),
        }
    }
}

impl From<usize> for Dimension {
    fn from(value: usize) -> Self {
        Dimension::Concrete(value)
    }
}

#[cfg(feature = "shadowing")]
impl From<Symbol> for Dimension {
    fn from(value: Symbol) -> Self {
        Dimension::Symbolic(value.into())
    }
}

#[cfg(feature = "shadowing")]
impl<'a> TryFrom<AtomView<'a>> for Dimension {
    type Error = DimensionError;
    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        match value {
            AtomView::Var(a) => Ok(Dimension::Symbolic(a.get_symbol().into())),
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, 1) => {
                    if n < 0 {
                        return Err(DimensionError::Negative);
                    }
                    Ok(Dimension::Concrete(n as usize))
                }
                _ => return Err(DimensionError::NotNatural),
            },
            a => Err(DimensionError::NotNatural),
        }
    }
}

#[allow(unreachable_patterns)]
impl PartialEq<usize> for Dimension {
    fn eq(&self, other: &usize) -> bool {
        match self {
            Self::Concrete(c) => c == other,
            _ => false,
        }
    }
}

#[allow(unreachable_patterns)]
impl PartialEq<Dimension> for usize {
    fn eq(&self, other: &Dimension) -> bool {
        match other {
            Dimension::Concrete(c) => c == self,
            _ => false,
        }
    }
}

#[cfg(feature = "shadowing")]
use anyhow::Result;

use serde::Deserialize;
use serde::Serialize;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::AddAssign;
#[cfg(feature = "shadowing")]
use symbolica::atom::{Atom, AtomView};
use symbolica::parse;

#[cfg(feature = "shadowing")]
use symbolica::coefficient::CoefficientView;

#[cfg(feature = "shadowing")]
use crate::symbolica_utils::SerializableSymbol;

use crate::utils::{to_subscript, to_superscript};

use thiserror::Error;

pub const ABSTRACTIND: &str = "aind";

pub const UPIND: &str = "uind";

pub const DOWNIND: &str = "dind";

pub const SELFDUALIND: &str = "sind";

/// A type that represents the name of an index in a tensor.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, Serialize, Deserialize)]
pub enum AbstractIndex {
    Normal(usize),
    Dualize(usize),
    #[cfg(feature = "shadowing")]
    Symbol(SerializableSymbol),
}

impl PartialEq for AbstractIndex {
    fn eq(&self, other: &Self) -> bool {
        usize::from(*self) == usize::from(*other)
    }
}

impl Hash for AbstractIndex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        usize::from(*self).hash(state)
    }
}

impl std::ops::Add<AbstractIndex> for AbstractIndex {
    type Output = AbstractIndex;
    fn add(self, rhs: AbstractIndex) -> Self::Output {
        match self {
            AbstractIndex::Normal(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Normal(l + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Normal(l + r),
                #[cfg(feature = "shadowing")]
                AbstractIndex::Symbol(r) => AbstractIndex::Normal(l + u32::from(r) as usize),
            },
            AbstractIndex::Dualize(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Normal(l + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Normal(l + r),
                #[cfg(feature = "shadowing")]
                AbstractIndex::Symbol(r) => AbstractIndex::Normal(l + u32::from(r) as usize),
            },
            #[cfg(feature = "shadowing")]
            AbstractIndex::Symbol(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Normal(u32::from(l) as usize + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Normal(u32::from(l) as usize + r),
                AbstractIndex::Symbol(r) => {
                    AbstractIndex::Normal(u32::from(l) as usize + u32::from(r) as usize)
                }
            },
        }
    }
}

impl AddAssign<AbstractIndex> for AbstractIndex {
    fn add_assign(&mut self, rhs: AbstractIndex) {
        *self = *self + rhs;
    }
}

impl std::fmt::Display for AbstractIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractIndex::Normal(v) => write!(f, "{}", to_subscript(*v as isize)),
            AbstractIndex::Dualize(v) => {
                write!(f, "{}", to_superscript(*v as isize))
            }
            #[cfg(feature = "shadowing")]
            AbstractIndex::Symbol(v) => write!(f, "{}", v),
        }
    }
}

impl From<usize> for AbstractIndex {
    fn from(value: usize) -> Self {
        AbstractIndex::Normal(value)
    }
}

#[cfg(feature = "shadowing")]
impl From<AbstractIndex> for Atom {
    fn from(value: AbstractIndex) -> Self {
        match value {
            AbstractIndex::Normal(v) => Atom::new_num(v as i64),
            AbstractIndex::Dualize(v) => Atom::new_num(-(v as i64)),
            #[cfg(feature = "shadowing")]
            AbstractIndex::Symbol(v) => Atom::new_var(v.into()),
        }
    }
}
impl From<AbstractIndex> for usize {
    fn from(value: AbstractIndex) -> Self {
        match value {
            AbstractIndex::Dualize(v) => v,
            AbstractIndex::Normal(v) => v,
            #[cfg(feature = "shadowing")]
            AbstractIndex::Symbol(v) => u32::from(v) as usize,
        }
    }
}

impl From<isize> for AbstractIndex {
    fn from(value: isize) -> Self {
        if value < 0 {
            AbstractIndex::Dualize(-value as usize)
        } else {
            AbstractIndex::Normal(value as usize)
        }
    }
}

impl From<i32> for AbstractIndex {
    fn from(value: i32) -> Self {
        if value < 0 {
            AbstractIndex::Dualize(-value as usize)
        } else {
            AbstractIndex::Normal(value as usize)
        }
    }
}

#[derive(Error, Debug)]
pub enum AbstractIndexError {
    #[error("Argument is not a natural number")]
    NotNatural,
    #[error("Argument  {0} is not a valid index")]
    NotIndex(String),
    #[error("parsing error")]
    ParsingError(String),
}

#[cfg(feature = "shadowing")]
impl TryFrom<AtomView<'_>> for AbstractIndex {
    type Error = AbstractIndexError;

    fn try_from(view: AtomView<'_>) -> Result<Self, Self::Error> {
        match view {
            AtomView::Num(n) => {
                if let CoefficientView::Natural(n, 1) = n.get_coeff_view() {
                    return Ok(AbstractIndex::from(n as i32));
                }
                Err(AbstractIndexError::NotNatural)
            }
            AtomView::Var(v) => Ok(AbstractIndex::Symbol(v.get_symbol().into())),
            _ => Err(AbstractIndexError::NotIndex(view.to_string())),
        }
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<std::string::String> for AbstractIndex {
    type Error = AbstractIndexError;

    fn try_from(value: std::string::String) -> Result<Self, Self::Error> {
        let atom = parse!(&value).map_err(AbstractIndexError::ParsingError)?;
        Self::try_from(atom.as_view())
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<&'_ str> for AbstractIndex {
    type Error = AbstractIndexError;

    fn try_from(value: &'_ str) -> Result<Self, Self::Error> {
        let atom = parse!(value).map_err(AbstractIndexError::ParsingError)?;
        Self::try_from(atom.as_view())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mem_size_test() {}
}

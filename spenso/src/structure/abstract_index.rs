#[cfg(feature = "shadowing")]
use anyhow::Result;

use serde::Deserialize;
use serde::Serialize;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::AddAssign;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::Symbol,
    atom::{Atom, AtomView},
    symbol, try_parse,
};

#[cfg(feature = "shadowing")]
use symbolica::coefficient::CoefficientView;

#[cfg(feature = "shadowing")]
use crate::shadowing::symbolica_utils::SerializableSymbol;
use crate::utils::{to_subscript, to_superscript};

use thiserror::Error;

use super::slot::AbsInd;
use super::slot::DummyAind;

pub const ABSTRACTIND: &str = "aind";

pub const UPIND: &str = "uind";

pub const DOWNIND: &str = "dind";

pub const SELFDUALIND: &str = "sind";

#[cfg(feature = "shadowing")]
pub struct AindSymbols {
    pub aind: Symbol,
    pub uind: Symbol,
    pub dind: Symbol,
    pub selfdualind: Symbol,
}

#[cfg(feature = "shadowing")]
#[cfg(test)]
mod test {

    use symbolica::{atom::AtomCore, function, id::Replacement, parse_lit};

    use super::*;

    #[test]
    fn normalisation() {
        let atom = function!(
            AIND_SYMBOLS.dind,
            function!(AIND_SYMBOLS.dind, function!(AIND_SYMBOLS.uind, Atom::Zero))
        );
        assert_eq!(atom, Atom::Zero, "{atom}");
        let atom = parse_lit!(dind(dind(f(1))));

        assert_eq!(atom, parse_lit!(f(1)), "{atom}");

        let f = symbol!("f");
        let fa = function!(f, symbol!("a__"));

        let atom = parse_lit!(g(dind(f(1)), f(2)));
        let tgt = parse_lit!(g(f(1), dind(f(2))));

        let rep = atom
            .replace(fa.clone())
            .with(function!(AIND_SYMBOLS.dind, fa));

        assert_eq!(rep, tgt, "{rep} not equal to {tgt}");

        let atom = parse_lit!(g(aind(f(1)), f(2)));
        let tgt = parse_lit!(g(f(1), aind(f(2))));
        let rep = atom.replace_multiple(&[
            Replacement::new(
                fa.clone().to_pattern(),
                function!(AIND_SYMBOLS.aind, fa).to_pattern(),
            ),
            Replacement::new(function!(AIND_SYMBOLS.aind, fa).to_pattern(), fa.clone()),
        ]);

        let rep2 = atom.replace_multiple(&[
            Replacement::new(function!(AIND_SYMBOLS.aind, fa).to_pattern(), fa.clone()),
            Replacement::new(
                fa.clone().to_pattern(),
                function!(AIND_SYMBOLS.aind, fa).to_pattern(),
            ),
        ]);

        assert_eq!(rep, tgt, "{rep} not equal to {tgt}");
        assert_eq!(rep2, tgt, "{rep2} not equal to {tgt}");
    }
}
#[cfg(feature = "shadowing")]
pub static AIND_SYMBOLS: std::sync::LazyLock<AindSymbols> =
    std::sync::LazyLock::new(|| AindSymbols {
        aind: symbol!(ABSTRACTIND),
        uind: symbol!(UPIND;;|view,out|{
            if let AtomView::Fun(f)=view{
                if f.get_nargs()==1{
                    *out=f.iter().next().unwrap().to_owned();
                    true
                }else{
                    // panic!("can only take one argument")
                    false
                }
            } else{
                false
            }
        }),
        dind: symbol!(DOWNIND;;|view,out|{
            if let AtomView::Fun(dind1)=view{
                if dind1.get_nargs()==1{
                    let arg = dind1.iter().next().unwrap();
                    if let AtomView::Fun(arg)=arg{
                        if arg.get_nargs()==1{
                            if arg.get_symbol()==symbol!(DOWNIND){
                                *out=arg.iter().next().unwrap().to_owned();
                            return true;
                            }else {
                                return false;
                            }
                        }
                    }
                    false
                }else{
                    // panic!("can only take one argument")
                    false
                }
            } else{
                false
            }
        }),
        selfdualind: symbol!(SELFDUALIND;;|view,out|{
            if let AtomView::Fun(f)=view{
                if f.get_nargs()==1{
                    *out=f.iter().next().unwrap().to_owned();
                    true
                }else{
                    // panic!("can only take one argument")
                    false
                }
            } else{
                false
            }
        }),
    });

static DUMMYCOUNTER: AtomicUsize = AtomicUsize::new(0);
/// A type that represents the name of an index in a tensor.
#[derive(
    Debug,
    Copy,
    Clone,
    Ord,
    PartialOrd,
    Eq,
    Serialize,
    Deserialize,
    bincode_trait_derive::Encode,
    bincode_trait_derive::Decode,
    // bincode_trait_derive::BorrowDecodeFromDecode,
)]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
pub enum AbstractIndex {
    Normal(usize),
    Dualize(usize),
    Double(u16, u16),
    Dummy(usize),
    Added(usize),
    #[cfg(feature = "shadowing")]
    Symbol(SerializableSymbol),
}

impl AbsInd for AbstractIndex {}

impl DummyAind for AbstractIndex {
    fn new_dummy() -> Self {
        AbstractIndex::Dummy(DUMMYCOUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

#[cfg(feature = "shadowing")]
impl From<Symbol> for AbstractIndex {
    fn from(value: Symbol) -> Self {
        AbstractIndex::Symbol(value.into())
    }
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
                AbstractIndex::Normal(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Added(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dummy(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Double(_, _) => panic!("cannot add double"),
                #[cfg(feature = "shadowing")]
                AbstractIndex::Symbol(r) => AbstractIndex::Added(l + r.get_id() as usize),
            },
            AbstractIndex::Dualize(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Added(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dummy(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Double(_, _) => panic!("cannot add double"),
                #[cfg(feature = "shadowing")]
                AbstractIndex::Symbol(r) => AbstractIndex::Added(l + r.get_id() as usize),
            },
            AbstractIndex::Added(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Added(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dummy(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Double(_, _) => panic!("cannot add double"),
                #[cfg(feature = "shadowing")]
                AbstractIndex::Symbol(r) => AbstractIndex::Added(l + r.get_id() as usize),
            },
            AbstractIndex::Dummy(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Added(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Dummy(r) => AbstractIndex::Added(l + r),
                AbstractIndex::Double(_, _) => panic!("cannot add double"),
                #[cfg(feature = "shadowing")]
                AbstractIndex::Symbol(r) => AbstractIndex::Added(l + r.get_id() as usize),
            },
            AbstractIndex::Double(_, _) => panic!("cannot add double"),

            #[cfg(feature = "shadowing")]
            AbstractIndex::Symbol(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Added(l.get_id() as usize + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Added(l.get_id() as usize + r),
                AbstractIndex::Added(r) => AbstractIndex::Added(l.get_id() as usize + r),
                AbstractIndex::Dummy(r) => AbstractIndex::Added(l.get_id() as usize + r),
                AbstractIndex::Double(_, _) => panic!("cannot add double"),
                AbstractIndex::Symbol(r) => {
                    AbstractIndex::Added(l.get_id() as usize + r.get_id() as usize)
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
            AbstractIndex::Normal(v) | AbstractIndex::Dualize(v) => {
                if f.sign_minus() {
                    write!(f, "{}", to_subscript(*v as isize))
                } else if f.sign_plus() {
                    write!(f, "{}", to_superscript(*v as isize))
                } else {
                    write!(f, "{}", v)
                }
            }
            AbstractIndex::Double(i, j) => {
                if f.sign_minus() {
                    write!(
                        f,
                        "{}.{}",
                        to_subscript(*i as isize),
                        to_subscript(*j as isize)
                    )
                } else if f.sign_plus() {
                    write!(
                        f,
                        "{}'{}",
                        to_superscript(*i as isize),
                        to_superscript(*j as isize)
                    )
                } else {
                    write!(f, "{}-{}", i, j)
                }
            }

            AbstractIndex::Dummy(v) => {
                if f.sign_minus() {
                    write!(f, "d{}", to_subscript(*v as isize))
                } else if f.sign_plus() {
                    write!(f, "d{}", to_superscript(*v as isize))
                } else {
                    write!(f, "d{}", v)
                }
            }
            AbstractIndex::Added(v) => {
                if f.sign_minus() {
                    write!(f, "(+){}", to_subscript(*v as isize))
                } else if f.sign_plus() {
                    write!(f, "(+){}", to_superscript(*v as isize))
                } else {
                    write!(f, "(+){}", v)
                }
            }
            #[cfg(feature = "shadowing")]
            AbstractIndex::Symbol(v) => {
                if f.sign_minus() {
                    write!(f, "_{}", v)
                } else if f.sign_plus() {
                    write!(f, "^{}", v)
                } else {
                    write!(f, "{}", v)
                }
            }
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
            AbstractIndex::Double(i, j) => Atom::num((i as i64, j as i64)),
            AbstractIndex::Normal(v) => Atom::num(v as i64),
            AbstractIndex::Dualize(v) => Atom::num(-(v as i64)),
            AbstractIndex::Added(v) => Atom::num(v as i64),
            AbstractIndex::Dummy(v) => Atom::var(symbol!(format!("d_{}", v))),
            AbstractIndex::Symbol(v) => Atom::var(v.into()),
        }
    }
}
#[cfg(feature = "shadowing")]
impl<'a> From<AbstractIndex> for symbolica::atom::AtomOrView<'a> {
    fn from(value: AbstractIndex) -> Self {
        symbolica::atom::AtomOrView::Atom(Atom::from(value))
    }
}
impl From<AbstractIndex> for usize {
    fn from(value: AbstractIndex) -> Self {
        match value {
            AbstractIndex::Double(i, j) => i as usize * u16::MAX as usize + j as usize,
            AbstractIndex::Dualize(v) => v,
            AbstractIndex::Normal(v) => v,
            AbstractIndex::Added(v) => v,
            AbstractIndex::Dummy(v) => v,
            #[cfg(feature = "shadowing")]
            AbstractIndex::Symbol(v) => v.get_id() as usize,
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
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, 1, _, _) => Ok(AbstractIndex::from(n as i32)),
                CoefficientView::Natural(n, d, _, _) => {
                    Ok(AbstractIndex::Double(n as u16, d as u16))
                }
                _ => Err(AbstractIndexError::NotNatural),
            },
            AtomView::Var(v) => Ok(AbstractIndex::Symbol(v.get_symbol().into())),
            _ => Err(AbstractIndexError::NotIndex(view.to_string())),
        }
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<std::string::String> for AbstractIndex {
    type Error = AbstractIndexError;

    fn try_from(value: std::string::String) -> Result<Self, Self::Error> {
        let atom = try_parse!(&value).map_err(AbstractIndexError::ParsingError)?;
        Self::try_from(atom.as_view())
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<&'_ str> for AbstractIndex {
    type Error = AbstractIndexError;

    fn try_from(value: &'_ str) -> Result<Self, Self::Error> {
        let atom = try_parse!(value).map_err(AbstractIndexError::ParsingError)?;
        Self::try_from(atom.as_view())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mem_size_test() {}
}

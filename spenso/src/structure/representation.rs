use super::{
    abstract_index::{AbstractIndex, AbstractIndexError},
    dimension::{Dimension, DimensionError},
    slot::Slot,
};
use ahash::AHashMap;
use append_only_vec::AppendOnlyVec;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use spenso_macros::SimpleRepresentation;
use std::{
    fmt::{Debug, Display},
    sync::RwLock,
};
use std::{hash::Hash, ops::Index};

#[cfg(feature = "shadowing")]
use crate::{
    structure::abstract_index::{DOWNIND, SELFDUALIND, UPIND},
    structure::slot::SlotError,
    tensor_library::ETS,
};

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, AtomCore, FunctionBuilder, Symbol},
    {function, symbol},
};

use thiserror::Error;

use anyhow::Result;

pub trait BaseRepName: RepName<Dual: RepName> + Default {
    const NAME: &'static str;
    // fn selfless_name() -> String;
    fn selfless_base() -> Self::Base;

    #[cfg(feature = "shadowing")]
    fn selfless_symbol() -> Symbol {
        symbol!(Self::NAME)
    }
    fn selfless_dual() -> Self::Dual;
    fn selfless_rep<D: Into<Dimension>>(dim: D) -> Representation<Self>
    where
        Self: Sized,
    {
        Representation {
            dim: dim.into(),
            rep: Self::default(),
        }
    }

    #[cfg(feature = "shadowing")]
    fn pattern(symbol: Symbol) -> Atom {
        Self::default().to_symbolic([Atom::new_var(symbol)])
    }

    fn slot<D: Into<Dimension>, A: Into<AbstractIndex>>(dim: D, aind: A) -> Slot<Self>
    where
        Self: Sized,
    {
        let aind: AbstractIndex = aind.into();
        Slot {
            rep: Self::selfless_rep(dim),
            aind,
        }
    }

    #[cfg(feature = "shadowing")]
    fn new_slot_from(
        sym: Symbol,
        dim: Dimension,
        aind: AbstractIndex,
    ) -> Result<Slot<Self>, SlotError> {
        if sym == Self::selfless_symbol() {
            Ok(Slot {
                rep: Representation {
                    dim,
                    rep: Self::default(),
                },
                aind,
            })
        } else {
            Err(SlotError::NotRepresentation)
        }
    }
}

#[derive(Error, Debug)]
pub enum RepresentationError {
    #[cfg(feature = "shadowing")]
    #[error("Symbol {0} isn't one of [sind,uind,dind]")]
    SymbolError(Symbol),
    #[cfg(feature = "shadowing")]
    #[error("Expected dual state: {0} but got {1}")]
    ExpectedDualStateError(Symbol, Symbol),
    #[cfg(feature = "shadowing")]
    #[error("{0} is not a possible Representation")]
    NotRepresentationError(Symbol),
    #[error("Wrong representation, expected {0},got {1}")]
    WrongRepresentationError(String, String),
    #[error("Abstract index error :{0}")]
    AindError(#[from] AbstractIndexError),
    #[error("{0}")]
    DimErr(#[from] DimensionError),
    #[error("{0}")]
    Any(#[from] anyhow::Error),
}

pub trait RepName:
    Copy + Clone + Debug + PartialEq + Eq + Hash + Display + Ord + Into<LibraryRep>
{
    type Dual: RepName<Dual = Self, Base = Self::Base>;
    type Base: RepName;
    fn dual(self) -> Self::Dual;
    fn is_dual(self) -> bool;
    fn base(&self) -> Self::Base;
    fn is_base(&self) -> bool;
    fn matches(&self, other: &Self::Dual) -> bool;
    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol, aind: Symbol) -> Result<Self, RepresentationError>;
    #[cfg(feature = "shadowing")]
    fn try_from_symbol_coerced(sym: Symbol) -> Result<Self, RepresentationError>;

    // fn try_from<B: BaseRepName>(b: B) -> Result<B, SlotError>;

    /// for the given concrete index, says whether it should have a minus sign during contraction
    ///
    /// for example see [`Self::negative`]
    #[must_use]
    fn is_neg(self, _i: usize) -> bool {
        false
    }

    #[allow(clippy::cast_possible_wrap)]
    #[cfg(feature = "shadowing")]
    /// yields a function builder for the representation, adding a first variable: the dimension.
    ///
    fn to_symbolic<It: Into<Atom>>(&self, args: impl IntoIterator<Item = It>) -> Atom;

    #[cfg(feature = "shadowing")]
    fn to_pattern(&self, symbol: Symbol) -> Atom {
        self.to_symbolic([Atom::new_var(symbol)])
    }

    fn slot<D: Into<Dimension>, A: Into<AbstractIndex>>(self, dim: D, aind: A) -> Slot<Self>
    where
        Self: Sized,
    {
        Slot {
            rep: self.rep(dim),
            aind: aind.into(),
        }
    }

    fn rep<D: Into<Dimension>>(&self, dim: D) -> Representation<Self>
    where
        Self: Sized,
    {
        Representation {
            dim: dim.into(),
            rep: *self,
        }
    }
}

#[rustfmt::skip]
#[derive(SimpleRepresentation)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Deserialize,
)]
#[representation(name = "euc", self_dual)] // Specify the dual name
pub struct Euclidean {}

#[rustfmt::skip]
#[derive(SimpleRepresentation)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Deserialize,
)]
#[representation(name = "lor")] // Specify the dual name
pub struct Lorentz {}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct Minkowski {}

impl From<Minkowski> for LibraryRep {
    fn from(_value: Minkowski) -> Self {
        ExtendibleReps::MINKOWSKI
    }
}

impl RepName for Minkowski {
    type Base = Minkowski;
    type Dual = Minkowski;

    fn base(&self) -> Self::Base {
        Minkowski::selfless_base()
    }

    fn is_base(&self) -> bool {
        true
    }

    fn dual(self) -> Self::Dual {
        Minkowski::selfless_dual()
    }

    fn is_dual(self) -> bool {
        true
    }

    fn matches(&self, _: &Self::Dual) -> bool {
        true
    }

    fn is_neg(self, i: usize) -> bool {
        i > 0
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol, aind: Symbol) -> Result<Self, RepresentationError> {
        let uind = symbol!(UPIND);
        let dind = symbol!(DOWNIND);
        let sind = symbol!(SELFDUALIND);
        if aind == sind {
            if Self::selfless_symbol() == sym {
                Ok(Minkowski::default())
            } else {
                Err(RepresentationError::NotRepresentationError(sym))
            }
        } else if aind == dind && aind == uind {
            Err(RepresentationError::ExpectedDualStateError(sind, aind))
        } else {
            Err(RepresentationError::SymbolError(aind))
        }
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol_coerced(sym: Symbol) -> Result<Self, RepresentationError> {
        if Self::selfless_symbol() == sym {
            Ok(Minkowski::default())
        } else {
            Err(RepresentationError::NotRepresentationError(sym))
        }
    }

    #[cfg(feature = "shadowing")]
    fn to_symbolic<It: Into<Atom>>(&self, args: impl IntoIterator<Item = It>) -> Atom {
        let mut fun = FunctionBuilder::new(symbol!(ExtendibleReps::mink_name()));
        for a in args {
            fun = fun.add_arg(&(a.into()));
        }
        fun.finish()
    }
}

impl Display for Minkowski {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mink")
    }
}

impl BaseRepName for Minkowski {
    const NAME: &'static str = ExtendibleReps::mink_name();

    // fn selfless_name() -> String {
    //     constname.to_string()
    // }

    fn selfless_base() -> Self::Base {
        Self::default()
    }

    fn selfless_dual() -> Self::Dual {
        Self::default()
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Representation<T: RepName> {
    pub dim: Dimension,
    pub rep: T,
}

impl<T: RepName> Representation<T> {
    #[cfg(feature = "shadowing")]
    pub fn pattern<A: AtomCore>(&self, aind: A) -> Atom {
        let mut atom = Atom::new();
        atom.set_from_view(&aind.as_atom_view());
        self.rep.to_symbolic([self.dim.to_symbolic(), atom])
    }

    #[cfg(feature = "shadowing")]
    pub fn to_pattern_wrapped(&self, aind: Symbol) -> Atom {
        self.rep.to_symbolic([
            self.dim.to_symbolic(),
            function!(symbol!("indexid"), Atom::new_var(aind)),
        ])
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LibraryRep {
    SelfDual(u16),
    Dualizable(i16),
}

pub type LibrarySlot = Slot<LibraryRep>;

pub(crate) static REPS: Lazy<RwLock<ExtendibleReps>> =
    Lazy::new(|| RwLock::new(ExtendibleReps::new()));
pub(crate) static SELF_DUAL: AppendOnlyVec<(LibraryRep, RepData)> = AppendOnlyVec::new();
pub(crate) static DUALIZABLE: AppendOnlyVec<(LibraryRep, RepData)> = AppendOnlyVec::new();

impl LibraryRep {
    pub fn new_dual(name: &str) -> Result<Self, RepError> {
        REPS.write().unwrap().new_dual_impl(name)
    }

    pub fn new_self_dual(name: &str) -> Result<Self, RepError> {
        REPS.write().unwrap().new_self_dual(name)
    }

    pub fn all_self_duals() -> impl Iterator<Item = &'static LibraryRep> {
        SELF_DUAL.iter().map(|(rep, _)| rep)
    }

    pub fn all_dualizables() -> impl Iterator<Item = &'static LibraryRep> {
        DUALIZABLE.iter().map(|(rep, _)| rep)
    }
}

pub struct RepData {
    // metric_data: Fn(Dimension)->SparseTensor<i8,IndexLess>
    name: String,
    #[cfg(feature = "shadowing")]
    symbol: Symbol,
}

pub struct ExtendibleReps {
    name_map: AHashMap<String, LibraryRep>,
    #[cfg(feature = "shadowing")]
    symbol_map: AHashMap<Symbol, LibraryRep>,
}

#[derive(Debug, Error)]
pub enum RepError {
    #[error("{0} Already exists and is of different type")]
    AlreadyExistsDifferentType(String),
}

impl ExtendibleReps {
    pub fn reps(&self) -> impl Iterator<Item = &LibraryRep> {
        self.name_map.values()
    }
    pub fn new_dual_impl(&mut self, name: &str) -> Result<LibraryRep, RepError> {
        if let Some(rep) = self.name_map.get(name) {
            if let LibraryRep::SelfDual(_) = rep {
                return Err(RepError::AlreadyExistsDifferentType(name.into()));
            } else {
                return Ok(*rep);
            }
        }
        let rep = LibraryRep::Dualizable(DUALIZABLE.len() as i16 + 1);

        self.name_map.insert(name.into(), rep);
        #[cfg(feature = "shadowing")]
        let symbol = symbol!(name);
        #[cfg(feature = "shadowing")]
        self.symbol_map.insert(symbol, rep);

        DUALIZABLE.push((
            rep,
            RepData {
                name: name.to_string(),
                #[cfg(feature = "shadowing")]
                symbol,
            },
        ));
        Ok(rep)
    }

    pub fn new_dual(name: &str) -> Result<LibraryRep, RepError> {
        REPS.write().unwrap().new_dual_impl(name)
    }

    pub fn new_self_dual(&mut self, name: &str) -> Result<LibraryRep, RepError> {
        if let Some(rep) = self.name_map.get(name) {
            if let LibraryRep::Dualizable(_) = rep {
                return Err(RepError::AlreadyExistsDifferentType(name.into()));
            } else {
                return Ok(*rep);
            }
        }

        let rep = LibraryRep::SelfDual(SELF_DUAL.len() as u16);
        self.name_map.insert(name.into(), rep);
        #[cfg(feature = "shadowing")]
        let symbol = symbol!(name);
        #[cfg(feature = "shadowing")]
        self.symbol_map.insert(symbol, rep);

        SELF_DUAL.push((
            rep,
            RepData {
                name: name.to_string(),
                #[cfg(feature = "shadowing")]
                symbol,
            },
        ));
        Ok(rep)
    }
}

impl Index<LibraryRep> for ExtendibleReps {
    type Output = RepData;

    fn index(&self, index: LibraryRep) -> &Self::Output {
        match index {
            LibraryRep::SelfDual(l) => &SELF_DUAL[l as usize].1,
            LibraryRep::Dualizable(l) => &DUALIZABLE[l.unsigned_abs() as usize - 1].1,
        }
    }
}

impl ExtendibleReps {
    pub const EUCLIDEAN: LibraryRep = LibraryRep::SelfDual(0);
    pub const BISPINOR: LibraryRep = LibraryRep::SelfDual(1);
    pub const COLORADJ: LibraryRep = LibraryRep::SelfDual(2);
    pub const MINKOWSKI: LibraryRep = LibraryRep::SelfDual(3);

    pub const LORENTZ_UP: LibraryRep = LibraryRep::Dualizable(1);
    pub const LORENTZ_DOWN: LibraryRep = LibraryRep::Dualizable(-1);
    pub const SPINFUND: LibraryRep = LibraryRep::Dualizable(2);
    pub const SPINANTIFUND: LibraryRep = LibraryRep::Dualizable(-2);
    pub const COLORFUND: LibraryRep = LibraryRep::Dualizable(3);
    pub const COLORANTIFUND: LibraryRep = LibraryRep::Dualizable(-3);
    pub const COLORSEXT: LibraryRep = LibraryRep::Dualizable(4);
    pub const COLORANTISEXT: LibraryRep = LibraryRep::Dualizable(-4);

    pub const BUILTIN_SELFDUAL_NAMES: [&'static str; 4] = ["euc", "bis", "coad", "mink"];
    pub const BUILTIN_DUALIZABLE_NAMES: [&'static str; 4] = ["lor", "spf", "cof", "cos"];

    pub const fn euc_name() -> &'static str {
        Self::BUILTIN_SELFDUAL_NAMES[0]
    }

    pub const fn bis_name() -> &'static str {
        Self::BUILTIN_SELFDUAL_NAMES[1]
    }

    pub const fn coad_name() -> &'static str {
        Self::BUILTIN_SELFDUAL_NAMES[2]
    }

    pub const fn mink_name() -> &'static str {
        Self::BUILTIN_SELFDUAL_NAMES[3]
    }

    pub const fn lor_name() -> &'static str {
        Self::BUILTIN_DUALIZABLE_NAMES[0]
    }

    pub const fn spf_name() -> &'static str {
        Self::BUILTIN_DUALIZABLE_NAMES[1]
    }

    pub const fn cof_name() -> &'static str {
        Self::BUILTIN_DUALIZABLE_NAMES[2]
    }

    pub const fn cos_name() -> &'static str {
        Self::BUILTIN_DUALIZABLE_NAMES[3]
    }

    pub fn new() -> Self {
        let mut new = Self {
            name_map: AHashMap::new(),
            #[cfg(feature = "shadowing")]
            symbol_map: AHashMap::new(),
        };

        #[cfg(feature = "shadowing")]
        ETS.id;

        for &name in Self::BUILTIN_SELFDUAL_NAMES.iter() {
            new.new_self_dual(name).unwrap();
        }

        for &name in Self::BUILTIN_DUALIZABLE_NAMES.iter() {
            new.new_dual_impl(name).unwrap();
        }
        new
    }

    #[cfg(feature = "shadowing")]
    pub fn find_symbol(&self, symbol: Symbol) -> Option<LibraryRep> {
        self.symbol_map.get(&symbol).cloned()
    }
}

impl Default for ExtendibleReps {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for LibraryRep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfDual(_) => write!(f, "{}", REPS.read().unwrap()[*self].name),
            Self::Dualizable(l) => {
                if *l < 0 {
                    write!(f, "{}🠓", REPS.read().unwrap()[*self].name)
                } else {
                    write!(f, "{}🠑", REPS.read().unwrap()[*self].name)
                }
            }
        }
    }
}

impl RepName for LibraryRep {
    type Dual = LibraryRep;
    type Base = LibraryRep;

    fn dual(self) -> Self::Dual {
        match self {
            Self::SelfDual(l) => Self::SelfDual(l),
            Self::Dualizable(l) => Self::Dualizable(-l),
        }
    }

    fn is_base(&self) -> bool {
        match self {
            Self::Dualizable(l) => *l > 0,
            _ => true,
        }
    }

    fn is_dual(self) -> bool {
        match self {
            Self::Dualizable(l) => l < 0,
            _ => true,
        }
    }

    fn base(&self) -> Self::Base {
        match self {
            Self::Dualizable(l) => Self::Dualizable(l.abs()),
            x => *x,
        }
    }

    fn matches(&self, other: &Self::Dual) -> bool {
        match (self, other) {
            (Self::SelfDual(s), Self::SelfDual(o)) => s == o,
            (Self::Dualizable(s), Self::Dualizable(o)) => *s == -o,
            _ => false,
        }
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol, aind: Symbol) -> Result<Self, RepresentationError> {
        let rep = REPS
            .read()
            .unwrap()
            .find_symbol(sym)
            .ok_or(RepresentationError::NotRepresentationError(sym))?;

        match rep {
            LibraryRep::Dualizable(_) => {
                if aind == symbol!(DOWNIND) {
                    Ok(rep.dual())
                } else if aind == symbol!(UPIND) {
                    Ok(rep)
                } else if aind == symbol!(SELFDUALIND) {
                    Err(RepresentationError::ExpectedDualStateError(
                        symbol!(UPIND),
                        aind,
                    ))
                } else {
                    Err(RepresentationError::SymbolError(aind))
                }
            }
            LibraryRep::SelfDual(_) => {
                if aind == symbol!(SELFDUALIND) {
                    Ok(rep)
                } else if aind == symbol!(UPIND) || aind == symbol!(DOWNIND) {
                    Err(RepresentationError::ExpectedDualStateError(
                        symbol!(SELFDUALIND),
                        aind,
                    ))
                } else {
                    Err(RepresentationError::SymbolError(aind))
                }
            }
        }
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol_coerced(sym: Symbol) -> Result<Self, RepresentationError> {
        REPS.read()
            .unwrap()
            .find_symbol(sym)
            .ok_or(RepresentationError::NotRepresentationError(sym))
    }

    #[cfg(feature = "shadowing")]
    /// yields a function builder for the representation, adding a first variable: the dimension.
    ///

    fn to_symbolic<It: Into<Atom>>(&self, args: impl IntoIterator<Item = It>) -> Atom {
        let mut fun = FunctionBuilder::new(REPS.read().unwrap()[*self].symbol);
        for a in args {
            fun = fun.add_arg(&a.into());
        }
        let inner = fun.finish();

        match self {
            Self::SelfDual(_) => inner,
            Self::Dualizable(l) => {
                if *l < 0 {
                    function!(symbol!(DOWNIND), &inner)
                } else {
                    inner
                }
            }
        }
    }
}

#[test]
fn extendible_reps() {
    let r = LibraryRep::new_dual("lor").unwrap();
    let rd = r.dual();
    let e = LibraryRep::new_self_dual("euc").unwrap();

    println!(
        "{r}{r:?}, {e}{e:?},{rd}{rd:?},{}",
        ExtendibleReps::BISPINOR.base()
    );

    assert!(ExtendibleReps::LORENTZ_UP.matches(&ExtendibleReps::LORENTZ_DOWN));
    assert!(!ExtendibleReps::LORENTZ_UP.matches(&ExtendibleReps::LORENTZ_UP));
    assert!(ExtendibleReps::BISPINOR.matches(&ExtendibleReps::BISPINOR));

    // let rs = r.new_slot(10, 1);
    // let rr = r.new_dimed_rep(1);

    // // println!("{}", rs.to_symbolic());
    // println!("{}", rs.dual());
    // println!("{}", rr)
}

// struct UserDefRep{
//     dual: usize,
//     name: String,
// }

// struct UserDefReps

// }

// impl RepName for usize{
//     type
// }

// pub trait HasDimension: RepName {
//     fn dim(&self) -> Dimension;

//     fn to_fnbuilder(&self) -> FunctionBuilder {
//         ::to_fnbuilder().add_arg(self.dim().to_symbolic().as_atom_view())
//     }
// }

// impl<T: BaseRepName<Base: BaseRepName, Dual: BaseRepName>> Representation<T> {
//     pub fn dual_pair(self) -> Representation<DualPair<T::Base>>
//     where
//         <T::Base as RepName>::Dual: RepName<Dual = T::Base, Base = T::Base>,
//         T::Base: RepName<Dual = T::Dual, Base = T::Base>,
//     {
//         Representation {
//             dim: self.dim,
//             rep: T::selfless_pair(),
//         }
//     }
// }

impl<T: RepName> Representation<T> {
    pub fn matches(&self, other: &Representation<T::Dual>) -> bool {
        self.dim == other.dim && self.rep.matches(&other.rep)
    }
    #[cfg(feature = "shadowing")]
    /// yields a function builder for the representation, adding a first variable: the dimension.
    ///
    pub fn to_symbolic(&self, args: impl IntoIterator<Item = Atom>) -> Atom {
        self.rep
            .to_symbolic([self.dim.to_symbolic()].into_iter().chain(args))
    }
    pub fn dual(self) -> Representation<T::Dual> {
        Representation {
            dim: self.dim,
            rep: self.rep.dual(),
        }
    }

    pub fn cast<U: RepName + From<T>>(self) -> Representation<U> {
        Representation {
            dim: self.dim,
            rep: U::from(self.rep),
        }
    }

    pub fn is_neg(&self, i: usize) -> bool {
        self.rep.is_neg(i)
    }

    pub fn slot<A: Into<AbstractIndex>>(&self, aind: A) -> Slot<T> {
        Slot {
            aind: aind.into(),
            rep: *self,
        }
    }

    #[inline]
    // this could be implemented directly in the fiberiterator.
    /// gives the vector of booleans, saying which concrete index along a Dimension/Abstract Index should have a minus sign during contraction.
    ///
    /// # Example
    /// ```
    /// # use spenso::structure::*;
    /// # use spenso::structure::representation::*;
    /// # use spenso::structure::dimension::*;
    /// # use spenso::structure::abstract_index::*;
    /// # use spenso::structure::slot::*;
    /// # use spenso::structure::concrete_index::*;
    /// let spin: Representation<Bispinor> = Bispinor::rep(5);
    ///
    /// let metric_diag: Vec<bool> = spin.negative().unwrap();
    ///
    /// let mut agree = true;
    ///
    /// for (i, r) in metric_diag.iter().enumerate() {
    ///   if r ^ spin.is_neg(i) {
    ///        agree = false;
    ///     }
    /// }
    ///
    /// assert!(agree);
    /// ```
    pub fn negative(&self) -> Result<Vec<bool>> {
        Ok((0..usize::try_from(self.dim)?)
            .map(|i| self.is_neg(i))
            .collect())
    }
}

#[test]
fn test_negative() {
    let spin: Representation<Euclidean> = Euclidean {}.rep(5);

    let metric_diag: Vec<bool> = spin.negative().unwrap();

    let mut agree = true;

    for (i, r) in metric_diag.iter().enumerate() {
        if r ^ spin.is_neg(i) {
            agree = false;
        }
    }

    assert!(agree);
}

impl<T: RepName> Display for Representation<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.rep, self.dim)
    }
}

impl From<Dimension> for Representation<Euclidean> {
    fn from(value: Dimension) -> Self {
        Representation {
            dim: value,
            rep: Euclidean {},
        }
    }
}

impl<T: RepName> From<Representation<T>> for Dimension {
    fn from(value: Representation<T>) -> Self {
        value.dim
    }
}

impl<T: RepName> TryFrom<Representation<T>> for usize {
    type Error = DimensionError;
    fn try_from(value: Representation<T>) -> std::result::Result<Self, Self::Error> {
        usize::try_from(value.dim)
    }
}

impl<'a, T: RepName> FromIterator<&'a Representation<T>> for Vec<Dimension> {
    fn from_iter<I: IntoIterator<Item = &'a Representation<T>>>(iter: I) -> Self {
        iter.into_iter().map(|rep| rep.dim).collect()
    }
}

#[cfg(test)]
#[cfg(feature = "shadowing")]
mod shadowing_tests {
    use symbolica::symbol;

    use crate::structure::representation::BaseRepName;

    // use super::Lorentz;

    // #[test]
    // fn rep_pattern() {
    //     println!("{}", Dual::<Lorentz>::pattern(symbol!("d_")));
    //     println!(
    //         "{}",
    //         Dual::<Lorentz>::rep(3).to_pattern_wrapped(symbol!("d_"))
    //     );
    // }
}

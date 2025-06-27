use super::{
    abstract_index::{AbstractIndex, AbstractIndexError},
    dimension::DimensionError,
    representation::{
        BaseRepName, LibraryRep, LibrarySlot, RepName, Representation, RepresentationError,
    },
};
use crate::structure::dimension::Dimension;
use bincode::{Decode, Encode};
// #[cfg(feature = "shadowing")]
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::{
    cmp::Ordering,
    fmt::{Debug, Display},
};
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, AtomView, ListIterator, Symbol},
    {function, symbol},
};

#[cfg(feature = "shadowing")]
use crate::network::library::symbolic::ETS;
use thiserror::Error;

#[derive(
    Debug,
    Copy,
    Clone,
    Ord,
    PartialOrd,
    Eq,
    PartialEq,
    Hash,
    Serialize,
    Deserialize,
    Encode,
    bincode_trait_derive::Decode,
    // bincode_trait_derive::BorrowDecodeFromDecode,
)]
#[cfg_attr(
    feature = "shadowing",
    trait_decode(trait = symbolica::state::HasStateMap),
)]
/// A [`Slot`] is an index, identified by a `usize` and a [`Representation`].
///
/// A vector of slots thus identifies the shape and type of the tensor.
/// Two indices are considered matching if *both* the `Dimension` and the [`Representation`] matches.
///
/// # Example
///
/// It can be built from a `Representation` calling one of the built in representations e.g.
/// ```
/// # use spenso::structure::*;
/// # use spenso::structure::representation::*;
/// # use spenso::structure::dimension::*;
/// # use spenso::structure::abstract_index::*;
/// # use spenso::structure::slot::*;
/// # use spenso::structure::concrete_index::*;
/// let mink: Representation<Lorentz> = Lorentz::rep(4);
/// let mud: Slot<Lorentz> = mink.new_slot(0);
/// let muu: Slot<Dual<Lorentz>> = mink.new_slot(0).dual();
/// assert!(mud.matches(&muu));
/// assert_eq!("lord4|₀", format!("{muu}"));
/// ```
/// Or one can define custom representations{}
/// ```
/// # use spenso::structure::*;
/// # use spenso::structure::representation::*;
/// # use spenso::structure::dimension::*;
/// # use spenso::structure::abstract_index::*;
/// # use spenso::structure::slot::*;
/// # use spenso::structure::concrete_index::*;
/// let custom_mink = Rep::new_dual("custom_lor").unwrap();
///
/// let nud: Slot<Rep> = custom_mink.new_slot(4, 0);
/// let nuu: Slot<Rep> = nud.dual();
///
/// assert!(nuu.matches(&nud));
/// assert_eq!("custom_lor🠓4|₀", format!("{nuu}"));
/// ```
pub struct Slot<T: RepName> {
    pub(crate) rep: Representation<T>,
    pub aind: AbstractIndex,
}

impl<T: RepName> Slot<T> {
    pub fn cast<U: RepName + From<T>>(self) -> Slot<U> {
        Slot {
            aind: self.aind,
            rep: Representation {
                dim: self.rep.dim,
                rep: U::from(self.rep.rep),
            },
        }
    }
    #[cfg(feature = "shadowing")]
    pub fn kroneker_atom(&self, other: &Slot<T::Dual>) -> Atom {
        function!(ETS.id, self.to_atom(), other.to_atom())
    }

    #[cfg(feature = "shadowing")]
    pub fn metric_atom(&self, other: &Slot<T>) -> Atom {
        function!(ETS.metric, self.to_atom(), other.to_atom())
    }
}

#[derive(Error, Debug)]
pub enum SlotError {
    #[error("Dimension is not concrete")]
    NotConcrete,
    #[error("Empty structure")]
    EmptyStructure,
    #[error("Argument is not a natural number")]
    NotNatural,
    #[error("Abstract index error :{0}")]
    AindError(#[from] AbstractIndexError),
    #[error("Representation error :{0}")]
    RepError(#[from] RepresentationError),
    #[error("Argument is not a number")]
    NotNumber,
    #[error("No more arguments")]
    NoMoreArguments,
    #[error("Too many arguments")]
    TooManyArguments,
    #[error("Not a slot, isn't a representation")]
    NotRepresentation,
    #[error("Not a slot, is composite")]
    Composite,
    #[error("{0}")]
    DimErr(#[from] DimensionError),
    #[error("{0}")]
    Any(#[from] anyhow::Error),
}

#[cfg(feature = "shadowing")]
/// Can possibly constuct a Slot from an `AtomView`, if it is of the form: <representation>(<dimension>,<index>)
///
/// # Example
///
/// ```
/// # use spenso::structure::*;
/// # use spenso::structure::representation::*;
/// # use spenso::structure::dimension::*;
/// # use spenso::structure::abstract_index::*;
/// # use spenso::structure::slot::*;
/// # use spenso::structure::concrete_index::*;
/// # use symbolica::atom::AtomView;

///    let mink = Lorentz::rep(4);
///    let mu = mink.new_slot(0);
///    let atom = mu.to_atom();
///    let slot = Slot::try_from(atom.as_view()).unwrap();
///    assert_eq!(slot, mu);
/// ```
impl<T: RepName> TryFrom<AtomView<'_>> for Slot<T> {
    type Error = SlotError;

    fn try_from(value: AtomView<'_>) -> Result<Self, Self::Error> {
        fn extract_num(iter: &mut ListIterator) -> Result<AbstractIndex, SlotError> {
            if let Some(a) = iter.next() {
                Ok(AbstractIndex::try_from(a)?)
            } else {
                Err(SlotError::NoMoreArguments)
            }
        }

        let (rep, mut iter) = if let AtomView::Fun(f) = value {
            let name = f.get_symbol();

            let innerf = f.iter().next().ok_or(SlotError::Composite)?;

            if let AtomView::Fun(innerf) = innerf {
                let rep =
                    T::try_from_symbol(innerf.get_symbol(), name).map_err(SlotError::RepError)?;

                (rep, innerf.iter())
            } else {
                let rep = T::try_from_symbol_coerced(name).map_err(SlotError::RepError)?;
                (rep, f.iter())
            }
        } else {
            return Err(SlotError::Composite);
        };

        let dim: Dimension = if let Some(a) = iter.next() {
            Dimension::try_from(a).map_err(SlotError::DimErr)?
        } else {
            return Err(SlotError::NoMoreArguments);
        };

        let index: AbstractIndex = extract_num(&mut iter)?;

        if extract_num(&mut iter).is_ok() {
            return Err(SlotError::TooManyArguments);
        }

        Ok(Slot {
            rep: Representation { dim, rep },
            aind: index,
        })
    }
}

// pub trait SlotFromRep<S:IsAbstractSlot>:Rep{

// }

pub trait ConstructibleSlot<T: RepName> {
    fn new(rep: T, dim: Dimension, aind: AbstractIndex) -> Self;
}

impl<T: BaseRepName> ConstructibleSlot<T> for Slot<T> {
    fn new(_: T, dim: Dimension, aind: AbstractIndex) -> Self {
        Slot {
            aind,
            rep: Representation {
                dim,
                rep: T::default(),
            },
        }
    }
}

pub trait IsAbstractSlot: Copy + PartialEq + Eq + Debug + Clone + Hash + Ord + Display {
    type R: RepName;

    fn reindex(self, id: AbstractIndex) -> Self;
    fn dim(&self) -> Dimension;
    fn to_dummy(&self) -> LibrarySlot {
        let rep = self.rep().to_dummy().to_lib();
        let aind = AbstractIndex::new_dummy();
        Slot { rep, aind }
    }
    fn to_lib(&self) -> LibrarySlot {
        let rep: LibraryRep = self.rep_name().into();
        rep.new_slot(self.dim(), self.aind())
    }
    fn aind(&self) -> AbstractIndex;
    fn set_aind(&mut self, aind: AbstractIndex);
    fn rep_name(&self) -> Self::R;
    fn rep(&self) -> Representation<Self::R> {
        Representation {
            dim: self.dim(),
            rep: self.rep_name(),
        }
    }

    #[cfg(feature = "shadowing")]
    /// using the function builder of the representation add the abstract index as an argument, and finish it to an Atom.
    /// # Example
    ///
    /// ```
    /// # use symbolica::state::{State, Workspace};
    /// # use spenso::structure::*;
    /// # use spenso::structure::representation::*;
    /// # use spenso::structure::dimension::*;
    /// # use spenso::structure::abstract_index::*;
    /// # use spenso::structure::slot::*;
    /// # use spenso::structure::concrete_index::*;
    /// let mink = Lorentz::rep(4);
    /// let mu = mink.new_slot(0);
    /// println!("{}", mu.to_atom());
    /// assert_eq!("lor(4,0)", mu.to_atom().to_string());
    /// assert_eq!("lor4|₀", mu.to_string());
    /// ```
    fn to_atom(&self) -> Atom;
    #[cfg(feature = "shadowing")]
    fn to_symbolic_wrapped(&self) -> Atom;
    #[cfg(feature = "shadowing")]
    fn try_from_view(v: AtomView<'_>) -> Result<Self, SlotError>;
}

pub trait DualSlotTo: IsAbstractSlot {
    type Dual: IsAbstractSlot;
    fn dual(&self) -> Self::Dual;
    fn matches(&self, other: &Self::Dual) -> bool;

    fn match_cmp(&self, other: &Self::Dual) -> Ordering;
}

impl<T: RepName> IsAbstractSlot for Slot<T> {
    type R = T;
    // type Dual = GenSlot<T::Dual>;
    fn dim(&self) -> Dimension {
        self.rep.dim
    }

    fn reindex(mut self, id: AbstractIndex) -> Self {
        self.aind = id;
        self
    }
    fn aind(&self) -> AbstractIndex {
        self.aind
    }
    fn rep_name(&self) -> Self::R {
        self.rep.rep
    }

    fn set_aind(&mut self, aind: AbstractIndex) {
        self.aind = aind;
    }
    #[cfg(feature = "shadowing")]
    fn to_atom(&self) -> Atom {
        self.rep.to_symbolic([Atom::from(self.aind)])
    }
    #[cfg(feature = "shadowing")]
    fn to_symbolic_wrapped(&self) -> Atom {
        use symbolica::function;

        self.rep
            .to_symbolic([function!(symbol!("indexid"), Atom::from(self.aind))])
    }
    #[cfg(feature = "shadowing")]
    fn try_from_view(v: AtomView<'_>) -> Result<Self, SlotError> {
        Slot::try_from(v)
    }
}

impl<T: RepName> std::fmt::Display for Slot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}|{}", self.rep, self.aind)
    }
}

impl<T: RepName> Slot<T> {
    #[cfg(feature = "shadowing")]
    pub fn to_pattern(&self, dimension: Symbol) -> Atom {
        self.rep
            .rep
            .to_symbolic([Atom::var(dimension), Atom::from(self.aind)])
    }
}

impl<T: RepName> DualSlotTo for Slot<T> {
    type Dual = Slot<T::Dual>;
    fn dual(&self) -> Slot<T::Dual> {
        Slot {
            rep: self.rep.dual(),
            aind: self.aind,
        }
    }
    fn matches(&self, other: &Self::Dual) -> bool {
        self.rep.matches(&other.rep) && self.aind() == other.aind()
    }

    fn match_cmp(&self, other: &Self::Dual) -> Ordering {
        self.rep
            .match_cmp(&other.rep)
            .then(self.aind.cmp(&other.aind))
    }
}

#[cfg(test)]
#[cfg(feature = "shadowing")]
mod shadowing_tests {
    use symbolica::{atom::AtomCore, parse, symbol};

    use crate::structure::{
        representation::{DualLorentz, LibraryRep, Lorentz, RepName, Representation},
        slot::{DualSlotTo, IsAbstractSlot, Slot},
    };

    #[test]
    fn doc_slot() {
        let mink: Representation<Lorentz> = Lorentz {}.new_rep(4);

        let mud: Slot<Lorentz> = mink.slot(0);
        let muu: Slot<DualLorentz> = mink.slot(0).dual();

        assert!(mud.matches(&muu));
        assert_eq!("lor🠓4|₀", format!("{muu}"));

        let custom_mink = LibraryRep::new_dual("custom_lor").unwrap();

        let nud: Slot<LibraryRep> = custom_mink.new_slot(4, 0);
        let nuu: Slot<LibraryRep> = nud.dual();

        assert!(nuu.matches(&nud));
        assert_eq!("custom_lor🠓4|₀", format!("{nuu}"));
    }

    #[test]
    fn to_symbolic() {
        let mink = Lorentz {}.new_rep(4);
        let mu = mink.slot(0);
        println!("{}", mu.to_atom());
        assert_eq!("spenso::lor(4,0)", mu.to_atom().to_canonical_string());
        assert_eq!("lor🠑4|₀", mu.to_string());

        let mink = Lorentz {}.new_rep(4);
        let mu = mink.slot(0);
        let atom = mu.to_atom();
        let slot = Slot::try_from(atom.as_view()).unwrap();
        assert_eq!(slot, mu);
    }

    #[test]
    fn slot_from_atom_view() {
        let mink = Lorentz {}.new_rep(4);
        let mu = mink.slot(0);
        let atom = mu.to_atom();
        assert_eq!(Slot::try_from(atom.as_view()).unwrap(), mu);
        assert_eq!(
            Slot::<Lorentz>::try_from(atom.as_view()).unwrap().dual(),
            mu.dual()
        );
        assert_eq!(
            Slot::try_from(mu.dual().to_atom().as_view()).unwrap(),
            mu.dual()
        );

        let expr = parse!("dind(lor(4,-1))");

        let _slot: Slot<LibraryRep> = Slot::try_from(expr.as_view()).unwrap();
        let _slot: Slot<DualLorentz> = Slot::try_from(expr.as_view()).unwrap();

        println!("{}", _slot.to_symbolic_wrapped());
        println!("{}", _slot.to_pattern(symbol!("d_")));
    }
}

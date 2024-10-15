use super::{
    abstract_index::AbstractIndex,
    dimension::DimensionError,
    representation::{BaseRepName, Dual, DualPair, PhysReps, RepName, Representation},
};
use crate::structure::{
    dimension::Dimension,
    representation::{Lorentz, Rep},
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::Hash;
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{AsAtomView, Atom, AtomView, FunctionBuilder, ListIterator, Symbol},
    coefficient::CoefficientView,
    state::State,
};

use thiserror::Error;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
/// A [`Slot`] is an index, identified by a `usize` and a [`Representation`].
///
/// A vector of slots thus identifies the shape and type of the tensor.
/// Two indices are considered matching if *both* the `Dimension` and the [`Representation`] matches.
///
/// # Example
///
/// It can be built from a `Representation` calling one of the built in representations e.g.
/// ```
/// # use spenso::structure::{Representation,Slot,Dimension,AbstractIndex,Lorentz,Dual,RepName,BaseRepName,DualSlotTo};
/// let mink: Representation<Lorentz> = Lorentz::rep(4);
/// let mud: Slot<Lorentz> = mink.new_slot(0);
/// let muu: Slot<Dual<Lorentz>> = mink.new_slot(0).dual();
/// assert!(mud.matches(&muu));
/// assert_eq!("lord4|₀", format!("{muu}"));
/// ```
/// Or one can define custom representations{}
/// ```
/// # use spenso::structure::{Rep,Slot,RepName,DualSlotTo};
/// let custom_mink = Rep::new_dual("custom_lor").unwrap();
///
/// let nud: Slot<Rep> = custom_mink.new_slot(4, 0);
/// let nuu: Slot<Rep> = nud.dual();
///
/// assert!(nuu.matches(&nud));
/// assert_eq!("custom_lor🠓4|₀", format!("{nuu}"));
/// ```
pub struct Slot<T: RepName> {
    pub aind: AbstractIndex,
    pub(crate) rep: Representation<T>,
}

#[test]
fn doc_slot() {
    let mink: Representation<Lorentz> = Lorentz::rep(4);

    let mud: Slot<Lorentz> = mink.new_slot(0);
    let muu: Slot<Dual<Lorentz>> = mink.new_slot(0).dual();

    assert!(mud.matches(&muu));
    assert_eq!("lord4|₀", format!("{muu}"));

    let custom_mink = Rep::new_dual("custom_lor").unwrap();

    let nud: Slot<Rep> = custom_mink.new_slot(4, 0);
    let nuu: Slot<Rep> = nud.dual();

    assert!(nuu.matches(&nud));
    assert_eq!("custom_lor🠓4|₀", format!("{nuu}"));
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
        let value_builder = FunctionBuilder::new(State::get_symbol("id"));

        let indices = FunctionBuilder::new(State::get_symbol("aind"))
            .add_arg(&self.to_symbolic())
            .add_arg(&other.to_symbolic())
            .finish();
        value_builder.add_arg(&indices).finish()
    }

    #[cfg(feature = "shadowing")]
    pub fn metric_atom(&self, other: &Slot<T>) -> Atom {
        let value_builder = FunctionBuilder::new(State::get_symbol("metric"));

        let indices = FunctionBuilder::new(State::get_symbol("aind"))
            .add_arg(&self.to_symbolic())
            .add_arg(&other.to_symbolic())
            .finish();
        value_builder.add_arg(&indices).finish()
    }
}

#[derive(Error, Debug)]
pub enum SlotError {
    #[error("Dimension is not concrete")]
    NotConcrete,
    #[error("Argument is not a natural number")]
    NotNatural,
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
///
/// # use spenso::structure::{Representation,Slot,Dimension,AbstractIndex,ToSymbolic,Lorentz,RepName,BaseRepName,IsAbstractSlot};
/// # use symbolica::atom::AtomView;

///    let mink = Lorentz::rep(4);
///    let mu = mink.new_slot(0);
///    let atom = mu.to_symbolic();
///    let slot = Slot::try_from(atom.as_view()).unwrap();
///    assert_eq!(slot, mu);
/// ```
impl<T: RepName> TryFrom<AtomView<'_>> for Slot<T> {
    type Error = SlotError;

    fn try_from(value: AtomView<'_>) -> Result<Self, Self::Error> {
        fn extract_num(iter: &mut ListIterator) -> Result<i64, SlotError> {
            if let Some(a) = iter.next() {
                if let AtomView::Num(n) = a {
                    if let CoefficientView::Natural(n, 1) = n.get_coeff_view() {
                        return Ok(n);
                    }
                    return Err(SlotError::NotNatural);
                }
                Err(SlotError::NotNumber)
            } else {
                Err(SlotError::NoMoreArguments)
            }
        }

        let mut iter = if let AtomView::Fun(f) = value {
            f.iter()
        } else {
            return Err(SlotError::Composite);
        };

        let dim: Dimension = Dimension::new_concrete(
            usize::try_from(extract_num(&mut iter)?).or(Err(DimensionError::TooLarge))?,
        );
        let index: AbstractIndex = usize::try_from(extract_num(&mut iter)?)
            .or(Err(DimensionError::TooLarge))?
            .into();

        if extract_num(&mut iter).is_ok() {
            return Err(SlotError::TooManyArguments);
        }

        if let AtomView::Fun(f) = value {
            let sym = f.get_symbol();
            let rep = T::try_from_symbol(sym)?;

            Ok(Slot {
                rep: Representation { dim, rep },
                aind: index,
            })
        } else {
            Err(SlotError::Composite)
        }
    }
}

#[cfg(feature = "shadowing")]
#[test]
fn feature() {
    let mink = Lorentz::rep(4);
    let mu = mink.new_slot(0);
    let atom = mu.to_symbolic();
    let slot = Slot::try_from(atom.as_view()).unwrap();
    assert_eq!(slot, mu);
}
impl<T: BaseRepName<Base = T>> Slot<T>
where
    Dual<T>: BaseRepName<Base = T, Dual = T>,
{
    pub fn dual_pair(self) -> Slot<DualPair<T::Base>>
    where
        T: RepName<Dual = Dual<T>>,
    {
        Slot {
            aind: self.aind,
            rep: self.rep.dual_pair(),
        }
    }
}

impl<T: BaseRepName<Dual = Dual<T>, Base = T>> Slot<Dual<T>>
where
    Dual<T>: BaseRepName<Dual = T, Base = T>,
{
    pub fn pair(self) -> Slot<DualPair<T>> {
        Slot {
            aind: self.aind,
            rep: self.rep.pair(),
        }
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

pub trait IsAbstractSlot: Copy + PartialEq + Eq + Debug + Clone + Hash {
    type R: RepName;
    fn dim(&self) -> Dimension;
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
    /// # use spenso::structure::{Representation,Slot,Dimension,AbstractIndex,Lorentz,ToSymbolic,IsAbstractSlot,BaseRepName};
    /// let mink = Lorentz::rep(4);
    /// let mu = mink.new_slot(0);
    /// println!("{}", mu.to_symbolic());
    /// assert_eq!("loru(4,0)", mu.to_symbolic().to_string());
    /// assert_eq!("loru4|₀", mu.to_string());
    /// ```
    fn to_symbolic(&self) -> Atom;
    #[cfg(feature = "shadowing")]
    fn to_symbolic_wrapped(&self) -> Atom;
    #[cfg(feature = "shadowing")]
    fn try_from_view(v: AtomView<'_>) -> Result<Self, SlotError>;
}

#[cfg(feature = "shadowing")]
#[test]
fn to_symbolic() {
    let mink = Lorentz::rep(4);
    let mu = mink.new_slot(0);
    println!("{}", mu.to_symbolic());
    assert_eq!("loru(4,0)", mu.to_symbolic().to_string());
    assert_eq!("loru4|₀", mu.to_string());
}

pub trait DualSlotTo: IsAbstractSlot {
    type Dual: IsAbstractSlot;
    fn dual(&self) -> Self::Dual;
    fn matches(&self, other: &Self::Dual) -> bool;
}

impl<T: RepName> IsAbstractSlot for Slot<T> {
    type R = T;
    // type Dual = GenSlot<T::Dual>;
    fn dim(&self) -> Dimension {
        self.rep.dim
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
    fn to_symbolic(&self) -> Atom {
        let mut value_builder = self.rep.to_fnbuilder();
        value_builder =
            value_builder.add_arg(Atom::new_num(usize::from(self.aind) as i64).as_atom_view());
        value_builder.finish()
    }
    #[cfg(feature = "shadowing")]
    fn to_symbolic_wrapped(&self) -> Atom {
        let mut value_builder = self.rep.to_fnbuilder();
        let id = Atom::parse(&format!("indexid({})", self.aind)).unwrap();
        value_builder = value_builder.add_arg(&id);
        value_builder.finish()
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
}

pub type PhysicalSlots = Slot<PhysReps>;

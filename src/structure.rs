use ahash::AHashMap;
use delegate::delegate;
use derive_more::Add;
use derive_more::AddAssign;
use derive_more::Display;
use derive_more::From;
use derive_more::Index;
use derive_more::Into;
use derive_more::IntoIterator;
use derive_more::Mul;
use derive_more::MulAssign;
use derive_more::Rem;
use derive_more::RemAssign;
use derive_more::Sub;
use derive_more::SubAssign;
use indexmap::IndexMap;
use serde::Deserialize;
use serde::Serialize;

use smartstring::LazyCompact;
use smartstring::SmartString;
use std::fmt::Debug;
#[cfg(feature = "shadowing")]
use std::fmt::Display;
use std::ops::Deref;

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{
        representation::FunView, AsAtomView, Atom, AtomView, FunctionBuilder, ListIterator,
        MulView, Symbol,
    },
    coefficient::CoefficientView,
    state::State,
};

use std::ops::Range;

use crate::Permutation;

use std::collections::HashSet;
use std::{cmp::Ordering, collections::HashMap};

use super::TensorStructureIndexIterator;
#[cfg(feature = "shadowing")]
use super::{ufo, DenseTensor, MixedTensor};
use smartstring::alias::String;
/// A type that represents the name of an index in a tensor.
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
    From,
    Into,
    Display,
    Add,
    AddAssign,
)]
#[display(fmt = "id{}", _0)]
pub struct AbstractIndex(pub usize);

#[cfg(feature = "shadowing")]
impl TryFrom<AtomView<'_>> for AbstractIndex {
    type Error = String;

    fn try_from(view: AtomView<'_>) -> Result<Self, Self::Error> {
        if let AtomView::Var(v) = view {
            Ok(AbstractIndex(v.get_symbol().get_id() as usize))
        } else {
            Err("Not a var".to_string().into())
        }
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<std::string::String> for AbstractIndex {
    type Error = String;

    fn try_from(value: std::string::String) -> Result<Self, Self::Error> {
        let atom = Atom::parse(&value)?;
        Self::try_from(atom.as_view())
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<&'_ str> for AbstractIndex {
    type Error = String;

    fn try_from(value: &'_ str) -> Result<Self, Self::Error> {
        let atom = Atom::parse(value)?;
        Self::try_from(atom.as_view())
    }
}

/// A Dimension
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
    From,
    Into,
    Add,
    Display,
)]
#[into(owned, ref, ref_mut)]
#[display(fmt = "{}", _0)]
pub struct Dimension(pub usize);

impl PartialEq<usize> for Dimension {
    fn eq(&self, other: &usize) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Dimension> for usize {
    fn eq(&self, other: &Dimension) -> bool {
        *self == other.0
    }
}

impl PartialOrd<usize> for Dimension {
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<Dimension> for usize {
    fn partial_cmp(&self, other: &Dimension) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}

/// A  concrete index, i.e. the concrete usize/index of the corresponding abstract index

pub type ConcreteIndex = usize;

#[derive(
    Debug,
    Clone,
    Ord,
    PartialOrd,
    Eq,
    PartialEq,
    Hash,
    Index,
    Serialize,
    Deserialize,
    From,
    Into,
    Display,
    IntoIterator,
)]
#[display(fmt = "{:?}", indices)]

pub struct ExpandedIndex {
    indices: Vec<ConcreteIndex>,
}

impl Deref for ExpandedIndex {
    type Target = [ConcreteIndex];

    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}

impl FromIterator<ConcreteIndex> for ExpandedIndex {
    fn from_iter<T: IntoIterator<Item = ConcreteIndex>>(iter: T) -> Self {
        ExpandedIndex {
            indices: iter.into_iter().collect(),
        }
    }
}

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
    From,
    Rem,
    RemAssign,
    Into,
    Display,
    Add,
    Mul,
    MulAssign,
    AddAssign,
    Sub,
    SubAssign,
)]
#[display(fmt = "{}", index)]
pub struct FlatIndex {
    index: usize,
}

pub const EUCLIDEAN: &str = "euc";
pub const LORENTZ: &str = "lor";
pub const BISPINOR: &str = "bis";
pub const SPINFUND: &str = "spin";
pub const SPINANTIFUND: &str = "spina";
pub const COLORADJ: &str = "coad";
pub const COLORFUND: &str = "cof";
pub const COLORANTIFUND: &str = "coaf";
pub const COLORSEXT: &str = "cos";
pub const COLORANTISEXT: &str = "coas";
pub const CONCRETEIND: &str = "cind";
pub const ABSTRACTIND: &str = "aind";

/// A Representation/Dimension of the index.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Representation {
    /// Represents a Euclidean space of the given dimension, with metric diag(1,1,1,1,...)
    Euclidean(Dimension),
    /// Represents a Minkowski space of the given dimension, with metric diag(1,-1,-1,-1,...)
    Lorentz(Dimension),
    Bispinor(Dimension),
    /// Represents a Spinor Fundamental space of the given dimension
    SpinFundamental(Dimension),
    /// Represents a Spinor Adjoint space of the given dimension
    SpinAntiFundamental(Dimension),
    /// Represents a Color Fundamental space of the given dimension
    ColorFundamental(Dimension),
    /// Represents a Color Anti-Fundamental space of the given dimension
    ColorAntiFundamental(Dimension),
    /// Represents a Color Adjoint space of the given dimension
    ColorAdjoint(Dimension),
    /// Represents a Color Sextet space of the given dimension
    ColorSextet(Dimension),
    /// Represents a Color Anti-Sextet space of the given dimension
    ColorAntiSextet(Dimension),
}

impl Representation {
    #[inline]
    // this could be implemented directly in the fiberiterator.
    /// gives the vector of booleans, saying which concrete index along a Dimension/Abstract Index should have a minus sign during contraction.
    ///
    /// # Example
    /// ```
    /// # use spenso::Representation;
    /// # use spenso::Dimension;
    /// let spin = Representation::Bispinor(Dimension(5));
    ///
    /// let metric_diag = spin.negative();
    ///
    /// let mut agree= true;
    ///
    ///     for (i,r) in metric_diag.iter().enumerate(){
    ///         if (r ^ spin.is_neg(i)) {
    ///             agree = false;
    ///         }
    ///     }
    ///
    /// assert!(agree);
    /// ```
    #[must_use]
    pub fn negative(&self) -> Vec<bool> {
        match *self {
            Self::Lorentz(value) => std::iter::once(false)
                .chain(std::iter::repeat(true).take(value.0 - 1))
                .collect::<Vec<_>>(),
            Self::Euclidean(value)
            | Self::Bispinor(value)
            | Self::SpinFundamental(value)
            | Self::SpinAntiFundamental(value) => {
                vec![false; value.into()]
            }
            Self::ColorAdjoint(value)
            | Self::ColorFundamental(value)
            | Self::ColorAntiFundamental(value)
            | Self::ColorSextet(value)
            | Self::ColorAntiSextet(value) => vec![false; value.into()],
        }
    }

    /// for the given concrete index, says whether it should have a minus sign during contraction
    ///
    /// for example see [`Self::negative`]
    #[inline]
    #[must_use]
    pub const fn is_neg(&self, i: usize) -> bool {
        match self {
            Self::Lorentz(_) => i > 0,
            _ => false,
        }
    }

    /// yields a function builder for the representation, adding a first variable: the dimension.
    ///
    /// for example see [`Slot::to_symbolic`]
    #[allow(clippy::cast_possible_wrap)]
    #[cfg(feature = "shadowing")]
    pub fn to_fnbuilder<'a, 'b: 'a>(&'a self) -> FunctionBuilder {
        let (value, id) = match *self {
            Self::Euclidean(value) => (value, State::get_symbol(EUCLIDEAN)),
            Self::Lorentz(value) => (value, State::get_symbol(LORENTZ)),
            Self::Bispinor(value) => (value, State::get_symbol(BISPINOR)),
            Self::SpinFundamental(value) => (value, State::get_symbol(SPINFUND)),
            Self::SpinAntiFundamental(value) => (value, State::get_symbol(SPINANTIFUND)),
            Self::ColorAdjoint(value) => (value, State::get_symbol(COLORADJ)),
            Self::ColorFundamental(value) => (value, State::get_symbol(COLORFUND)),
            Self::ColorAntiFundamental(value) => (value, State::get_symbol(COLORANTIFUND)),
            Self::ColorSextet(value) => (value, State::get_symbol(COLORSEXT)),
            Self::ColorAntiSextet(value) => (value, State::get_symbol(COLORANTISEXT)),
        };

        let mut value_builder = FunctionBuilder::new(id);

        value_builder =
            value_builder.add_arg(Atom::new_num(usize::from(value) as i64).as_atom_view());

        value_builder
    }

    /// Finishes the function builder into an Atom
    ///
    /// # Example
    ///
    /// ```
    /// # use symbolica::state::{State, Workspace};
    /// # use spenso::Representation;
    /// # use spenso::Dimension;
    ///
    /// let mink = Representation::Lorentz(Dimension(4));
    ///
    /// assert_eq!("lor(4)",format!("{}",mink.to_symbolic()));
    /// assert_eq!("lor4",format!("{}",mink));
    /// ```
    #[cfg(feature = "shadowing")]
    pub fn to_symbolic(&self) -> Atom {
        self.to_fnbuilder().finish()
    }
}

impl From<Dimension> for Representation {
    fn from(value: Dimension) -> Self {
        Self::Euclidean(value)
    }
}

impl From<usize> for Representation {
    fn from(value: usize) -> Self {
        Self::Euclidean(value.into())
    }
}

impl<'a> std::iter::FromIterator<&'a Representation> for Vec<Dimension> {
    fn from_iter<T: IntoIterator<Item = &'a Representation>>(iter: T) -> Self {
        iter.into_iter()
            .map(|&rep| -> Dimension { (&rep).into() })
            .collect()
    }
}

impl From<&Representation> for Dimension {
    fn from(rep: &Representation) -> Self {
        match rep {
            Representation::Euclidean(value)
            | Representation::Lorentz(value)
            | Representation::Bispinor(value)
            | Representation::SpinFundamental(value)
            | Representation::SpinAntiFundamental(value) => *value,
            Representation::ColorAdjoint(value) => *value, //Dimension(8),
            Representation::ColorFundamental(value)
            | Representation::ColorAntiFundamental(value) => {
                *value // Dimension(3)
            }
            Representation::ColorSextet(value) | Representation::ColorAntiSextet(value) => *value,
        }
    }
}

impl From<&Representation> for usize {
    fn from(value: &Representation) -> Self {
        usize::from(Dimension::from(value))
    }
}

impl From<Representation> for Dimension {
    fn from(rep: Representation) -> Self {
        match rep {
            Representation::Euclidean(value)
            | Representation::Lorentz(value)
            | Representation::Bispinor(value)
            | Representation::SpinFundamental(value)
            | Representation::SpinAntiFundamental(value) => value,
            Representation::ColorAdjoint(value) => value,
            Representation::ColorFundamental(value)
            | Representation::ColorAntiFundamental(value) => {
                value // Dimension(3)
            }
            Representation::ColorSextet(value) | Representation::ColorAntiSextet(value) => value, //Dimension(6),
        }
    }
}

impl From<Representation> for usize {
    fn from(value: Representation) -> Self {
        usize::from(Dimension::from(value))
    }
}

impl std::fmt::Display for Representation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Euclidean(value) => write!(f, "{EUCLIDEAN}{value}"),
            Self::Lorentz(value) => write!(f, "{LORENTZ}{value}"),
            Self::Bispinor(value) => write!(f, "{BISPINOR}{value}"),
            Self::SpinFundamental(value) => write!(f, "{SPINFUND}{value}"),
            Self::SpinAntiFundamental(value) => write!(f, "{SPINANTIFUND}{value}"),
            Self::ColorAdjoint(value) => write!(f, "{COLORADJ}{value}"),
            Self::ColorFundamental(value) => write!(f, "{COLORFUND}{value}"),
            Self::ColorAntiFundamental(value) => write!(f, "{COLORANTIFUND}{value}"),
            Self::ColorSextet(value) => write!(f, "{COLORSEXT}{value}"),
            Self::ColorAntiSextet(value) => write!(f, "{COLORANTISEXT}{value}"),
        }
    }
}

/// A [`Slot`] is an index, identified by a `usize` and a [`Representation`].
///
/// A vector of slots thus identifies the shape and type of the tensor.
/// Two indices are considered matching if *both* the `usize` and the [`Representation`] matches.
///
/// # Example
///
/// It can be built from a tuple of `usize` and `Representation`
/// ```
/// # use spenso::{Representation,Slot,Dimension,AbstractIndex};
/// let mink = Representation::Lorentz(Dimension(4));
/// let mu = Slot::from((AbstractIndex(0),mink));
/// let nu = Slot::from((AbstractIndex(1),mink));
///
/// assert_eq!("id0lor4",format!("{}",mu));
/// assert_eq!("id1lor4",format!("{}",nu));
/// ```
///
/// It can also be built from a tuple of `usize` and `usize`, where we default to `Representation::Euclidean`
/// ```
/// # use spenso::{Representation,Slot};
/// let mu = Slot::from((0,4));
/// assert_eq!("id0euc4",format!("{}",mu));
/// ```
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    pub index: AbstractIndex,
    pub representation: Representation,
}

/// Can possibly constuct a Slot from an `AtomView`, if it is of the form: <representation>(<dimension>,<index>)
///
/// # Example
///
/// ```
///
/// # use spenso::{Representation,Slot,Dimension,AbstractIndex};
/// # use symbolica::atom::AtomView;

///    let mink = Representation::Lorentz(Dimension(4));
///    let mu = Slot::from((AbstractIndex(0), mink));
///    let atom = mu.to_symbolic();
///    let slot = Slot::try_from(atom.as_view()).unwrap();
///    assert_eq!(slot, mu);
/// ```
#[cfg(feature = "shadowing")]
impl TryFrom<AtomView<'_>> for Slot {
    type Error = &'static str;

    fn try_from(value: AtomView<'_>) -> Result<Self, Self::Error> {
        fn extract_num(iter: &mut ListIterator) -> Result<i64, &'static str> {
            if let Some(a) = iter.next() {
                if let AtomView::Num(n) = a {
                    if let CoefficientView::Natural(n, 1) = n.get_coeff_view() {
                        return Ok(n);
                    }
                    return Err("Argument is not a natural number");
                }
                Err("Argument is not a number")
            } else {
                Err("No more arguments")
            }
        }

        let mut iter = if let AtomView::Fun(f) = value {
            f.iter()
        } else {
            return Err("Not a slot, is composite");
        };

        let dim: Dimension = usize::try_from(extract_num(&mut iter)?)
            .or(Err("Dimension too large"))?
            .into();
        let index: AbstractIndex = usize::try_from(extract_num(&mut iter)?)
            .or(Err("Dimension too large"))?
            .into();

        if extract_num(&mut iter).is_ok() {
            return Err("Too many arguments");
        }

        let euc = State::get_symbol(EUCLIDEAN);

        let lor = State::get_symbol(LORENTZ);
        let bis = State::get_symbol(BISPINOR);
        let spin = State::get_symbol(SPINFUND);
        let spina = State::get_symbol(SPINANTIFUND);
        let coad = State::get_symbol(COLORADJ);
        let cof = State::get_symbol(COLORFUND);
        let coaf = State::get_symbol(COLORANTIFUND);
        let cos = State::get_symbol(COLORSEXT);
        let coas = State::get_symbol(COLORANTISEXT);

        let representation = if let AtomView::Fun(f) = value {
            let sym = f.get_symbol();
            match sym {
                _ if sym == euc => Representation::Euclidean(dim),
                _ if sym == lor => Representation::Lorentz(dim),
                _ if sym == bis => Representation::Bispinor(dim),
                _ if sym == spin => Representation::SpinFundamental(dim),
                _ if sym == spina => Representation::SpinAntiFundamental(dim),
                _ if sym == coad => Representation::ColorAdjoint(dim),
                _ if sym == cof => Representation::ColorFundamental(dim),
                _ if sym == coaf => Representation::ColorAntiFundamental(dim),
                _ if sym == cos => Representation::ColorSextet(dim),
                _ if sym == coas => Representation::ColorAntiSextet(dim),
                _ => return Err("Not a slot, isn't a representation"),
            }
        } else {
            return Err("Not a slot, is composite");
        };

        Ok(Slot {
            index,
            representation,
        })
    }
}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.representation.cmp(&other.representation) {
            Ordering::Equal => self.index.cmp(&other.index),
            other => other,
        }
    }
}

impl From<(AbstractIndex, Representation)> for Slot {
    fn from(value: (AbstractIndex, Representation)) -> Self {
        Self {
            index: value.0,
            representation: value.1,
        }
    }
}

impl From<(usize, usize)> for Slot {
    fn from(value: (usize, usize)) -> Self {
        Self {
            index: value.0.into(),
            representation: value.1.into(),
        }
    }
}

#[allow(clippy::cast_possible_wrap)]
impl Slot {
    /// using the function builder of the representation add the abstract index as an argument, and finish it to an Atom.
    /// # Example
    ///
    /// ```
    /// # use symbolica::state::{State, Workspace};
    /// # use spenso::{Representation,Slot,Dimension,AbstractIndex};
    /// let mink = Representation::Lorentz(Dimension(4));
    /// let mu = Slot::from((AbstractIndex(0),mink));
    ///
    /// assert_eq!("lor(4,0)",format!("{}",mu.to_symbolic()));
    /// assert_eq!("id0lor4",format!("{}",mu));
    /// ```
    #[cfg(feature = "shadowing")]
    pub fn to_symbolic(&self) -> Atom {
        let mut value_builder = self.representation.to_fnbuilder();
        value_builder =
            value_builder.add_arg(Atom::new_num(usize::from(self.index) as i64).as_atom_view());
        value_builder.finish()
    }
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.index, self.representation)
    }
}

/// A trait for a any struct that functions as a tensor structure.
/// Only three methods are required to be implemented, the rest are default implementations.
///
/// The associated type `Structure` is the type of the structure. This is usefull for containers of structures, like a datatensor.
/// The two methods `structure` and `mut_structure` are used to get a reference to the structure, and a mutable reference to the structure.
///
///
///
///
pub trait HasStructure {
    type Structure: TensorStructure;
    type Scalar;
    fn structure<'a>(&'a self) -> &'a Self::Structure;
    fn mut_structure(&mut self) -> &mut Self::Structure;
    fn set_structure_name<N>(&mut self, name: N)
    where
        Self::Structure: HasName<Name = N>,
    {
        self.mut_structure().set_name(name);
    }
    fn structure_name(&self) -> Option<<Self::Structure as HasName>::Name>
    where
        Self::Structure: HasName,
    {
        self.structure().name()
    }
    fn structure_id(&self) -> Option<<Self::Structure as HasName>::Args>
    where
        Self::Structure: HasName,
    {
        self.structure().id()
    }
    // fn cast_structure<O, S>(self) -> O
    // where
    //     O: HasStructure<Structure = S, Scalar = Self::Scalar>,
    //     S: TensorStructure + From<Self::Structure>;
}

pub struct Tensor<Store, Structure> {
    pub store: Store,
    pub structure: Structure,
}

#[allow(dead_code)]
impl<Store, Structure> Tensor<Store, Structure> {
    fn cast<NewStructure>(self) -> Tensor<Store, NewStructure>
    where
        NewStructure: TensorStructure + From<Structure>,
    {
        Tensor {
            store: self.store,
            structure: self.structure.into(),
        }
    }
}

impl<T> TensorStructure for T
where
    T: HasStructure,
{
    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = &Representation>;
            fn external_indices_iter(&self)-> impl Iterator<Item = &AbstractIndex>;
            fn get_slot(&self, i: usize)-> Option<Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn order(&self)-> usize;
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T> ToSymbolic for T where T: HasStructure {}

#[cfg(feature = "shadowing")]
pub trait ToSymbolic: TensorStructure {
    fn concrete_atom(&self, id: FlatIndex) -> Atom {
        let exp = self.expanded_index(id).unwrap();
        let mut cind = FunctionBuilder::new(State::get_symbol(CONCRETEIND));
        for i in exp.iter() {
            cind = cind.add_arg(Atom::new_num(*i as i64).as_atom_view());
        }
        cind.finish()
    }
    fn shadow_with(self, name: Symbol, args: &[Atom]) -> DenseTensor<Atom, Self>
    where
        Self: std::marker::Sized + Clone,
    {
        let mut data = vec![];
        for index in self.index_iter() {
            data.push(atomic_expanded_label_id(&index, name, args));
        }

        DenseTensor {
            data,
            structure: self,
        }
    }
    fn to_explicit_rep(self, name: Symbol, args: &[Atom]) -> MixedTensor<f64, Self>
    where
        Self: std::marker::Sized + Clone,
    {
        let identity = State::get_symbol("id");
        let gamma = State::get_symbol("γ");
        let gamma5 = State::get_symbol("γ5");
        let proj_m = State::get_symbol("ProjM");
        let proj_p = State::get_symbol("ProjP");
        let sigma = State::get_symbol("σ");
        let metric = State::get_symbol("Metric");

        match name {
            _ if name == identity => ufo::identity_data::<f64, Self>(self).into(),

            _ if name == gamma => ufo::gamma_data(self).into(),
            _ if name == gamma5 => ufo::gamma5_data(self).into(),
            _ if name == proj_m => ufo::proj_m_data(self).into(),
            _ if name == proj_p => ufo::proj_p_data(self).into(),
            _ if name == sigma => ufo::sigma_data(self).into(),
            _ if name == metric => ufo::metric_data::<f64, Self>(self).into(),
            name => MixedTensor::param(self.shadow_with(name, args).into()),
        }
    }
}

pub trait TensorStructure {
    fn external_reps_iter(&self) -> impl Iterator<Item = &Representation>;
    fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex>;
    fn get_slot(&self, i: usize) -> Option<Slot>;
    fn get_rep(&self, i: usize) -> Option<Representation>;
    fn get_dim(&self, i: usize) -> Option<Dimension>;
    fn order(&self) -> usize;
    /// returns the list of slots that are the external indices of the tensor
    fn external_structure_iter(&self) -> impl Iterator<Item = Slot> {
        self.external_indices_iter()
            .zip(self.external_reps_iter())
            .map(|(i, r)| Slot::from((*i, *r)))
    }

    fn external_structure(&self) -> Vec<Slot> {
        self.external_structure_iter().collect()
    }

    fn to_shell(self) -> TensorShell<Self>
    where
        Self: Sized,
    {
        TensorShell::new(self)
    }

    fn contains(&self, slot: &Slot) -> bool {
        self.external_structure_iter().any(|s| s == *slot)
    }

    fn external_reps(&self) -> Vec<Representation> {
        self.external_reps_iter().cloned().collect()
    }

    fn external_indices(&self) -> Vec<AbstractIndex> {
        self.external_indices_iter().cloned().collect()
    }

    // fn iter_index_along_fiber(&self,fiber_position: &[bool]  )-> TensorStructureMultiFiberIterator where Self: Sized{
    //     TensorStructureMultiFiberIterator::new(self, fiber_position)
    // }

    // fn single_fiber_at(&self,fiber_pos:usize)->Fiber{
    //     let mut  f =Fiber::zeros(self.external_structure().len());
    //     f.free(fiber_pos);
    //     f.is_single();
    //     f
    // }

    /// checks if the tensor has the same exact structure as another tensor
    fn same_content(&self, other: &Self) -> bool {
        self.same_external(other)
    }

    /// Given two [`TensorStructure`]s, returns the index of the first matching slot in each external index list, along with a boolean indicating if there is a single match
    fn match_index(&self, other: &Self) -> Option<(bool, usize, usize)> {
        let posmap = self
            .external_structure_iter()
            .enumerate()
            .map(|(i, slot)| (slot, i))
            .collect::<AHashMap<_, _>>();

        let mut first_pair: Option<(usize, usize)> = None;

        for (j, slot) in other.external_structure_iter().enumerate() {
            if let Some(&i) = posmap.get(&slot) {
                if let Some((i, j)) = first_pair {
                    // Found a second match, return early with false indicating non-unique match
                    return Some((false, i, j));
                }
                first_pair = Some((i, j));
            }
        }

        first_pair.map(|(i, j)| (true, i, j)) // Maps the found pair to Some with true indicating a unique match, or None if no match was found
    }

    /// Given two [`TensorStructure`]s, returns the index of the first matching slot in each external index list
    fn match_indices(&self, other: &Self) -> Option<(Permutation, Vec<bool>, Vec<bool>)> {
        let mut self_matches = vec![false; self.order()];
        let mut perm = Vec::new();
        let mut other_matches = vec![false; other.order()];

        let posmap = self
            .external_structure_iter()
            .enumerate()
            .map(|(i, slot)| (slot, i))
            .collect::<AHashMap<_, _>>();

        for (j, slot_other) in other.external_structure_iter().enumerate() {
            if let Some(&i) = posmap.get(&slot_other) {
                self_matches[i] = true;
                other_matches[j] = true;
                perm.push(i);
            }
        }

        if perm.is_empty() {
            None
        } else {
            let p: Permutation = Permutation::sort(&perm);
            Some((p, self_matches, other_matches))
        }
    }
    /// Identify the repeated slots in the external index list
    fn traces(&self) -> Vec<[usize; 2]> {
        let mut positions = HashMap::new();

        // Track the positions of each element
        for (index, value) in self.external_structure_iter().enumerate() {
            positions.entry(value).or_insert_with(Vec::new).push(index);
        }

        // Collect only the positions of repeated elements
        positions
            .into_iter()
            .filter_map(|(_, indices)| {
                if indices.len() == 2 {
                    Some([indices[0], indices[1]])
                } else {
                    None
                }
            })
            .collect()
    }

    /// yields the (outwards facing) shape of the tensor as a list of dimensions
    fn shape(&self) -> Vec<Dimension> {
        self.external_reps_iter().collect()
    }

    fn reps(&self) -> Vec<Representation> {
        self.external_reps_iter().map(|r| r.clone()).collect()
    }

    /// yields the order/total valence of the tensor, i.e. the number of indices
    /// (or misnamed : rank)

    /// checks if externally, the two tensors are the same
    fn same_external(&self, other: &Self) -> bool {
        let set1: HashSet<_> = self.external_structure_iter().collect();
        let set2: HashSet<_> = other.external_structure_iter().collect();
        set1 == set2
    }

    /// find the permutation of the external indices that would make the two tensors the same. Applying the permutation to other should make it the same as self
    fn find_permutation(&self, other: &Self) -> Option<Vec<ConcreteIndex>> {
        if self.order() != other.order() {
            return None;
        }

        let mut index_map = HashMap::new();
        for (i, item) in other.external_structure_iter().enumerate() {
            index_map.entry(item).or_insert_with(Vec::new).push(i);
        }

        let mut permutation = Vec::with_capacity(self.order());
        let mut used_indices = HashSet::new();
        for item in self.external_structure_iter() {
            if let Some(indices) = index_map.get_mut(&item) {
                // Find an index that hasn't been used yet
                if let Some(&index) = indices.iter().find(|&&i| !used_indices.contains(&i)) {
                    permutation.push(index);
                    used_indices.insert(index);
                } else {
                    // No available index for this item
                    return None;
                }
            } else {
                // Item not found in other
                return None;
            }
        }

        Some(permutation)
    }

    /// yields the strides of the tensor in column major order
    fn strides_column_major(&self) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![1; self.order()];

        if self.order() == 0 {
            return strides;
        }

        for i in 0..self.order() - 1 {
            strides[i + 1] = strides[i] * usize::from((self.shape()[i]).0);
        }

        strides
    }

    /// yields the strides of the tensor in row major order
    fn strides_row_major(&self) -> Vec<usize> {
        let mut strides = vec![1; self.order()];
        if self.order() == 0 {
            return strides;
        }

        for i in (0..self.order() - 1).rev() {
            strides[i] = strides[i + 1] * usize::from(self.shape()[i + 1].0);
        }

        strides
    }

    /// By default, the strides are row major
    fn strides(&self) -> Vec<usize> {
        self.strides_row_major()
    }

    /// Verifies that the list of indices provided are valid for the tensor
    ///
    /// # Errors
    ///
    /// `Mismatched order` = if the length of the indices is different from the order of the tensor,
    ///
    /// `Index out of bounds` = if the index is out of bounds for the dimension of that index   
    ///
    fn verify_indices(&self, indices: &[ConcreteIndex]) -> Result<(), String> {
        if indices.len() != self.order() {
            return Err("Mismatched order".into());
        }

        for (i, dim_len) in self
            .external_structure_iter()
            .map(|slot| slot.representation)
            .enumerate()
        {
            if indices[i] >= usize::from(dim_len) {
                return Err(format!(
                    "Index {} out of bounds for dimension {} of size {}",
                    indices[i],
                    i,
                    usize::from(dim_len)
                )
                .into());
            }
        }
        Ok(())
    }

    /// yields the flat index of the tensor given a list of indices
    ///
    /// # Errors
    ///
    /// Same as [`Self::verify_indices`]
    fn flat_index(&self, indices: &[ConcreteIndex]) -> Result<FlatIndex, String> {
        let strides = self.strides();
        self.verify_indices(indices)?;

        let mut idx = 0;
        for (i, &index) in indices.iter().enumerate() {
            idx += index * strides[i];
        }
        Ok(FlatIndex { index: idx })
    }

    /// yields the expanded index of the tensor given a flat index
    ///
    /// # Errors
    ///
    /// `Index out of bounds` = if the flat index is out of bounds for the tensor
    fn expanded_index(&self, flat_index: FlatIndex) -> Result<ExpandedIndex, String> {
        let mut indices = vec![];
        let mut index = flat_index.index;
        for &stride in &self.strides() {
            indices.push(index / stride);
            index %= stride;
        }
        if flat_index.index < self.size() {
            Ok(indices.into())
        } else {
            Err(format!("Index {flat_index} out of bounds").into())
        }
    }

    /// yields an iterator over the indices of the tensor
    fn index_iter(&self) -> TensorStructureIndexIterator<Self>
    where
        Self: Sized,
    {
        TensorStructureIndexIterator::new(self)
    }

    /// if the tensor has no (external) indices, it is a scalar
    fn is_scalar(&self) -> bool {
        self.order() == 0
    }

    /// get the metric along the i-th index
    fn get_ith_metric(&self, i: usize) -> Option<Vec<bool>> {
        Some(self.get_rep(i)?.negative())
    }

    /// yields the size of the tensor, i.e. the product of the dimensions. This is the length of the vector of the data in a dense tensor
    fn size(&self) -> usize {
        self.shape().iter().map(|x| usize::from(*x)).product()
    }
}

// impl<'a> HasStructure for &'a [Slot] {
//     type Structure = &'a [Slot];
//     type Scalar = ();

//     fn order(&self) -> usize {
//         self.len()
//     }

//     fn structure(&self) -> &Self::Structure {
//         self
//     }

//     fn mut_structure(&mut self) -> &mut Self::Structure {
//         self
//     }
// }

impl TensorStructure for Vec<Slot> {
    fn external_reps_iter(&self) -> impl Iterator<Item = &Representation> {
        self.iter().map(|s| &s.representation)
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex> {
        self.iter().map(|s| &s.index)
    }

    fn order(&self) -> usize {
        self.len()
    }

    fn get_slot(&self, i: usize) -> Option<Slot> {
        self.get(i).cloned()
    }

    fn get_rep(&self, i: usize) -> Option<Representation> {
        self.get(i).map(|s| s.representation)
    }

    fn get_dim(&self, i: usize) -> Option<Dimension> {
        self.get(i).map(|s| s.representation.into())
    }
}

#[cfg(feature = "shadowing")]
impl ToSymbolic for Vec<Slot> {}

/// A trait for a structure that can be traced and merged, during a contraction.
pub trait StructureContract {
    fn trace(&mut self, i: usize, j: usize);

    fn trace_out(&mut self);

    fn merge(&mut self, other: &Self) -> Option<usize>;

    #[must_use]
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self;
}

impl StructureContract for Vec<Slot> {
    fn trace(&mut self, i: usize, j: usize) {
        if i < j {
            self.trace(j, i);
            return;
        }
        let a = self.remove(i);
        let b = self.remove(j);
        assert_eq!(a, b);
    }

    fn trace_out(&mut self) {
        let mut positions = IndexMap::new();

        // Track the positions of each element
        for (index, &value) in self.iter().enumerate() {
            positions.entry(value).or_insert_with(Vec::new).push(index);
        }
        // Collect only the positions of non- repeated elements

        *self = positions
            .into_iter()
            .filter_map(|(value, indices)| {
                if indices.len() == 1 {
                    Some(value)
                } else {
                    None
                }
            })
            .collect();
    }

    fn merge(&mut self, other: &Self) -> Option<usize> {
        let mut positions = IndexMap::new();
        let mut i = 0;

        self.retain(|x| {
            let e = positions.get(x);
            if e.is_some() {
                return false;
            }
            positions.insert(*x, (Some(i), None));
            i += 1;
            true
        });

        let mut first = true;
        let mut first_other = 0;

        for (index, &value) in self.iter().enumerate() {
            positions.entry(value).or_insert((Some(index), None));
        }

        for (index, &value) in other.iter().enumerate() {
            let e = positions.get(&value);
            if let Some((Some(selfi), None)) = e {
                positions.insert(value, (Some(*selfi), Some(index)));
            } else {
                positions.insert(value, (None, Some(index)));
                self.push(value);
            }
        }

        let mut i = 0;

        self.retain(|x| {
            let pos = positions.get(x).unwrap();
            if pos.1.is_none() {
                i += 1;
                return true;
            }
            if pos.0.is_none() {
                if first {
                    first = false;
                    first_other = i;
                }
                return true;
            }
            false
        });

        if first {
            None
        } else {
            Some(first_other)
        }
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        let mut slots_b = other.clone();
        let mut slots_a = self.clone();

        slots_a.remove(positions.0);
        slots_b.remove(positions.1);

        slots_a.append(&mut slots_b);
        slots_a
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct IndexLess {
    pub structure: Vec<Representation>,
}

impl IndexLess {
    pub fn new(structure: Vec<Representation>) -> Self {
        Self { structure }
    }

    pub fn empty() -> Self {
        Self { structure: vec![] }
    }

    pub fn to_indexed(self, indices: &[AbstractIndex]) -> VecStructure {
        indices
            .iter()
            .cloned()
            .zip(self.structure.iter().cloned())
            .map(Slot::from)
            .collect()
    }
}

impl TensorStructure for IndexLess {
    fn external_reps_iter(&self) -> impl Iterator<Item = &Representation> {
        self.structure.iter()
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex> {
        [].iter()
    }

    fn order(&self) -> usize {
        self.structure.len()
    }

    fn get_slot(&self, _: usize) -> Option<Slot> {
        None
    }

    fn find_permutation(&self, other: &Self) -> Option<Vec<ConcreteIndex>> {
        if self.order() != other.order() {
            return None;
        }

        let mut index_map = HashMap::new();
        for (i, item) in other.structure.iter().enumerate() {
            index_map.entry(item).or_insert_with(Vec::new).push(i);
        }

        let mut permutation = Vec::with_capacity(self.order());
        let mut used_indices = HashSet::new();
        for item in self.structure.iter() {
            if let Some(indices) = index_map.get_mut(&item) {
                // Find an index that hasn't been used yet
                if let Some(&index) = indices.iter().find(|&&i| !used_indices.contains(&i)) {
                    permutation.push(index);
                    used_indices.insert(index);
                } else {
                    // No available index for this item
                    return None;
                }
            } else {
                // Item not found in other
                return None;
            }
        }

        Some(permutation)
    }

    fn get_rep(&self, i: usize) -> Option<Representation> {
        self.structure.get(i).cloned()
    }

    fn get_dim(&self, i: usize) -> Option<Dimension> {
        self.structure.get(i).clone().map(|r| r.into())
    }
}

#[cfg(feature = "shadowing")]
impl ToSymbolic for IndexLess {}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
pub struct VecStructure {
    pub structure: Vec<Slot>,
}

#[cfg(feature = "shadowing")]
impl TryFrom<AtomView<'_>> for VecStructure {
    type Error = String;
    fn try_from(value: AtomView) -> Result<Self, Self::Error> {
        match value {
            AtomView::Mul(mul) => return mul.try_into(),
            AtomView::Fun(fun) => return fun.try_into(),
            AtomView::Pow(_) => {
                return Ok(VecStructure::default()); // powers do not have a structure
            }
            _ => return Err(format!("Not a structure: {value}").into()), // could check if it
        }
    }
}

#[cfg(test)]
#[test]
#[cfg(feature = "shadowing")]
fn test_from_atom() {
    let a = Atom::parse("f(aind(lor(4,1)))").unwrap();

    let b = VecStructure::try_from(a.as_atom_view()).unwrap();

    print!("{}", b);
}

#[cfg(feature = "shadowing")]
impl TryFrom<FunView<'_>> for VecStructure {
    type Error = String;
    fn try_from(value: FunView) -> Result<Self, Self::Error> {
        if value.get_symbol() == State::get_symbol(ABSTRACTIND) {
            let mut structure: Vec<Slot> = vec![];

            for arg in value.iter() {
                structure.push(arg.try_into()?);
            }

            Ok(structure.into())
        } else {
            let mut structure: Self = vec![].into();
            for arg in value.iter() {
                structure.extend(arg.try_into()?); // append all the structures found
            }
            Ok(structure)
        }
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<MulView<'_>> for VecStructure {
    type Error = String;
    fn try_from(value: MulView) -> Result<Self, Self::Error> {
        let mut structure: Self = vec![].into();

        for arg in value.iter() {
            structure.extend(arg.try_into()?);
        }
        Ok(structure.into())
    }
}

impl FromIterator<Slot> for VecStructure {
    fn from_iter<T: IntoIterator<Item = Slot>>(iter: T) -> Self {
        Self {
            structure: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for VecStructure {
    type Item = Slot;
    type IntoIter = std::vec::IntoIter<Slot>;
    fn into_iter(self) -> std::vec::IntoIter<Slot> {
        self.structure.into_iter()
    }
}

impl<'a> IntoIterator for &'a VecStructure {
    type Item = &'a Slot;
    type IntoIter = std::slice::Iter<'a, Slot>;
    fn into_iter(self) -> std::slice::Iter<'a, Slot> {
        self.structure.iter()
    }
}

impl<'a> IntoIterator for &'a mut VecStructure {
    type Item = &'a mut Slot;
    type IntoIter = std::slice::IterMut<'a, Slot>;
    fn into_iter(self) -> std::slice::IterMut<'a, Slot> {
        self.structure.iter_mut()
    }
}

impl VecStructure {
    pub fn new(structure: Vec<Slot>) -> Self {
        Self { structure }
    }

    fn extend(&mut self, other: Self) {
        self.structure.extend(other.structure.into_iter())
    }

    pub fn to_named<N, A>(self, name: N, args: Option<A>) -> NamedStructure<N, A> {
        NamedStructure::from_iter(self.into_iter(), name, args)
    }

    pub fn empty() -> Self {
        Self { structure: vec![] }
    }
}

impl From<ContractionCountStructure> for VecStructure {
    fn from(structure: ContractionCountStructure) -> Self {
        structure.structure
    }
}

impl<N, A> From<NamedStructure<N, A>> for VecStructure {
    fn from(structure: NamedStructure<N, A>) -> Self {
        structure.structure
    }
}

impl<N, A> From<SmartShadowStructure<N, A>> for VecStructure {
    fn from(structure: SmartShadowStructure<N, A>) -> Self {
        structure.structure
    }
}

impl<N, A> From<HistoryStructure<N, A>> for VecStructure {
    fn from(structure: HistoryStructure<N, A>) -> Self {
        structure.external.into()
    }
}

impl From<VecStructure> for ContractionCountStructure {
    fn from(structure: VecStructure) -> Self {
        Self {
            structure,
            contractions: 0,
        }
    }
}

impl From<Vec<Slot>> for VecStructure {
    fn from(structure: Vec<Slot>) -> Self {
        Self { structure }
    }
}

impl From<VecStructure> for Vec<Slot> {
    fn from(structure: VecStructure) -> Self {
        structure.structure
    }
}

// const IDPRINTER: Lazy<BlockId<char>> = Lazy::new(|| BlockId::new(Alphabet::alphanumeric(), 1, 1));

impl std::fmt::Display for VecStructure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, item) in self.structure.iter().enumerate() {
            if index != 0 {
                // To avoid a newline at the start
                writeln!(f)?;
            }
            write!(
                f,
                "{:<3} ({})",
                usize::from(item.index),
                // IDPRINTER
                //     .encode_string(usize::from(item.index) as u64)
                //     .unwrap(),
                item.representation
            )?;
        }
        Ok(())
    }
}

impl TensorStructure for VecStructure {
    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = &Representation>;
            fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}
#[cfg(feature = "shadowing")]
impl ToSymbolic for VecStructure {}

impl StructureContract for VecStructure {
    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.structure.merge(&other.structure)
    }

    fn trace_out(&mut self) {
        self.structure.trace_out();
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        Self {
            structure: self.structure.merge_at(&other.structure, positions),
        }
    }

    fn trace(&mut self, i: usize, j: usize) {
        self.structure.trace(i, j);
    }
}

/// A named structure is a structure with a global name, and a list of slots
///
/// It is useful when you want to shadow tensors, to nest tensor network contraction operations.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
pub struct NamedStructure<Name = SmartString<LazyCompact>, Args = usize> {
    pub structure: VecStructure,
    pub global_name: Option<Name>,
    pub additional_args: Option<Args>,
}

impl<Name, Args> NamedStructure<Name, Args> {
    #[must_use]
    pub fn from_iter<I, T>(iter: T, name: Name, args: Option<Args>) -> Self
    where
        I: Into<Slot>,
        T: IntoIterator<Item = I>,
    {
        Self {
            structure: iter.into_iter().map(I::into).collect(),
            global_name: Some(name),
            additional_args: args,
        }
    }
}

#[cfg(feature = "shadowing")]
impl<'a> TryFrom<FunView<'a>> for NamedStructure<Symbol, Vec<Atom>> {
    type Error = String;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == State::get_symbol(ABSTRACTIND) => {
                let mut structure: Vec<Slot> = vec![];

                for arg in value.iter() {
                    structure.push(arg.try_into()?);
                }

                Ok(VecStructure::from(structure).into())
            }
            name => {
                let mut structure: NamedStructure<Symbol, Vec<Atom>> =
                    VecStructure::default().into();
                structure.set_name(name);
                let mut args = vec![];

                for arg in value.iter() {
                    if let AtomView::Fun(fun) = arg {
                        structure.structure.extend(fun.try_into()?);
                    } else {
                        args.push(arg.to_owned());
                    }
                }

                structure.additional_args = Some(args);

                Ok(structure)
            }
        }
    }
}

impl<N, A> NamedStructure<N, Vec<A>> {
    pub fn extend(&mut self, other: Self) {
        let result = match (self.additional_args.take(), other.additional_args) {
            (Some(mut v1), Some(v2)) => {
                v1.extend(v2);
                Some(v1)
            }
            (None, Some(v2)) => Some(v2),
            (Some(v1), None) => Some(v1),
            (None, None) => None,
        };
        self.additional_args = result;
        self.structure.extend(other.structure);
    }
}

impl<N, A> From<VecStructure> for NamedStructure<N, A> {
    fn from(value: VecStructure) -> Self {
        Self {
            structure: value,
            global_name: None,
            additional_args: None,
        }
    }
}

/// A trait for a structure that has a name

impl<N, A> HasName for NamedStructure<N, A>
where
    N: Clone,
    A: Clone,
{
    type Name = N;
    type Args = A;

    fn name(&self) -> Option<Self::Name> {
        self.global_name.clone()
    }
    fn set_name(&mut self, name: Self::Name) {
        self.global_name = Some(name);
    }
    fn id(&self) -> Option<Self::Args> {
        self.additional_args.clone()
    }
}

pub trait HasName {
    type Name: Clone;
    type Args: Clone;
    fn name(&self) -> Option<Self::Name>;
    fn id(&self) -> Option<Self::Args>;
    fn set_name(&mut self, name: Self::Name);
}

impl<N, A> TensorStructure for NamedStructure<N, A> {
    delegate! {
        to self.structure {
            fn external_reps_iter(&self) -> impl Iterator<Item = &Representation>;
            fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}

#[cfg(feature = "shadowing")]
impl<N: IntoSymbol, A: IntoArgs> ToSymbolic for NamedStructure<N, A> {
    fn concrete_atom(&self, id: FlatIndex) -> Atom {
        let exp_atom = self.structure.concrete_atom(id);
        if let Some(ref f) = self.global_name {
            let mut fun = FunctionBuilder::new(f.into_symbol());
            if let Some(ref args) = self.additional_args {
                for arg in args.into_args() {
                    fun = fun.add_arg(arg.as_atom_view());
                }
            }
            fun.add_arg(exp_atom.as_atom_view()).finish()
        } else {
            exp_atom
        }
    }
}

impl<N, A> StructureContract for NamedStructure<N, A> {
    delegate! {
        to self.structure{
            fn trace_out(&mut self);
            fn trace(&mut self, i: usize, j: usize);
        }
    }

    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.structure.merge(&other.structure)
    }

    /// when merging two named structures, the global name is lost
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        Self {
            structure: self.structure.merge_at(&other.structure, positions),
            global_name: None,
            additional_args: None,
        }
    }
}

/// A contraction count structure
///
/// Useful for tensor network contraction algorithm.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ContractionCountStructure {
    pub structure: VecStructure,
    pub contractions: usize,
}

impl<I: Into<Slot>> FromIterator<I> for ContractionCountStructure {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        Self {
            structure: iter.into_iter().map(I::into).collect(),
            contractions: 0,
        }
    }
}

pub trait TracksCount {
    fn contractions_num(&self) -> usize;

    fn is_composite(&self) -> bool {
        self.contractions_num() > 0
    }
}

impl TracksCount for ContractionCountStructure {
    fn contractions_num(&self) -> usize {
        self.contractions
    }
}

impl TensorStructure for ContractionCountStructure {
    delegate! {
        to self.structure {
            fn external_reps_iter(&self) -> impl Iterator<Item = &Representation>;
            fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}

#[cfg(feature = "shadowing")]
impl ToSymbolic for ContractionCountStructure {}

impl StructureContract for ContractionCountStructure {
    delegate! {
        to self.structure{
            fn trace_out(&mut self);
            fn trace(&mut self, i: usize, j: usize);
        }
    }
    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.contractions += other.contractions + 1;
        self.structure.merge(&other.structure)
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        Self {
            structure: self.structure.merge_at(&other.structure, positions),
            contractions: self.contractions + other.contractions + 1,
        }
    }
}

/// A structure to enable smart shadowing of tensors in a tensor network contraction algorithm.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct SmartShadowStructure<Name = SmartString<LazyCompact>, Args = usize> {
    pub structure: VecStructure,
    pub contractions: usize,
    pub global_name: Option<Name>,
    additional_args: Option<Args>,
}

impl<Name, Args> SmartShadowStructure<Name, Args> {
    /// Constructs a new [`SmartShadow`] from a list of tuples of indices and dimension (assumes they are all euclidean), along with a name
    #[must_use]
    pub fn from_iter<I, T>(iter: T, name: Option<Name>, args: Option<Args>) -> Self
    where
        I: Into<Slot>,
        T: IntoIterator<Item = I>,
    {
        Self {
            structure: iter.into_iter().map(I::into).collect(),
            global_name: name,
            additional_args: args,
            contractions: 0,
        }
    }
}

impl<N, A> HasName for SmartShadowStructure<N, A>
where
    N: Clone,
    A: Clone,
{
    type Name = N;
    type Args = A;

    fn name(&self) -> Option<Self::Name> {
        self.global_name.clone()
    }
    fn set_name(&mut self, name: Self::Name) {
        self.global_name = Some(name);
    }
    fn id(&self) -> Option<Self::Args> {
        self.additional_args.clone()
    }
}

impl<N, A> TensorStructure for SmartShadowStructure<N, A> {
    delegate! {
        to self.structure {
            fn external_reps_iter(&self) -> impl Iterator<Item = &Representation>;
            fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
        }
    }
}

#[cfg(feature = "shadowing")]
impl<N: IntoSymbol, A: IntoArgs> ToSymbolic for SmartShadowStructure<N, A> {
    fn concrete_atom(&self, id: FlatIndex) -> Atom {
        let exp_atom = self.structure.concrete_atom(id);
        if let Some(ref f) = self.global_name {
            let mut fun = FunctionBuilder::new(f.into_symbol());
            if let Some(ref args) = self.additional_args {
                for arg in args.into_args() {
                    fun = fun.add_arg(arg.as_atom_view());
                }
            }
            fun.add_arg(exp_atom.as_atom_view()).finish()
        } else {
            exp_atom
        }
    }
}
impl<N, A> TracksCount for SmartShadowStructure<N, A> {
    fn contractions_num(&self) -> usize {
        self.contractions
    }
}

impl<N, A> StructureContract for SmartShadowStructure<N, A> {
    fn merge(&mut self, other: &Self) -> Option<usize> {
        self.contractions += other.contractions;
        self.structure.merge(&other.structure)
    }

    delegate! {
        to self.structure{
            fn trace_out(&mut self);
            fn trace(&mut self, i: usize, j: usize);
        }
    }

    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        SmartShadowStructure {
            structure: self.structure.merge_at(&other.structure, positions),
            contractions: self.contractions + other.contractions,
            global_name: None,
            additional_args: None,
        }
    }
}

/// A tracking structure
///
/// It contains two vecs of [`Slot`]s, one for the internal structure, simply extended during each contraction, and one external, coresponding to all the free indices
///
/// It enables keeping track of the contraction history of the tensor, mostly for debugging and display purposes.
/// A [`SymbolicTensor`] can also be used in this way, however it needs a symbolica state and workspace during contraction.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct HistoryStructure<Name, Args = ()> {
    internal: VecStructure,
    pub names: AHashMap<Range<usize>, Name>, //ideally this is a named partion.. maybe a btreemap<usize, N>, and the range is from previous to next
    external: NamedStructure<Name, Args>,
}

impl<N, A> From<NamedStructure<N, A>> for HistoryStructure<N, A>
where
    N: Clone,
{
    fn from(external: NamedStructure<N, A>) -> Self {
        Self {
            internal: external.structure.clone(),
            names: AHashMap::from([(0..external.order(), external.global_name.clone().unwrap())]),
            external,
        }
    }
}

impl<N, A> HistoryStructure<N, A> {
    /// make the indices in the internal index list of self independent from the indices in the internal index list of other
    /// This is done by shifting the indices in the internal index list of self by the the maximum index present.
    pub fn independentize_internal(&mut self, other: &Self) {
        let internal_set: HashSet<Slot> = self
            .internal
            .clone()
            .into_iter()
            .filter(|s| self.external.contains(s))
            .collect();

        let other_set: HashSet<Slot> = other.internal.clone().into_iter().collect();

        let mut replacement_value = internal_set
            .union(&other_set)
            .map(|s| s.index)
            .max()
            .unwrap_or(0.into())
            + 1.into();

        for item in &mut self.internal {
            if other_set.contains(item) {
                item.index = replacement_value;
                replacement_value += 1.into();
            }
        }
    }
}

impl<N, A> HasName for HistoryStructure<N, A>
where
    N: Clone,
    A: Clone,
{
    type Name = N;
    type Args = A;
    delegate! {
        to self.external {
            fn name(&self) -> Option<Self::Name>;
            fn set_name(&mut self, name: Self::Name);
            fn id(&self) -> Option<Self::Args>;
        }
    }
}

impl<N, A> TracksCount for HistoryStructure<N, A> {
    /// Since each time we contract, we merge the name maps, the amount of contractions, is the size of the name map
    /// This function returns the number of contractions thus computed
    fn contractions_num(&self) -> usize {
        self.names.len()
    }
}

impl<N, A> TensorStructure for HistoryStructure<N, A> {
    delegate! {
        to self.external {
            fn external_reps_iter(&self) -> impl Iterator<Item = &Representation>;
            fn external_indices_iter(&self) -> impl Iterator<Item = &AbstractIndex>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;

        }
    }
    /// checks if internally, the two tensors are the same. This implies that the external indices are the same
    fn same_content(&self, other: &Self) -> bool {
        let set1: HashSet<_> = (&self.internal).into_iter().collect();
        let set2: HashSet<_> = (&other.internal).into_iter().collect();
        set1 == set2
        // TODO: check names
    }
}

#[cfg(feature = "shadowing")]
impl<N: IntoSymbol, A: IntoArgs> ToSymbolic for HistoryStructure<N, A> {
    delegate! {
        to self.external{

        }
    }
}

// impl TensorStructure for [Slot] {
//     type Structure = [Slot];

//     fn external_structure(&self) -> &[Slot] {
//         self
//     }
// }

impl<N, A> StructureContract for HistoryStructure<N, A>
where
    N: Clone,
    A: Clone,
{
    /// remove the repeated indices in the external index list
    fn trace_out(&mut self) {
        let mut positions = IndexMap::new();

        // Track the positions of each element
        for (index, value) in (&self.external).external_structure_iter().enumerate() {
            positions.entry(value).or_insert_with(Vec::new).push(index);
        }
        // Collect only the positions of non- repeated elements

        self.external.structure = positions
            .into_iter()
            .filter_map(|(value, indices)| {
                if indices.len() == 1 {
                    Some(value)
                } else {
                    None
                }
            })
            .collect();
    }

    /// remove the given indices from the external index list
    fn trace(&mut self, i: usize, j: usize) {
        if i < j {
            self.trace(j, i);
            return;
        }
        let a = self.external.structure.structure.remove(i);
        let b = self.external.structure.structure.remove(j);
        assert_eq!(a, b);
    }

    /// essentially contract.
    fn merge(&mut self, other: &Self) -> Option<usize> {
        let shift = self.internal.order();
        for (range, name) in &other.names {
            self.names
                .insert((range.start + shift)..(range.end + shift), name.clone());
        }
        self.trace_out();
        self.independentize_internal(other);
        self.internal
            .structure
            .append(&mut other.internal.structure.clone());
        self.external.merge(&other.external)
    }

    /// Merge two [`HistoryStructure`] at the given positions of the external index list. Ideally the internal index list should be independentized before merging
    /// This is essentially a contraction of only one index. The name maps are merged, and shifted accordingly. The global name is lost, since the resulting tensor is composite
    /// The global name can be set again with the [`Self::set_global_name`] function
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        let external = self.external.merge_at(&other.external, positions);

        let mut slots_self_int = self.internal.clone();
        let slots_other_int = other.internal.clone();
        slots_self_int.extend(slots_other_int);

        let mut names = self.names.clone();
        let shift = self.internal.order();
        for (range, name) in &other.names {
            names.insert((range.start + shift)..(range.end + shift), name.clone());
        }
        HistoryStructure {
            internal: slots_self_int,
            external,
            names,
        }
    }
}
#[cfg(feature = "shadowing")]
pub fn atomic_expanded_label<I: IntoSymbol>(indices: &[ConcreteIndex], name: I) -> Atom {
    let id = name.into_symbol();
    atomic_expanded_label_id(indices, id, &[])
}
#[cfg(feature = "shadowing")]
pub fn atomic_flat_label<I: IntoSymbol>(index: usize, name: I) -> Atom {
    let id = name.into_symbol();
    atomic_flat_label_id(index, id)
}

#[allow(clippy::cast_possible_wrap)]
#[cfg(feature = "shadowing")]
pub fn atomic_flat_label_id(index: usize, id: Symbol) -> Atom {
    let mut value_builder = FunctionBuilder::new(id);
    value_builder = value_builder.add_arg(Atom::new_num(index as i64).as_atom_view());
    value_builder.finish()
}
#[cfg(feature = "shadowing")]
#[allow(clippy::cast_possible_wrap)]
pub fn atomic_expanded_label_id(indices: &[ConcreteIndex], name: Symbol, args: &[Atom]) -> Atom {
    let mut value_builder = FunctionBuilder::new(name);
    let mut index_func = FunctionBuilder::new(State::get_symbol("cind"));
    for arg in args {
        value_builder = value_builder.add_arg(arg);
    }
    for &index in indices {
        index_func = index_func.add_arg(Atom::new_num(index as i64).as_atom_view());
    }

    let indices = index_func.finish();
    value_builder.add_arg(&indices).finish()
}

#[cfg(feature = "shadowing")]
pub trait IntoSymbol {
    fn into_symbol(&self) -> Symbol;
}

#[cfg(feature = "shadowing")]
pub trait IntoArgs {
    fn into_args<'a>(&self) -> impl Iterator<Item = Atom>;
    fn args(&self) -> Vec<Atom> {
        self.into_args().collect()
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for usize {
    fn into_args<'a>(&self) -> impl Iterator<Item = Atom> {
        std::iter::once(Atom::new_num(*self as i64))
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for () {
    fn into_args<'a>(&self) -> impl Iterator<Item = Atom> {
        std::iter::empty()
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for Atom {
    fn into_args<'a>(&self) -> impl Iterator<Item = Atom> {
        std::iter::once(self.clone())
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for Vec<Atom> {
    fn into_args<'a>(&self) -> impl Iterator<Item = Atom> {
        self.iter().cloned()
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for SmartString<LazyCompact> {
    fn into_symbol(&self) -> Symbol {
        State::get_symbol(self)
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for Symbol {
    fn into_symbol(&self) -> Symbol {
        self.clone()
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for &str {
    fn into_symbol(&self) -> Symbol {
        State::get_symbol(self)
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for std::string::String {
    fn into_symbol(&self) -> Symbol {
        State::get_symbol(self)
    }
}

/// Trait that enables shadowing of a tensor
///
/// This creates a dense tensor of atoms, where the atoms are the expanded indices of the tensor, with the global name as the name of the labels.
#[cfg(feature = "shadowing")]
pub trait Shadowable: HasStructure + TensorStructure + HasName {
    fn shadow(&self) -> Option<DenseTensor<Atom, Self::Structure>>
    where
        Self: std::marker::Sized,
        Self::Name: IntoSymbol + Clone,
        Self::Args: IntoArgs,
        Self::Structure: Clone + TensorStructure + ToSymbolic,
    {
        let name = self.name()?;
        let args = self.id().map(|s| s.args()).unwrap_or(vec![]);

        Some(
            self.structure()
                .clone()
                .shadow_with(name.into_symbol(), &args),
        )
    }

    fn smart_shadow(&self) -> Option<MixedTensor<f64, Self::Structure>>
    where
        Self: std::marker::Sized,
        Self::Args: IntoArgs,
        Self::Structure: Clone + TensorStructure + ToSymbolic,
        Self::Name: IntoSymbol + Clone,
    {
        let name = self.name()?;
        let args = self.id()?.args();
        Some(
            self.structure()
                .clone()
                .to_explicit_rep(name.into_symbol(), &args),
        )
    }

    fn to_symbolic(&self) -> Option<Atom>
    where
        Self::Name: IntoSymbol + Clone,
        Self::Args: IntoArgs,
    {
        let args = self.id().map(|s| s.args()).unwrap_or(vec![]);

        Some(self.to_symbolic_with(self.name()?.into_symbol(), &args))
    }

    fn to_symbolic_with(&self, name: Symbol, args: &[Atom]) -> Atom {
        let slots = self
            .external_structure_iter()
            .map(|slot| slot.to_symbolic())
            .collect::<Vec<_>>();

        let mut value_builder = FunctionBuilder::new(name.into_symbol());

        let mut index_func = FunctionBuilder::new(State::get_symbol("aind"));
        for arg in args {
            value_builder = value_builder.add_arg(arg);
        }

        for s in slots {
            index_func = index_func.add_arg(&s);
        }
        let indices = index_func.finish();
        value_builder.add_arg(&indices).finish()
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct TensorShell<S: TensorStructure> {
    structure: S,
}

impl<S: TensorStructure> HasStructure for TensorShell<S> {
    type Structure = S;
    type Scalar = ();
    fn structure(&self) -> &S {
        &self.structure
    }
    fn mut_structure(&mut self) -> &mut S {
        &mut self.structure
    }
}

impl<S: TensorStructure> HasName for TensorShell<S>
where
    S: HasName,
{
    type Args = S::Args;
    type Name = S::Name;

    fn id(&self) -> Option<Self::Args> {
        self.structure.id()
    }

    fn name(&self) -> Option<Self::Name> {
        self.structure.name()
    }

    fn set_name(&mut self, name: Self::Name) {
        self.structure.set_name(name);
    }
}

// impl<I> HasName for I
// where
//     I: HasStructure,
//     I::Structure: HasName,
// {
//     type Name = <I::Structure as HasName>::Name;
//     fn name(&self) -> Option<Cow<Self::Name>> {
//         self.structure().name()
//     }
//     fn set_name(&mut self, name: &Self::Name) {
//         self.mut_structure().set_name(name);
//     }
// }

impl<S: TensorStructure> TensorShell<S> {
    pub fn new(structure: S) -> Self {
        Self { structure }
    }
}

impl<S: TensorStructure> From<S> for TensorShell<S> {
    fn from(structure: S) -> Self {
        Self::new(structure)
    }
}

#[cfg(feature = "shadowing")]
impl<N> Shadowable for N
where
    N: HasStructure + HasName + TensorStructure,
    N::Name: IntoSymbol + Clone,
    N::Structure: Clone + TensorStructure,
{
}

#[cfg(feature = "shadowing")]
impl<N, A> std::fmt::Display for HistoryStructure<N, A>
where
    N: Display + Clone + IntoSymbol,
    A: Display + Clone + IntoArgs,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut string = String::new();
        if let Some(global_name) = self.name() {
            string.push_str(&format!("{global_name}:"));
        }
        for (range, name) in self
            .names
            .iter()
            .filter(|(r, _)| *r != &(0..self.internal.order()) || !self.is_composite())
        {
            string.push_str(&format!("{name}("));
            for slot in &self.internal.structure[range.clone()] {
                string.push_str(&format!("{slot},"));
            }
            string.pop();
            string.push(')');
        }
        write!(f, "{string}")
    }
}

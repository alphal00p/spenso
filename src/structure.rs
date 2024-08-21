use ahash::AHashMap;
use anyhow::{anyhow, Result};
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
use duplicate::duplicate;
use indexmap::IndexMap;
use num::One;
use num::Zero;
use serde::Deserialize;
use serde::Serialize;
use smartstring::LazyCompact;
use smartstring::SmartString;
// use std::fmt::write;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::AddAssign;
use std::ops::Deref;
use std::ops::Neg;
use thiserror::Error;

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{
        representation::FunView, AsAtomView, Atom, AtomView, FunctionBuilder, ListIterator,
        MulView, Symbol,
    },
    coefficient::CoefficientView,
    evaluate::FunctionMap,
    state::State,
};

use crate::data::SparseTensor;
use crate::parametric::SerializableAtom;
use crate::permutation::Permutation;
#[cfg(feature = "shadowing")]
use crate::{
    data::DenseTensor,
    parametric::{ExpandedCoefficent, FlatCoefficent, MixedTensor, ParamTensor, TensorCoefficient},
    ufo,
};
use std::ops::Range;

use std::collections::HashMap;
use std::collections::HashSet;

use crate::data::SetTensorData;
use crate::iterators::TensorStructureIndexIterator;

// use smartstring::alias::String;
/// A type that represents the name of an index in a tensor.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, Serialize, Deserialize)]
pub enum AbstractIndex {
    Normal(usize),
    Dualize(usize),
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
            },
            AbstractIndex::Dualize(l) => match rhs {
                AbstractIndex::Normal(r) => AbstractIndex::Normal(l + r),
                AbstractIndex::Dualize(r) => AbstractIndex::Normal(l + r),
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
        write!(f, "{}", usize::from(*self))
    }
}

impl From<usize> for AbstractIndex {
    fn from(value: usize) -> Self {
        AbstractIndex::Normal(value)
    }
}

impl From<AbstractIndex> for usize {
    fn from(value: AbstractIndex) -> Self {
        match value {
            AbstractIndex::Dualize(v) => v,
            AbstractIndex::Normal(v) => v,
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

#[cfg(feature = "shadowing")]
impl TryFrom<AtomView<'_>> for AbstractIndex {
    type Error = String;

    fn try_from(view: AtomView<'_>) -> Result<Self, Self::Error> {
        if let AtomView::Var(v) = view {
            Ok(AbstractIndex::Normal(v.get_symbol().get_id() as usize))
        } else {
            Err("Not a var".to_string())
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

#[cfg(feature = "shadowing")]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Display)]
pub struct SerializableSymbol {
    symbol: Symbol,
}

#[cfg(feature = "shadowing")]
impl Serialize for SerializableSymbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.symbol.to_string().serialize(serializer)
    }
}

#[cfg(feature = "shadowing")]
impl<'d> Deserialize<'d> for SerializableSymbol {
    fn deserialize<D>(deserializer: D) -> Result<SerializableSymbol, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(SerializableSymbol {
            symbol: State::get_symbol(value),
        })
    }
}

#[cfg(feature = "shadowing")]
impl From<Symbol> for SerializableSymbol {
    fn from(value: Symbol) -> Self {
        Self { symbol: value }
    }
}

#[cfg(feature = "shadowing")]
impl From<SerializableSymbol> for Symbol {
    fn from(value: SerializableSymbol) -> Self {
        value.symbol
    }
}

/// A Dimension
#[derive(
    Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Display, Serialize, Deserialize,
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

// impl PartialOrd<usize> for Dimension {
//     fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
//         self.0.partial_cmp(other)
//     }
// }

// impl PartialOrd<Dimension> for usize {
//     fn partial_cmp(&self, other: &Dimension) -> Option<Ordering> {
//         self.partial_cmp(&other.0)
//     }
// }

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

#[cfg(feature = "shadowing")]
impl From<ExpandedIndex> for Atom {
    fn from(value: ExpandedIndex) -> Self {
        let mut cind = FunctionBuilder::new(State::get_symbol(CONCRETEIND));
        for i in value.iter() {
            cind = cind.add_arg(Atom::new_num(*i as i64).as_atom_view());
        }
        cind.finish()
    }
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

#[cfg(feature = "shadowing")]
impl From<FlatIndex> for Atom {
    fn from(value: FlatIndex) -> Self {
        let mut cind = FunctionBuilder::new(State::get_symbol(FLATIND));
        cind = cind.add_arg(Atom::new_num(value.index as i64).as_atom_view());
        cind.finish()
    }
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
pub const FLATIND: &str = "find";
pub const ABSTRACTIND: &str = "aind";

// /// A Representation/Dimension of the index.
// #[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
// pub enum Representation {
//     /// Represents a Euclidean space of the given dimension, with metric diag(1,1,1,1,...)
//     Euclidean(Dimension),
//     /// Represents a Minkowski space of the given dimension, with metric diag(1,-1,-1,-1,...)
//     Lorentz(Dimension),
//     Bispinor(Dimension),
//     /// Represents a Spinor Fundamental space of the given dimension
//     SpinFundamental(Dimension),
//     /// Represents a Spinor Adjoint space of the given dimension
//     SpinAntiFundamental(Dimension),
//     /// Represents a Color Fundamental space of the given dimension
//     ColorFundamental(Dimension),
//     /// Represents a Color Anti-Fundamental space of the given dimension
//     ColorAntiFundamental(Dimension),
//     /// Represents a Color Adjoint space of the given dimension
//     ColorAdjoint(Dimension),
//     /// Represents a Color Sextet space of the given dimension
//     ColorSextet(Dimension),
//     /// Represents a Color Anti-Sextet space of the given dimension
//     ColorAntiSextet(Dimension),
// }

pub trait BaseRepName: RepName<Dual: RepName> + Default {
    const NAME: &'static str;
    // fn selfless_name() -> String;
    fn selfless_base() -> Self::Base;
    fn selfless_pair() -> DualPair<Self::Base>
    where
        Self::Base: BaseRepName<Dual: BaseRepName>;

    #[cfg(feature = "shadowing")]
    fn selfless_symbol() -> Symbol {
        State::get_symbol(Self::NAME)
    }
    fn selfless_dual() -> Self::Dual;
    fn new_dimed_rep_selfless<D: Into<Dimension>>(dim: D) -> Representation<Self>
    where
        Self: Sized,
    {
        Representation {
            dim: dim.into(),
            rep: Self::default(),
        }
    }

    fn new_slot_selfless<D: Into<Dimension>, A: Into<AbstractIndex>>(dim: D, aind: A) -> Slot<Self>
    where
        Self: Sized,
    {
        let aind: AbstractIndex = aind.into();
        Slot {
            rep: Self::new_dimed_rep_selfless(dim),
            aind,
        }
    }

    fn base_metric<S, V: One + Neg<Output = V>>(&self, structure: S) -> SparseTensor<V, S>
    where
        S: TensorStructure;

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

// impl<B: BaseRepName<Dual = Dual<B>>> BaseRepName for Dual<B> {
//     type Dual = B;
//     type Base = B;
//     fn selfless_name() -> String {
//         format!("dual{}", B::selfless_name())
//     }

//     fn selfless_dual() -> Self::Dual {
//         B::default()
//     }

//     fn selfless_base() -> Self::Base {
//         B::default()
//     }

//     fn selfless_pair() -> DualPair<Self::Base> {
//         DualPair::DualRep(Dual {
//             inner: B::default(),
//         })
//     }
// }

// impl<B: BaseRepName> Display for B {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", B::selfless_name())
//     }
// }

pub trait RepName: Copy + Clone + Debug + PartialEq + Eq + Hash + Display {
    type Dual: RepName<Dual = Self, Base = Self::Base>;
    type Base: RepName;
    fn dual(self) -> Self::Dual;
    fn base(&self) -> Self::Base;
    fn matches(&self, other: &Self::Dual) -> bool;
    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol) -> Result<Self>;
    fn metric<S, V: One + Zero + Neg<Output = V>>(
        &self,
        size: usize,
        aind: [AbstractIndex; 2],
    ) -> SparseTensor<V, S>
    where
        S: TensorStructure + FromIterator<Slot<Self>> + FromIterator<Slot<Self::Dual>>,
    {
        let structure: S = [self.new_slot(size, aind[0]), self.new_slot(size, aind[1])]
            .into_iter()
            .collect();
        self.metric_data(structure)
    }

    fn metric_data<V: One + Neg<Output = V>, S: TensorStructure>(
        &self,
        structure: S,
    ) -> SparseTensor<V, S>;

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
    fn to_fnbuilder(&self) -> FunctionBuilder {
        FunctionBuilder::new(State::get_symbol(self.to_string()))
    }

    #[cfg(feature = "shadowing")]
    fn to_symbol(&self) -> Symbol {
        State::get_symbol(self.to_string())
    }

    fn new_slot<D: Into<Dimension>, A: Into<AbstractIndex>>(self, dim: D, aind: A) -> Slot<Self>
    where
        Self: Sized,
    {
        Slot {
            rep: self.new_dimed_rep(dim),
            aind: aind.into(),
        }
    }

    fn new_dimed_rep<D: Into<Dimension>>(&self, dim: D) -> Representation<Self>
    where
        Self: Sized,
    {
        Representation {
            dim: dim.into(),
            rep: *self,
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct Dual<T> {
    inner: T,
}

// impl<T: RepName<Dual: RepName>> Display for Dual<T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.inner.dual())
//     }
// }

// impl<T: BaseRepName<Dual = Dual<T>, Base = T>> BaseRepName for Dual<T> {
//     fn selfless_name() -> String {
//         format!("dual{}", T::selfless_name())
//     }

//     fn selfless_base() -> Self::Base {
//         T::selfless_base()
//     }

//     fn selfless_pair() -> DualPair<Self::Base> {
//         DualPair::DualRep(Dual {
//             inner: T::selfless_base(),
//         })
//     }

//     fn selfless_dual() -> Self::Dual {
//         T::selfless_base()
//     }
// }

// impl<T: BaseRepName<Dual = Dual<T>, Base = T>> RepName for Dual<T> {
//     type Dual = T;
//     type Base = T::Base;
//     fn dual(self) -> Self::Dual {
//         self.inner
//     }

//     fn metric<S,V:One+Zero+Neg<Output=V>>
//     (&self,size:usize,aind:[AbstractIndex;2])-> SparseTensor<V,S> where S:TensorStructure+FromIterator<Slot<Self>>+FromIterator<Slot<Self::Dual>> {
//         self.inner.metric(size,aind)
//     }

//     fn base(&self) -> Self::Base {
//         self.inner.base()
//     }

//     fn try_from_symbol(sym: Symbol) -> Result<Self> {
//         if Self::selfless_symbol() == sym {
//             Ok(Dual {
//                 inner: T::default(),
//             })
//         } else {
//             Err(anyhow!("Not a dual representation"))
//         }
//     }

//     fn is_neg(self, i: usize) -> bool {
//         self.is_neg(i)
//     }
// }

duplicate! {
   [isnotselfdual isneg constname varname varnamedual dualconstname;
    [Lorentz] [_i > 0] ["loru"] [LorentzUp] [LorentzDown] ["lord"];
    [SpinFundamental] [false] ["spin"] [SpinFund] [SpinAntiFund] ["spina"];
    [ColorFundamental] [false] ["cof"] [ColorFund] [ColorAntiFund] ["coaf"];
    [ColorSextet] [false] ["cos"] [ColorSextet] [ColorAntiSextet]["coas"]]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,Default)]

    pub struct isnotselfdual {}
    #[allow(unused_variables)]
    impl BaseRepName for isnotselfdual {
        const NAME: &'static str = constname;


        // fn selfless_name() -> String {
        //     constname.to_string()
        // }

        fn selfless_base() -> Self::Base {
            Self::default()
        }

        fn selfless_pair() -> DualPair<Self::Base> {
            DualPair::Rep(Self::default())
        }

        fn selfless_dual() -> Self::Dual {
            Dual{inner:isnotselfdual::default()}
        }



        fn base_metric<S,V:One+Neg<Output=V>>
        (&self,structure:S)-> SparseTensor<V,S> where S:TensorStructure {
        let size = usize::try_from(structure.get_dim(0).unwrap()).unwrap();
        let mut tensor= SparseTensor::empty(structure);
            for i in 0..size{
                if self.is_neg(i){

                    tensor.set(&[i,i],V::one().neg()).unwrap();
                } else {
                    tensor.set(&[i,i],V::one()).unwrap();
                }
            }
        tensor
        }


    }

    impl RepName for isnotselfdual {
    type Base = isnotselfdual;
    type Dual = Dual<isnotselfdual>;

    fn base(&self) -> Self::Base {
        isnotselfdual::selfless_base()
    }

    fn matches(&self, _: &Self::Dual) -> bool {
        true
    }

    fn metric_data<V: One + Neg<Output = V>, S: TensorStructure>(
            &self,
            structure: S,
        ) -> SparseTensor<V, S> {
        self.base_metric(structure)
    }

    fn dual(self) -> Self::Dual {
        isnotselfdual::selfless_dual()
    }

    fn is_neg(self,_i:usize)->bool{
        isneg
    }
    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol) -> Result<Self> {
        if Self::selfless_symbol() == sym {
            Ok(isnotselfdual::default())
        } else {
            Err(anyhow!("Not a representation"))
        }
    }



}


    impl BaseRepName for Dual<isnotselfdual>{

        const NAME: &'static str = dualconstname;

        // fn selfless_name() -> String {
        //     format!("dual{}",constname)
        // }
        fn selfless_base() -> Self::Base {
            isnotselfdual::default()
        }
        fn selfless_dual()->Self::Dual{
            isnotselfdual::default()
        }
        fn selfless_pair() -> DualPair<Self::Base> {
            DualPair::DualRep(Dual{inner:isnotselfdual::default()})
        }

        fn base_metric<S,V:One+Neg<Output=V>>
        (&self,structure:S)-> SparseTensor<V,S> where S:TensorStructure {
 let size = usize::try_from(structure.get_dim(0).unwrap()).unwrap();
        let mut tensor= SparseTensor::empty(structure);
            for i in 0..size{
                if self.is_neg(i){

                    tensor.set(&[i,i],V::one().neg()).unwrap();
                } else {
                    tensor.set(&[i,i],V::one()).unwrap();
                }
            }
        tensor
        }
    }

    impl RepName for Dual<isnotselfdual>{
        type Base = isnotselfdual;
        type Dual = isnotselfdual;

        fn dual(self) -> Self::Dual {
            self.inner
        }

       fn metric_data<V: One + Neg<Output = V>, S: TensorStructure>(
               &self,
               structure: S,
           ) -> SparseTensor<V, S> {
           self.base_metric(structure)
       }

        fn base(&self) -> Self::Base {
            self.inner
        }

        fn matches(&self, _: &Self::Dual) -> bool {
        true
    }
        #[cfg(feature = "shadowing")]
        fn try_from_symbol(sym:Symbol)->Result<Self>{
            if Self::selfless_symbol() == sym {
            Ok(Dual {
                inner: isnotselfdual::default(),
            })
        } else {
            Err(anyhow!("Not a dual representation"))
        }
        }

        fn is_neg(self, i: usize) -> bool {
            self.dual().is_neg(i)
        }




    }

    impl Display for isnotselfdual{
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", constname)
        }
    }

    impl Display for Dual<isnotselfdual>{
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", dualconstname)
        }
    }




    impl From<isnotselfdual> for PhysReps {
        fn from(value: isnotselfdual) -> Self {
            PhysReps::varname(value)
        }
    }

    impl From<Dual<isnotselfdual>> for PhysReps{
        fn from(value: Dual<isnotselfdual>) -> Self {
            PhysReps::varnamedual(value)
        }
    }

     impl From<Slot<isnotselfdual>> for PhysicalSlots {
    fn from(value: Slot<isnotselfdual>) -> Self {
        value.cast() //RecSlotEnum::A(value.dual_pair()).into()
    }}

    impl From<Slot<Dual<isnotselfdual>>> for PhysicalSlots {
    fn from(value: Slot<Dual<isnotselfdual>>) -> Self {
        value.cast() //RecSlotEnum::A(value.dual_pair()).into()
    }
    }
}

duplicate! {
   [isselfdual constname;
    [Euclidean] ["euc"];
    [Bispinor] ["bis"];
    [ColorAdjoint] ["coad"]]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,Default)]
    pub struct isselfdual {}

    impl BaseRepName for isselfdual {
        const NAME: &'static str = constname;

        // fn selfless_name() -> String {
        //     constname.to_string()
        // }

        fn selfless_base() -> Self::Base {
            Self::default()
        }

        fn selfless_pair() -> DualPair<Self::Base> {
            DualPair::Rep(Self::default())
        }

        fn selfless_dual() -> Self::Dual {
            Self::default()
        }

fn base_metric<S,V:One+Neg<Output=V>>
        (&self,structure:S)-> SparseTensor<V,S> where S:TensorStructure {
 let size = usize::try_from(structure.get_dim(0).unwrap()).unwrap();
        let mut tensor= SparseTensor::empty(structure);
            for i in 0..size{
                if self.is_neg(i){

                    tensor.set(&[i,i],V::one().neg()).unwrap();
                } else {
                    tensor.set(&[i,i],V::one()).unwrap();
                }
            }
        tensor
        }

    }

    impl RepName for isselfdual {
    type Base = isselfdual;
    type Dual = isselfdual;

    fn base(&self) -> Self::Base {
        isselfdual::selfless_base()
    }

    fn dual(self) -> Self::Dual {
        isselfdual::selfless_dual()
    }

    fn matches(&self, _: &Self::Dual) -> bool {
        true
    }

    #[cfg(feature = "shadowing")]
  fn try_from_symbol(sym: Symbol) -> Result<Self> {
        if Self::selfless_symbol() == sym {
            Ok(isselfdual::default())
        } else {
            Err(anyhow!("Not a representation"))
        }
    }

    fn metric_data<V: One + Neg<Output = V>, S: TensorStructure>(
               &self,
               structure: S,
           ) -> SparseTensor<V, S> {
           self.base_metric(structure)
       }

}





    impl Display for isselfdual{
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", constname)
        }
    }

    impl From<isselfdual> for PhysReps {
        fn from(value: isselfdual) -> Self {
            PhysReps::isselfdual(value)
        }
    }

    impl From<Slot<isselfdual>> for PhysicalSlots {
    fn from(value: Slot<isselfdual>) -> Self {
        value.cast() //RecSlotEnum::A(value.dual_pair()).into()
    }
}

}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhysReps {
    Euclidean(Euclidean),
    LorentzUp(Lorentz),
    LorentzDown(Dual<Lorentz>),
    SpinFund(SpinFundamental),
    SpinAntiFund(Dual<SpinFundamental>),
    ColorFund(ColorFundamental),
    ColorAntiFund(Dual<ColorFundamental>),
    ColorSextet(ColorSextet),
    ColorAntiSextet(Dual<ColorSextet>),
    Bispinor(Bispinor),
    ColorAdjoint(ColorAdjoint),
}

// impl From<Lorentz> for PhysReps {
//     fn from(value: Lorentz) -> Self {
//         PhysReps::LorentzUp(value)
//     }
// }

impl Default for PhysReps {
    fn default() -> Self {
        Self::Euclidean(Euclidean {})
    }
}

impl Display for PhysReps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bispinor(_) => write!(f, "bis"),
            Self::Euclidean(_) => write!(f, "euc"),
            Self::ColorAdjoint(_) => write!(f, "coad"),
            Self::ColorFund(_) => write!(f, "cof"),
            Self::ColorAntiFund(_) => write!(f, "coaf"),
            Self::ColorSextet(_) => write!(f, "cos"),
            Self::ColorAntiSextet(_) => write!(f, "coas"),
            Self::SpinFund(_) => write!(f, "spin"),
            Self::SpinAntiFund(_) => write!(f, "spina"),
            Self::LorentzUp(_) => write!(f, "loru"),
            Self::LorentzDown(_) => write!(f, "lord"),
        }
    }
}

impl RepName for PhysReps {
    type Base = PhysReps;
    type Dual = PhysReps;

    fn dual(self) -> Self::Dual {
        match self {
            Self::LorentzUp(l) => Self::LorentzDown(l.dual()),
            Self::LorentzDown(l) => Self::LorentzUp(l.dual()),
            Self::SpinFund(l) => Self::SpinAntiFund(l.dual()),
            Self::SpinAntiFund(l) => Self::SpinFund(l.dual()),
            Self::ColorFund(l) => Self::ColorAntiFund(l.dual()),
            Self::ColorAntiFund(l) => Self::ColorFund(l.dual()),
            Self::ColorSextet(l) => Self::ColorAntiSextet(l.dual()),
            Self::ColorAntiSextet(l) => Self::ColorSextet(l.dual()),
            x => x,
        }
    }

    fn matches(&self, other: &Self::Dual) -> bool {
        matches!(
            (self, other),
            (PhysReps::Euclidean(_), PhysReps::Euclidean(_))
                | (PhysReps::LorentzUp(_), PhysReps::LorentzDown(_))
                | (PhysReps::LorentzDown(_), PhysReps::LorentzUp(_))
                | (PhysReps::Bispinor(_), PhysReps::Bispinor(_))
                | (PhysReps::ColorAdjoint(_), PhysReps::ColorAdjoint(_))
                | (PhysReps::ColorFund(_), PhysReps::ColorAntiFund(_))
                | (PhysReps::ColorAntiFund(_), PhysReps::ColorFund(_))
                | (PhysReps::ColorSextet(_), PhysReps::ColorAntiSextet(_))
                | (PhysReps::ColorAntiSextet(_), PhysReps::ColorSextet(_))
                | (PhysReps::SpinAntiFund(_), PhysReps::SpinFund(_))
                | (PhysReps::SpinFund(_), PhysReps::SpinAntiFund(_))
        )
    }

    fn base(&self) -> Self::Base {
        match self {
            Self::LorentzUp(l) => Self::LorentzUp(l.base()),
            Self::LorentzDown(l) => Self::LorentzUp(l.base()),
            Self::SpinFund(l) => Self::SpinFund(l.base()),
            Self::SpinAntiFund(l) => Self::SpinFund(l.base()),
            Self::ColorFund(l) => Self::ColorFund(l.base()),
            Self::ColorAntiFund(l) => Self::ColorFund(l.base()),
            Self::ColorSextet(l) => Self::ColorSextet(l.base()),
            Self::ColorAntiSextet(l) => Self::ColorSextet(l.base()),
            x => *x,
        }
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol) -> Result<Self> {
        match State::get_name(sym) {
            EUCLIDEAN => Ok(Self::Euclidean(Euclidean {})),
            LORENTZ => Ok(Self::LorentzUp(Lorentz {})),
            "loru" => Ok(Self::LorentzUp(Lorentz {})),
            "lord" => Ok(Self::LorentzDown(Dual::default())),
            BISPINOR => Ok(Self::Bispinor(Bispinor {})),
            SPINFUND => Ok(Self::SpinFund(SpinFundamental {})),
            SPINANTIFUND => Ok(Self::SpinAntiFund(Dual::default())),
            COLORADJ => Ok(Self::ColorAdjoint(ColorAdjoint {})),
            COLORFUND => Ok(Self::ColorFund(ColorFundamental {})),
            COLORANTIFUND => Ok(Self::ColorAntiFund(Dual::default())),
            COLORSEXT => Ok(Self::ColorSextet(ColorSextet {})),
            COLORANTISEXT => Ok(Self::ColorAntiSextet(Dual::default())),
            _ => Err(anyhow!("Not a representation")),
        }
    }

    fn metric_data<V: One + Neg<Output = V>, S: TensorStructure>(
        &self,
        structure: S,
    ) -> SparseTensor<V, S> {
        match self {
            Self::Bispinor(s) => s.base_metric(structure),
            Self::ColorAntiFund(s) => s.base_metric(structure),
            Self::ColorAntiSextet(s) => s.base_metric(structure),
            Self::ColorFund(s) => s.base_metric(structure),
            Self::ColorSextet(s) => s.base_metric(structure),
            Self::Euclidean(s) => s.base_metric(structure),
            Self::ColorAdjoint(s) => s.base_metric(structure),
            Self::LorentzDown(s) => s.base_metric(structure),
            Self::SpinFund(s) => s.base_metric(structure),
            Self::SpinAntiFund(s) => s.base_metric(structure),
            Self::LorentzUp(s) => s.base_metric(structure),
        }
    }

    fn is_neg(self, i: usize) -> bool {
        match self {
            Self::LorentzUp(l) => l.is_neg(i),
            Self::LorentzDown(l) => l.is_neg(i),
            _ => false,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Representation<T: RepName> {
    pub dim: Dimension,
    pub rep: T,
}

// pub trait HasDimension: RepName {
//     fn dim(&self) -> Dimension;

//     fn to_fnbuilder(&self) -> FunctionBuilder {
//         ::to_fnbuilder().add_arg(self.dim().to_symbolic().as_atom_view())
//     }
// }

impl<T: BaseRepName<Base: BaseRepName, Dual: BaseRepName>> Representation<T> {
    pub fn dual_pair(self) -> Representation<DualPair<T::Base>>
    where
        <T::Base as RepName>::Dual: RepName<Dual = T::Base, Base = T::Base>,
        T::Base: RepName<Dual = T::Dual, Base = T::Base>,
    {
        Representation {
            dim: self.dim,
            rep: T::selfless_pair(),
        }
    }
}

impl<T: RepName> Representation<T> {
    pub fn matches(&self, other: &Representation<T::Dual>) -> bool {
        self.dim == other.dim && self.rep.matches(&other.rep)
    }
    #[cfg(feature = "shadowing")]
    /// yields a function builder for the representation, adding a first variable: the dimension.
    ///
    pub fn to_fnbuilder(&self) -> FunctionBuilder {
        self.rep
            .to_fnbuilder()
            .add_arg(self.dim.to_symbolic().as_atom_view())
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
            rep: self.rep.into(),
        }
    }

    pub fn metric<S, V: One + Zero + Neg<Output = V>>(
        &self,
        aind: [AbstractIndex; 2],
    ) -> SparseTensor<V, S>
    where
        S: TensorStructure + FromIterator<Slot<T>> + FromIterator<Slot<T::Dual>>,
    {
        self.rep.metric(self.dim.try_into().unwrap(), aind)
    }

    pub fn metric_data<V: One + Neg<Output = V>, S: TensorStructure>(
        &self,
        structure: S,
    ) -> SparseTensor<V, S> {
        self.rep.metric_data(structure)
    }

    pub fn is_neg(&self, i: usize) -> bool {
        self.rep.is_neg(i)
    }

    pub fn new_slot<A: Into<AbstractIndex>>(&self, aind: A) -> Slot<T> {
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
    pub fn negative(&self) -> Result<Vec<bool>> {
        Ok((0..usize::try_from(self.dim)?)
            .map(|i| self.is_neg(i))
            .collect())
    }
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

impl<T: BaseRepName<Dual = Dual<T>, Base = T>> Representation<Dual<T>>
where
    Dual<T>: BaseRepName<Dual = T, Base = T>,
{
    pub fn pair(self) -> Representation<DualPair<T>> {
        Representation {
            dim: self.dim,
            rep: Dual::<T>::selfless_pair(),
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
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
pub struct Slot<T: RepName> {
    pub aind: AbstractIndex,
    rep: Representation<T>,
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
/// # use spenso::{Representation,Slot,Dimension,AbstractIndex};
/// # use symbolica::atom::AtomView;

///    let mink = Representation::Lorentz(Dimension(4));
///    let mu = Slot::from((AbstractIndex(0), mink));
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
    /// # use spenso::{Representation,Slot,Dimension,AbstractIndex};
    /// let mink = Representation::Lorentz(Dimension(4));
    /// let mu = Slot::from((AbstractIndex(0),mink));
    ///
    /// assert_eq!("lor(4,0)",format!("{}",mu.to_symbolic()));
    /// assert_eq!("id0lor4",format!("{}",mu));
    /// ```
    fn to_symbolic(&self) -> Atom;
    #[cfg(feature = "shadowing")]
    fn to_symbolic_wrapped(&self) -> Atom;
    #[cfg(feature = "shadowing")]
    fn try_from_view(v: AtomView<'_>) -> Result<Self, SlotError>;
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
        write!(f, "{}{}", self.aind, self.rep)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DualPair<R: BaseRepName<Dual: BaseRepName>> {
    Rep(R),
    DualRep(R::Dual),
}

impl<R: BaseRepName<Dual: BaseRepName>> Default for DualPair<R> {
    fn default() -> Self {
        DualPair::Rep(R::default())
    }
}

impl<R: BaseRepName<Dual: BaseRepName>> Display for DualPair<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DualPair::DualRep(r) => <R::Dual as Display>::fmt(r, f),
            DualPair::Rep(r) => <R as Display>::fmt(r, f),
        }
    }
}

impl<R: BaseRepName<Dual: BaseRepName>> RepName for DualPair<R>
where
    R::Dual: RepName<Dual = R, Base = R::Base>,
    R::Base: RepName<Dual = R::Dual, Base = R::Base>,
{
    type Dual = DualPair<R>;
    type Base = R::Base;

    fn metric_data<V: One + Neg<Output = V>, S: TensorStructure>(
        &self,
        structure: S,
    ) -> SparseTensor<V, S> {
        match self {
            Self::DualRep(d) => d.base_metric(structure),
            Self::Rep(r) => r.base_metric(structure),
        }
    }
    fn dual(self) -> Self::Dual {
        match self {
            Self::Rep(r) => Self::DualRep(r.dual()),
            Self::DualRep(d) => Self::Rep(d.dual()),
        }
    }
    fn matches(&self, other: &Self::Dual) -> bool {
        matches!(
            (self, other),
            (Self::DualRep(_), Self::Rep(_)) | (Self::Rep(_), Self::DualRep(_))
        )
    }
    fn is_neg(self, i: usize) -> bool {
        match self {
            Self::Rep(r) => r.is_neg(i),
            Self::DualRep(d) => d.is_neg(i),
        }
    }

    fn base(&self) -> Self::Base {
        match self {
            Self::Rep(r) => r.base(),
            Self::DualRep(d) => d.base(),
        }
    }
    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol) -> Result<Self> {
        if let Ok(r) = R::try_from_symbol(sym) {
            Ok(DualPair::Rep(r))
        } else if let Ok(d) = R::Dual::try_from_symbol(sym) {
            Ok(DualPair::DualRep(d))
        } else {
            Err(anyhow!("Not a representation"))
        }
    }
}

pub type PhysicalSlots = Slot<PhysReps>;

pub trait ScalarTensor: HasStructure<Structure: ScalarStructure> {
    fn new_scalar(scalar: Self::Scalar) -> Self;
}
pub trait HasStructure {
    type Structure: TensorStructure;
    type Scalar;

    fn structure(&self) -> &Self::Structure;
    fn mut_structure(&mut self) -> &mut Self::Structure;
    fn scalar(self) -> Option<Self::Scalar>;

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
        self.structure().args()
    }
    // fn cast_structure<O, S>(self) -> O
    // where
    //     O: HasStructure<Structure = S, Scalar = Self::Scalar>,
    //     S: TensorStructure + From<Self::Structure>;
}

pub trait CastStructure<O: HasStructure<Structure: From<Self::Structure>>>: HasStructure {
    fn cast(self) -> O;
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
    // type R = <T::Structure as TensorStructure>::R;
    type Slot = <T::Structure as TensorStructure>::Slot;

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

#[cfg(feature = "shadowing")]
impl<T: HasName> ToSymbolic for T
where
    T: TensorStructure,
    T::Name: IntoSymbol,
    T::Args: IntoArgs,
{
    fn concrete_atom(&self, id: FlatIndex) -> ExpandedCoefficent<()> {
        ExpandedCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: self.expanded_index(id).unwrap(),
            args: None,
        }
    }

    fn flat_atom(&self, id: FlatIndex) -> FlatCoefficent<()> {
        FlatCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: id,
            args: None,
        }
    }
}

#[cfg(feature = "shadowing")]
pub trait ToSymbolic: TensorStructure {
    fn concrete_atom(&self, id: FlatIndex) -> ExpandedCoefficent<()> {
        ExpandedCoefficent {
            name: None,
            index: self.expanded_index(id).unwrap(),
            args: None,
        }
    }

    fn flat_atom(&self, id: FlatIndex) -> FlatCoefficent<()> {
        FlatCoefficent {
            name: None,
            index: id,
            args: None,
        }
    }

    fn to_dense_expanded_labels(self) -> Result<DenseTensor<Atom, Self>>
    where
        Self: std::marker::Sized + Clone,
    {
        self.to_dense_labeled(Self::concrete_atom)
    }

    fn to_dense_labeled<T>(
        self,
        index_to_atom: impl Fn(&Self, FlatIndex) -> T,
    ) -> Result<DenseTensor<Atom, Self>>
    where
        Self: Sized,
        T: TensorCoefficient,
    {
        let mut data = vec![];
        for index in 0..self.size()? {
            data.push(index_to_atom(&self, index.into()).to_atom().unwrap());
        }

        Ok(DenseTensor {
            data,
            structure: self,
        })
    }

    fn to_dense_labeled_complex<T>(
        self,
        index_to_atom: impl Fn(&Self, FlatIndex) -> T,
    ) -> Result<DenseTensor<Atom, Self>>
    where
        Self: Sized,
        T: TensorCoefficient,
    {
        let mut data = vec![];
        for index in 0..self.size()? {
            let re = index_to_atom(&self, index.into()).to_atom_re().unwrap();
            let im = index_to_atom(&self, index.into()).to_atom_im().unwrap();
            let i = Atom::new_var(State::I);
            data.push(&re + i * &im);
        }

        Ok(DenseTensor {
            data,
            structure: self,
        })
    }

    fn to_dense_flat_labels(self) -> Result<DenseTensor<Atom, Self>>
    where
        Self: std::marker::Sized + Clone,
    {
        self.to_dense_labeled(Self::flat_atom)
    }
    fn to_explicit_rep(self) -> Result<MixedTensor<f64, Self>>
    where
        Self: std::marker::Sized + Clone + HasName,
        Self::Name: IntoSymbol,
    {
        let identity = State::get_symbol("id");
        let gamma = State::get_symbol("γ");
        let gamma5 = State::get_symbol("γ5");
        let proj_m = State::get_symbol("ProjM");
        let proj_p = State::get_symbol("ProjP");
        let sigma = State::get_symbol("σ");
        let metric = State::get_symbol("Metric");

        if let Some(name) = self.name() {
            let name = name.ref_into_symbol();
            Ok(match name {
                _ if name == identity => ufo::identity_data::<f64, Self>(self).into(),

                _ if name == gamma => ufo::gamma_data(self).into(),
                _ if name == gamma5 => ufo::gamma5_data(self).into(),
                _ if name == proj_m => ufo::proj_m_data(self).into(),
                _ if name == proj_p => ufo::proj_p_data(self).into(),
                _ if name == sigma => ufo::sigma_data(self).into(),
                _ if name == metric => ufo::metric_data::<f64, Self>(self).into(),
                _ => MixedTensor::param(self.to_dense_expanded_labels()?.into()),
            })
        } else {
            Err(anyhow!("No name"))
        }
    }

    fn to_symbolic(&self) -> Option<Atom>
    where
        Self: HasName<Name: IntoSymbol, Args: IntoArgs>,
    {
        let args = self.args().map(|s| s.args()).unwrap_or_default();

        Some(self.to_symbolic_with(self.name()?.ref_into_symbol(), &args))
    }

    fn to_symbolic_with(&self, name: Symbol, args: &[Atom]) -> Atom {
        let slots = self
            .external_structure_iter()
            .map(|slot| slot.to_symbolic())
            .collect::<Vec<_>>();

        let mut value_builder = FunctionBuilder::new(name.ref_into_symbol());

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

pub trait ScalarStructure {
    fn scalar_structure() -> Self;
}
pub trait TensorStructure {
    type Slot: IsAbstractSlot + DualSlotTo<Dual = Self::Slot>;
    // type R: Rep;

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension>;
    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
    fn get_aind(&self, i: usize) -> Option<AbstractIndex>;
    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
    fn get_dim(&self, i: usize) -> Option<Dimension>;
    fn get_slot(&self, i: usize) -> Option<Self::Slot>;
    fn order(&self) -> usize;
    /// returns the list of slots that are the external indices of the tensor

    fn external_structure(&self) -> Vec<Self::Slot> {
        self.external_structure_iter().collect()
    }

    fn to_shell(self) -> TensorShell<Self>
    where
        Self: Sized,
    {
        TensorShell::new(self)
    }

    fn contains_matching(&self, slot: &Self::Slot) -> bool {
        self.external_structure_iter().any(|s| s.matches(slot))
    }

    fn external_reps(&self) -> Vec<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.external_reps_iter().collect()
    }

    fn external_indices(&self) -> Vec<AbstractIndex> {
        self.external_indices_iter().collect()
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
            if let Some(&i) = posmap.get(&slot.dual()) {
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
            if let Some(&i) = posmap.get(&slot_other.dual()) {
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
        let mut positions: HashMap<<Self as TensorStructure>::Slot, Vec<usize>> = HashMap::new();

        // Track the positions of each element
        for (index, key) in self.external_structure_iter().enumerate() {
            if let Some(v) = positions.get_mut(&key.dual()) {
                v.push(index);
            } else {
                positions.insert(key, vec![index]);
            }
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
        self.external_dims_iter().collect()
    }

    fn reps(&self) -> Vec<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.external_reps_iter().collect()
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
    fn find_permutation(&self, other: &Self) -> Result<Vec<ConcreteIndex>> {
        if self.order() != other.order() {
            return Err(anyhow!(
                "Mismatched order: {} vs {}",
                self.order(),
                other.order()
            ));
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
                    return Err(anyhow!("No available index for {:?}", item));
                }
            } else {
                // Item not found in other
                return Err(anyhow!("Item {:?} not found in other", item));
            }
        }

        Ok(permutation)
    }

    /// yields the strides of the tensor in column major order
    fn strides_column_major(&self) -> Result<Vec<usize>> {
        let mut strides: Vec<usize> = vec![1; self.order()];

        if self.order() == 0 {
            return Ok(strides);
        }

        for i in 0..self.order() - 1 {
            strides[i + 1] = strides[i] * usize::try_from(self.shape()[i])?;
        }

        Ok(strides)
    }

    /// yields the strides of the tensor in row major order
    fn strides_row_major(&self) -> Result<Vec<usize>> {
        let mut strides = vec![1; self.order()];
        if self.order() == 0 {
            return Ok(strides);
        }

        for i in (0..self.order() - 1).rev() {
            strides[i] = strides[i + 1] * usize::try_from(self.shape()[i + 1])?;
        }

        Ok(strides)
    }

    /// By default, the strides are row major
    fn strides(&self) -> Result<Vec<usize>> {
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
    fn verify_indices(&self, indices: &[ConcreteIndex]) -> Result<()> {
        if indices.len() != self.order() {
            return Err(anyhow!(
                "Mismatched order: {} indices, vs order {}",
                indices.len(),
                self.order()
            ));
        }

        for (i, dim_len) in self
            .external_structure_iter()
            .map(|slot| slot.dim())
            .enumerate()
        {
            if indices[i] >= usize::try_from(dim_len)? {
                return Err(anyhow!(
                    "Index {} out of bounds for dimension {} of size {}",
                    indices[i],
                    i,
                    usize::try_from(dim_len)?
                ));
            }
        }
        Ok(())
    }

    /// yields the flat index of the tensor given a list of indices
    ///
    /// # Errors
    ///
    /// Same as [`Self::verify_indices`]
    fn flat_index(&self, indices: &[ConcreteIndex]) -> Result<FlatIndex> {
        let strides = self.strides()?;
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
    fn expanded_index(&self, flat_index: FlatIndex) -> Result<ExpandedIndex> {
        let mut indices = vec![];
        let mut index = flat_index.index;
        for &stride in &self.strides()? {
            indices.push(index / stride);
            index %= stride;
        }
        if flat_index.index < self.size()? {
            Ok(indices.into())
        } else {
            Err(anyhow!("Index {flat_index} out of bounds"))
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

    // /// get the metric along the i-th index
    // fn get_ith_metric(&self, i: usize) -> Result<Vec<bool>> {
    //     self.get_rep(i)
    //         .ok_or(anyhow!("out of bounds access"))?
    //         .negative()
    // }

    /// yields the size of the tensor, i.e. the product of the dimensions. This is the length of the vector of the data in a dense tensor
    fn size(&self) -> Result<usize> {
        if self.order() == 0 {
            return Ok(1);
        }
        let mut size = 1;
        for dim in self.shape() {
            size *= usize::try_from(dim)?;
        }
        Ok(size)
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

impl<S: IsAbstractSlot<R: RepName> + DualSlotTo<Dual = S>> ScalarStructure for Vec<S> {
    fn scalar_structure() -> Self {
        vec![]
    }
}

impl<S: IsAbstractSlot<R: RepName> + DualSlotTo<Dual = S>> TensorStructure for Vec<S> {
    type Slot = S;
    // type R = S::R;

    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.iter().map(|s| s.rep())
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex> {
        self.iter().map(|s| s.aind())
    }

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot> {
        self.iter().cloned()
    }

    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension> {
        self.iter().map(|s| s.dim())
    }

    fn order(&self) -> usize {
        self.len()
    }

    fn get_slot(&self, i: usize) -> Option<S> {
        self.get(i).cloned()
    }

    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.get(i).map(|s| s.rep())
    }

    fn get_dim(&self, i: usize) -> Option<Dimension> {
        self.get(i).map(|s| s.dim())
    }

    fn get_aind(&self, i: usize) -> Option<AbstractIndex> {
        self.get(i).map(|s| s.aind())
    }
}

#[cfg(feature = "shadowing")]
impl<S: IsAbstractSlot<R: RepName> + ConstructibleSlot<S::R> + DualSlotTo<Dual = S>> ToSymbolic
    for Vec<S>
{
}

/// A trait for a structure that can be traced and merged, during a contraction.
pub trait StructureContract {
    fn trace(&mut self, i: usize, j: usize);

    fn trace_out(&mut self);

    fn merge(&mut self, other: &Self) -> Option<usize>;

    #[must_use]
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self;
}

impl<S: DualSlotTo<Dual = S, R: RepName>> StructureContract for Vec<S> {
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
            let e = positions.get(&value.dual());
            if let Some((Some(selfi), None)) = e {
                positions.insert(value.dual(), (Some(*selfi), Some(index)));
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
pub struct IndexLess<T: RepName> {
    pub structure: Vec<Representation<T>>,
}

impl<T: RepName> IndexLess<T> {
    pub fn new(structure: Vec<Representation<T>>) -> Self {
        Self { structure }
    }

    pub fn empty() -> Self {
        Self { structure: vec![] }
    }

    pub fn to_indexed(self, indices: &[AbstractIndex]) -> Vec<Slot<T>> {
        indices
            .iter()
            .cloned()
            .zip(self.structure.iter().cloned())
            .map(|(i, r)| Representation::new_slot(&r, i))
            .collect()
    }
}

impl<T: RepName<Dual = T>> ScalarStructure for IndexLess<T> {
    fn scalar_structure() -> Self {
        Self::empty()
    }
}

impl<T: RepName<Dual = T>> TensorStructure for IndexLess<T>
// where
//     T::Base: Rep<Dual = T::Dual, Base = T::Base>,
{
    type Slot = Slot<T>;
    // type R = T;

    fn external_reps_iter(
        &self,
    ) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.structure.iter().copied()
    }

    fn external_dims_iter(&self) -> impl Iterator<Item = Dimension> {
        self.structure.iter().map(|s| s.dim)
    }

    fn get_aind(&self, _: usize) -> Option<AbstractIndex> {
        None
    }

    fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex> {
        [].iter().cloned()
    }

    fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot> {
        [].iter().cloned()
    }

    fn order(&self) -> usize {
        self.structure.len()
    }

    fn get_slot(&self, _: usize) -> Option<Self::Slot> {
        None
    }

    fn find_permutation(&self, other: &Self) -> Result<Vec<ConcreteIndex>> {
        if self.order() != other.order() {
            return Err(anyhow!(
                "Mismatched order: {} vs {}",
                self.order(),
                other.order()
            ));
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
                    return Err(anyhow!("No available index for {:?}", item));
                }
            } else {
                // Item not found in other
                return Err(anyhow!("Item {:?} not found in other", item));
            }
        }

        Ok(permutation)
    }

    fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>> {
        self.structure.get(i).copied()
    }

    fn get_dim(&self, i: usize) -> Option<Dimension> {
        self.structure.get(i).map(|&r| r.into())
    }
}

#[cfg(feature = "shadowing")]
impl<T: RepName<Dual = T>> ToSymbolic for IndexLess<T> {}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Default, Hash)]
pub struct VecStructure {
    pub structure: Vec<PhysicalSlots>,
}

impl FromIterator<Slot<Lorentz>> for VecStructure {
    fn from_iter<T: IntoIterator<Item = Slot<Lorentz>>>(iter: T) -> Self {
        let vec = iter.into_iter().map(Slot::<PhysReps>::from).collect();
        Self { structure: vec }
    }
}

impl FromIterator<Slot<Dual<Lorentz>>> for VecStructure {
    fn from_iter<T: IntoIterator<Item = Slot<Dual<Lorentz>>>>(iter: T) -> Self {
        let vec = iter.into_iter().map(Slot::<PhysReps>::from).collect();
        Self { structure: vec }
    }
}

impl FromIterator<Slot<Euclidean>> for VecStructure {
    fn from_iter<T: IntoIterator<Item = Slot<Euclidean>>>(iter: T) -> Self {
        let vec = iter.into_iter().map(Slot::<PhysReps>::from).collect();
        Self { structure: vec }
    }
}

#[cfg(feature = "shadowing")]
impl TryFrom<AtomView<'_>> for VecStructure {
    type Error = SlotError;
    fn try_from(value: AtomView) -> Result<Self, Self::Error> {
        match value {
            AtomView::Mul(mul) => mul.try_into(),
            AtomView::Fun(fun) => fun.try_into(),
            AtomView::Pow(_) => {
                Ok(VecStructure::default()) // powers do not have a structure
            }
            _ => Err(anyhow!("Not a structure: {value}").into()), // could check if it
        }
    }
}

// impl From<Vec<PhysicalSlots>> for VecStructure {
//     fn from(value: Vec<PhysicalSlots>) -> Self {
//         VecStructure { structure: value }
//     }
// }

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
    type Error = SlotError;
    fn try_from(value: FunView) -> Result<Self, Self::Error> {
        if value.get_symbol() == State::get_symbol(ABSTRACTIND) {
            let mut structure: Vec<PhysicalSlots> = vec![];

            for arg in value.iter() {
                structure.push(arg.try_into()?);
            }

            Ok(VecStructure { structure })
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
    type Error = SlotError;
    fn try_from(value: MulView) -> Result<Self, Self::Error> {
        let mut structure: Self = vec![].into();

        for arg in value.iter() {
            structure.extend(arg.try_into()?);
        }
        Ok(structure)
    }
}

impl FromIterator<PhysicalSlots> for VecStructure {
    fn from_iter<T: IntoIterator<Item = PhysicalSlots>>(iter: T) -> Self {
        Self {
            structure: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<PhysicalSlots>> for VecStructure {
    fn from(structure: Vec<PhysicalSlots>) -> Self {
        Self { structure }
    }
}

impl IntoIterator for VecStructure {
    type Item = PhysicalSlots;
    type IntoIter = std::vec::IntoIter<PhysicalSlots>;
    fn into_iter(self) -> std::vec::IntoIter<PhysicalSlots> {
        self.structure.into_iter()
    }
}

impl<'a> IntoIterator for &'a VecStructure {
    type Item = &'a PhysicalSlots;
    type IntoIter = std::slice::Iter<'a, PhysicalSlots>;
    fn into_iter(self) -> std::slice::Iter<'a, PhysicalSlots> {
        self.structure.iter()
    }
}

impl<'a> IntoIterator for &'a mut VecStructure {
    type Item = &'a mut PhysicalSlots;
    type IntoIter = std::slice::IterMut<'a, PhysicalSlots>;
    fn into_iter(self) -> std::slice::IterMut<'a, PhysicalSlots> {
        self.structure.iter_mut()
    }
}

impl VecStructure {
    pub fn new(structure: Vec<PhysicalSlots>) -> Self {
        Self { structure }
    }

    fn extend(&mut self, other: Self) {
        self.structure.extend(other.structure)
    }

    pub fn to_named<N, A>(self, name: N, args: Option<A>) -> NamedStructure<N, A> {
        NamedStructure::from_iter(self, name, args)
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

impl From<VecStructure> for Vec<PhysicalSlots> {
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
                usize::from(item.aind()),
                // IDPRINTER
                //     .encode_string(usize::from(item.index) as u64)
                //     .unwrap(),
                item.rep()
            )?;
        }
        Ok(())
    }
}

impl ScalarStructure for VecStructure {
    fn scalar_structure() -> Self {
        VecStructure { structure: vec![] }
    }
}

impl TensorStructure for VecStructure {
    type Slot = PhysicalSlots;
    // type R = PhysicalReps;

    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
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
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Default, Hash)]
pub struct NamedStructure<Name = SmartString<LazyCompact>, Args = usize> {
    pub structure: VecStructure,
    pub global_name: Option<Name>,
    pub additional_args: Option<Args>,
}

#[cfg(feature = "shadowing")]
pub type AtomStructure = NamedStructure<SerializableSymbol, Vec<SerializableAtom>>;

impl<Name, Args> NamedStructure<Name, Args> {
    #[must_use]
    pub fn from_iter<I, T>(iter: T, name: Name, args: Option<Args>) -> Self
    where
        I: Into<PhysicalSlots>,
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
impl<'a> TryFrom<FunView<'a>> for AtomStructure {
    type Error = SlotError;
    fn try_from(value: FunView<'a>) -> Result<Self, Self::Error> {
        match value.get_symbol() {
            s if s == State::get_symbol(ABSTRACTIND) => {
                let mut structure: Vec<PhysicalSlots> = vec![];

                for arg in value.iter() {
                    structure.push(arg.try_into()?);
                }

                Ok(VecStructure::from(structure).into())
            }
            name => {
                let mut structure: AtomStructure = VecStructure::default().into();
                structure.set_name(name.into());
                let mut args = vec![];

                for arg in value.iter() {
                    if let AtomView::Fun(fun) = arg {
                        structure.structure.extend(fun.try_into()?);
                    } else {
                        args.push(arg.to_owned().into());
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
    fn args(&self) -> Option<Self::Args> {
        self.additional_args.clone()
    }
}

pub trait HasName {
    type Name: Clone;
    type Args: Clone;
    fn name(&self) -> Option<Self::Name>;
    fn args(&self) -> Option<Self::Args>;
    fn set_name(&mut self, name: Self::Name);

    #[cfg(feature = "shadowing")]
    fn expanded_coef(&self, id: FlatIndex) -> ExpandedCoefficent<Self::Args>
    where
        Self: TensorStructure,
        Self::Name: IntoSymbol,
        Self::Args: IntoArgs,
    {
        ExpandedCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: self.expanded_index(id).unwrap(),
            args: self.args(),
        }
    }

    #[cfg(feature = "shadowing")]
    fn flat_coef(&self, id: FlatIndex) -> FlatCoefficent<Self::Args>
    where
        Self: TensorStructure,
        Self::Name: IntoSymbol,
        Self::Args: IntoArgs,
    {
        FlatCoefficent {
            name: self.name().map(|n| n.ref_into_symbol()),
            index: id,
            args: self.args(),
        }
    }
}

impl<N, A> ScalarStructure for NamedStructure<N, A> {
    fn scalar_structure() -> Self {
        NamedStructure {
            structure: VecStructure::default(),
            global_name: None,
            additional_args: None,
        }
    }
}

impl<N, A> TensorStructure for NamedStructure<N, A> {
    type Slot = PhysicalSlots;
    // type R = PhysicalReps;

    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
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

impl<I: Into<PhysicalSlots>> FromIterator<I> for ContractionCountStructure {
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

impl ScalarStructure for ContractionCountStructure {
    fn scalar_structure() -> Self {
        ContractionCountStructure {
            structure: VecStructure::default(),
            contractions: 0,
        }
    }
}

impl TensorStructure for ContractionCountStructure {
    type Slot = PhysicalSlots;
    // type R = PhysicalReps;

    delegate! {
        to self.structure{
            fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
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
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize, Hash)]
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
        I: Into<PhysicalSlots>,
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
    fn args(&self) -> Option<Self::Args> {
        self.additional_args.clone()
    }
}

#[cfg(feature = "shadowing")]
impl<N: IntoSymbol, A: IntoArgs> Display for SmartShadowStructure<N, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref name) = self.global_name {
            write!(f, "{}", name.ref_into_symbol())?
        }
        write!(f, "(")?;
        if let Some(ref args) = self.additional_args {
            let args: Vec<std::string::String> =
                args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }

        write!(f, "{})", self.structure)?;
        Result::Ok(())
    }
}

impl<N, A> ScalarStructure for SmartShadowStructure<N, A> {
    fn scalar_structure() -> Self {
        SmartShadowStructure {
            structure: VecStructure::default(),
            contractions: 0,
            global_name: None,
            additional_args: None,
        }
    }
}

impl<N, A> TensorStructure for SmartShadowStructure<N, A> {
    type Slot = PhysicalSlots;
    // type R = PhysicalReps;

    delegate! {
        to self.structure{
           fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
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

impl<N, A> From<NamedStructure<N, A>> for SmartShadowStructure<N, A> {
    fn from(value: NamedStructure<N, A>) -> Self {
        Self {
            structure: value.structure,
            contractions: 0,
            global_name: value.global_name,
            additional_args: value.additional_args,
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
        let internal_set: HashSet<<Self as TensorStructure>::Slot> = self
            .internal
            .clone()
            .into_iter()
            .filter(|s| self.external.contains_matching(s))
            .collect();

        let other_set: HashSet<<Self as TensorStructure>::Slot> =
            other.internal.clone().into_iter().collect();

        let mut replacement_value = internal_set
            .union(&other_set)
            .map(|s| s.aind())
            .max()
            .unwrap_or(0.into())
            + 1.into();

        for item in &mut self.internal {
            if other_set.contains(item) {
                item.set_aind(replacement_value);
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
            fn args(&self) -> Option<Self::Args>;
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

impl<N, A> ScalarStructure for HistoryStructure<N, A> {
    fn scalar_structure() -> Self {
        HistoryStructure {
            internal: VecStructure::default(),
            names: AHashMap::default(),
            external: NamedStructure::scalar_structure(),
        }
    }
}

impl<N, A> TensorStructure for HistoryStructure<N, A> {
    type Slot = PhysicalSlots;
    // type R = PhysicalReps;

    delegate! {
        to self.external{
           fn external_reps_iter(&self) -> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)->impl Iterator<Item=Dimension>;
            fn external_structure_iter(&self) -> impl Iterator<Item = Self::Slot>;
            fn order(&self) -> usize;
            fn get_slot(&self, i: usize) -> Option<Self::Slot>;
            fn get_rep(&self, i: usize) -> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_aind(&self,i:usize)->Option<AbstractIndex>;
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
        for (index, value) in (self.external).external_structure_iter().enumerate() {
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
    let id = name.ref_into_symbol();
    atomic_expanded_label_id(indices, id, &[])
}
#[cfg(feature = "shadowing")]
pub fn atomic_flat_label<I: IntoSymbol>(index: usize, name: I) -> Atom {
    let id = name.ref_into_symbol();
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
    fn ref_into_symbol(&self) -> Symbol;

    fn from_str(s: &str) -> Self;
}

#[cfg(feature = "shadowing")]
pub trait IntoArgs {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom>;
    fn args(&self) -> Vec<Atom> {
        self.ref_into_args().collect()
    }
    fn cooked_name(&self) -> std::string::String;
}

#[cfg(feature = "shadowing")]
impl IntoArgs for usize {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom> {
        std::iter::once(Atom::new_num(*self as i64))
    }
    fn cooked_name(&self) -> std::string::String {
        format!("{self}")
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for Vec<SerializableAtom> {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom> {
        self.iter().map(|x| x.0.clone())
    }
    fn cooked_name(&self) -> std::string::String {
        let init = "".into();
        self.iter()
            .fold(init, |acc, x| acc + x.to_string().as_str())
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for () {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom> {
        std::iter::empty()
    }
    fn cooked_name(&self) -> std::string::String {
        "".into()
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct NoArgs;

impl Display for NoArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for NoArgs {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom> {
        std::iter::empty()
    }
    fn cooked_name(&self) -> std::string::String {
        "".into()
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for Atom {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom> {
        std::iter::once(self.clone())
    }

    fn cooked_name(&self) -> std::string::String {
        self.to_string()
    }
}

#[cfg(feature = "shadowing")]
impl IntoArgs for Vec<Atom> {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom> {
        self.iter().cloned()
    }

    fn cooked_name(&self) -> std::string::String {
        let init = "".into();
        self.iter()
            .fold(init, |acc, x| acc + x.to_string().as_str())
    }
}

#[cfg(feature = "shadowing")]
impl<const N: usize> IntoArgs for [Atom; N] {
    fn ref_into_args(&self) -> impl Iterator<Item = Atom> {
        self.iter().cloned()
    }

    fn cooked_name(&self) -> std::string::String {
        let init = "".into();
        self.iter()
            .fold(init, |acc, x| acc + x.to_string().as_str())
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for SmartString<LazyCompact> {
    fn ref_into_symbol(&self) -> Symbol {
        State::get_symbol(self)
    }

    fn from_str(s: &str) -> Self {
        s.into()
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for Symbol {
    fn ref_into_symbol(&self) -> Symbol {
        *self
    }

    fn from_str(s: &str) -> Self {
        State::get_symbol(s)
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for SerializableSymbol {
    fn ref_into_symbol(&self) -> Symbol {
        self.symbol
    }

    fn from_str(s: &str) -> Self {
        Self {
            symbol: State::get_symbol(s),
        }
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for std::string::String {
    fn ref_into_symbol(&self) -> Symbol {
        State::get_symbol(self)
    }
    fn from_str(s: &str) -> Self {
        s.into()
    }
}

/// Trait that enables shadowing of a tensor
///
/// This creates a dense tensor of atoms, where the atoms are the expanded indices of the tensor, with the global name as the name of the labels.
#[cfg(feature = "shadowing")]
pub trait Shadowable:
    HasStructure<
        Structure: TensorStructure + HasName<Name: IntoSymbol, Args: IntoArgs> + Clone + Sized,
    > + Sized
{
    // type Const;
    fn expanded_shadow(&self) -> Result<DenseTensor<Atom, Self::Structure>> {
        self.shadow(Self::Structure::expanded_coef)
    }

    fn flat_shadow(&self) -> Result<DenseTensor<Atom, Self::Structure>> {
        self.shadow(Self::Structure::flat_coef)
    }

    fn shadow<T>(
        &self,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    ) -> Result<DenseTensor<Atom, Self::Structure>>
    where
        T: TensorCoefficient,
    {
        self.structure().clone().to_dense_labeled(index_to_atom)
    }

    fn to_explicit(&self) -> Option<MixedTensor<f64, Self::Structure>> {
        let identity = State::get_symbol("id");
        let gamma = State::get_symbol("γ");
        let gamma5 = State::get_symbol("γ5");
        let proj_m = State::get_symbol("ProjM");
        let proj_p = State::get_symbol("ProjP");
        let sigma = State::get_symbol("σ");
        let metric = State::get_symbol("Metric");

        if let Some(name) = self.structure().name() {
            let name = name.ref_into_symbol();
            Some(match name {
                _ if name == identity => {
                    ufo::identity_data::<f64, Self::Structure>(self.structure().clone()).into()
                }

                _ if name == gamma => ufo::gamma_data(self.structure().clone()).into(),
                _ if name == gamma5 => ufo::gamma5_data(self.structure().clone()).into(),
                _ if name == proj_m => ufo::proj_m_data(self.structure().clone()).into(),
                _ if name == proj_p => ufo::proj_p_data(self.structure().clone()).into(),
                _ if name == sigma => ufo::sigma_data(self.structure().clone()).into(),
                _ if name == metric => {
                    ufo::metric_data::<f64, Self::Structure>(self.structure().clone()).into()
                }
                _ => MixedTensor::param(self.expanded_shadow().unwrap().into()),
            })
        } else {
            None
        }
        // self.structure().clone().to_explicit_rep()
    }
}

#[cfg(feature = "shadowing")]
pub trait ShadowMapping<Const>: Shadowable {
    fn expanded_shadow_with_map<'a>(
        &'a self,
        fn_map: &mut FunctionMap<'a, Const>,
    ) -> Result<ParamTensor<Self::Structure>> {
        self.shadow_with_map(fn_map, Self::Structure::expanded_coef)
    }

    fn shadow_with_map<'a, T, F>(
        &'a self,
        fn_map: &mut FunctionMap<'a, Const>,
        index_to_atom: F,
    ) -> Result<ParamTensor<Self::Structure>>
    where
        T: TensorCoefficient,
        F: Fn(&Self::Structure, FlatIndex) -> T + Clone,
    {
        // Some(ParamTensor::Param(self.shadow(index_to_atom)?.into()))
        self.append_map(fn_map, index_to_atom.clone());
        self.shadow(index_to_atom)
            .map(|x| ParamTensor::param(x.into()))
    }

    fn append_map<'a, T>(
        &'a self,
        fn_map: &mut FunctionMap<'a, Const>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    ) where
        T: TensorCoefficient;

    fn flat_append_map<'a>(&'a self, fn_map: &mut FunctionMap<'a, Const>) {
        self.append_map(fn_map, Self::Structure::flat_coef)
    }

    fn expanded_append_map<'a>(&'a self, fn_map: &mut FunctionMap<'a, Const>) {
        self.append_map(fn_map, Self::Structure::expanded_coef)
    }

    fn flat_shadow_with_map<'a>(
        &'a self,
        fn_map: &mut FunctionMap<'a, Const>,
    ) -> Result<ParamTensor<Self::Structure>> {
        self.shadow_with_map(fn_map, Self::Structure::flat_coef)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct TensorShell<S: TensorStructure> {
    structure: S,
}

impl<S: TensorStructure + ScalarStructure> ScalarTensor for TensorShell<S> {
    fn new_scalar(_scalar: Self::Scalar) -> Self {
        TensorShell {
            structure: S::scalar_structure(),
        }
    }
}

impl<S: TensorStructure> HasStructure for TensorShell<S> {
    type Structure = S;
    type Scalar = ();
    fn structure(&self) -> &S {
        &self.structure
    }

    fn scalar(self) -> Option<Self::Scalar> {
        if self.structure.is_scalar() {
            Some(())
        } else {
            None
        }
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

    fn args(&self) -> Option<Self::Args> {
        self.structure.args()
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

impl<S: TensorStructure, O: From<S> + TensorStructure> CastStructure<TensorShell<O>>
    for TensorShell<S>
{
    fn cast(self) -> TensorShell<O> {
        TensorShell {
            structure: self.structure.into(),
        }
    }
}

impl<S: TensorStructure> From<S> for TensorShell<S> {
    fn from(structure: S) -> Self {
        Self::new(structure)
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure + HasName + Clone> Shadowable for TensorShell<S>
where
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
{
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure + HasName + Clone, Const> ShadowMapping<Const> for TensorShell<S>
where
    S::Name: IntoSymbol + Clone,
    S::Args: IntoArgs,
{
    fn append_map<'a, T>(
        &'a self,
        _fn_map: &mut FunctionMap<'a, Const>,
        _index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    ) where
        T: TensorCoefficient,
    {
    }
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

// pub struct Kroneker {
//     structure: VecStructure,
// }

// impl Kroneker {
//     pub fn new<T: Rep>(i: GenSlot<T>, j: GenSlot<T::Dual>) -> Self {}
// }

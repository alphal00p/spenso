use super::{
    abstract_index::{AbstractIndex, AbstractIndexError},
    dimension::{Dimension, DimensionError},
    slot::{PhysicalSlots, Slot},
};
use ahash::AHashMap;
use append_only_vec::AppendOnlyVec;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    sync::RwLock,
};
use std::{hash::Hash, ops::Index};

#[cfg(feature = "shadowing")]
use crate::{
    structure::abstract_index::{DOWNIND, SELFDUALIND, UPIND},
    structure::slot::SlotError,
};

#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{Atom, AtomCore, FunctionBuilder, Symbol},
    {function, symbol},
};

use thiserror::Error;

use anyhow::Result;
use duplicate::duplicate;

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
        symbol!(Self::NAME)
    }
    fn selfless_dual() -> Self::Dual;
    fn rep<D: Into<Dimension>>(dim: D) -> Representation<Self>
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
            rep: Self::rep(dim),
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
//
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
    #[error("Abstract index error :{0}")]
    AindError(#[from] AbstractIndexError),
    #[error("{0}")]
    DimErr(#[from] DimensionError),
    #[error("{0}")]
    Any(#[from] anyhow::Error),
}

pub trait RepName:
    Copy + Clone + Debug + PartialEq + Eq + Hash + Display + Ord + Into<Rep>
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
    fn to_symbolic(&self, args: impl IntoIterator<Item = Atom>) -> Atom;
    // {
    //     FunctionBuilder::new(symb!(self.to_string()))
    //         .add_args(args)
    //         .finish()
    // }
    //

    #[cfg(feature = "shadowing")]
    fn to_pattern(&self, symbol: Symbol) -> Atom {
        self.to_symbolic([Atom::new_var(symbol)])
    }

    fn new_slot<D: Into<Dimension>, A: Into<AbstractIndex>>(self, dim: D, aind: A) -> Slot<Self>
    where
        Self: Sized,
    {
        Slot {
            rep: self.new_rep(dim),
            aind: aind.into(),
        }
    }

    fn new_rep<D: Into<Dimension>>(&self, dim: D) -> Representation<Self>
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

duplicate! {
   [isnotselfdual repconst repconstdual constname varname varnamedual dualconstname;
       [Lorentz] [LORENTZ_UP] [LORENTZ_DOWN][ExtendibleReps::lor_name()] [LorentzUp] [LorentzDown] ["lord"];
    [SpinFundamental] [SPINFUND] [LORENTZ_DOWN][ExtendibleReps::spf_name()] [SpinFund] [SpinAntiFund] ["spina"];
    [ColorFundamental] [COLORFUND] [LORENTZ_DOWN][ExtendibleReps::cof_name()] [ColorFund] [ColorAntiFund] ["coaf"];
    [ColorSextet]  [COLORSEXT][LORENTZ_DOWN][ExtendibleReps::cos_name()] [ColorSextet] [ColorAntiSextet]["coas"]]

   impl From<isnotselfdual> for Rep {
        fn from(_rep: isnotselfdual) -> Self {
            ExtendibleReps::repconst
        }
    }

    impl From<Dual<isnotselfdual>> for Rep {
         fn from(_rep: Dual<isnotselfdual>) -> Self {
             ExtendibleReps::repconstdual
         }
     }
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,Default)]
    pub struct isnotselfdual {}

    #[allow(unused_variables)]
    impl BaseRepName for isnotselfdual {
        const NAME: &'static str = constname;

        fn selfless_base() -> Self::Base {
            Self::default()
        }

        fn selfless_pair() -> DualPair<Self::Base> {
            DualPair::Rep(Self::default())
        }

        fn selfless_dual() -> Self::Dual {
            Dual{inner:isnotselfdual::default()}
        }
    }

    impl RepName for isnotselfdual {
    type Base = isnotselfdual;
    type Dual = Dual<isnotselfdual>;

    fn base(&self) -> Self::Base {
        isnotselfdual::selfless_base()
    }

    fn is_base(&self) -> bool {
        true
    }

    fn is_dual(self) -> bool {
        false
    }

    fn matches(&self, _: &Self::Dual) -> bool {
        true
    }



    fn dual(self) -> Self::Dual {
        isnotselfdual::selfless_dual()
    }
    #[cfg(feature = "shadowing")]
    fn to_symbolic(&self, args: impl IntoIterator<Item=Atom>) -> Atom{
        let mut fun = FunctionBuilder::new(symbol!(constname));
        for a in args {
            fun = fun.add_arg(&a);
        }
        fun.finish()
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol,aind:Symbol) -> Result<Self,RepresentationError> {
        let uind = symbol!(UPIND);
        let dind = symbol!(DOWNIND);
        let sind = symbol!(SELFDUALIND);
        if aind == uind {
            if Self::selfless_symbol() == sym {
                Ok(isnotselfdual::default())
            } else {
                Err(RepresentationError::NotRepresentationError(sym))
            }
        } else if aind == dind && aind == sind {
            Err(RepresentationError::ExpectedDualStateError(uind,aind))
        } else {
            Err(RepresentationError::SymbolError(aind))
        }

    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol_coerced(sym: Symbol) -> Result<Self, RepresentationError> {
        if Self::selfless_symbol() == sym {
            Ok(isnotselfdual::default())
        } else {
            Err(RepresentationError::NotRepresentationError(sym))
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


    }

    impl RepName for Dual<isnotselfdual>{
        type Base = isnotselfdual;
        type Dual = isnotselfdual;

        fn dual(self) -> Self::Dual {
            self.inner
        }

        fn is_base(&self) -> bool {
            false
        }

        fn is_dual(self) -> bool {
            true
        }


        fn base(&self) -> Self::Base {
            self.inner
        }

        fn matches(&self, _: &Self::Dual) -> bool {
        true
    }
        #[cfg(feature = "shadowing")]
        fn try_from_symbol(sym: Symbol,aind:Symbol) -> Result<Self,RepresentationError> {
            let uind = symbol!(UPIND);
            let dind = symbol!(DOWNIND);
            let sind = symbol!(SELFDUALIND);
            if aind == dind {
                if symbol!(constname) == sym {
                    Ok(Dual{inner:isnotselfdual::default()})
                } else {
                    Err(RepresentationError::NotRepresentationError(sym))
                }
            } else if aind == uind && aind == sind{
                Err(RepresentationError::ExpectedDualStateError(dind,aind))
            } else {
                Err(RepresentationError::SymbolError(aind))
            }

        }


        #[cfg(feature = "shadowing")]
        fn try_from_symbol_coerced(sym: Symbol) -> Result<Self, RepresentationError> {
            if Self::selfless_symbol() == sym {
                Ok(Dual{inner:isnotselfdual::default()})
            } else {
                Err(RepresentationError::NotRepresentationError(sym))
            }
        }

        #[cfg(feature = "shadowing")]
        fn to_symbolic(&self, args: impl IntoIterator<Item= Atom>) -> Atom{
            let mut fun = FunctionBuilder::new(symbol!(constname));
            for a in args {
                fun = fun.add_arg(&a);
            }
            FunctionBuilder::new(symbol!(DOWNIND)).add_arg(fun.finish()).finish()
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

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct Minkowski {}

impl From<Minkowski> for Rep {
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
    fn to_symbolic(&self, args: impl IntoIterator<Item = Atom>) -> Atom {
        let mut fun = FunctionBuilder::new(symbol!(ExtendibleReps::mink_name()));
        for a in args {
            fun = fun.add_arg(&a);
        }
        fun.finish()
    }
}

impl Display for Minkowski {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mink")
    }
}

impl From<Minkowski> for PhysReps {
    fn from(value: Minkowski) -> Self {
        PhysReps::Minkowski(value)
    }
}

impl From<Slot<Minkowski>> for PhysicalSlots {
    fn from(value: Slot<Minkowski>) -> Self {
        value.cast() //RecSlotEnum::A(value.dual_pair()).into()
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

    fn selfless_pair() -> DualPair<Self::Base> {
        DualPair::Rep(Self::default())
    }

    fn selfless_dual() -> Self::Dual {
        Self::default()
    }
}

duplicate! {
   [isselfdual constrep constname;
       [Euclidean] [EUCLIDEAN][ExtendibleReps::euc_name()];
    [Bispinor][BISPINOR] [ExtendibleReps::bis_name()];
    [ColorAdjoint][COLORADJ] [ExtendibleReps::coad_name()]
   ]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,Default)]
    pub struct isselfdual {}

    impl From<isselfdual> for Rep {
        fn from(_value: isselfdual) -> Self {
            ExtendibleReps::constrep
        }
    }
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

    }

    impl RepName for isselfdual {
    type Base = isselfdual;
    type Dual = isselfdual;

    fn base(&self) -> Self::Base {
        isselfdual::selfless_base()
    }

    fn is_base(&self) -> bool {
        true
    }

    fn dual(self) -> Self::Dual {
        isselfdual::selfless_dual()
    }

    fn is_dual(self) -> bool {
        true
    }

    fn matches(&self, _: &Self::Dual) -> bool {
        true
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol, aind: Symbol) -> Result<Self, RepresentationError> {
        let uind = symbol!(UPIND);
        let dind = symbol!(DOWNIND);
        let sind = symbol!(SELFDUALIND);
        if aind == sind {
            if Self::selfless_symbol() == sym {
                Ok(isselfdual::default())
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
            Ok(isselfdual::default())
        } else {
            Err(RepresentationError::NotRepresentationError(sym))
        }}

    #[cfg(feature = "shadowing")]
    fn to_symbolic(&self, args: impl IntoIterator<Item=Atom>) -> Atom{
        let mut fun = FunctionBuilder::new(symbol!(constname));
        for a in args {
            fun = fun.add_arg(&a);
        }
        fun.finish()
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
    Minkowski(Minkowski),
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
            Self::Bispinor(a) => write!(f, "{a}"),
            Self::Minkowski(a) => write!(f, "{a}"),
            Self::Euclidean(a) => write!(f, "{a}"),
            Self::ColorAdjoint(a) => write!(f, "{a}"),
            Self::ColorFund(a) => write!(f, "{a}"),
            Self::ColorAntiFund(a) => write!(f, "{a}"),
            Self::ColorSextet(a) => write!(f, "{a}"),
            Self::ColorAntiSextet(a) => write!(f, "{a}"),
            Self::SpinFund(a) => write!(f, "{a}"),
            Self::SpinAntiFund(a) => write!(f, "{a}"),
            Self::LorentzUp(a) => write!(f, "{a}"),
            Self::LorentzDown(a) => write!(f, "{a}"),
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

    fn is_base(&self) -> bool {
        matches!(
            self,
            PhysReps::Euclidean(_)
                | PhysReps::Minkowski(_)
                | PhysReps::SpinFund(_)
                | PhysReps::ColorFund(_)
                | PhysReps::ColorSextet(_)
                | PhysReps::Bispinor(_)
                | PhysReps::ColorAdjoint(_)
                | PhysReps::LorentzUp(_)
        )
    }

    fn is_dual(self) -> bool {
        matches!(
            self,
            PhysReps::LorentzDown(_)
                | PhysReps::SpinAntiFund(_)
                | PhysReps::ColorAntiFund(_)
                | PhysReps::ColorAntiSextet(_)
                | PhysReps::Bispinor(_)
                | PhysReps::Euclidean(_)
                | PhysReps::Minkowski(_)
                | PhysReps::ColorAdjoint(_)
        )
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
                | (PhysReps::Minkowski(_), PhysReps::Minkowski(_))
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
    fn to_symbolic(&self, args: impl IntoIterator<Item = Atom>) -> Atom {
        match self {
            Self::Bispinor(b) => b.to_symbolic(args),
            Self::Minkowski(b) => b.to_symbolic(args),
            Self::Euclidean(b) => b.to_symbolic(args),
            Self::ColorAdjoint(b) => b.to_symbolic(args),
            Self::ColorFund(b) => b.to_symbolic(args),
            Self::ColorAntiFund(b) => b.to_symbolic(args),
            Self::ColorSextet(b) => b.to_symbolic(args),
            Self::ColorAntiSextet(b) => b.to_symbolic(args),
            Self::SpinFund(b) => b.to_symbolic(args),
            Self::SpinAntiFund(b) => b.to_symbolic(args),
            Self::LorentzUp(b) => b.to_symbolic(args),
            Self::LorentzDown(b) => b.to_symbolic(args),
        }
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol(sym: Symbol, aind: Symbol) -> Result<Self, RepresentationError> {
        match aind.get_name() {
            UPIND => match sym.get_name() {
                Euclidean::NAME | Minkowski::NAME | Bispinor::NAME | ColorAdjoint::NAME => Err(
                    RepresentationError::ExpectedDualStateError(symbol!(SELFDUALIND), aind),
                ),
                Lorentz::NAME => Ok(Self::LorentzUp(Lorentz {})),
                SpinFundamental::NAME => Ok(Self::SpinFund(SpinFundamental {})),
                ColorFundamental::NAME => Ok(Self::ColorFund(ColorFundamental {})),
                ColorSextet::NAME => Ok(Self::ColorSextet(ColorSextet {})),
                _ => Err(RepresentationError::NotRepresentationError(sym)),
            },
            DOWNIND => match sym.get_name() {
                Euclidean::NAME | Minkowski::NAME | Bispinor::NAME | ColorAdjoint::NAME => Err(
                    RepresentationError::ExpectedDualStateError(symbol!(SELFDUALIND), aind),
                ),
                Lorentz::NAME => Ok(Self::LorentzDown(Lorentz {}.dual())),
                SpinFundamental::NAME => Ok(Self::SpinAntiFund(SpinFundamental {}.dual())),
                ColorFundamental::NAME => Ok(Self::ColorAntiFund(ColorFundamental {}.dual())),
                ColorSextet::NAME => Ok(Self::ColorAntiSextet(ColorSextet {}.dual())),
                _ => Err(RepresentationError::NotRepresentationError(sym)),
            },
            SELFDUALIND => match sym.get_name() {
                Lorentz::NAME
                | SpinFundamental::NAME
                | ColorFundamental::NAME
                | ColorSextet::NAME => Err(RepresentationError::ExpectedDualStateError(
                    symbol!(SELFDUALIND),
                    aind,
                )),
                Euclidean::NAME => Ok(Self::Euclidean(Euclidean::default())),
                Minkowski::NAME => Ok(Self::Minkowski(Minkowski::default())),
                Bispinor::NAME => Ok(Self::Bispinor(Bispinor::default())),
                ColorAdjoint::NAME => Ok(Self::ColorAdjoint(ColorAdjoint::default())),
                _ => Err(RepresentationError::NotRepresentationError(sym)),
            },

            _ => Err(RepresentationError::SymbolError(aind)),
        }
    }

    #[cfg(feature = "shadowing")]
    fn try_from_symbol_coerced(sym: Symbol) -> Result<Self, RepresentationError> {
        match sym.get_name() {
            Euclidean::NAME => Ok(Self::Euclidean(Euclidean::default())),
            Minkowski::NAME => Ok(Self::Minkowski(Minkowski::default())),
            Bispinor::NAME => Ok(Self::Bispinor(Bispinor::default())),
            ColorAdjoint::NAME => Ok(Self::ColorAdjoint(ColorAdjoint::default())),
            Lorentz::NAME => Ok(Self::LorentzUp(Lorentz {})),
            SpinFundamental::NAME => Ok(Self::SpinFund(SpinFundamental {})),
            ColorFundamental::NAME => Ok(Self::ColorFund(ColorFundamental {})),
            ColorSextet::NAME => Ok(Self::ColorSextet(ColorSextet {})),
            _ => Err(RepresentationError::NotRepresentationError(sym)),
        }
    }

    fn is_neg(self, i: usize) -> bool {
        match self {
            Self::Minkowski(m) => m.is_neg(i),
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
pub enum Rep {
    SelfDual(u16),
    Dualizable(i16),
}

pub(crate) static REPS: Lazy<RwLock<ExtendibleReps>> =
    Lazy::new(|| RwLock::new(ExtendibleReps::new()));
static SELF_DUAL: AppendOnlyVec<RepData> = AppendOnlyVec::new();
static DUALIZABLE: AppendOnlyVec<RepData> = AppendOnlyVec::new();

impl Rep {
    pub fn new_dual(name: &str) -> Result<Self, RepError> {
        REPS.write().unwrap().new_dual_impl(name)
    }

    pub fn new_self_dual(name: &str) -> Result<Self, RepError> {
        REPS.write().unwrap().new_self_dual(name)
    }
}

impl From<PhysReps> for Rep {
    fn from(value: PhysReps) -> Self {
        match value {
            PhysReps::ColorFund(a) => a.into(),
            PhysReps::Minkowski(a) => a.into(),
            PhysReps::ColorAntiFund(a) => a.into(),
            PhysReps::ColorSextet(a) => a.into(),
            PhysReps::ColorAntiSextet(a) => a.into(),
            PhysReps::SpinFund(a) => a.into(),
            PhysReps::SpinAntiFund(a) => a.into(),
            PhysReps::Bispinor(a) => a.into(),
            PhysReps::ColorAdjoint(a) => a.into(),
            PhysReps::LorentzUp(a) => a.into(),
            PhysReps::LorentzDown(a) => a.into(),
            PhysReps::Euclidean(a) => a.into(),
        }
    }
}

pub struct RepData {
    // metric_data: Fn(Dimension)->SparseTensor<i8,IndexLess>
    name: String,
    #[cfg(feature = "shadowing")]
    symbol: Symbol,
}

pub struct ExtendibleReps {
    name_map: AHashMap<String, Rep>,
    #[cfg(feature = "shadowing")]
    symbol_map: AHashMap<Symbol, Rep>,
}

#[derive(Debug, Error)]
pub enum RepError {
    #[error("{0} Already exists and is of different type")]
    AlreadyExistsDifferentType(String),
}

impl ExtendibleReps {
    pub fn reps(&self) -> impl Iterator<Item = &Rep> {
        self.name_map.values()
    }
    pub fn new_dual_impl(&mut self, name: &str) -> Result<Rep, RepError> {
        if let Some(rep) = self.name_map.get(name) {
            if let Rep::SelfDual(_) = rep {
                return Err(RepError::AlreadyExistsDifferentType(name.into()));
            } else {
                return Ok(*rep);
            }
        }
        let rep = Rep::Dualizable(DUALIZABLE.len() as i16 + 1);

        self.name_map.insert(name.into(), rep);
        #[cfg(feature = "shadowing")]
        let symbol = symbol!(name);
        #[cfg(feature = "shadowing")]
        self.symbol_map.insert(symbol, rep);

        DUALIZABLE.push(RepData {
            name: name.to_string(),
            #[cfg(feature = "shadowing")]
            symbol,
        });
        Ok(rep)
    }

    pub fn new_dual(name: &str) -> Result<Rep, RepError> {
        REPS.write().unwrap().new_dual_impl(name)
    }

    pub fn new_self_dual(&mut self, name: &str) -> Result<Rep, RepError> {
        if let Some(rep) = self.name_map.get(name) {
            if let Rep::Dualizable(_) = rep {
                return Err(RepError::AlreadyExistsDifferentType(name.into()));
            } else {
                return Ok(*rep);
            }
        }

        let rep = Rep::SelfDual(SELF_DUAL.len() as u16);
        self.name_map.insert(name.into(), rep);
        #[cfg(feature = "shadowing")]
        let symbol = symbol!(name);
        #[cfg(feature = "shadowing")]
        self.symbol_map.insert(symbol, rep);

        SELF_DUAL.push(RepData {
            name: name.to_string(),
            #[cfg(feature = "shadowing")]
            symbol,
        });
        Ok(rep)
    }
}

impl Index<Rep> for ExtendibleReps {
    type Output = RepData;

    fn index(&self, index: Rep) -> &Self::Output {
        match index {
            Rep::SelfDual(l) => &SELF_DUAL[l as usize],
            Rep::Dualizable(l) => &DUALIZABLE[l.unsigned_abs() as usize - 1],
        }
    }
}

impl ExtendibleReps {
    pub const EUCLIDEAN: Rep = Rep::SelfDual(0);
    pub const BISPINOR: Rep = Rep::SelfDual(1);
    pub const COLORADJ: Rep = Rep::SelfDual(2);
    pub const MINKOWSKI: Rep = Rep::SelfDual(3);

    pub const LORENTZ_UP: Rep = Rep::Dualizable(1);
    pub const LORENTZ_DOWN: Rep = Rep::Dualizable(-1);
    pub const SPINFUND: Rep = Rep::Dualizable(2);
    pub const SPINANTIFUND: Rep = Rep::Dualizable(-2);
    pub const COLORFUND: Rep = Rep::Dualizable(3);
    pub const COLORANTIFUND: Rep = Rep::Dualizable(-3);
    pub const COLORSEXT: Rep = Rep::Dualizable(4);
    pub const COLORANTISEXT: Rep = Rep::Dualizable(-4);

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

        for &name in Self::BUILTIN_SELFDUAL_NAMES.iter() {
            new.new_self_dual(name).unwrap();
        }

        for &name in Self::BUILTIN_DUALIZABLE_NAMES.iter() {
            new.new_dual_impl(name).unwrap();
        }
        new
    }

    #[cfg(feature = "shadowing")]
    pub fn find_symbol(&self, symbol: Symbol) -> Option<Rep> {
        self.symbol_map.get(&symbol).cloned()
    }
}

impl Default for ExtendibleReps {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Rep {
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

impl RepName for Rep {
    type Dual = Rep;
    type Base = Rep;

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
            Rep::Dualizable(_) => {
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
            Rep::SelfDual(_) => {
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

    fn to_symbolic(&self, args: impl IntoIterator<Item = Atom>) -> Atom {
        let mut fun = FunctionBuilder::new(REPS.read().unwrap()[*self].symbol);
        for a in args {
            fun = fun.add_arg(&a);
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
    let r = Rep::new_dual("lor").unwrap();
    let rd = r.dual();
    let e = Rep::new_self_dual("euc").unwrap();

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
    let spin: Representation<Bispinor> = Bispinor::rep(5);

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

impl<R: BaseRepName<Dual: BaseRepName>> From<DualPair<R>> for Rep {
    fn from(value: DualPair<R>) -> Self {
        match value {
            DualPair::Rep(r) => r.into(),
            DualPair::DualRep(d) => d.into(),
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

    fn dual(self) -> Self::Dual {
        match self {
            Self::Rep(r) => Self::DualRep(r.dual()),
            Self::DualRep(d) => Self::Rep(d.dual()),
        }
    }

    fn is_base(&self) -> bool {
        match self {
            Self::Rep(r) => r.is_base(),
            Self::DualRep(d) => d.is_base(),
        }
    }

    fn is_dual(self) -> bool {
        match self {
            Self::Rep(r) => r.is_dual(),
            Self::DualRep(d) => d.is_dual(),
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
    fn try_from_symbol(sym: Symbol, aind: Symbol) -> Result<Self, RepresentationError> {
        if let Ok(r) = R::try_from_symbol(sym, aind) {
            Ok(DualPair::Rep(r))
        } else if let Ok(d) = R::Dual::try_from_symbol(sym, aind) {
            Ok(DualPair::DualRep(d))
        } else {
            Err(RepresentationError::NotRepresentationError(sym))
        }
    }
    #[cfg(feature = "shadowing")]
    fn try_from_symbol_coerced(sym: Symbol) -> Result<Self, RepresentationError> {
        if let Ok(r) = R::try_from_symbol_coerced(sym) {
            Ok(DualPair::Rep(r))
        } else if let Ok(d) = R::Dual::try_from_symbol_coerced(sym) {
            Ok(DualPair::DualRep(d))
        } else {
            Err(RepresentationError::NotRepresentationError(sym))
        }
    }
    #[cfg(feature = "shadowing")]
    fn to_symbolic(&self, args: impl IntoIterator<Item = Atom>) -> Atom {
        match self {
            Self::Rep(r) => r.to_symbolic(args),
            Self::DualRep(d) => d.to_symbolic(args),
        }
    }
}
#[cfg(test)]
#[cfg(feature = "shadowing")]
mod shadowing_tests {
    use symbolica::symbol;

    use crate::structure::representation::{BaseRepName, Dual};

    use super::Lorentz;

    #[test]
    fn rep_pattern() {
        println!("{}", Dual::<Lorentz>::pattern(symbol!("d_")));
        println!(
            "{}",
            Dual::<Lorentz>::rep(3).to_pattern_wrapped(symbol!("d_"))
        );
    }
}

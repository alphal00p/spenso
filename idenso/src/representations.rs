use spenso::structure::{
    abstract_index::AIND_SYMBOLS,
    representation::{Euclidean, Minkowski, RepName},
};
use spenso_macros::SimpleRepresentation;
use symbolica::atom::Atom;

use super::{color::CS, gamma::AGS, metric::MS, rep_symbols::RS};

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
)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
)]
#[representation(name = "spf", dual_name = "SpinAntiFundamental")] // Specify the dual name
pub struct SpinFundamental {}

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
)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
)]
#[representation(name = "lor", dual_name = "LorentzUp")] // Specify the dual name
pub struct Lorentz {}

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
)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
)]
#[representation(name = "cof", dual_name = "ColorAntiFundamental")] // Specify the dual name
pub struct ColorFundamental {}

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
)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
)]
#[representation(name = "cos", dual_name = "ColorAntiSextet")] // Specify the dual name
pub struct ColorSextet {}

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
)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
)]
#[representation(name = "bis", self_dual)] // Specify the dual name
pub struct Bispinor {}

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
)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
)]
#[representation(name = "coad", self_dual)] // Specify the dual name
pub struct ColorAdjoint {}

pub fn initialize() {
    let _ = AIND_SYMBOLS.dind;
    let _ = Minkowski {}.to_symbolic([Atom::Zero]);
    let _ = Euclidean {}.to_symbolic([Atom::Zero]);
    let _ = Lorentz {}.to_symbolic([Atom::Zero]);
    let _ = SpinFundamental {}.to_symbolic([Atom::Zero]);
    let _ = Bispinor {}.to_symbolic([Atom::Zero]);
    let _ = ColorAdjoint {}.to_symbolic([Atom::Zero]);
    let _ = ColorFundamental {}.to_symbolic([Atom::Zero]);
    let _ = ColorSextet {}.to_symbolic([Atom::Zero]);
    let _ = RS.a_;
    let _ = MS.dot;
    let _ = AGS.gamma;
    let _ = CS.f;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_spin_fundamental() {
        let spin_fundamental = SpinFundamental::default();

        // let dual = spin_fundamental.dual();

        assert_eq!(spin_fundamental.dual(), SpinAntiFundamental {});
        // assert_eq!(spin_fundamental.dual_name(), "SpinAntiFundamental");
    }
}

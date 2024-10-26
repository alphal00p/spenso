use std::ops::Deref;

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
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::{AsAtomView, Atom, FunctionBuilder},
    state::State,
    {fun, symb},
};

use serde::{Deserialize, Serialize};

pub const CONCRETEIND: &str = "cind";
pub const FLATIND: &str = "find";
pub const UP: &str = "u";
pub const DOWN: &str = "d";

/// A  concrete index, i.e. the concrete usize/index of the corresponding abstract index
pub type ConcreteIndex = usize;

#[derive(
    Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Serialize, Deserialize, Display,
)]
pub enum DualConciousIndex {
    Up(ConcreteIndex),
    Down(ConcreteIndex),
    SelfDual(ConcreteIndex),
}

#[cfg(feature = "shadowing")]
impl From<DualConciousIndex> for Atom {
    fn from(value: DualConciousIndex) -> Self {
        match value {
            DualConciousIndex::Up(s) => Atom::new_num(s as i64),
            DualConciousIndex::Down(s) => fun!(symb!(DOWN), Atom::new_num(s as i64)),
            DualConciousIndex::SelfDual(s) => Atom::new_num(s as i64),
        }
    }
}

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
pub struct DualConciousExpandedIndex {
    indices: Vec<DualConciousIndex>,
}

impl Deref for DualConciousExpandedIndex {
    type Target = [DualConciousIndex];

    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}

#[cfg(feature = "shadowing")]
impl From<DualConciousExpandedIndex> for Atom {
    fn from(value: DualConciousExpandedIndex) -> Self {
        let mut cind = FunctionBuilder::new(State::get_symbol(CONCRETEIND));
        for i in value.iter() {
            cind = cind.add_arg(Atom::from(*i).as_atom_view());
        }
        cind.finish()
    }
}

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

// #[cfg(feature = "shadowing")]
// impl From<ExpandedIndex> for Atom {
//     fn from(value: ExpandedIndex) -> Self {
//         let mut cind = FunctionBuilder::new(State::get_symbol(CONCRETEIND));
//         for i in value.iter() {
//             cind = cind.add_arg(Atom::new_num(*i as i64).as_atom_view());
//         }
//         cind.finish()
//     }
// }

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
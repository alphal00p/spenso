use crate::structure::concrete_index::ConcreteIndex;
use derive_more::Display;
use serde::{Deserialize, Serialize};
use smartstring::{LazyCompact, SmartString};
use symbolica::{
    atom::{AsAtomView, Atom, AtomView, FunctionBuilder, Symbol},
    id::{Pattern, PatternOrMap, Replacement},
    state::State,
};

extern crate derive_more;

use std::{
    fmt::{Debug, Display},
    io::Cursor,
};

use anyhow::Result;

// use anyhow::Ok;
use serde::ser::SerializeStruct;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Display)]
pub struct SerializableSymbol {
    symbol: Symbol,
}

impl Serialize for SerializableSymbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.symbol.to_string().serialize(serializer)
    }
}

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

impl From<Symbol> for SerializableSymbol {
    fn from(value: Symbol) -> Self {
        Self { symbol: value }
    }
}

impl From<SerializableSymbol> for Symbol {
    fn from(value: SerializableSymbol) -> Self {
        value.symbol
    }
}

impl From<SerializableSymbol> for u32 {
    fn from(value: SerializableSymbol) -> Self {
        value.symbol.get_id()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SerializableAtom(pub Atom);

impl SerializableAtom {
    pub fn replace_repeat(&mut self, lhs: Pattern, rhs: PatternOrMap) {
        let atom = self.0.replace_all(&lhs, &rhs, None, None);
        if atom != self.0 {
            self.0 = atom;
            self.replace_repeat(lhs, rhs);
        }
    }

    pub fn replace_repeat_multiple(&mut self, reps: &[Replacement<'_>]) {
        let atom = self.0.replace_all_multiple(reps);
        // info!("expanded rep");
        if atom != self.0 {
            // info!("applied replacement");
            self.0 = atom;
            self.replace_repeat_multiple(reps);
        }
    }

    pub fn replace_repeat_multiple_atom(expr: &mut Atom, reps: &[Replacement<'_>]) {
        let atom = expr.replace_all_multiple(reps);
        if atom != *expr {
            *expr = atom;
            Self::replace_repeat_multiple_atom(expr, reps)
        }
    }

    pub fn replace_repeat_multiple_atom_expand(expr: &mut Atom, reps: &[Replacement<'_>]) {
        let a = expr.expand();
        let atom = a.replace_all_multiple(reps);
        if atom != *expr {
            *expr = atom;
            Self::replace_repeat_multiple_atom_expand(expr, reps)
        }
    }
}

impl Display for SerializableAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Serialize for SerializableAtom {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("SerializableAtom", 3)?;

        let mut serialized_atom: Vec<u8> = Vec::new();

        self.0.as_view().write(&mut serialized_atom).unwrap();

        state.serialize_field("atom", &serialized_atom)?;

        let mut symbolica_state = Vec::new();

        State::export(&mut symbolica_state).unwrap();

        state.serialize_field("state", &symbolica_state)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SerializableAtom {
    fn deserialize<D>(deserializer: D) -> std::result::Result<SerializableAtom, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SerializableAtomHelper {
            atom: Vec<u8>,
            state: Vec<u8>,
        }

        let helper = SerializableAtomHelper::deserialize(deserializer)?;

        let state = helper.state;

        let map = State::import(&mut Cursor::new(&state), None).unwrap();

        let atom = Atom::import_with_map(Cursor::new(&helper.atom), &map).unwrap();

        Ok(SerializableAtom(atom))
    }
}

impl<'a> TryFrom<AtomView<'a>> for SerializableAtom {
    type Error = anyhow::Error;

    fn try_from(value: AtomView<'a>) -> Result<Self> {
        Ok(SerializableAtom(value.to_owned()))
    }
}

impl From<Atom> for SerializableAtom {
    fn from(atom: Atom) -> Self {
        SerializableAtom(atom)
    }
}

impl From<SerializableAtom> for Atom {
    fn from(atom: SerializableAtom) -> Self {
        atom.0
    }
}

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

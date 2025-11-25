use crate::{
    network::parsing::SPENSO_TAG, structure::concrete_index::ConcreteIndex,
    tensors::parametric::atomcore::PatternReplacement,
};
use derive_more::Display;
use serde::{Deserialize, Serialize};
use symbolica::{
    atom::{
        AddView, Atom, AtomCore, AtomView, FunctionBuilder, MulView, NumView, PowView, Symbol,
        VarView, representation::FunView,
    },
    coefficient::CoefficientView,
    domains::{float::Complex, rational::Rational},
    id::Context,
    printer::{CanonicalOrderingSettings, PrintOptions, PrintState},
    state::State,
    symbol,
    utils::Settable,
};

use symbolica::domains::SelfRing;

extern crate derive_more;

use std::{
    fmt::{self, Debug, Display, Error},
    io::Cursor,
};

use anyhow::Result;

pub trait AtomCoreExt {
    fn to_bare_ordered_string(&self) -> String;

    fn typst(&self) -> String;

    fn is_upper(&self) -> bool;
    fn is_lower(&self) -> bool;
}

impl<A: AtomCore> AtomCoreExt for A {
    fn to_bare_ordered_string(&self) -> String {
        self.to_canonically_ordered_string(CanonicalOrderingSettings {
            include_namespace: false,
            include_attributes: false,
            hide_namespace: None,
        })
    }

    fn typst(&self) -> String {
        let mut out = String::new();
        self.as_atom_view()
            .fmt_output(
                &mut out,
                &PrintOptions {
                    custom_print_mode: Some(("typst", 1)),
                    ..Default::default()
                },
                PrintState::new(),
            )
            .unwrap();
        out
    }

    fn is_upper(&self) -> bool {
        match self.as_atom_view() {
            AtomView::Fun(a) => a.get_symbol().has_tag(&SPENSO_TAG.upper),
            AtomView::Var(a) => a.get_symbol().has_tag(&SPENSO_TAG.upper),
            _ => false,
        }
    }

    fn is_lower(&self) -> bool {
        match self.as_atom_view() {
            AtomView::Fun(a) => a.get_symbol().has_tag(&SPENSO_TAG.lower),
            AtomView::Var(a) => a.get_symbol().has_tag(&SPENSO_TAG.lower),
            _ => false,
        }
    }
}
pub struct Typst;

pub trait FormatWithState {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, Error>;
}

impl FormatWithState for AtomView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        fmt: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        match self {
            AtomView::Num(n) => n.as_view().format(fmt, opts, print_state),
            AtomView::Var(v) => v.as_view().format(fmt, opts, print_state),
            AtomView::Fun(f) => f.fmt_output(fmt, opts, print_state),
            AtomView::Pow(p) => p.fmt_output(fmt, opts, print_state),
            AtomView::Mul(t) => t.fmt_output(fmt, opts, print_state),
            AtomView::Add(e) => e.fmt_output(fmt, opts, print_state),
        }
    }
}

impl FormatWithState for FunView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            f.write_char('+')?;
        }

        let id = self.get_symbol();
        // if let Some(custom_print) = &id.fo {
        //     if let Some(s) = custom_print(self.as_view(), opts) {
        //         f.write_str(&s)?;
        //         return Ok(false);
        //     }
        // }
        let mut uppers = vec![];
        let mut lowers = vec![];
        // f.write_str("attach(")?;
        for a in self.iter() {
            if a.is_upper() {
                let mut out = String::new();
                a.fmt_output(&mut out, opts, print_state.clone())?;
                let out_low = format!("#hide(${}$)", out);
                uppers.push(out);
                lowers.push(out_low);
            }

            if a.is_lower() {
                let mut out = String::new();
                a.fmt_output(&mut out, opts, print_state.clone())?;
                let out_low = format!("#hide(${}$)", out);
                lowers.push(out);
                uppers.push(out_low);
            }
        }

        if uppers.is_empty() {
            f.write_str("op(\"")?;
            self.get_symbol().format(opts, f)?;
            f.write_str("\")")?;
            let n_args = self.get_nargs();

            if n_args > 0 {
                f.write_char('(')?;
            }
            for (i, a) in self.iter().enumerate() {
                if i + 1 < n_args {
                    f.write_char(',')?;
                }
                a.fmt_output(f, opts, print_state)?;
            }
            if n_args > 0 {
                f.write_char(')')?;
            }
        } else {
            f.write_str("scripts(attach(")?;
            f.write_str("op(\"")?;

            self.get_symbol().format(opts, f)?;
            f.write_str("\")")?;
            f.write_str(", tr: ")?;
            f.write_str(&uppers.join(" "))?;
            f.write_str(", br: ")?;
            f.write_str(&lowers.join(" "))?;
            f.write_char(')')?;
            f.write_char(')')?;
            let n_args = self.get_nargs() - lowers.len();

            if n_args > 0 {
                f.write_char('(')?;
            }

            for (i, a) in self
                .iter()
                .filter(|a| !(a.is_lower() || a.is_upper()))
                .enumerate()
            {
                if i + 1 < n_args {
                    f.write_char(',')?;
                }
                a.fmt_output(f, opts, print_state.clone())?;
            }
            // f.write_char(')');
            if n_args > 0 {
                f.write_char(')')?;
            }
        }
        Ok(false)
    }
}

impl FormatWithState for MulView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        let add_paren = print_state.in_exp || print_state.in_exp_base;
        if add_paren {
            if print_state.in_sum {
                print_state.in_sum = false;
                f.write_char('+')?;
            }

            f.write_char('(')?;
            print_state.in_exp = false;
            print_state.in_exp_base = false;
        }

        print_state.in_product = true;

        // write the coefficient first
        let mut first = true;
        let mut skip_num = false;
        if let Some(AtomView::Num(n)) = self.iter().last() {
            print_state.suppress_one = true;
            first = n.as_view().format(f, opts, print_state)?;
            print_state.suppress_one = false;
            skip_num = true;
        } else if print_state.in_sum {
            f.write_char('+')?;
        }

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;

        for x in self.iter().take(if skip_num {
            self.get_nargs() - 1
        } else {
            self.get_nargs()
        }) {
            if !first {
                f.write_char(' ')?;
            }
            first = false;

            x.fmt_output(f, opts, print_state)?;
        }

        if add_paren {
            f.write_char(')')?;
        }
        Ok(false)
    }
}

impl FormatWithState for PowView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            f.write_char('+')?;
        }

        let add_paren = print_state.in_exp_base; // right associative
        if add_paren {
            f.write_char('(')?;
            print_state.in_exp = false;
            print_state.in_exp_base = false;
        }

        let b = self.get_base();
        let e = self.get_exp();

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;
        print_state.in_product = false;
        print_state.suppress_one = false;

        if let AtomView::Num(n) = e
            && n.get_coeff_view() == CoefficientView::Natural(-1, 1, 0, 1)
        {
            // TODO: construct the numerator
            f.write_str("1/(")?;
            b.fmt_output(f, opts, print_state)?;
            f.write_char(')')?;
            return Ok(false);
        }

        print_state.in_exp_base = true;

        b.fmt_output(f, opts, print_state)?;

        print_state.in_exp_base = false;
        print_state.in_exp = true;

        f.write_char('^')?;

        f.write_char('(')?;
        print_state.in_exp = false;
        e.fmt_output(f, opts, print_state)?;
        f.write_char(')')?;

        if add_paren {
            f.write_char(')')?;
        }

        Ok(false)
    }
}

impl FormatWithState for AddView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        let mut first = true;
        print_state.top_level_add_child = print_state.level == 0;
        print_state.level += 1;
        print_state.suppress_one = false;

        let add_paren = print_state.in_product || print_state.in_exp || print_state.in_exp_base;
        if add_paren {
            if print_state.in_sum {
                f.write_char('+')?;
            }

            print_state.in_sum = false;
            print_state.in_product = false;
            print_state.in_exp = false;
            print_state.in_exp_base = false;

            f.write_char('(')?;
        }

        let mut count = 0;
        for x in self.iter() {
            if !first && print_state.top_level_add_child && opts.terms_on_new_line {
                f.write_char('\n')?;
            }
            first = false;

            x.fmt_output(f, opts, print_state)?;
            print_state.in_sum = true;
            count += 1;
        }

        if opts.max_terms.is_some() && count < self.get_nargs() {
            if print_state.top_level_add_child && opts.terms_on_new_line {
                f.write_char('\n')?;
            }

            f.write_str("+...")?;
        }

        if add_paren {
            f.write_char(')')?;
        }
        Ok(false)
    }
}
// // fn print_fun_view(view:FunView<'_>)-?

// use anyhow::Ok;
use serde::ser::SerializeStruct;

#[derive(
    Debug,
    Copy,
    Clone,
    Ord,
    PartialOrd,
    Eq,
    PartialEq,
    Hash,
    Display,
    bincode_trait_derive::Encode,
    bincode_trait_derive::Decode,
    bincode_trait_derive::BorrowDecodeFromDecode,
)]
#[trait_decode(trait = symbolica::state::HasStateMap)]
pub struct SerializableSymbol {
    symbol: Symbol,
}

impl SerializableSymbol {
    pub fn get_id(&self) -> u32 {
        self.symbol.get_id()
    }

    pub fn get_name(&self) -> &str {
        self.symbol.get_name()
    }
}

impl Serialize for SerializableSymbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.symbol.get_name().serialize(serializer)
    }
}

impl<'d> Deserialize<'d> for SerializableSymbol {
    fn deserialize<D>(deserializer: D) -> Result<SerializableSymbol, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(SerializableSymbol {
            symbol: symbol!(&value),
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

impl PatternReplacement for SerializableAtom {
    fn replace_map_mut<F: Fn(AtomView, &Context, &mut Settable<'_, Atom>)>(&mut self, m: &F) {
        self.0.replace_map_mut(m)
    }

    fn replace_multiple_mut<T: symbolica::id::BorrowReplacement>(&mut self, replacements: &[T]) {
        self.0.replace_multiple_mut(replacements)
    }

    fn replace_multiple_repeat<T: symbolica::id::BorrowReplacement>(
        &self,
        replacements: &[T],
    ) -> Self {
        self.0.replace_multiple_repeat(replacements).into()
    }

    fn replace_multiple_repeat_mut<T: symbolica::id::BorrowReplacement>(
        &mut self,
        replacements: &[T],
    ) {
        self.0.replace_multiple_repeat_mut(replacements)
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
    value_builder = value_builder.add_arg(Atom::num(index as i64).as_atom_view());
    value_builder.finish()
}
#[cfg(feature = "shadowing")]
#[allow(clippy::cast_possible_wrap)]
pub fn atomic_expanded_label_id(indices: &[ConcreteIndex], name: Symbol, args: &[Atom]) -> Atom {
    let mut value_builder = FunctionBuilder::new(name);
    let mut index_func = FunctionBuilder::new(symbol!("cind"));
    for arg in args {
        value_builder = value_builder.add_arg(arg);
    }
    for &index in indices {
        index_func = index_func.add_arg(Atom::num(index as i64).as_atom_view());
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
        std::iter::once(Atom::num(*self as i64))
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

// #[cfg(feature = "shadowing")]
// impl IntoSymbol for String {
//     fn ref_into_symbol(&self) -> Symbol {
//         symbol!(self)
//     }

//     fn from_str(s: &str) -> Self {
//         s.into()
//     }
// }

#[cfg(feature = "shadowing")]
impl IntoSymbol for Symbol {
    fn ref_into_symbol(&self) -> Symbol {
        *self
    }

    fn from_str(s: &str) -> Self {
        symbol!(s)
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for SerializableSymbol {
    fn ref_into_symbol(&self) -> Symbol {
        self.symbol
    }

    fn from_str(s: &str) -> Self {
        Self { symbol: symbol!(s) }
    }
}

#[cfg(feature = "shadowing")]
impl IntoSymbol for std::string::String {
    fn ref_into_symbol(&self) -> Symbol {
        symbol!(self)
    }
    fn from_str(s: &str) -> Self {
        s.into()
    }
}

#[cfg(test)]
mod test {
    use crate::{network::parsing::SPENSO_TAG, shadowing::symbolica_utils::AtomCoreExt};
    use itertools::Itertools;
    use std::fmt::Write;
    use symbolica::{
        atom::{AtomCore, AtomView},
        domains::{
            float::{Complex, FloatLike},
            rational::Rational,
        },
        evaluate::{FunctionMap, Instruction, OptimizationSettings, Slot},
        parse, parse_lit,
        printer::PrintState,
        symbol, tag,
    };
    #[test]
    fn print() {
        let lower = symbol!(
            "lower",
            tag = SPENSO_TAG.lower,
            print = |a, opt| {
                let mut fmt = "".to_string();
                if let AtomView::Fun(f) = a {
                    let n_args = f.get_nargs();
                    for (i, a) in f.iter().enumerate() {
                        a.format(&mut fmt, opt, PrintState::new()).unwrap();
                        if i < n_args - 1 {
                            fmt.push_str(",");
                        }
                    }
                }

                Some(fmt)
            }
        );

        let upper = symbol!(
            "upper",
            tags = [SPENSO_TAG.upper.clone(), tag!("Real")],
            print = |a, opt| {
                let mut fmt = "".to_string();
                if let AtomView::Fun(f) = a {
                    let n_args = f.get_nargs();
                    for (i, a) in f.iter().enumerate() {
                        a.format(&mut fmt, opt, PrintState::new()).unwrap();
                        if i < n_args - 1 {
                            fmt.push_str(",");
                        }
                    }
                }

                Some(fmt)
            } // ; Real
        );

        let expr = parse!(
            "a*f(lower(f(upper(x),lower(y,c))))^(sin(x)*cos(x))*g(x,lower(y),upper(x+1),lower(1))"
        )
        .replace(parse_lit!(_x ^ _y))
        .with(parse_lit!(pow(_x, _y)));

        let mut params = vec![];
        let mut fn_map = FunctionMap::new();

        expr.visitor(&mut |a| {
            if let AtomView::Var(a) = a {
                params.push(a.as_view().to_owned());
                false
            } else if let AtomView::Fun(a) = a {
                let mut tags = vec![];
                for i in a.iter() {
                    if i.is_lower() || i.is_upper() {
                        tags.push(i.to_owned())
                    }
                }
                // fn_map.add_tagged_function(a.get_symbol(),tags, a.get_symbol().get_name().into(), vec![], body)
                fn_map.add_external_function(a.get_symbol(), a.get_symbol().get_name().into());
                true
            } else {
                true
            }
        });
        let eval_tree = expr
            .evaluator(
                &fn_map,
                &params,
                OptimizationSettings {
                    horner_iterations: 10,
                    ..Default::default()
                },
            )
            .unwrap();

        let mut out = String::new();
        writeln!(out, "#{{");
        fn typst_rat(r: &Rational) -> String {
            if r.is_integer() {
                r.numerator().to_string()
            } else {
                format!(" {}/{}", r.numerator(), r.denominator())
            }
        }

        fn typst_slot(
            s: Slot,
            params: &[symbolica::atom::Atom],
            consts: &[Complex<Rational>],
        ) -> String {
            match s {
                Slot::Const(c) => {
                    let Complex { re, im } = &consts[c];

                    match (re.is_zero(), im.is_zero()) {
                        (true, true) => "0".into(),
                        (true, false) => format!("i {}", typst_rat(im)),
                        (false, true) => format!(" {}", typst_rat(re)),
                        _ => format!("({} + i {})", typst_rat(re), typst_rat(im)),
                    }
                }
                Slot::Out(c) => format!("out{c}"),
                Slot::Param(c) => format!("\"{}\"", params[c].typst()),
                Slot::Temp(c) => format!("tmp{c}"),
            }
        }

        let (instr, size, consts) = eval_tree.export_instructions();

        for i in instr {
            match i {
                Instruction::Add(s, args) => {
                    writeln!(
                        out,
                        "let {} = ${}$",
                        typst_slot(s, &params, &consts),
                        args.into_iter()
                            .map(|a| typst_slot(a, &params, &consts))
                            .join(" + ")
                    );
                }
                Instruction::Mul(s, args) => {
                    writeln!(
                        out,
                        "let {} = ${}$",
                        typst_slot(s, &params, &consts),
                        args.into_iter()
                            .map(|a| typst_slot(a, &params, &consts))
                            .join(" ")
                    );
                }
                Instruction::ExternalFun(o, name, args) => {
                    writeln!(
                        out,
                        "let {} = $op(\"{name}\")({})$",
                        typst_slot(o, &params, &consts),
                        args.into_iter()
                            .map(|a| typst_slot(a, &params, &consts))
                            .join(" ")
                    );
                }
                Instruction::Fun(o, builtin, s) => {
                    let name = builtin.get_symbol().get_stripped_name().to_string();
                    writeln!(
                        out,
                        "let {} = $op(\"{name}\")({})$",
                        typst_slot(o, &params, &consts),
                        typst_slot(s, &params, &consts)
                    );
                }
                Instruction::Powf(o, b, e) => {
                    writeln!(
                        out,
                        "let {} = ${}^{}$",
                        typst_slot(o, &params, &consts),
                        typst_slot(b, &params, &consts),
                        typst_slot(e, &params, &consts)
                    );
                }
                _ => {
                    println!("{i:?}")
                }
            }
        }
        writeln!(out, "out0");
        writeln!(out, "}}");

        println!("{}", out);
        println!("{}", expr.typst())
    }
}

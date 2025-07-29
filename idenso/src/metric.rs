use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use spenso::{
    network::library::symbolic::ETS,
    shadowing::symbolica_utils::{IntoArgs, IntoSymbol},
    structure::{
        HasName, PermutedStructure, TensorStructure, ToSymbolic,
        abstract_index::AIND_SYMBOLS,
        permuted::Perm,
        representation::{LibraryRep, LibrarySlot, RepName},
        slot::{AbsInd, DummyAind},
    },
    tensors::symbolic::SymbolicTensor,
};
use symbolica::{
    atom::{Atom, AtomCore, AtomType, AtomView, Symbol, representation::FunView},
    coefficient::CoefficientView,
    function,
    id::{
        Condition, FilterFn, Match, MatchSettings, PatternRestriction, Replacement,
        WildcardRestriction,
    },
    symbol,
};

use crate::parsing_ind::Parsind;

use super::rep_symbols::RS;

pub struct MetricSymbols {
    pub dim: Symbol,
    pub dot: Symbol,
    pub dummy: Symbol,
}

pub static MS: LazyLock<MetricSymbols> = LazyLock::new(|| MetricSymbols {
    dim: symbol!("spenso::dim"),
    dot: symbol!("spenso::dot";Symmetric, Linear),
    dummy: symbol!("spenso::dummy"),
});

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum CookingError {
    Add,
    Mul,
    Pow,
    RatCoeff,
    FiniteField,
    Float,
}

pub fn cook_function_view(view: AtomView) -> Result<Atom, CookingError> {
    match view {
        AtomView::Var(_) | AtomView::Num(_) => Ok(view.to_owned()),
        AtomView::Mul(_) => Err(CookingError::Mul),
        AtomView::Add(_) => Err(CookingError::Add),
        AtomView::Pow(_) => Err(CookingError::Pow),
        AtomView::Fun(f) => {
            let s = cook_function_impl(f)?;
            Ok(Atom::var(s))
        }
    }
}

pub fn cook_function_impl(fun: FunView) -> Result<Symbol, CookingError> {
    let mut name = fun.get_symbol().get_name().to_string();

    for arg in fun.iter() {
        name.push('_');
        match arg {
            AtomView::Fun(f) => {
                let arg_sym = cook_function_impl(f)?;
                name.push_str(arg_sym.get_stripped_name());
            }
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::FiniteField(_, _) => {
                    return Err(CookingError::FiniteField);
                }
                CoefficientView::Natural(n, d, imnum, imden) => {
                    name.push_str(&n.to_string());
                    if d != 1 {
                        name.push(':');
                        name.push_str(&d.to_string());
                    }
                    if imnum != 0 {
                        name.push('i');
                        name.push_str(&imnum.to_string());
                        if d != 1 {
                            name.push(':');
                            name.push_str(&imden.to_string());
                        }
                    }
                }
                CoefficientView::Float(_, _) => {
                    return Err(CookingError::Float);
                }
                CoefficientView::Large(r, imr) => {
                    let rat = r.to_rat();
                    name.push_str(&rat.numerator().to_string());
                    if !rat.is_integer() {
                        name.push(':');
                        name.push_str(&rat.denominator().to_string());
                    }

                    if !imr.is_zero() {
                        let rat = imr.to_rat();
                        name.push('i');
                        name.push_str(&rat.numerator().to_string());
                        if !rat.is_integer() {
                            name.push(':');
                            name.push_str(&rat.denominator().to_string());
                        }
                    }
                }
                CoefficientView::RationalPolynomial(_) => {
                    return Err(CookingError::RatCoeff);
                }
            },
            AtomView::Var(s) => {
                name.push_str(s.get_symbol().get_stripped_name());
            }
            AtomView::Pow(_) => {
                return Err(CookingError::Pow);
            }
            AtomView::Add(_) => {
                return Err(CookingError::Add);
            }
            AtomView::Mul(_) => {
                return Err(CookingError::Mul);
            }
        }
    }

    Ok(symbol!(&name))
}

pub fn cook_indices_impl(view: AtomView) -> Atom {
    let mut expr = view.to_owned();

    let settings = MatchSettings {
        level_range: (0, Some(0)),
        ..Default::default()
    };

    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
        expr = expr.replace_map(|term, ctx, out| {
            if ctx.function_level < 2 && ctx.function_level > 0 {
                if let Some(c) = term.pattern_match(&ipat, None, &settings).next() {
                    if let Ok(aind) = cook_function_view(c[&RS.a_].as_view()) {
                        *out = i.to_symbolic([c[&RS.d_].clone(), aind]);
                        return true;
                    }
                }
                false
            } else {
                false
            }
        });
    }

    for i in LibraryRep::all_dualizables() {
        let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
        // println!("{ipat}");
        let ipat_dual = i.dual().to_symbolic([RS.d_, RS.a_]).to_pattern();
        expr = expr.replace_map(|term, ctx, out| {
            if ctx.function_level == 1 {
                if let Some(c) = term.pattern_match(&ipat_dual, None, &settings).next() {
                    if let Ok(aind) = cook_function_view(c[&RS.a_].as_view()) {
                        // println!("{aind}");
                        *out = i.dual().to_symbolic([c[&RS.d_].clone(), aind]);
                        return true;
                    }
                }
                false
            } else {
                false
            }
        });
        expr = expr.replace_map(|term, ctx, out| {
            if ctx.function_level == 1 {
                if let Some(c) = term.pattern_match(&ipat, None, &settings).next() {
                    if let Ok(aind) = cook_function_view(c[&RS.a_].as_view()) {
                        *out = i.to_symbolic([c[&RS.d_].clone(), aind]);
                        return true;
                    }
                }
                false
            } else {
                false
            }
        });
    }

    expr
}

pub fn not_wraped_aind(header: Symbol) -> impl FilterFn + 'static {
    move |a| match a {
        Match::FunctionName(f) => *f != header,
        Match::Single(a) => {
            if let Some(n) = a.get_symbol() {
                n != header
            } else {
                true
            }
        }
        _ => false,
    }
}

pub fn wrap_indices_impl(view: AtomView, header: Symbol) -> Atom {
    let mut expr = view.expand();
    let dim = RS.d_;
    let dima = Atom::var(dim);
    let settings = MatchSettings {
        level_range: (0, Some(1)),
        ..Default::default()
    };

    let mut reps = vec![];
    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        reps.push(
            Replacement::new(
                i.to_symbolic([dim, RS.a_]).to_pattern(),
                i.to_symbolic([dima.clone(), function!(header, Atom::var(RS.a_))]),
            )
            .with_conditions(RS.a_.filter(not_wraped_aind(header)))
            .with_settings(settings.clone()),
        );
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();
        reps.push(
            Replacement::new(
                i.to_symbolic([dim, RS.a_]).to_pattern(),
                i.to_symbolic([dima.clone(), function!(header, Atom::var(RS.a_))]),
            )
            .with_conditions(RS.a_.filter(not_wraped_aind(header)))
            .with_settings(settings.clone()),
        );
        reps.push(
            Replacement::new(
                di.to_symbolic([dim, RS.a_]).to_pattern(),
                di.to_symbolic([dima.clone(), function!(header, Atom::var(RS.a_))]),
            )
            .with_conditions(RS.a_.filter(not_wraped_aind(header)))
            .with_settings(settings.clone()),
        );
    }
    let mut atom = Atom::new();
    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
    }
    expr
}

pub fn list_dangling_impl(view: AtomView) -> Vec<Atom> {
    let a = view.expand();
    let settings = MatchSettings {
        level_range: (0, Some(1)),
        ..Default::default()
    };
    let mut dangling = HashMap::new();
    let first_term = if let AtomView::Add(a) = a.as_view() {
        if let Some(ft) = a.iter().next() {
            ft
        } else {
            Atom::Zero.as_view()
        }
    } else {
        a.as_view()
    };

    // println!("First term: {}", first_term);
    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
        for p in first_term.pattern_match(&ipat, None, &settings) {
            *dangling.entry(ipat.replace_wildcards(&p)).or_insert(0) += 1;
        }
    }
    for i in LibraryRep::all_dualizables() {
        let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
        let ipat_dual = i.dual().to_symbolic([RS.d_, RS.a_]).to_pattern();

        for p in first_term.pattern_match(&ipat, None, &settings) {
            *dangling.entry(ipat.replace_wildcards(&p)).or_insert(0) += 1;
        }
        for p in first_term.pattern_match(&ipat_dual, None, &settings) {
            *dangling.entry(ipat.replace_wildcards(&p)).or_insert(0) -= 1;
        }
    }

    dangling
        .into_iter()
        .filter_map(|(k, v)| {
            // println!("Dangling: {}, Value: {}", k, v);
            match v {
                1 => Some(k),
                -1 => Some(function!(AIND_SYMBOLS.dind, k)),
                _ => None,
            }
        })
        .collect()
}

pub fn wrap_dummies_impl(view: AtomView, header: Symbol) -> Atom {
    let externals: HashSet<_> = list_dangling_impl(view).into_iter().collect();

    let mut expr = view.to_owned();
    let settings = MatchSettings {
        level_range: (0, Some(0)),
        ..Default::default()
    };

    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
        expr = expr.replace_map(|term, ctx, out| {
            if ctx.function_level < 2 && ctx.function_level > 0 {
                if let Some(c) = term.pattern_match(&ipat, None, &settings).next() {
                    let atom = ipat.replace_wildcards(&c);
                    if !externals.contains(&atom) {
                        *out = i
                            .to_symbolic([c[&RS.d_].clone(), function!(header, c[&RS.a_].clone())]);
                        return true;
                    }
                }
                false
            } else {
                false
            }
        });
    }
    for i in LibraryRep::all_dualizables() {
        let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
        let ipat_dual = i.dual().to_symbolic([RS.d_, RS.a_]).to_pattern();

        expr = expr.replace_map(|term, ctx, out| {
            if ctx.function_level < 2 && ctx.function_level > 0 {
                if let Some(c) = term.pattern_match(&ipat, None, &settings).next() {
                    let atom = ipat.replace_wildcards(&c);
                    if !externals.contains(&atom) {
                        *out = i
                            .to_symbolic([c[&RS.d_].clone(), function!(header, c[&RS.a_].clone())]);
                        return true;
                    }
                } else if let Some(c) = term.pattern_match(&ipat_dual, None, &settings).next() {
                    let atom = ipat_dual.replace_wildcards(&c);
                    if !externals.contains(&atom) {
                        *out = i
                            .dual()
                            .to_symbolic([c[&RS.d_].clone(), function!(header, c[&RS.a_].clone())]);
                        return true;
                    }
                }
                false
            } else {
                false
            }
        });
    }

    expr
}

pub fn simplify_metrics_impl(view: AtomView) -> Atom {
    let mut reps = vec![];

    reps.push((
        function!(ETS.metric, RS.a_, LibraryRep::Dummy.to_symbolic([RS.i__]))
            * function!(
                RS.f_,
                RS.a___,
                LibraryRep::Dummy.to_symbolic([RS.i__]),
                RS.b___
            ),
        function!(RS.f_, RS.a___, RS.a_, RS.b___),
    ));

    for i in LibraryRep::all_representations() {
        reps.extend([(i.id_atom([RS.d_, RS.i_], [RS.d_, RS.i_]), Atom::var(RS.d_))]);
    }

    // You can only really simplify kroneckers when the metric is the identity or when you have the concept of dualizability

    // The metric acts like the identity when there is no dual
    for i in LibraryRep::all_inline_metrics().chain(LibraryRep::all_self_duals()) {
        reps.extend([
            (
                i.metric_atom([RS.a__], [RS.i__])
                    * function!(RS.f_, RS.a___, i.to_symbolic([RS.i__]), RS.b___),
                function!(RS.f_, RS.a___, i.to_symbolic([RS.a__]), RS.b___),
            ),
            //Only when the you have no concept of dualizability is a squared metric acceptable
            (
                i.metric_atom([RS.d_, RS.i_], [RS.d_, RS.a_])
                    .pow(Atom::num(2)),
                Atom::var(RS.d_),
            ),
        ]);
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();

        reps.extend([
            (
                i.id_atom([RS.d_, RS.i_], [RS.d_, RS.a_]).pow(Atom::num(2)),
                Atom::var(RS.d_),
            ),
            (
                di.id_atom([RS.a__], [RS.i__])
                    * function!(RS.f_, RS.a___, i.to_symbolic([RS.i__]), RS.b___),
                function!(RS.f_, RS.a___, i.to_symbolic([RS.a__]), RS.b___),
            ),
            (
                i.id_atom([RS.a__], [RS.i__])
                    * function!(RS.f_, RS.a___, di.to_symbolic([RS.i__]), RS.b___),
                function!(RS.f_, RS.a___, di.to_symbolic([RS.a__]), RS.b___),
            ),
            (
                i.metric_atom([RS.a__], [RS.i__])
                    * function!(RS.f_, RS.a___, di.to_symbolic([RS.i__]), RS.b___),
                function!(RS.f_, RS.a___, i.to_symbolic([RS.a__]), RS.b___),
            ),
            (
                di.metric_atom([RS.a__], [RS.i__])
                    * function!(RS.f_, RS.a___, i.to_symbolic([RS.i__]), RS.b___),
                function!(RS.f_, RS.a___, di.to_symbolic([RS.a__]), RS.b___),
            ),
        ]);
    }

    let reps: Vec<_> = reps
        .into_iter()
        .map(|(lhs, rhs)| Replacement::new(lhs.to_pattern(), rhs.to_pattern()))
        .collect();

    // for rep in &reps {
    //     println!("{}", rep)
    // }

    let mut atom = Atom::new();
    let mut expr = view.to_owned();

    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
        // expr = expr.expand();
    }

    expr
}

pub fn not_slot(sym: Symbol) -> Condition<PatternRestriction> {
    sym.restrict(WildcardRestriction::IsAtomType(AtomType::Var))
        | sym.restrict(WildcardRestriction::IsAtomType(AtomType::Num))
        | sym.restrict(WildcardRestriction::filter(|a| match a {
            Match::FunctionName(f) => {
                // println!("FunctionName{f}");
                LibraryRep::all_representations().all(|r| r.symbol() != *f)
            }
            Match::Multiple(_, views) => {
                // println!("Multiple:");
                // for v in views {
                //     print!("{v}");
                // }
                views
                    .iter()
                    .all(|a| LibrarySlot::<Parsind>::try_from(*a).is_err())
            }
            Match::Single(s) => {
                // println!("Single{s}");
                LibrarySlot::<Parsind>::try_from(*s).is_err()
            }
        }))
}

// pub fn not_aind(sym: Symbol) -> Condition<PatternRestriction> {
//     sym.restrict(WildcardRestriction::IsAtomType(AtomType::Var))
//         | sym.restrict(WildcardRestriction::IsAtomType(AtomType::Num))
//         | sym.restrict(WildcardRestriction::filter(|a| match a {
//             Match::FunctionName(f) => {
//                 println!("FunctionName{f}");
//                 LibraryRep::all_representations().all(|r| r.symbol() != *f)
//             }
//             Match::Multiple(_, views) => {
//                 println!("Multiple:");
//                 for v in views {
//                     print!("{v}");
//                 }
//                 views
//                     .iter()
//                     .all(|a| LibrarySlot::<Parsind>::try_from(*a).is_err())
//             }
//             Match::Single(s) => {
//                 println!("Single{s}");
//                 LibrarySlot::<Parsind>::try_from(*s).is_err()
//             }
//         }))
// }

pub fn to_dots_impl(expr: AtomView) -> Atom {
    let mut reps = vec![];

    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        reps.push(Replacement::new(
            (function!(RS.f_, i.to_symbolic([RS.i__])) * function!(RS.g_, i.to_symbolic([RS.i__])))
                .to_pattern(),
            function!(MS.dot, RS.f_, RS.g_),
        ));

        reps.push(Replacement::new(
            (function!(RS.f_, i.to_symbolic([RS.i__])).pow(Atom::num(2))).to_pattern(),
            function!(MS.dot, RS.f_, RS.f_),
        ));

        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x___, i.to_symbolic([RS.i__]))
                    * function!(RS.g_, RS.y___, i.to_symbolic([RS.i__])))
                .to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x___), function!(RS.g_, RS.y___)),
            )
            .with_conditions(not_slot(RS.x___) & not_slot(RS.y___)),
        );

        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x___, i.to_symbolic([RS.i__])).pow(Atom::num(2))).to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x___), function!(RS.f_, RS.x___)),
            )
            .with_conditions(not_slot(RS.x___)),
        );
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();
        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x___, i.to_symbolic([RS.i__]))
                    * function!(RS.g_, RS.y___, di.to_symbolic([RS.i__])))
                .to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x___), function!(RS.g_, RS.y___)),
            )
            .with_conditions(not_slot(RS.x___) & not_slot(RS.y___)),
        );
    }

    let mut atom = Atom::new();
    let mut expr = expr.expand();
    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
    }

    expr
}

/// Trait for simplifying expressions involving metric tensors and converting
/// index contractions to dot product notation.
///
/// Provides methods for contracting indices with metric tensors (`g(mu, nu)`) or
/// identity tensors (`id(mu, nu)`), and for replacing contracted index patterns
/// (like `p(mu)*q(mu)`) with dot products (`dot(p, q)`).
pub trait MetricSimplifier {
    /// Simplifies contractions involving metric tensors (`g` or `metric`) and identity tensors (`id` or `𝟙`).
    ///
    /// Applies rules like `g(mu, nu) * p(nu) -> p(mu)`, `g(mu, mu) -> D`, etc.
    ///
    /// # Returns
    /// An [`Atom`] representing the expression after metric simplification.
    fn simplify_metrics(&self) -> Atom;

    /// Converts contracted index patterns into dot product notation `dot(...)`.
    ///
    /// Replaces expressions like `p(mu) * q(mu)` or `p(mu) * M(mu, nu) * q(nu)` (implicitly via metric rules)
    /// with `dot(p, q)`. Assumes standard representations for vectors and tensors involved
    /// in the contractions.
    ///
    /// # Returns
    /// An [`Atom`] where contractions have been replaced by `dot` functions where possible.
    fn to_dots(&self) -> Atom;
}

impl MetricSimplifier for Atom {
    fn to_dots(&self) -> Atom {
        to_dots_impl(self.as_view())
    }
    fn simplify_metrics(&self) -> Atom {
        simplify_metrics_impl(self.as_view())
    }
}

pub trait PermuteWithMetric {
    fn permute_with_metric(self) -> Atom;
}

impl<N, Aind: AbsInd + DummyAind> PermuteWithMetric for PermutedStructure<N>
where
    N: ToSymbolic + HasName + TensorStructure<Slot = LibrarySlot<Aind>>,
    N::Name: IntoSymbol + Clone,
    N::Args: IntoArgs,
    Atom: From<Aind>,
{
    fn permute_with_metric(self) -> Atom {
        self.map_structure(|a| SymbolicTensor::from_named(&a).unwrap())
            .permute_inds()
            .expression
            .simplify_metrics()
    }
}

impl MetricSimplifier for AtomView<'_> {
    fn to_dots(&self) -> Atom {
        to_dots_impl(*self)
    }
    fn simplify_metrics(&self) -> Atom {
        simplify_metrics_impl(*self)
    }
}
#[cfg(test)]
mod test {

    use crate::{
        IndexTooling,
        representations::{Bispinor, initialize},
    };

    use super::*;

    use spenso::{
        network::parsing::ShadowedStructure,
        structure::{
            IndexlessNamedStructure, PermutedStructure,
            abstract_index::AbstractIndex,
            permuted::Perm,
            representation::{Euclidean, Lorentz},
        },
        tensors::symbolic::SymbolicTensor,
    };
    use symbolica::parse_lit;

    #[test]
    fn cook() {
        initialize();
        let expr = parse_lit!(
            spenso::g(spenso::mink(4, f(0)), spenso::dind(spenso::cof(4, f(1))))
                * p(spenso::mink(4, 1))
        )
        .cook_indices();

        println!("{}", expr);
    }

    #[test]
    fn metric_contract() {
        initialize();
        let expr =
            parse_lit!(spenso::g(spenso::mink(4, 0), spenso::mink(4, 1)) * p(spenso::mink(4, 1)))
                .simplify_metrics();

        assert_eq!(expr, parse_lit!(p(spenso::mink(4, 0))), "got {:#}", expr);
    }

    #[test]
    fn permute() {
        initialize();
        let f = IndexlessNamedStructure::<Symbol, ()>::from_iter(
            [
                Lorentz {}.new_rep(8).to_lib(),
                Lorentz {}.new_rep(2).cast(),
                Lorentz {}.new_rep(2).cast(),
                Euclidean {}.new_rep(2).cast(),
                Lorentz {}.new_rep(4).cast(),
                Lorentz {}.new_rep(2).cast(),
                Lorentz {}.new_rep(7).cast(),
            ],
            symbol!("test"),
            None,
        )
        .clone()
        .reindex([6, 4, 5, 2, 3, 1, 0])
        .unwrap()
        .map_structure(|a| SymbolicTensor::from_named(&a).unwrap());

        println!("{}\n", f.structure.expression);

        let f_p = f.clone().permute_inds();

        println!("{}\n", f_p.expression);
        println!("{}\n", f_p.expression.simplify_metrics());
        let f_parsed = PermutedStructure::<ShadowedStructure<AbstractIndex>>::try_from(
            &f_p.expression.simplify_metrics(),
        )
        .unwrap();

        assert_eq!(f.index_permutation, f_parsed.index_permutation);
        assert!(f_parsed.rep_permutation.is_identity());

        let f_p = f.clone().permute_reps_wrapped().permute_inds();

        let f_parsed = PermutedStructure::<ShadowedStructure<AbstractIndex>>::try_from(
            &f_p.expression.simplify_metrics(),
        )
        .unwrap();

        println!("{}\n", f_p.expression);
        println!("{}\n", f_p.expression.simplify_metrics());
        assert_eq!(f.index_permutation, f_parsed.index_permutation);
        assert_eq!(f.rep_permutation, f_parsed.rep_permutation);
    }

    #[test]
    fn id_trace() {
        initialize();
        let bis = Bispinor {}.new_rep(symbol!("dim"));

        let expr = bis.g(9, 9).simplify_metrics();

        assert_eq!(expr, Atom::var(symbol!("dim")), "got {:#}", expr);
    }
    #[test]
    fn dots() {
        initialize();

        let a =
            parse_lit!(P(spenso::mink(4, -1 * g(2)), spenso::mink(4, 2)) * P(spenso::mink(4, 2)))
                .to_dots();
        assert_eq!(
            a,
            parse_lit!(P(spenso::mink(4, 2)) * P(spenso::mink(4, -g(2)), spenso::mink(4, 2)))
        );
        let a =
            parse_lit!(P(wrong::mink(4, -1 * g(2)), spenso::mink(4, 2)) * P(spenso::mink(4, 2)))
                .to_dots();
        println!("{a}");
        assert_eq!(a, parse_lit!(spenso::dot(P(), P(wrong::mink(4, -g(2))))));
    }
}

use std::sync::LazyLock;

use spenso::{
    network::library::symbolic::{ETS, ExplicitKey},
    structure::{
        dimension::Dimension,
        representation::{LibraryRep, Minkowski, RepName},
    },
};
use symbolica::{
    atom::{Atom, AtomCore, AtomOrView, AtomView, FunctionBuilder, Symbol},
    function,
    id::{Context, Replacement},
    symbol,
};

use crate::{color::SelectiveExpand, metric::MetricSimplifier, rep_symbols::RS};

use super::representations::Bispinor;

pub struct GammaLibrary {
    pub gamma: Symbol,
    pub projp: Symbol,
    pub projm: Symbol,
    pub gamma5: Symbol,
    pub sigma: Symbol,
}

impl GammaLibrary {
    pub fn replace_with(&self, rep: &Self) -> Vec<Replacement> {
        vec![
            Replacement::new(
                function!(self.gamma, RS.i__).to_pattern(),
                function!(rep.gamma, RS.i__),
            ),
            Replacement::new(
                function!(self.projp, RS.i__).to_pattern(),
                function!(rep.projp, RS.i__),
            ),
            Replacement::new(
                function!(self.projm, RS.i__).to_pattern(),
                function!(rep.projm, RS.i__),
            ),
            Replacement::new(
                function!(self.gamma5, RS.i__).to_pattern(),
                function!(rep.gamma5, RS.i__),
            ),
            Replacement::new(
                function!(self.sigma, RS.i__).to_pattern(),
                function!(rep.sigma, RS.i__),
            ),
        ]
    }
}

pub struct GammaSymbolsInternal {
    pub gamma_chain: Symbol,
    pub gamma_trace: Symbol,
}

pub struct PolSymbols {
    pub eps: Symbol,
    pub ebar: Symbol,
    pub u: Symbol,
    pub ubar: Symbol,
    pub v: Symbol,
    pub vbar: Symbol,
}

pub static PS: LazyLock<PolSymbols> = LazyLock::new(|| PolSymbols {
    eps: symbol!("spenso::ϵ"),
    ebar: symbol!("spenso::ϵbar"),
    u: symbol!("spenso::u"),
    ubar: symbol!("spenso::ubar"),
    v: symbol!("spenso::v"),
    vbar: symbol!("spenso::vbar"),
});

pub fn factor_conj_impl(expression: AtomView) -> Atom {
    expression
        .to_owned()
        .replace(Atom::i().to_pattern())
        .with((-Atom::i()).to_pattern())
}
pub fn pol_conj_impl(expression: AtomView) -> Atom {
    let expr = expression.to_owned().expand();

    expr.replace_multiple(&[
        Replacement::new(
            function!(PS.ebar, RS.i__).to_pattern(),
            function!(PS.eps, RS.i__),
        ),
        Replacement::new(
            function!(PS.eps, RS.i__).to_pattern(),
            function!(PS.ebar, RS.i__),
        ),
        Replacement::new(
            function!(PS.u, RS.i__).to_pattern(),
            function!(PS.ubar, RS.i__),
        ),
        Replacement::new(
            function!(PS.ubar, RS.i__).to_pattern(),
            function!(PS.u, RS.i__),
        ),
        Replacement::new(
            function!(PS.v, RS.i__).to_pattern(),
            function!(PS.vbar, RS.i__),
        ),
        Replacement::new(
            function!(PS.vbar, RS.i__).to_pattern(),
            function!(PS.v, RS.i__),
        ),
    ])
}

pub fn gamma_conj_impl(expression: AtomView) -> Atom {
    let expr = expression.to_owned();
    let bis = Bispinor {};

    expr.replace(
        function!(
            AGS.gamma,
            RS.i_,
            bis.to_symbolic([RS.d_, RS.a_]),
            bis.to_symbolic([RS.d_, RS.b_])
        )
        .to_pattern(),
    )
    .with(-function!(
        AGS.gamma,
        RS.i_,
        bis.to_symbolic([RS.d_, RS.b_]),
        bis.to_symbolic([RS.d_, RS.a_])
    ))
    .replace(
        function!(
            AGS.gamma5,
            bis.to_symbolic([RS.d_, RS.a_]),
            bis.to_symbolic([RS.d_, RS.b_])
        )
        .to_pattern(),
    )
    .with(
        function!(
            AGS.gamma5,
            bis.to_symbolic([RS.d_, RS.b_]),
            bis.to_symbolic([RS.d_, RS.a_])
        )
        .to_pattern(),
    )
}
pub static GS: LazyLock<GammaSymbolsInternal> = LazyLock::new(|| GammaSymbolsInternal {
    gamma_chain: symbol!("spenso::gamma_chain"),
    gamma_trace: symbol!("spenso::gamma_trace"),
});

pub static AGS: LazyLock<GammaLibrary> = LazyLock::new(|| GammaLibrary {
    gamma: symbol!("spenso::gamma"),
    projp: symbol!("spenso::projp"),
    projm: symbol!("spenso::projm"),
    gamma5: symbol!("spenso::gamma5"),
    sigma: symbol!("spenso::sigma"),
});

impl GammaLibrary {
    pub fn projp<'a, 'b>(
        &self,
        a: impl Into<AtomOrView<'a>>,
        b: impl Into<AtomOrView<'b>>,
    ) -> Atom {
        function!(self.projp, a.into().as_view(), b.into().as_view())
    }
    pub fn projm<'a, 'b>(
        &self,
        a: impl Into<AtomOrView<'a>>,
        b: impl Into<AtomOrView<'b>>,
    ) -> Atom {
        function!(self.projm, a.into().as_view(), b.into().as_view())
    }

    pub fn gamma_strct(&self, dim: impl Into<Dimension>) -> ExplicitKey {
        let gamma = ExplicitKey::from_iter(
            [
                LibraryRep::from(Minkowski {}).new_rep(dim),
                Bispinor {}.new_rep(4).cast(),
                Bispinor {}.new_rep(4).cast(),
            ],
            self.gamma,
            None,
        );
        gamma.structure
    }

    pub fn gamma5_strct(&self, dim: impl Into<Dimension>) -> ExplicitKey {
        let dim = dim.into();
        let gamma5 = ExplicitKey::from_iter(
            [Bispinor {}.new_rep(dim), Bispinor {}.new_rep(dim)],
            self.gamma5,
            None,
        );
        gamma5.structure
    }

    pub fn projm_strct(&self, dim: impl Into<Dimension>) -> ExplicitKey {
        let dim = dim.into();
        let projm = ExplicitKey::from_iter(
            [Bispinor {}.new_rep(dim), Bispinor {}.new_rep(dim)],
            self.projm,
            None,
        );
        projm.structure
    }

    pub fn projp_strct(&self, dim: impl Into<Dimension>) -> ExplicitKey {
        let dim = dim.into();
        let projp_strct = ExplicitKey::from_iter(
            [Bispinor {}.new_rep(dim), Bispinor {}.new_rep(dim)],
            self.projm,
            None,
        );

        projp_strct.structure
    }
}

pub fn gamma_simplify_impl(expr: AtomView) -> Atom {
    let mink = Minkowski {};

    let mut expr = expr.expand_mink_bis();

    let reps: Vec<_> = [
        (
            function!(AGS.projp, RS.a_, RS.b_),
            (function!(ETS.id, RS.a_, RS.b_) - function!(AGS.gamma5, RS.a_, RS.b_)) / 2,
        ),
        (
            function!(AGS.projm, RS.a_, RS.b_),
            (function!(ETS.id, RS.a_, RS.b_) + function!(AGS.gamma5, RS.a_, RS.b_)) / 2,
        ),
        (
            function!(AGS.gamma, RS.a_, RS.b_, RS.c_) * function!(AGS.gamma, RS.d_, RS.c_, RS.e_),
            function!(GS.gamma_chain, RS.a_, RS.d_, RS.b_, RS.e_),
        ),
        (function!(AGS.gamma, RS.a_, RS.b_, RS.b_), Atom::Zero),
        (
            function!(GS.gamma_chain, RS.a__, RS.a_, RS.b_)
                * function!(GS.gamma_chain, RS.b__, RS.b_, RS.c_),
            function!(GS.gamma_chain, RS.a__, RS.b__, RS.a_, RS.c_),
        ),
        (
            function!(GS.gamma_chain, RS.a__, RS.a_, RS.b_)
                * function!(AGS.gamma, RS.y_, RS.b_, RS.c_),
            function!(GS.gamma_chain, RS.a__, RS.y_, RS.a_, RS.c_),
        ),
        (
            function!(AGS.gamma, RS.a_, RS.a_, RS.b_)
                * function!(GS.gamma_chain, RS.y__, RS.b_, RS.c_),
            function!(GS.gamma_chain, RS.a_, RS.y__, RS.a_, RS.c_),
        ),
    ]
    .iter()
    .map(|(a, b)| {
        // println!("{}->{}", a, b);

        Replacement::new(a.to_pattern(), b.to_pattern())
    })
    .collect();

    let mut atom = Atom::new();

    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
        expr = expr.expand_mink_bis();
        expr = expr.simplify_metrics();
    }

    let reps: Vec<_> = [
        (
            function!(
                GS.gamma_chain,
                RS.a___,
                mink.to_symbolic([RS.d_, RS.a_]),
                mink.to_symbolic([RS.d_, RS.a_]),
                RS.b__
            ),
            function!(GS.gamma_chain, RS.a___, RS.b__) * RS.d_,
        ),
        (
            function!(GS.gamma_chain, RS.a_, RS.b_),
            function!(ETS.id, RS.a_, RS.b_),
        ),
        (
            function!(
                GS.gamma_chain,
                RS.a___,
                RS.a_,
                RS.b___,
                RS.b_,
                RS.a_,
                RS.a__
            ),
            function!(GS.gamma_chain, RS.a___, RS.b_, RS.b___, RS.a__) * 2
                - function!(
                    GS.gamma_chain,
                    RS.a___,
                    RS.a_,
                    RS.b___,
                    RS.a_,
                    RS.b_,
                    RS.a__
                ),
        ),
    ]
    .iter()
    .map(|(a, b)| Replacement::new(a.to_pattern(), b.to_pattern()))
    .collect();

    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
    }

    fn gamma_chain_accumulator(arg: AtomView, _context: &Context, out: &mut Atom) -> bool {
        if let AtomView::Fun(f) = arg {
            if f.get_symbol() == GS.gamma_chain {
                let mut args = f.iter().collect::<Vec<_>>();
                if args.len() >= 4 {
                    for i in 0..args.len().saturating_sub(3) {
                        // println!("{}", args[i]);
                        // println!("{}?{}", args[i], args[i + 1]);
                        if args[i] > args[i + 1] {
                            // println!("{}>{}", args[i], args[i + 1]);
                            args.swap(i, i + 1);
                            let swapped = FunctionBuilder::new(GS.gamma_chain)
                                .add_args(&args)
                                .finish();
                            let mu = args.remove(i);
                            let nu = args.remove(i);
                            let metric = function!(ETS.metric, mu, nu)
                                * 2
                                * FunctionBuilder::new(GS.gamma_chain)
                                    .add_args(&args)
                                    .finish();
                            *out = metric - swapped;
                            // println!("{}->{}", a, c);
                            return true;
                        }
                    }
                    return false;
                } else {
                    return false;
                }
            }
            false
        } else {
            false
        }
    }

    loop {
        let new = expr
            .replace_map(&gamma_chain_accumulator)
            .replace_multiple(&reps);
        if new == expr {
            break;
        } else {
            expr = new;
        }
    }

    expr = expr
        .replace(function!(GS.gamma_chain, RS.a__, RS.x_, RS.x_).to_pattern())
        .repeat()
        .with(function!(GS.gamma_trace, RS.a__).to_pattern());

    // //Chisholm identity:
    // expr.replace_all_repeat_mut(
    //     &(function!(AGS.gamma, RS.a_, RS.x_, RS.y_) * function!(gamma_trace, RS.a_, RS.a__)).to_pattern(),
    //     (function!(gamma_chain, RS.a__)).to_pattern(),
    //     None,
    //     None,
    // );
    //
    fn gamma_tracer(arg: AtomView, _context: &Context, out: &mut Atom) -> bool {
        let gamma_trace = GS.gamma_trace;

        let mut found = false;
        if let AtomView::Fun(f) = arg {
            if f.get_symbol() == gamma_trace {
                // println!("{arg}");
                found = true;
                let mut sum = Atom::Zero;

                if f.get_nargs() == 1 {
                    *out = Atom::Zero;
                }
                let args = f.iter().collect::<Vec<_>>();

                for i in 1..args.len() {
                    let sign = if i % 2 == 0 { -1 } else { 1 };

                    let mut gcn = FunctionBuilder::new(gamma_trace);
                    #[allow(clippy::needless_range_loop)]
                    for j in 1..args.len() {
                        if i != j {
                            gcn = gcn.add_arg(args[j]);
                        }
                    }

                    let metric = if args[0] == args[i] {
                        if let AtomView::Fun(f) = args[0].as_atom_view() {
                            f.iter().next().unwrap().to_owned()
                        } else {
                            panic!("aaaa")
                        }
                        // Atom::num(4)
                    } else {
                        function!(ETS.metric, args[0], args[i])
                    };
                    if args.len() == 2 {
                        sum = sum + metric * sign * Atom::num(4);
                    } else {
                        sum = sum + metric * gcn.finish() * sign;
                    }
                }
                *out = sum;

                // println!("{}->{}", arg, out);
            }
        }

        found
    }

    loop {
        let new = expr.replace_map(&gamma_tracer);
        if new == expr {
            break;
        } else {
            expr = new;
        }
    }

    expr = expr
        .replace(
            function!(AGS.gamma, mink.to_symbolic([RS.d_, RS.b_]), RS.a__)
                .pow(Atom::num(2))
                .to_pattern(),
        )
        .repeat()
        .with(Atom::var(RS.d_) * 4)
        .expand_mink_bis()
        .simplify_metrics();

    expr
}
/// Trait for simplifying expressions involving Dirac gamma matrices using Clifford algebra.
///
/// Implementors provide a method to apply gamma matrix identities, such as
/// anticommutation relations and trace evaluations.
pub trait GammaSimplifier {
    /// Simplifies gamma matrix structures within the expression.
    ///
    /// Uses the Clifford algebra relation `{gamma^mu, gamma^nu} = 2 * g^{mu nu}`
    /// and evaluates traces of products of gamma matrices. It handles intermediate
    /// simplification steps involving metric tensors.
    ///
    /// # Returns
    /// An [`Atom`] representing the expression after gamma matrix simplification.
    fn simplify_gamma(&self) -> Atom;
}

impl GammaSimplifier for Atom {
    fn simplify_gamma(&self) -> Atom {
        gamma_simplify_impl(self.as_atom_view())
    }
}

impl<'a> GammaSimplifier for AtomView<'a> {
    fn simplify_gamma(&self) -> Atom {
        gamma_simplify_impl(self.as_atom_view())
    }
}

pub fn id_atom(i: impl Into<Atom>, j: impl Into<Atom>) -> Atom {
    function!(ETS.id, i.into(), j.into())
}

#[macro_export]
macro_rules! id {
    ($i: expr, $j: expr) => {{
        let i = symbolica::parse_lit!($i);
        let j = symbolica::parse_lit!($j);
        id_atom(i, j)
    }};
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::id;
    use symbolica::{
        atom::{Atom, AtomCore},
        parse_lit,
    };

    use crate::representations::initialize;

    #[test]
    fn gamma_alg() {
        initialize();
        let expr = parse_lit!(
            spenso::gamma_chain(mink(4, 0), mink(4, 0), b(1), b(2)),
            "spenso"
        )
        .simplify_gamma();

        assert_eq!(expr, id!(spenso::b(1), spenso::b(2)) * 4, "got {:#}", expr);

        let expr = parse_lit!(
            p(mink(4, nu1))
                * (p(mink(4, nu3)) + q(mink(4, nu3)))
                * spenso::gamma_chain(
                    mink(4, nu1),
                    mink(4, mu),
                    mink(4, nu3),
                    mink(4, nu),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .simplify_gamma()
        .expand();
        assert_eq!(
            expr,
            parse_lit!(
                -4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1))
                    ^ 2 + 8 * p(mink(4, mu)) * p(mink(4, nu))
                        + 4 * p(mink(4, mu)) * q(mink(4, nu))
                        + 4 * p(mink(4, nu)) * q(mink(4, mu))
                        - 4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1)) * q(mink(4, nu1)),
                "spenso"
            ),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            g(mink(dim, 5), mink(dim, 6))
                * (g(mink(dim, 1), mink(dim, 2))
                    * g(mink(dim, 3), mink(dim, 4))
                    * g(mink(dim, 5), mink(dim, 6))
                    - g(mink(dim, 1), mink(dim, 3))
                        * g(mink(dim, 2), mink(dim, 6))
                        * g(mink(dim, 5), mink(dim, 4)))
                * (g(mink(dim, 1), mink(dim, 2)) * g(mink(dim, 3), mink(dim, 4))
                    - g(mink(dim, 1), mink(dim, 3)) * g(mink(dim, 2), mink(dim, 4))),
            "spenso"
        )
        .simplify_gamma();
        assert_eq!(expr, parse_lit!(-dim + dim ^ 3, "spenso"), "got {}", expr);

        let expr = parse_lit!(
            p(mink(4, nu1))
                * (p(mink(4, nu3)) + q(mink(4, nu3)))
                * spenso::gamma_chain(
                    mink(4, nu1),
                    mink(4, mu),
                    mink(4, nu),
                    mink(4, nu3),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .simplify_gamma();
        assert_eq!(
            expr,
            parse_lit!(
                4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1))
                    ^ 2 + 4 * p(mink(4, mu)) * q(mink(4, nu)) - 4 * q(mink(4, mu)) * p(mink(4, nu))
                        + 4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1)) * q(mink(4, nu1)),
                "spenso"
            ),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            p(mink(dim, nu1))
                * (p(mink(dim, nu3)) + q(mink(dim, nu3)))
                * spenso::gamma_chain(
                    mink(dim, nu1),
                    mink(dim, nu),
                    mink(dim, nu),
                    mink(dim, nu3),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .simplify_gamma();
        assert_eq!(
            expr,
            parse_lit!(
                4 * dim * p(mink(dim, nu1)) ^ 2 + 4 * dim * p(mink(dim, nu1)) * q(mink(dim, nu1)),
                "spenso"
            ),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            p(mink(dim, nu1))
                * (p(mink(dim, nu3)) + q(mink(dim, nu3)))
                * spenso::gamma_chain(
                    mink(dim, nu1),
                    mink(dim, nu),
                    mink(dim, nu3),
                    mink(dim, nu),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .simplify_gamma();
        assert_eq!(
            expr,
            parse_lit!(
                8 * p(mink(dim, nu1))
                    ^ 2 - 4 * dim * p(mink(dim, nu1))
                    ^ 2 + 8 * p(mink(dim, nu1)) * q(mink(dim, nu1))
                        - 4 * dim * p(mink(dim, nu1)) * q(mink(dim, nu1)),
                "spenso"
            ),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            symbolica_community::p(mink(dim, nu1))
                * symbolica_community::q(mink(dim, nu2))
                * (symbolica_community::p(mink(dim, nu3)) + symbolica_community::q(mink(dim, nu3)))
                * symbolica_community::q(mink(dim, nu4))
                * spenso::gamma_chain(
                    mink(dim, nu1),
                    mink(dim, nu4),
                    mink(dim, nu3),
                    mink(dim, nu2),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .simplify_gamma()
        .to_dots();
        assert_eq!(
            expr,
            parse_lit!(8 * dot(p, q) ^ 2 - 4 * dot(p, p) * dot(q, q) + 4 * dot(p, q) * dot(q, q)),
            "got {}",
            expr
        );

        let expr = parse_lit!(
            spenso::gamma_chain(
                mink(dim, mu),
                mink(dim, nu),
                mink(dim, mu),
                mink(dim, nu),
                b(1),
                b(2)
            ),
            "spenso"
        )
        .simplify_gamma()
        .to_dots();

        let dim = Atom::var(symbol!("spenso::dim"));
        assert_eq!(
            expr,
            &dim * id!(spenso::b(1), spenso::b(2)) * 2
                - dim.pow(Atom::num(2)) * id!(spenso::b(1), spenso::b(2)),
            "got {}",
            expr
        );
    }
}

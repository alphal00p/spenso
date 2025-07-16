use std::sync::LazyLock;

use spenso::{
    network::library::symbolic::{ETS, ExplicitKey},
    structure::{
        dimension::Dimension,
        representation::{LibraryRep, Minkowski, RepName},
        slot::AbsInd,
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

impl GammaSymbolsInternal {
    pub fn chain_pattern<'a, It: Into<AtomOrView<'a>>>(
        &self,
        i: impl Into<AtomOrView<'a>>,
        j: impl Into<AtomOrView<'a>>,
        mus: impl IntoIterator<Item = It>,
    ) -> Atom {
        let mus: Vec<_> = mus
            .into_iter()
            .enumerate()
            .map(|(idx, a)| {
                if idx % 2 == 0 {
                    Minkowski {}.to_symbolic([a])
                } else {
                    Bispinor {}.to_symbolic([a])
                }
            })
            .collect();
        if mus.len() % 2 == 0 {
            panic!("Should intersperse dummies")
        }
        FunctionBuilder::new(self.gamma_chain)
            .add_args(&mus)
            .add_arg(Bispinor {}.to_symbolic([i]))
            .add_arg(Bispinor {}.to_symbolic([j]))
            .finish()
    }
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

pub fn pol_conj_impl(expression: AtomView) -> Atom {
    let expr = expression.to_owned();

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

    expr.replace(AGS.gamma_pattern(RS.a__, RS.i__, RS.j__).to_pattern())
        .with((-AGS.gamma_pattern(RS.a__, RS.j__, RS.i__)).to_pattern())
        .replace(AGS.gamma5_pattern(RS.i__, RS.j__).to_pattern())
        .with(AGS.gamma5_pattern(RS.j__, RS.i__).to_pattern())
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

    pub fn projp_pattern<'a>(
        &self,
        i: impl Into<AtomOrView<'a>>,
        j: impl Into<AtomOrView<'a>>,
    ) -> Atom {
        function!(
            self.projp,
            Bispinor {}.to_symbolic([i]),
            Bispinor {}.to_symbolic([j])
        )
    }

    pub fn projm<'a, 'b>(
        &self,
        a: impl Into<AtomOrView<'a>>,
        b: impl Into<AtomOrView<'b>>,
    ) -> Atom {
        function!(self.projm, a.into().as_view(), b.into().as_view())
    }
    pub fn projm_pattern<'a>(
        &self,
        i: impl Into<AtomOrView<'a>>,
        j: impl Into<AtomOrView<'a>>,
    ) -> Atom {
        function!(
            self.projm,
            Bispinor {}.to_symbolic([i]),
            Bispinor {}.to_symbolic([j])
        )
    }

    pub fn gamma_strct<Aind: AbsInd>(&self, dim: impl Into<Dimension>) -> ExplicitKey<Aind> {
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

    pub fn gamma_pattern<'a>(
        &self,
        mu: impl Into<AtomOrView<'a>>,
        i: impl Into<AtomOrView<'a>>,
        j: impl Into<AtomOrView<'a>>,
    ) -> Atom {
        function!(
            self.gamma,
            Bispinor {}.to_symbolic([i]),
            Bispinor {}.to_symbolic([j]),
            Minkowski {}.to_symbolic([mu])
        )
    }
    pub fn gamma5_pattern<'a>(
        &self,
        i: impl Into<AtomOrView<'a>>,
        j: impl Into<AtomOrView<'a>>,
    ) -> Atom {
        function!(
            self.gamma5,
            Bispinor {}.to_symbolic([i]),
            Bispinor {}.to_symbolic([j])
        )
    }

    pub fn gamma5_strct<Aind: AbsInd>(&self, dim: impl Into<Dimension>) -> ExplicitKey<Aind> {
        let dim = dim.into();
        let gamma5 = ExplicitKey::from_iter(
            [Bispinor {}.new_rep(dim), Bispinor {}.new_rep(dim)],
            self.gamma5,
            None,
        );
        gamma5.structure
    }

    pub fn projm_strct<Aind: AbsInd>(&self, dim: impl Into<Dimension>) -> ExplicitKey<Aind> {
        let dim = dim.into();
        let projm = ExplicitKey::from_iter(
            [Bispinor {}.new_rep(dim), Bispinor {}.new_rep(dim)],
            self.projm,
            None,
        );
        projm.structure
    }

    pub fn projp_strct<Aind: AbsInd>(&self, dim: impl Into<Dimension>) -> ExplicitKey<Aind> {
        let dim = dim.into();
        let projp_strct = ExplicitKey::from_iter(
            [Bispinor {}.new_rep(dim), Bispinor {}.new_rep(dim)],
            self.projm,
            None,
        );

        projp_strct.structure
    }
}

fn collect_gammas(expr: &mut Atom) {
    let reps: Vec<_> = [
        (
            AGS.projp_pattern(RS.a__, RS.b__),
            (Bispinor {}.id_atom([RS.a__], [RS.b__]) - AGS.gamma5_pattern(RS.a__, RS.b__)) / 2,
        ),
        (
            AGS.projm_pattern(RS.a__, RS.b__),
            (Bispinor {}.id_atom([RS.a__], [RS.b__]) + AGS.gamma5_pattern(RS.a__, RS.b__)) / 2,
        ),
        (
            AGS.gamma_pattern(RS.a__, RS.b__, RS.c__) * AGS.gamma_pattern(RS.d__, RS.c__, RS.e__),
            GS.chain_pattern(RS.b__, RS.e__, [RS.a__, RS.c__, RS.d__]),
        ),
        (AGS.gamma_pattern(RS.a__, RS.b__, RS.b__), Atom::Zero),
        (
            function!(GS.gamma_chain, RS.a__, RS.a_, RS.b_)
                * function!(GS.gamma_chain, RS.b__, RS.b_, RS.c_),
            function!(GS.gamma_chain, RS.a__, RS.b_, RS.b__, RS.a_, RS.c_),
        ),
        (
            function!(
                GS.gamma_chain,
                RS.a___,
                Bispinor {}.to_symbolic([RS.a__]),
                Bispinor {}.to_symbolic([RS.b__])
            ) * AGS.gamma_pattern(RS.y__, RS.b__, RS.c__),
            function!(
                GS.gamma_chain,
                RS.a___,
                Bispinor {}.to_symbolic([RS.b__]),
                Minkowski {}.to_symbolic([RS.y__]),
                Bispinor {}.to_symbolic([RS.a__]),
                Bispinor {}.to_symbolic([RS.c__])
            ),
        ),
        (
            AGS.gamma_pattern(RS.y__, RS.a__, RS.b__)
                * function!(
                    GS.gamma_chain,
                    RS.a___,
                    Bispinor {}.to_symbolic([RS.b__]),
                    Bispinor {}.to_symbolic([RS.c__])
                ),
            function!(
                GS.gamma_chain,
                RS.a___,
                Bispinor {}.to_symbolic([RS.b__]),
                Minkowski {}.to_symbolic([RS.y__]),
                Bispinor {}.to_symbolic([RS.a__]),
                Bispinor {}.to_symbolic([RS.c__])
            ),
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
        // println!("collecting:{atom}");
        std::mem::swap(expr, &mut atom);
        *expr = expr.expand();
        // println!("expanding::{expr}");
        *expr = expr.simplify_metrics();
        // println!("simplifying::{expr}");
    }
}

fn normalise_gammas(expr: &mut Atom) {
    // Uses the anti commutation rule of the gamma chain to sort the minkowski indices
    fn gamma_chain_normalisation(arg: AtomView, _context: &Context, out: &mut Atom) -> bool {
        if let AtomView::Fun(f) = arg {
            if f.get_symbol() == GS.gamma_chain {
                // println!("Gamma chain:{}", arg);
                let mut args = f.iter().collect::<Vec<_>>();
                if args.len() >= 5 {
                    let more = args.len() > 5;
                    for i in 0..args.len().saturating_sub(4) {
                        if i % 2 == 0 {
                            // println!("{}", args[i]);
                            // println!("{}?{}", args[i], args[i + 1]);
                            if args[i] > args[i + 2] {
                                // println!("{}>{}", args[i], args[i + 1]);
                                args.swap(i, i + 2);
                                let swapped = FunctionBuilder::new(GS.gamma_chain)
                                    .add_args(&args)
                                    .finish();

                                // println!("{}", swapped);
                                let mu = args.remove(i);
                                let _bis = args.remove(i);
                                let nu = args.remove(i);
                                if more {
                                    if i == 0 {
                                        args.remove(i);
                                    } else {
                                        args.remove(i - 1);
                                    }
                                }

                                // println!("mu:{}bis:{}nu:{}", mu, bis, nu);
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
                    }
                }
            }
        }
        false
    }

    loop {
        let new = expr.replace_map(&gamma_chain_normalisation);
        // .replace_multiple(&reps);
        if new == *expr {
            break;
        } else {
            *expr = new;
        }
    }
}

fn undo_gamma_chain(expr: &mut Atom) {
    let reps: Vec<_> = [
        (
            GS.chain_pattern(RS.b__, RS.e__, [RS.a__, RS.c__, RS.d__]),
            AGS.gamma_pattern(RS.a__, RS.b__, RS.c__) * AGS.gamma_pattern(RS.d__, RS.c__, RS.e__),
        ),
        (
            function!(
                GS.gamma_chain,
                RS.a___,
                Bispinor {}.to_symbolic([RS.b__]),
                Minkowski {}.to_symbolic([RS.y__]),
                Bispinor {}.to_symbolic([RS.a__]),
                Bispinor {}.to_symbolic([RS.c__])
            ),
            function!(
                GS.gamma_chain,
                RS.a___,
                Bispinor {}.to_symbolic([RS.a__]),
                Bispinor {}.to_symbolic([RS.b__])
            ) * AGS.gamma_pattern(RS.y__, RS.b__, RS.c__),
        ),
        (
            function!(
                GS.gamma_chain,
                RS.a___,
                Bispinor {}.to_symbolic([RS.b__]),
                Minkowski {}.to_symbolic([RS.y__]),
                Bispinor {}.to_symbolic([RS.a__]),
                Bispinor {}.to_symbolic([RS.c__])
            ),
            AGS.gamma_pattern(RS.y__, RS.a__, RS.b__)
                * function!(
                    GS.gamma_chain,
                    RS.a___,
                    Bispinor {}.to_symbolic([RS.b__]),
                    Bispinor {}.to_symbolic([RS.c__])
                ),
        ),
    ]
    .iter()
    .map(|(a, b)| {
        // println!("{}->{}", a, b);

        Replacement::new(a.to_pattern(), b.to_pattern())
    })
    .collect();

    loop {
        let new = expr.replace_multiple(&reps);
        if new == *expr {
            break;
        } else {
            *expr = new;
        }
    }
}

pub fn gamma_simplify_impl(expr: AtomView) -> Atom {
    let mink = Minkowski {};

    let mut coef_list_spin = expr.expand_mink_bis();

    for (expr, _) in &mut coef_list_spin {
        // println!("gs:{expr}");
        *expr = expr.simplify_metrics();
        collect_gammas(expr);
        // expr = normalise_gammas(expr);

        let reps: Vec<_> = [
            (
                function!(
                    GS.gamma_chain,
                    RS.a___,
                    Bispinor {}.to_symbolic([RS.i__]),
                    Minkowski {}.to_symbolic([RS.d_, RS.a_]),
                    Bispinor {}.to_symbolic([RS.j__]),
                    Minkowski {}.to_symbolic([RS.d_, RS.a_]),
                    RS.b__
                ),
                function!(GS.gamma_chain, RS.a___, RS.b__) * RS.d_,
            ),
            (
                function!(
                    GS.gamma_chain,
                    Minkowski {}.to_symbolic([RS.d_, RS.a_]),
                    Bispinor {}.to_symbolic([RS.b__]),
                    Minkowski {}.to_symbolic([RS.d_, RS.a_]),
                    Bispinor {}.to_symbolic([RS.i__]),
                    Bispinor {}.to_symbolic([RS.j__])
                ),
                function!(
                    GS.gamma_chain,
                    Bispinor {}.to_symbolic([RS.i__]),
                    Bispinor {}.to_symbolic([RS.j__])
                ) * RS.d_,
            ),
            (
                function!(
                    GS.gamma_chain,
                    Minkowski {}.to_symbolic([RS.a__]),
                    Bispinor {}.to_symbolic([RS.i__]),
                    Bispinor {}.to_symbolic([RS.j__])
                ),
                function!(
                    AGS.gamma,
                    Bispinor {}.to_symbolic([RS.i__]),
                    Bispinor {}.to_symbolic([RS.j__]),
                    Minkowski {}.to_symbolic([RS.a__])
                ),
            ),
            (
                function!(
                    AGS.gamma,
                    Bispinor {}.to_symbolic([RS.i__]),
                    Bispinor {}.to_symbolic([RS.i__]),
                    Minkowski {}.to_symbolic([RS.a__])
                ),
                Atom::Zero,
            ),
            (
                function!(
                    GS.gamma_chain,
                    Bispinor {}.to_symbolic([RS.a__]),
                    Bispinor {}.to_symbolic([RS.b__])
                ),
                Bispinor {}.id_atom([RS.a__], [RS.b__]),
            ),
        ]
        .iter()
        .map(|(a, b)| {
            // println!("{a:#}->\n\t{b:#}");
            Replacement::new(a.to_pattern(), b.to_pattern())
        })
        .collect();

        normalise_gammas(expr);
        loop {
            let new = expr.replace_multiple(&reps).expand().simplify_metrics();
            if new == *expr {
                break;
            } else {
                *expr = new;
            }
        }

        *expr = expr
            .replace(function!(GS.gamma_chain, RS.a__, RS.x_, RS.x_).to_pattern())
            .repeat()
            .with(function!(GS.gamma_trace, RS.a__).to_pattern())
            .replace(function!(
                GS.gamma_trace,
                RS.a___,
                Bispinor {}.to_symbolic([RS.a__]),
                RS.b___
            ))
            .repeat()
            .with(function!(GS.gamma_trace, RS.a___, RS.b___));

        undo_gamma_chain(expr);
        // println!("Before tracer:{expr}");

        // expr = expr.replace_map(|term, ctx, out| {

        //     if let AtomView::Fun(f)= term{
        //         if f.get_symbol()= GS.gamma_chain{

        //         }
        //     }

        //     false});

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
                            sum += metric * sign * Atom::num(4);
                        } else {
                            sum += metric * gcn.finish() * sign;
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
            if new == *expr {
                break;
            } else {
                *expr = new;
            }
        }

        *expr = expr
            .replace(
                function!(AGS.gamma, RS.a__, mink.to_symbolic([RS.d_, RS.b_]))
                    .pow(Atom::num(2))
                    .to_pattern(),
            )
            .repeat()
            .with(Atom::var(RS.d_) * 4)
            .expand()
            .simplify_metrics();
    }

    coef_list_spin
        .iter()
        .fold(Atom::Zero, |a, (c, s)| a + c * s)
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

impl GammaSimplifier for AtomView<'_> {
    fn simplify_gamma(&self) -> Atom {
        gamma_simplify_impl(self.as_atom_view())
    }
}

pub fn id_atom(i: impl Into<Atom>, j: impl Into<Atom>) -> Atom {
    function!(ETS.metric, i.into(), j.into())
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

    use spenso::structure::IndexlessNamedStructure;
    use spenso::structure::PermutedStructure;

    static GG: LazyLock<PermutedStructure<IndexlessNamedStructure<Symbol, ()>>> =
        LazyLock::new(|| {
            IndexlessNamedStructure::from_iter(
                [
                    Bispinor {}.new_rep(4).to_lib(),
                    Bispinor {}.new_rep(4).cast(),
                    Minkowski {}.new_rep(4).cast(),
                ],
                AGS.gamma,
                None,
            )
        });

    use super::*;

    use crate::id;
    use spenso::{
        structure::{abstract_index::AbstractIndex, permuted::Perm},
        tensors::symbolic::SymbolicTensor,
    };
    use symbolica::{
        atom::{Atom, AtomCore},
        parse_lit,
    };

    use crate::representations::initialize;

    #[test]
    fn gamma_construct() {
        println!("{}", AGS.gamma_pattern(RS.a_, RS.b_, RS.c_));

        let f = GG
            .clone()
            .reindex([4, 3, 2])
            .unwrap()
            .map_structure(|a| SymbolicTensor::from_named(&a).unwrap());

        let f_s = f.structure.structure.clone();

        // f.rep_permutation = f.rep_permutation.inverse();

        let f_p = f.permute_reps_wrapped().permute_inds();

        println!(
            "Structure:{}\nPermuted:{}\nPermuted Structure{}\nMetric simplified{}",
            f_s,
            f_p,
            f_p.structure,
            f_p.expression.simplify_metrics()
        );
    }

    #[test]
    fn chain_test() {
        initialize();

        let expr = parse_lit!(
            (-1 * spenso::P(2, spenso::mink(python::dim, python::l(20)))
                + spenso::P(1, spenso::mink(python::dim, python::l(20))))
                * 1𝑖
                * spenso::G
                ^ 2 * spenso::g(spenso::bis(4, python::l(2)), spenso::bis(4, python::l(4)))
                    * spenso::g(spenso::bis(4, python::l(3)), spenso::bis(4, python::l(7)))
                    * spenso::g(
                        spenso::mink(python::dim, python::l(0)),
                        spenso::mink(python::dim, python::l(5))
                    )
                    * spenso::g(
                        spenso::mink(python::dim, python::l(1)),
                        spenso::mink(python::dim, python::l(4))
                    )
                    * spenso::gamma(
                        spenso::bis(4, python::l(5)),
                        spenso::bis(4, python::l(4)),
                        spenso::mink(python::dim, python::l(4))
                    )
                    * spenso::gamma(
                        spenso::bis(4, python::l(6)),
                        spenso::bis(4, python::l(5)),
                        spenso::mink(python::dim, python::l(20))
                    )
                    * spenso::gamma(
                        spenso::bis(4, python::l(7)),
                        spenso::bis(4, python::l(6)),
                        spenso::mink(python::dim, python::l(5))
                    ),
            "spenso"
        );

        println!("Bef:{}", expr);
        println!("Aft:{}", expr.simplify_gamma());
    }

    #[test]
    fn gamma_alg() {
        initialize();
        let expr = parse_lit!(
            spenso::gamma_chain(mink(4, 0), bis(3), mink(4, 0), bis(1), bis(2)),
            "spenso"
        )
        .simplify_gamma();

        assert_eq!(
            expr,
            id!(spenso::bis(1), spenso::bis(2)) * 4,
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            p(mink(4, nu1))
                * (p(mink(4, nu3)) + q(mink(4, nu3)))
                * spenso::gamma_chain(
                    mink(4, nu1),
                    bis(3),
                    mink(4, mu),
                    bis(4),
                    mink(4, nu3),
                    bis(5),
                    mink(4, nu),
                    bis(1),
                    bis(1)
                ),
            "spenso"
        )
        .simplify_gamma()
        .expand()
        .replace(symbol!("spenso::nu3"))
        .with(symbol!("spenso::nu1"));
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
            "got \n{:>}",
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
                    bis(4, 3),
                    mink(4, mu),
                    bis(4, 4),
                    mink(4, nu),
                    bis(4, 5),
                    mink(4, nu3),
                    bis(4, 1),
                    bis(4, 1)
                ),
            "spenso"
        )
        .simplify_gamma()
        .replace(symbol!("spenso::nu3"))
        .with(symbol!("spenso::nu1"));
        assert_eq!(
            expr,
            parse_lit!(
                4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1))
                    ^ 2 + 4 * p(mink(4, mu)) * q(mink(4, nu)) - 4 * q(mink(4, mu)) * p(mink(4, nu))
                        + 4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1)) * q(mink(4, nu1)),
                "spenso"
            ),
            "got \n{:>+}",
            expr
        );

        let expr = parse_lit!(
            p(mink(dim, nu1))
                * (p(mink(dim, nu3)) + q(mink(dim, nu3)))
                * spenso::gamma_chain(
                    mink(dim, nu1),
                    bis(3),
                    mink(dim, nu),
                    bis(4),
                    mink(dim, nu),
                    bis(5),
                    mink(dim, nu3),
                    bis(1),
                    bis(1)
                ),
            "spenso"
        )
        .simplify_gamma()
        .replace(symbol!("spenso::nu1"))
        .with(symbol!("spenso::nu3"));
        assert_eq!(
            expr.expand(),
            parse_lit!(
                4 * dim * p(mink(dim, nu3)) ^ 2 + 4 * dim * p(mink(dim, nu3)) * q(mink(dim, nu3)),
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
                    bis(3),
                    mink(dim, nu),
                    bis(4),
                    mink(dim, nu3),
                    bis(5),
                    mink(dim, nu),
                    bis(1),
                    bis(1)
                ),
            "spenso"
        )
        .simplify_gamma()
        .replace(symbol!("spenso::nu1"))
        .with(symbol!("spenso::nu3"));
        assert_eq!(
            expr.expand(),
            parse_lit!(
                8 * p(mink(dim, nu3))
                    ^ 2 - 4 * dim * p(mink(dim, nu3))
                    ^ 2 + 8 * p(mink(dim, nu3)) * q(mink(dim, nu3))
                        - 4 * dim * p(mink(dim, nu3)) * q(mink(dim, nu3)),
                "spenso"
            ),
            "got {}",
            expr
        );

        let expr = parse_lit!(
            p(mink(dim, nu1))
                * q(mink(dim, nu2))
                * (p(mink(dim, nu3)) + q(mink(dim, nu3)))
                * q(mink(dim, nu4))
                * gamma_chain(
                    mink(dim, nu1),
                    bis(3),
                    mink(dim, nu4),
                    bis(4),
                    mink(dim, nu3),
                    bis(5),
                    mink(dim, nu2),
                    bis(1),
                    bis(1)
                ),
            "spenso"
        )
        .simplify_gamma()
        .to_dots();
        assert_eq!(
            expr,
            parse_lit!(
                8 * dot(p, q) ^ 2 - 4 * dot(p, p) * dot(q, q) + 4 * dot(p, q) * dot(q, q),
                "spenso"
            ),
            "got {}",
            expr
        );

        let expr = parse_lit!(
            spenso::gamma_chain(
                mink(dim, mu),
                bis(4, 3),
                mink(dim, nu),
                bis(4, 4),
                mink(dim, mu),
                bis(4, 5),
                mink(dim, nu),
                bis(4, 1),
                bis(4, 2)
            ),
            "spenso"
        )
        .simplify_gamma()
        .to_dots();

        let dim = Atom::var(symbol!("spenso::dim"));
        assert_eq!(
            expr,
            &dim * id!(spenso::bis(4, 1), spenso::bis(4, 2)) * 2
                - dim.pow(Atom::num(2)) * id!(spenso::bis(4, 1), spenso::bis(4, 2)),
            "got {:#}",
            expr
        );
    }
    #[test]
    fn gamma_alg_structures() {
        initialize();

        fn gamma(
            i: impl Into<AbstractIndex>,
            j: impl Into<AbstractIndex>,
            mu: impl Into<AbstractIndex>,
        ) -> Atom {
            let dim = symbol!("spenso::dim");
            let gamma_strct = IndexlessNamedStructure::<Symbol, ()>::from_iter(
                [
                    Bispinor {}.new_rep(dim).to_lib(),
                    Bispinor {}.new_rep(dim).cast(),
                    Minkowski {}.new_rep(dim).cast(),
                ],
                AGS.gamma,
                None,
            );
            gamma_strct
                .reindex([i.into(), j.into(), mu.into()])
                .unwrap()
                .map_structure(|a| SymbolicTensor::from_named(&a).unwrap())
                .permute_inds()
                .expression
                .simplify_metrics()
        }

        fn p(m: impl Into<AbstractIndex>) -> Atom {
            let m_atom: AbstractIndex = m.into();
            let m_atom: Atom = m_atom.into();
            let dim = symbol!("spenso::dim");
            let mink = Minkowski {}.new_rep(dim);
            function!(symbol!("spenso::p"), mink.to_symbolic([m_atom]))
        }
        fn q(m: impl Into<AbstractIndex>) -> Atom {
            let m_atom: AbstractIndex = m.into();
            let m_atom: Atom = m_atom.into();
            let dim = symbol!("spenso::dim");
            let mink = Minkowski {}.new_rep(dim);
            function!(symbol!("spenso::q"), mink.to_symbolic([m_atom]))
        }

        let mink = Minkowski {}.new_rep(symbol!("spenso::dim"));
        // gamma.reindex([1,2,3]).unwrap().map_structure(|a|)

        let expr = (p(1)
            * (p(3) + q(3))
            * gamma(1, 2, 1)
            * gamma(2, 3, 2)
            * gamma(3, 4, 3)
            * gamma(4, 1, 4))
        .simplify_gamma()
        .expand()
        .replace(mink.pattern(3))
        .with(mink.pattern(1));
        assert_eq!(
            expr,
            mink.g(2, 4) * p(1).npow(2) * -4 + p(2) * p(4) * 8 + p(2) * q(4) * 4 + p(4) * q(2) * 4
                - mink.g(2, 4) * p(1) * q(1) * 4,
            "got \n{:>+} diff:{}",
            expr,
            (&expr
                - (mink.g(2, 4) * p(1).npow(2) * -4
                    + p(2) * p(4) * 8
                    + p(2) * q(4) * 4
                    + p(4) * q(2) * 4
                    - mink.g(2, 4) * p(1) * q(1) * 4))
                .expand()
        );

        let expr = (mink.g(5, 6)
            * (mink.g(1, 2) * mink.g(3, 4) * mink.g(5, 6)
                - mink.g(1, 3) * mink.g(2, 6) * mink.g(5, 4))
            * (mink.g(1, 2) * mink.g(3, 4) - mink.g(1, 3) * mink.g(2, 4)))
        .simplify_gamma();

        assert_eq!(expr, parse_lit!(-dim + dim ^ 3, "spenso"), "got {}", expr);

        let expr = (p(1)
            * (p(3) + q(3))
            * gamma(1, 2, 1)
            * gamma(2, 3, 2)
            * gamma(3, 4, 4)
            * gamma(4, 1, 3))
        .simplify_gamma()
        .replace(mink.pattern(3))
        .with(mink.pattern(1));
        assert_eq!(
            expr,
            mink.g(2, 4) * p(1).npow(2) * 4 + p(2) * q(4) * 4 - q(2) * p(4) * 4
                + mink.g(2, 4) * p(1) * q(1) * 4,
            "got {}",
            expr
        );

        let dim = symbol!("spenso::dim");

        let expr = (p(1)
            * (p(3) + q(3))
            * gamma(1, 2, 1)
            * gamma(2, 3, 2)
            * gamma(3, 4, 2)
            * gamma(4, 1, 3))
        .simplify_gamma()
        .replace(mink.pattern(3))
        .with(mink.pattern(1));

        assert_eq!(
            expr,
            p(1).npow(2) * dim * 4 + p(1) * q(1) * dim * 4,
            "got {:#}",
            expr
        );

        let expr = (p(1)
            * q(2)
            * (p(3) + q(3))
            * q(4)
            * gamma(1, 2, 1)
            * gamma(2, 3, 4)
            * gamma(3, 4, 3)
            * gamma(4, 1, 2))
        .simplify_gamma()
        .to_dots();

        assert_eq!(
            expr,
            parse_lit!(
                8 * dot(p, q) ^ 2 - 4 * dot(p, p) * dot(q, q) + 4 * dot(p, q) * dot(q, q),
                "spenso"
            ),
            "got {}",
            expr
        );
    }

    #[test]
    fn normalise_g() {
        let expr = parse_lit!(
            gamma_chain(mink(dim, mu), bis(5), mink(dim, mu), bis(1), bis(2)),
            "spenso"
        );

        println!("{}", expr.simplify_gamma())
    }

    #[test]
    fn collect_expand_chain() {
        initialize();
        let expr = parse_lit!(gamma_chain(
            mink(dim, nu1),
            bis(dim, 3),
            mink(dim, nu12),
            bis(dim, 31),
            mink(dim, nu13),
            bis(dim, 32),
            mink(dim, nu14),
            bis(dim, 33),
            mink(dim, nu),
            bis(dim, 34),
            mink(dim, nu16),
            bis(dim, 35),
            mink(dim, nu17),
            bis(dim, 36),
            mink(dim, nu),
            bis(dim, 4),
            mink(dim, nu3),
            bis(dim, 5),
            mink(dim, nu2),
            bis(dim, 1),
            bis(dim, 2)
        ));

        let mut a = expr.clone();
        collect_gammas(&mut a);
        undo_gamma_chain(&mut a);
        collect_gammas(&mut a);

        assert_eq!(a, expr);
        // println!("{a}");
    }

    #[test]
    fn val_test() {
        initialize();
        // let expr = parse_lit!(
        //     (MB * g(bis(4, hedge(0, 0)), bis(4, hedge(1, 0)))
        //         + gamma(
        //             bis(4, hedge(0, 0)),
        //             bis(4, hedge(1, 0)),
        //             mink(4, edge(0, 1))
        //         ) * Q(0, mink(4, edge(0, 1))))
        //         * (MB * g(bis(4, hedge(4, 0)), bis(4, hedge(5, 0)))
        //             + gamma(
        //                 bis(4, hedge(4, 0)),
        //                 bis(4, hedge(5, 0)),
        //                 mink(4, edge(2, 1))
        //             ) * Q(2, mink(4, edge(2, 1))))
        //         * (MB * g(bis(4, hedge(8, 0)), bis(4, hedge(9, 0)))
        //             + gamma(
        //                 bis(4, hedge(8, 0)),
        //                 bis(4, hedge(9, 0)),
        //                 mink(4, edge(5, 1))
        //             ) * Q(5, mink(4, edge(5, 1))))
        //         * gamma(
        //             bis(4, hedge(1, 0)),
        //             bis(4, hedge(4, 0)),
        //             mink(4, hedge(10, 0))
        //         )
        //         * gamma(
        //             bis(4, hedge(5, 0)),
        //             bis(4, hedge(8, 0)),
        //             mink(4, hedge(2, 0))
        //         )
        //         * gamma(
        //             bis(4, hedge(9, 0)),
        //             bis(4, hedge(11, 0)),
        //             mink(4, hedge(7, 0))
        //         )
        //         * gamma(
        //             bis(4, hedge(11, 0)),
        //             bis(4, hedge(0, 0)),
        //             mink(4, hedge(2, 0))
        //         )
        //         * p(1, mink(4, hedge(10, 0)))
        //         * p(7, mink(4, hedge(7, 0))),
        //     "spenso"
        // );

        let expr = parse_lit!(
            (MB * g(bis(4, hedge(0, 0)), bis(4, hedge(1, 0)))
                + gamma(
                    bis(4, hedge(0, 0)),
                    bis(4, hedge(1, 0)),
                    mink(4, edge(0, 1))
                ) * Q(0, mink(4, edge(0, 1))))
                * (MB * g(bis(4, hedge(2, 0)), bis(4, hedge(3, 0)))
                    + gamma(
                        bis(4, hedge(2, 0)),
                        bis(4, hedge(3, 0)),
                        mink(4, edge(1, 1))
                    ) * Q(1, mink(4, edge(1, 1))))
                * (MB * g(bis(4, hedge(5, 0)), bis(4, hedge(6, 0)))
                    + gamma(
                        bis(4, hedge(5, 0)),
                        bis(4, hedge(6, 0)),
                        mink(4, edge(3, 1))
                    ) * Q(3, mink(4, edge(3, 1))))
                * (gamma(
                    bis(4, hedge(9, 0)),
                    bis(4, hedge(10, 0)),
                    mink(4, edge(5, 1))
                ) * Q(5, mink(4, edge(5, 1))))
                * gamma(
                    bis(4, hedge(1, 0)),
                    bis(4, hedge(9, 0)),
                    mink(4, hedge(7, 0))
                )
                * gamma(
                    bis(4, hedge(3, 0)),
                    bis(4, hedge(5, 0)),
                    mink(4, hedge(7, 0))
                )
                * gamma(
                    bis(4, hedge(6, 0)),
                    bis(4, hedge(0, 0)),
                    mink(4, hedge(11, 0))
                )
                * gamma(
                    bis(4, hedge(10, 0)),
                    bis(4, hedge(2, 0)),
                    mink(4, hedge(4, 0))
                )
                * p(1, mink(4, hedge(4, 0)))
                * p(7, mink(4, hedge(11, 0))),
            "spenso"
        );

        let expr = parse_lit!(
            g(mink(D, left(2)), mink(D, left(5)))
                * g(mink(D, left(2)), mink(D, right(2)))
                * g(mink(D, left(3)), mink(D, left(6)))
                * g(mink(D, left(3)), mink(D, right(3)))
                * g(mink(D, left(4)), mink(D, left(7)))
                * g(mink(D, left(5)), mink(D, left(6)))
                * g(mink(D, right(2)), mink(D, right(5)))
                * g(mink(D, right(3)), mink(D, right(6)))
                * g(mink(D, right(4)), mink(D, right(7)))
                * g(mink(D, right(6)), mink(D, right(7)))
                * g(bis(D, left(0)), bis(D, left(5)))
                * g(bis(D, left(1)), bis(D, left(4)))
                * g(bis(D, right(0)), bis(D, right(5)))
                * g(bis(D, right(1)), bis(D, right(4)))
                * gamma(bis(D, left(1)), bis(D, right(1)), mink(D, 1337))
                * gamma(bis(D, right(0)), bis(D, left(0)), mink(D, 1338))
                * gamma(bis(D, left(5)), bis(D, left(4)), mink(D, left(4)))
                * gamma(bis(D, right(4)), bis(D, right(5)), mink(D, right(4)))
                * Q(0, mink(D, 1338))
                * Q(1, mink(D, 1337))
                * Q(3, mink(D, left(7)))
                * Q(3, mink(D, right(5))),
            "spenso"
        );

        println!("Simplified:{:>}", expr.simplify_gamma().to_dots());

        // assert_eq!(
        //     expr.simplify_gamma().to_dots(),
        //     expr.simplify_gamma().simplify_gamma().to_dots(),
        //     "\n{:>}\n not equal to \n{:>}\n diff:\n{:>}",
        //     expr.simplify_gamma().to_dots(),
        //     expr.simplify_gamma().simplify_gamma().to_dots(),
        //     (expr.simplify_gamma().simplify_gamma().to_dots() - expr.simplify_gamma().to_dots())
        //         .expand()
        // )
        // println!("{}", expr.simplify_gamma().to_dots());

        // println!("{}", SpinAntiFundamental {}.to_symbolic([RS.a_]))
    }
}

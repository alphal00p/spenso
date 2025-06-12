use std::{collections::HashSet, sync::LazyLock};

use itertools::Itertools;
use spenso::{
    network::library::symbolic::ETS,
    structure::representation::{LibraryRep, Minkowski, RepName},
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    function,
    id::{MatchSettings, Pattern, Replacement},
    symbol,
};

use super::{
    metric::MetricSimplifier,
    representations::{Bispinor, ColorAdjoint, ColorFundamental},
};
use super::{rep_symbols::RS, representations::ColorSextet};

#[derive(Debug)]
pub enum ColorError {
    NotFully(Atom),
}

pub struct ColorSymbols {
    pub t: Symbol,
    pub f: Symbol,
    pub tr: Symbol,
    pub nc: Symbol,
}

pub static CS: LazyLock<ColorSymbols> = LazyLock::new(|| ColorSymbols {
    t: symbol!("spenso::t"),
    f: symbol!("spenso::f"),
    tr: symbol!("spenso::TR"),
    nc: symbol!("spenso::Nc"),
});

pub fn color_conj_impl(expression: AtomView) -> Atom {
    let expr = expression.to_owned();
    let cof = ColorFundamental {};
    let coaf = ColorFundamental {}.dual();

    let expr = expr
        .replace(
            function!(
                CS.t,
                RS.i_,
                cof.to_symbolic([RS.d_, RS.a_]),
                coaf.to_symbolic([RS.d_, RS.b_])
            )
            .to_pattern(),
        )
        .with(function!(
            CS.t,
            RS.i_,
            coaf.to_symbolic([RS.d_, RS.b_]),
            cof.to_symbolic([RS.d_, RS.a_])
        ));

    expr.replace_multiple(&[
        Replacement::new(
            coaf.to_symbolic([RS.a__]).to_pattern(),
            cof.to_symbolic([RS.a__]),
        ),
        Replacement::new(
            cof.to_symbolic([RS.a__]).to_pattern(),
            coaf.to_symbolic([RS.a__]),
        ),
    ])
}

pub trait SelectiveExpand {
    fn expand_in_patterns(&self, pats: &[Pattern]) -> Atom;
    fn expand_metrics(&self) -> Atom {
        let metric_pat = function!(ETS.metric, RS.a__).to_pattern();
        let id_pat = function!(ETS.id, RS.a__).to_pattern();

        self.expand_in_patterns(&[metric_pat, id_pat])
    }

    fn expand_color(&self) -> Atom {
        let cof = ColorFundamental {};
        let coaf = ColorFundamental {}.dual();
        let coad = ColorAdjoint {};

        let cof_pat = function!(RS.f_, RS.a___, cof.to_symbolic([RS.b__]), RS.c___).to_pattern();
        let coaf_pat = function!(RS.f_, RS.a___, coaf.to_symbolic([RS.b__]), RS.c___).to_pattern();
        let coad_pat = function!(RS.f_, RS.a___, coad.to_symbolic([RS.b__]), RS.c___).to_pattern();

        self.expand_in_patterns(&[cof_pat, coad_pat, coaf_pat])
    }

    fn expand_bis(&self) -> Atom {
        let bis = Bispinor {};

        let bis_pat = function!(RS.f_, RS.a___, bis.to_symbolic([RS.b__]), RS.c___).to_pattern();

        self.expand_in_patterns(&[bis_pat])
    }

    fn expand_mink(&self) -> Atom {
        let mink = Minkowski {};

        let mink_pat = function!(RS.f_, RS.a___, mink.to_symbolic([RS.b__]), RS.c___).to_pattern();

        self.expand_in_patterns(&[mink_pat])
    }

    fn expand_mink_bis(&self) -> Atom {
        let mink = Minkowski {};

        let mink_pat = function!(RS.f_, RS.a___, mink.to_symbolic([RS.b__]), RS.c___).to_pattern();

        let bis = Bispinor {};

        let bis_pat = function!(RS.f_, RS.a___, bis.to_symbolic([RS.b__]), RS.c___).to_pattern();

        self.expand_in_patterns(&[mink_pat, bis_pat])
    }
}
impl SelectiveExpand for Atom {
    fn expand_in_patterns(&self, pats: &[Pattern]) -> Atom {
        self.as_view().expand_in_patterns(pats)
    }
}

impl<'a> SelectiveExpand for AtomView<'a> {
    fn expand_in_patterns(&self, pats: &[Pattern]) -> Atom {
        let mut coefs = HashSet::new();

        for p in pats {
            for m in self.pattern_match(&p, None, None) {
                coefs.insert(p.replace_wildcards(&m));
            }
        }

        let coefs = coefs.into_iter().collect_vec();

        self.coefficient_list::<i8>(&coefs)
            .into_iter()
            .map(|(a, b)| a * b)
            .fold(Atom::Zero, |acc, b| acc + b)
    }
}

pub fn color_simplify_impl(expression: AtomView) -> Result<Atom, ColorError> {
    let cof = ColorFundamental {};
    let coaf = ColorFundamental {}.dual();
    let coad = ColorAdjoint {};
    let tr = Atom::var(CS.tr);
    let nc = Atom::var(CS.nc);
    let reps = vec![
        (
            function!(RS.f_, RS.a___, cof.to_symbolic([RS.b__]), RS.c___)
                * function!(
                    ETS.id,
                    coaf.to_symbolic([RS.b__]),
                    cof.to_symbolic([RS.c__])
                ),
            function!(RS.f_, RS.a___, cof.to_symbolic([RS.c__]), RS.c___),
        ),
        (
            function!(RS.f_, RS.a___, coaf.to_symbolic([RS.b__]), RS.c___)
                * function!(
                    ETS.id,
                    cof.to_symbolic([RS.b__]),
                    coaf.to_symbolic([RS.c__])
                ),
            function!(RS.f_, RS.a___, coaf.to_symbolic([RS.c__]), RS.c___),
        ),
        (
            function!(RS.f_, RS.a___, coad.to_symbolic([RS.b__]), RS.c___)
                * function!(
                    ETS.id,
                    coad.to_symbolic([RS.b__]),
                    coad.to_symbolic([RS.a__])
                ),
            function!(RS.f_, RS.a___, coad.to_symbolic([RS.a__]), RS.c___),
        ),
        (
            function!(RS.f_, RS.a___, coad.to_symbolic([RS.a__]), RS.c___)
                * function!(
                    ETS.id,
                    coad.to_symbolic([RS.b__]),
                    coad.to_symbolic([RS.a__])
                ),
            function!(RS.f_, RS.a___, coad.to_symbolic([RS.b__]), RS.c___),
        ),
        (
            function!(
                ETS.id,
                coaf.to_symbolic([RS.a__]),
                cof.to_symbolic([RS.a__])
            ),
            nc.clone(),
        ),
        (
            function!(
                ETS.id,
                cof.to_symbolic([RS.a__]),
                coaf.to_symbolic([RS.a__])
            ),
            nc.clone(),
        ),
        (
            function!(
                ETS.id,
                coad.to_symbolic([RS.a__]),
                coad.to_symbolic([RS.a__])
            ),
            (&nc * &nc) - 1,
        ),
        (
            function!(
                CS.t,
                RS.a_,
                cof.to_symbolic([RS.b__]),
                coaf.to_symbolic([RS.b__])
            ),
            Atom::num(0),
        ),
        (
            function!(
                CS.t,
                RS.a_,
                cof.to_symbolic([RS.c__]),
                coaf.to_symbolic([RS.e__])
            ) * function!(
                CS.t,
                RS.b_,
                cof.to_symbolic([RS.e__]),
                coaf.to_symbolic([RS.c__])
            ),
            &tr * function!(ETS.id, RS.a_, RS.b_),
        ),
        (
            function!(
                CS.t,
                RS.a_,
                cof.to_symbolic([RS.c__]),
                coaf.to_symbolic([RS.e__])
            )
            .pow(Atom::num(2)),
            &tr * function!(ETS.id, RS.a_, RS.a_),
        ),
        (
            function!(CS.t, RS.e_, RS.a_, RS.b_) * function!(CS.t, RS.e_, RS.c_, RS.d_),
            &tr * (function!(ETS.id, RS.a_, RS.d_) * function!(ETS.id, RS.c_, RS.b_)
                - (function!(ETS.id, RS.a_, RS.b_) * function!(ETS.id, RS.c_, RS.d_) / &nc)),
        ),
        (
            function!(CS.t, RS.i_, RS.a_, coaf.to_symbolic([RS.b__]))
                * function!(
                    CS.t,
                    RS.e_,
                    cof.to_symbolic([RS.b__]),
                    coaf.to_symbolic([RS.c__])
                )
                * function!(CS.t, RS.i_, cof.to_symbolic([RS.c__]), RS.d_),
            -(&tr / &nc) * function!(CS.t, RS.e_, RS.a_, RS.d_),
        ),
        (
            function!(
                CS.f,
                coad.to_symbolic([RS.a__]),
                coad.to_symbolic([RS.b__]),
                coad.to_symbolic([RS.c__])
            )
            .pow(Atom::num(2)),
            &nc * (&nc * &nc - 1),
        ),
    ];

    let i = symbol!("i");
    let j = symbol!("j");
    let k = symbol!("k");

    let frep = [Replacement::new(
        function!(
            CS.f,
            coad.to_symbolic([RS.d_, RS.a_]),
            coad.to_symbolic([RS.d_, RS.b_]),
            coad.to_symbolic([RS.d_, RS.c_])
        )
        .to_pattern(),
        (((function!(
            CS.t,
            coad.to_symbolic([RS.d_, RS.a_]),
            cof.to_symbolic([Atom::num(3), function!(i, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([Atom::num(3), function!(j, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_symbolic([RS.d_, RS.b_]),
            cof.to_symbolic([Atom::num(3), function!(j, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([Atom::num(3), function!(k, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_symbolic([RS.d_, RS.c_]),
            cof.to_symbolic([Atom::num(3), function!(k, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([Atom::num(3), function!(i, RS.a_, RS.b_, RS.c_)])
        ) - function!(
            CS.t,
            coad.to_symbolic([RS.d_, RS.a_]),
            cof.to_symbolic([Atom::num(3), function!(i, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([Atom::num(3), function!(j, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_symbolic([RS.d_, RS.c_]),
            cof.to_symbolic([Atom::num(3), function!(j, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([Atom::num(3), function!(k, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_symbolic([RS.d_, RS.b_]),
            cof.to_symbolic([Atom::num(3), function!(k, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([Atom::num(3), function!(i, RS.a_, RS.b_, RS.c_)])
        )) / &tr)
            * -Atom::i())
        .to_pattern(),
    )];

    let settings = MatchSettings {
        rhs_cache_size: 0,
        ..Default::default()
    };
    let replacements: Vec<Replacement> = reps
        .into_iter()
        .map(|(a, b)| {
            Replacement::new(a.to_pattern(), b.to_pattern()).with_settings(settings.clone())
        })
        .collect();

    let mut atom = Atom::num(0);
    // for r in &replacements {
    //     println!("{r}")
    // }

    let mut expression = expression.to_owned();
    let mut first = true;
    while first || expression.replace_multiple_into(&replacements, &mut atom) {
        if !first {
            std::mem::swap(&mut expression, &mut atom)
        };
        first = false;
        expression = expression.replace_multiple(&frep);
        expression = expression.expand_color();
        expression = expression.simplify_metrics();
    }

    let pats: Vec<LibraryRep> = vec![ColorAdjoint {}.into()];
    let dualizablepats: Vec<LibraryRep> = vec![ColorFundamental {}.into(), ColorSextet {}.into()];

    let mut fully_simplified = true;
    for p in pats.iter().chain(&dualizablepats) {
        if expression
            .pattern_match(&p.to_symbolic([RS.a__]).to_pattern(), None, None)
            .next()
            .is_some()
        {
            fully_simplified = false;
        }
    }

    if fully_simplified {
        Ok(expression)
    } else {
        Err(ColorError::NotFully(expression))
    }
}
/// Trait for applying SU(N) color algebra simplification rules to a symbolic expression.
///
/// Implementors provide a method to simplify expressions containing color factors
/// like structure constants (`f_abc`), generators (`T^a`), traces (`TR`), and the
/// number of colors (`Nc`).
pub trait ColorSimplifier {
    /// Attempts to simplify the color structure of the expression.
    ///
    /// Applies various identities of SU(N) algebra, including Fierz identities,
    /// Casimir relations, and contractions involving `f_abc` and `T^a`.
    ///
    /// # Returns
    /// - `Ok(Atom)`: The simplified expression, ideally with only color-scalar factors remaining.
    /// - `Err(ColorError::NotFully(Atom))`: If the simplification could not fully remove all
    ///   explicit color index structures (like `cof(...)`, `coad(...)`). The partially
    ///   simplified `Atom` is included in the error.
    fn simplify_color(&self) -> Result<Atom, ColorError>;
}
impl ColorSimplifier for Atom {
    fn simplify_color(&self) -> Result<Atom, ColorError> {
        color_simplify_impl(self.as_atom_view())
    }
}

impl<'a> ColorSimplifier for AtomView<'a> {
    fn simplify_color(&self) -> Result<Atom, ColorError> {
        color_simplify_impl(self.as_atom_view())
    }
}

#[cfg(test)]
mod test {
    use symbolica::{parse, parse_lit};

    use crate::{
        IndexTooling, gamma::GammaSimplifier, metric::MetricSimplifier, representations::initialize,
    };

    use super::*;

    #[test]
    fn test_color_simplification() {
        let atom = parse_lit!(spenso::f(coad(8, 2), coad(8, 2), coad(8, 1)), "spenso");
        println!("{atom}");
        let simplified = atom.simplify_color().unwrap();
        println!("{simplified}");
        assert_eq!(simplified, Atom::num(0));
    }

    fn colored_matrix_element() -> (Atom, Atom) {
        (
            parse!(
                "-G^2
            *(
                -g(mink(D,5),mink(D,6))*Q(2,mink(D,7))
                +g(mink(D,5),mink(D,6))*Q(3,mink(D,7))
                +g(mink(D,5),mink(D,7))*Q(2,mink(D,6))
                +g(mink(D,5),mink(D,7))*Q(4,mink(D,6))
                -g(mink(D,6),mink(D,7))*Q(3,mink(D,5))
                -g(mink(D,6),mink(D,7))*Q(4,mink(D,5))
            )
            *g(mink(D,4),mink(D,7))
            *spenso::t(coad(spenso::Nc^2-1,6),cof(spenso::Nc,5),dind(cof(spenso::Nc,4)))
            *spenso::f(coad(spenso::Nc^2-1,7),coad(spenso::Nc^2-1,8),coad(spenso::Nc^2-1,9))
            *𝟙(bis(D,0),bis(D,5))
            *𝟙(bis(D,1),bis(D,4))
            *𝟙(mink(D,2),mink(D,5))
            *𝟙(mink(D,3),mink(D,6))
            *𝟙(coad(spenso::Nc^2-1,2),coad(spenso::Nc^2-1,7))
            *𝟙(coad(spenso::Nc^2-1,3),coad(spenso::Nc^2-1,8))
            *𝟙(coad(spenso::Nc^2-1,6),coad(spenso::Nc^2-1,9))
            *𝟙(cof(spenso::Nc,0),dind(cof(spenso::Nc,5)))
            *𝟙(cof(spenso::Nc,4),dind(cof(spenso::Nc,1)))
            *spenso::gamma(mink(D,4),bis(D,5),bis(D,4))
            *spenso::vbar(1,bis(D,1))
            *spenso::u(0,bis(D,0))
            *spenso::ϵbar(2,mink(D,2))
            *spenso::ϵbar(3,mink(D,3))",
                "spenso"
            ),
            parse_lit!(
                -12 * spenso::TR
                    ^ 2 * spenso::Nc
                    ^ -1 * spenso::G
                    ^ 4 * (spenso::D - 2)
                    ^ -2 * (-2 * spenso::Nc + spenso::Nc ^ 3 + 3)
                        * (-2 * dot(spenso::Q(0), spenso::Q(1)) * dot(spenso::Q(2), spenso::Q(2))
                            + dot(spenso::Q(0), spenso::Q(1)) * dot(spenso::Q(2), spenso::Q(3))
                            - 3 * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(2), spenso::Q(4))
                            - 2 * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(3), spenso::Q(3))
                            - 3 * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(3), spenso::Q(4))
                            - 3 * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(4), spenso::Q(4))
                            + 2 * dot(spenso::Q(0), spenso::Q(2))
                                * dot(spenso::Q(1), spenso::Q(2))
                            - dot(spenso::Q(0), spenso::Q(2)) * dot(spenso::Q(1), spenso::Q(3))
                            + dot(spenso::Q(0), spenso::Q(2)) * dot(spenso::Q(1), spenso::Q(4))
                            - dot(spenso::Q(0), spenso::Q(3)) * dot(spenso::Q(1), spenso::Q(2))
                            + 2 * dot(spenso::Q(0), spenso::Q(3))
                                * dot(spenso::Q(1), spenso::Q(3))
                            + dot(spenso::Q(0), spenso::Q(3)) * dot(spenso::Q(1), spenso::Q(4))
                            + dot(spenso::Q(0), spenso::Q(4)) * dot(spenso::Q(1), spenso::Q(2))
                            + dot(spenso::Q(0), spenso::Q(4)) * dot(spenso::Q(1), spenso::Q(3))
                            + 2 * dot(spenso::Q(0), spenso::Q(4))
                                * dot(spenso::Q(1), spenso::Q(4))
                            + spenso::D
                                * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(2), spenso::Q(2))
                            - spenso::D
                                * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(2), spenso::Q(3))
                            + spenso::D
                                * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(2), spenso::Q(4))
                            + spenso::D
                                * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(3), spenso::Q(3))
                            + spenso::D
                                * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(3), spenso::Q(4))
                            + spenso::D
                                * dot(spenso::Q(0), spenso::Q(1))
                                * dot(spenso::Q(4), spenso::Q(4))
                            - spenso::D
                                * dot(spenso::Q(0), spenso::Q(2))
                                * dot(spenso::Q(1), spenso::Q(2))
                            + spenso::D
                                * dot(spenso::Q(0), spenso::Q(2))
                                * dot(spenso::Q(1), spenso::Q(3))
                            + spenso::D
                                * dot(spenso::Q(0), spenso::Q(3))
                                * dot(spenso::Q(1), spenso::Q(2))
                            - spenso::D
                                * dot(spenso::Q(0), spenso::Q(3))
                                * dot(spenso::Q(1), spenso::Q(3))),
                "symbolica_community"
            ),
        )
    }

    #[test]
    fn test_color_matrix_element() {
        initialize();
        let spin_sum_rule = parse!(
            "
            𝟙(coad(Nc^2-1, left(3)), coad(Nc^2-1, right(3)))
                * 𝟙(coad(Nc^2-1, left(2)), coad(Nc^2-1, right(2)))
                * 𝟙(cof(Nc, right(0)), dind(cof(Nc, left(0))))
                * 𝟙(cof(Nc, left(1)), dind(cof(Nc, right(1))))",
            "spenso"
        );

        let amplitude_color = parse!(
            "
            t(coad(Nc^2-1, 6), cof(Nc, 5), dind(cof(Nc, 4)))
                * f(coad(Nc^2-1, 7), coad(Nc^2-1, 8), coad(Nc^2-1, 9))
                * 𝟙(coad(Nc^2-1, 2), coad(Nc^2-1, 7))
                * 𝟙(coad(Nc^2-1, 3), coad(Nc^2-1, 8))
                * 𝟙(coad(Nc^2-1, 6), coad(Nc^2-1, 9))
                * 𝟙(cof(Nc, 0), dind(cof(Nc, 5)))
                * 𝟙(cof(Nc, 4), dind(cof(Nc, 1)))",
            "spenso"
        );
        let amplitude_color_left = amplitude_color.wrap_indices(symbol!("spenso::left"));
        let amplitude_color_right = amplitude_color
            .conj()
            .wrap_indices(symbol!("spenso::right"));

        println!("left{amplitude_color_left}");

        println!("right{amplitude_color_right}");
        let amp_squared_color = amplitude_color_left * spin_sum_rule * amplitude_color_right;
        let simplified_color = amp_squared_color
            .simplify_metrics()
            .simplify_color()
            .map_or_else(
                |a| match a {
                    ColorError::NotFully(a) => a,
                },
                |a| a,
            );
        println!("simplified_color={}", simplified_color);

        let spin_sum_rule_src = parse_lit!(
            spenso::vbar(1, bis(D, left(1)))
                * spenso::v(1, bis(D, right(1)))
                * spenso::u(0, bis(D, left(0)))
                * spenso::ubar(0, bis(D, right(0)))
                * spenso::ϵbar(2, mink(D, left(2)))
                * spenso::ϵ(2, mink(D, right(2)))
                * spenso::ϵbar(3, mink(D, left(3)))
                * spenso::ϵ(3, mink(D, right(3))),
            "spenso"
        );

        let spin_sum_rule_trg = parse!(
            "
            1/4*1/(D-2)^2*
            (
                (-1) * gamma(mink(D,1337),bis(D,left(1)),bis(D,right(1)))*Q(1,mink(D,1337))
                * gamma(mink(D,1338),bis(D,right(0)),bis(D,left(0)))*Q(0,mink(D,1338))
                * (-1) * g(mink(D,left(2)),mink(D,right(2)))
                * (-1) * g(mink(D,left(3)),mink(D,right(3)))
                )
                * (
                    𝟙(coad(Nc^2-1, left(3)), coad(Nc^2-1, right(3)))
                    * 𝟙(coad(Nc^2-1, left(2)), coad(Nc^2-1, right(2)))
                    * 𝟙(cof(Nc, right(0)), dind(cof(Nc, left(0))))
                    * 𝟙(cof(Nc, left(1)), dind(cof(Nc, right(1))))
                )",
            "spenso"
        );

        let (amplitude, tgt) = colored_matrix_element();

        let amplitude_left = amplitude.wrap_indices(symbol!("spenso::left"));

        println!("Amplitude left:\n{}", amplitude_left.collect_factors());

        println!(
            "Amplitude left cooked:\n{}",
            amplitude_left.collect_factors().cook_indices()
        );
        let amplitude_right = amplitude.wrap_indices(symbol!("spenso::right"));

        println!("Amplitude right:\n{}", amplitude_right.factor());

        println!("Amplitude right conj:\n{}", amplitude_right.conj().factor());

        let mut amp_squared = amplitude_left * amplitude_right.conj();

        println!("Amplitude squared:\n{}", amp_squared.factor());

        let dangling_atoms = amp_squared.list_dangling();

        assert_eq!(dangling_atoms.len(), 8);

        amp_squared = amp_squared
            .expand()
            .replace(spin_sum_rule_src.to_pattern())
            .with(spin_sum_rule_trg.to_pattern());

        println!("Amplitude squared spin-summed:\n{}", amp_squared);

        let mut simplified_amp_squared = amp_squared.clone();

        simplified_amp_squared = simplified_amp_squared.simplify_color().map_or_else(
            |a| match a {
                ColorError::NotFully(a) => a,
            },
            |a| a,
        );

        println!(
            "Color-simplified amplitude squared:\n{}",
            simplified_amp_squared
        );

        simplified_amp_squared = simplified_amp_squared.simplify_gamma();

        println!(
            "Gamma+color-simplified amplitude squared:\n{}",
            simplified_amp_squared
        );

        simplified_amp_squared = simplified_amp_squared.to_dots();

        assert_eq!(
            tgt,
            simplified_amp_squared.factor(),
            "{}\nnot equal to\n{}",
            tgt,
            simplified_amp_squared.factor()
        );
    }

    #[test]
    fn test_color_matrix_element_two() {
        initialize();

        let spin_sum_rule_src = parse_lit!(
            vbar(1, bis(D, left(1)))
                * v(1, bis(D, right(1)))
                * u(0, bis(D, left(0)))
                * ubar(0, bis(D, right(0)))
                * ϵbar(2, mink(D, left(2)))
                * ϵ(2, mink(D, right(2)))
                * ϵbar(3, mink(D, left(3)))
                * ϵ(3, mink(D, right(3))),
            "spenso"
        );

        let spin_sum_rule_trg = parse!(
            "
            1/4*1/(D-2)^2*
            (
                (-1) * gamma(mink(D,1337),bis(D,left(1)),bis(D,right(1)))*Q(1,mink(D,1337))
                * gamma(mink(D,1338),bis(D,right(0)),bis(D,left(0)))*Q(0,mink(D,1338))
                * (-1) * g(mink(D,left(2)),mink(D,right(2)))
                * (-1) * g(mink(D,left(3)),mink(D,right(3)))
                )
                ",
            "spenso"
        );

        let (amplitude, tgt) = colored_matrix_element();

        let amplitude_left = amplitude.wrap_dummies(symbol!("spenso::left"));

        println!("Amplitude left:\n{}", amplitude_left.collect_factors());

        let amplitude_right = amplitude.wrap_dummies(symbol!("spenso::right"));

        println!("Amplitude right:\n{}", amplitude_right.conj().factor());

        let mut amp_squared = amplitude_left * amplitude_right.conj();

        println!("Amplitude squared:\n{}", amp_squared.factor());

        let _spin_sum_pat = parse!(
            "gamma(mink(D,1337),bis(D,left(1)),bis(D,right(1)))",
            "spenso"
        )
        .to_pattern();
        amp_squared = amp_squared
            .expand_bis()
            .expand_mink()
            .replace(spin_sum_rule_src.to_pattern())
            .with(spin_sum_rule_trg.to_pattern());

        println!("Amplitude squared spin-summed:\n{}", amp_squared.factor());

        let mut simplified_amp_squared = amp_squared.clone();

        simplified_amp_squared = simplified_amp_squared.simplify_color().map_or_else(
            |a| match a {
                ColorError::NotFully(a) => a,
            },
            |a| a,
        );

        println!(
            "Color-simplified amplitude squared:\n{}",
            simplified_amp_squared
        );

        simplified_amp_squared = simplified_amp_squared
            .simplify_gamma()
            .expand_mink()
            .simplify_metrics();

        println!(
            "Gamma+color-simplified amplitude squared:\n{}",
            simplified_amp_squared
        );

        simplified_amp_squared = simplified_amp_squared.to_dots();

        assert_eq!(
            tgt,
            simplified_amp_squared.factor(),
            "{:#}\nnot equal to\n{:#}",
            tgt,
            simplified_amp_squared.factor()
        );
    }
}

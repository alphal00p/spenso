#![allow(uncommon_codepoints)]

use metric::{
    CookingError, cook_function_view, cook_indices_impl, list_dangling_impl, wrap_dummies_impl,
    wrap_indices_impl,
};
use spenso::structure::{
    representation::{Minkowski, RepName},
    slot::{AbsInd, DummyAind, ParseableAind},
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    function, symbol,
};
use thiserror::Error;

use crate::{gamma::AGS, metric::MetricSimplifier, rep_symbols::RS, representations::Bispinor};

pub mod color;
pub mod gamma;
pub mod metric;
pub mod parsing_ind;
#[cfg(feature = "python")]
pub mod python;
pub mod rep_symbols;
pub mod representations;

/// Defines operations related to manipulating abstract indices within symbolic expressions,
/// particularly relevant for physics calculations involving tensor structures and diagrams.
///
/// This trait provides methods for conjugating expressions, wrapping indices (both all and
/// only dummy/contracted ones), simplifying indices ("cooking"), and identifying external
/// ("dangling") indices.
pub trait IndexTooling {
    /// Wraps all abstract indices within the expression using a specified header symbol.
    ///
    /// This transforms indices like `mink(dim,idx)` into `mink(dim,header(idx))`. Useful for distinguishing
    /// between different copies of an expression, e.g., an amplitude and its complex conjugate.
    ///
    /// # Arguments
    /// * `header` - The [`Symbol`] to use as the wrapping function name.
    ///
    /// # Returns
    /// A new [`Atom`] with all indices wrapped.
    fn wrap_indices(&self, header: Symbol) -> Atom;

    /// Wraps only the dummy (contracted) abstract indices within the expression using a header symbol.
    ///
    /// Identifies indices that appear contracted (e.g., one covariant, one contravariant)
    /// and wraps only those, leaving external indices unchanged. Transforms `idx -> header(idx)`
    /// for dummy indices `idx`.
    ///
    /// # Arguments
    /// * `header` - The [`Symbol`] to use as the wrapping function name for dummy indices.
    ///
    /// # Returns
    /// A new [`Atom`] with only dummy indices wrapped.
    fn wrap_dummies<Aind: AbsInd + ParseableAind>(&self, header: Symbol) -> Atom;

    /// Simplifies structured indices within function arguments into flattened variable symbols.
    ///
    /// Replaces indices like `mink(4, mu)` inside function arguments (not top-level) with
    /// unique variable symbols like `var_mink_4_mu`. This can aid pattern matching.
    ///
    /// # Returns
    /// A new [`Atom`] with "cooked" indices.
    fn cook_indices(&self) -> Atom;

    /// Converts a single function [`Atom`] into a flattened variable symbol based on its name and arguments.
    ///
    /// Expects the input `Atom` to be a function. Returns a variable `Atom` whose symbol name
    /// encodes the original function and its arguments (e.g., `f(a, b)` might become `var_f_a_b`).
    /// Fails if the input is not a function or if arguments are not convertible (e.g., polynomials).
    ///
    /// # Returns
    /// `Ok(Atom)` containing the new variable symbol on success.
    /// `Err(CookingError)` if the input cannot be cooked.
    fn cook_function(&self) -> Result<Atom, CookingError>;

    /// Computes the physics-aware conjugate of the expression.
    ///
    /// Applies conjugation rules specific to physics objects like spinors, gamma matrices,
    /// color representations, and the imaginary unit `i`. See implementation details for
    /// specific rules applied.
    ///
    /// # Returns
    /// A new [`Atom`] representing the conjugated expression.
    fn dirac_adjoint<Aind: DummyAind + for<'a> TryFrom<AtomView<'a>> + Into<Atom>>(
        &self,
    ) -> Result<Atom, AdjointError>;

    fn conjugate_transpose(&self, rep: impl RepName) -> Atom;

    /// Identifies and returns a list of dangling (external, uncontracted) indices.
    ///
    /// Analyzes the expression to find indices that are not summed over. Returns them
    /// as `Atom`s. Note that dual indices might be represented wrapped in a `dind` function.
    ///
    /// # Returns
    /// A `Vec<Atom>` where each `Atom` represents a dangling index.
    fn list_dangling<Aind: AbsInd + ParseableAind>(&self) -> Vec<Atom>;
}

impl IndexTooling for Atom {
    fn wrap_indices(&self, header: Symbol) -> Atom {
        self.as_view().wrap_indices(header)
    }
    fn wrap_dummies<Aind: AbsInd + ParseableAind>(&self, header: Symbol) -> Atom {
        self.as_view().wrap_dummies::<Aind>(header)
    }
    fn cook_indices(&self) -> Atom {
        self.as_view().cook_indices()
    }

    fn cook_function(&self) -> Result<Atom, CookingError> {
        self.as_view().cook_function()
    }

    fn dirac_adjoint<Aind: DummyAind + for<'a> TryFrom<AtomView<'a>> + Into<Atom>>(
        &self,
    ) -> Result<Atom, AdjointError> {
        self.as_view().dirac_adjoint::<Aind>()
    }

    fn conjugate_transpose(&self, rep: impl RepName) -> Atom {
        self.as_view().conjugate_transpose(rep)
    }
    fn list_dangling<Aind: AbsInd + ParseableAind>(&self) -> Vec<Atom> {
        self.as_view().list_dangling::<Aind>()
    }
}

#[derive(Error, Debug)]
pub enum AdjointError {
    #[error("Dummies already present:{0}")]
    DummiesAlready(Atom),
}

impl IndexTooling for AtomView<'_> {
    fn conjugate_transpose(&self, rep: impl RepName) -> Atom {
        let transpose_pat = function!(
            RS.a_,
            RS.a___,
            rep.to_symbolic([RS.d_, RS.i_]),
            rep.to_symbolic([RS.d_, RS.j_]),
            RS.b___
        )
        .to_pattern();

        let transpose_rhs = function!(
            RS.a_,
            RS.a___,
            rep.to_symbolic([RS.d_, RS.j_]),
            rep.to_symbolic([RS.d_, RS.i_]),
            RS.b___
        )
        .to_pattern();
        self.conj().replace(transpose_pat).with(transpose_rhs)
    }

    fn dirac_adjoint<Aind: DummyAind + for<'a> TryFrom<AtomView<'a>> + Into<Atom>>(
        &self,
    ) -> Result<Atom, AdjointError> {
        let mut a = self.conj();

        let dummy = symbol!("dummy");

        let vector_pat = function!(
            RS.a_,
            RS.a___,
            Bispinor {}.to_symbolic([RS.d_, RS.i_]),
            RS.b___
        )
        .to_pattern();

        let vector_rhs = (function!(
            RS.a_,
            RS.a___,
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.i_)]),
            RS.b___
        ) * function!(
            AGS.gamma0,
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.i_)]),
            Bispinor {}.to_symbolic([RS.d_, RS.i_])
        ))
        .to_pattern();

        let transpose_pat = function!(
            RS.a_,
            RS.a___,
            Bispinor {}.to_symbolic([RS.d_, RS.i_]),
            Bispinor {}.to_symbolic([RS.d_, RS.j_]),
            RS.b___
        )
        .to_pattern();

        let transpose_rhs = (function!(
            AGS.gamma0,
            Bispinor {}.to_symbolic([RS.d_, RS.j_]),
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.j_)])
        ) * function!(
            RS.a_,
            RS.a___,
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.j_)]),
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.i_)]),
            RS.b___
        ) * function!(
            AGS.gamma0,
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.i_)]),
            Bispinor {}.to_symbolic([RS.d_, RS.i_])
        ))
        .to_pattern();

        let bispat = Bispinor {}.to_symbolic([RS.e__]).to_pattern();
        let bispatc = Bispinor {}.to_symbolic([RS.e__]).to_pattern();
        let dummypati = function!(dummy, RS.i_).to_pattern();
        let dummypatic = dummypati.clone();
        let dummypatj = function!(dummy, RS.j_).to_pattern();

        let cond = RS.a___.filter(move |a| {
            a.to_atom()
                .pattern_match(&bispat, None, None)
                .next()
                .is_none()
        }) & RS.b___.filter(move |a| {
            a.to_atom()
                .pattern_match(&bispatc, None, None)
                .next()
                .is_none()
        });
        a = a
            .replace(transpose_pat)
            .when(&cond)
            .with_map(move |m| {
                let a = transpose_rhs.replace_wildcards_with_matches(m);
                let i = dummypati.replace_wildcards_with_matches(m);
                let j = dummypatj.replace_wildcards_with_matches(m);
                a.replace(i)
                    .with(Aind::new_dummy().into())
                    .replace(j)
                    .with(Aind::new_dummy().into())
            })
            .replace(vector_pat)
            .when(&cond)
            .with_map(move |m| {
                let a = vector_rhs.replace_wildcards_with_matches(m);
                let i = dummypatic.replace_wildcards_with_matches(m);
                a.replace(i).with(Aind::new_dummy().into())
            });

        let repeated_gamma0 =
            AGS.gamma0_pattern(RS.a__, RS.b__) * AGS.gamma0_pattern(RS.b__, RS.c__);

        let conj_gamma = function!(
            AGS.gamma,
            Bispinor {}.to_symbolic([RS.d_, RS.j_]),
            Bispinor {}.to_symbolic([RS.d_, RS.i_]),
            Minkowski {}.to_symbolic([RS.a__])
        )
        .conj();

        let conj_gamma_rhs = (function!(
            AGS.gamma0,
            Bispinor {}.to_symbolic([RS.d_, RS.j_]),
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.j_)])
        ) * function!(
            AGS.gamma,
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.j_)]),
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.i_)]),
            Minkowski {}.to_symbolic([RS.a__])
        ) * function!(
            AGS.gamma0,
            Bispinor {}.to_symbolic([Atom::var(RS.d_), function!(dummy, RS.i_)]),
            Bispinor {}.to_symbolic([RS.d_, RS.i_])
        ));

        Ok(a.replace(conj_gamma)
            .with(conj_gamma_rhs)
            .replace(repeated_gamma0)
            .repeat()
            .with(Bispinor {}.metric_atom([RS.a__], [RS.c__]))
            .simplify_metrics())
    }

    fn cook_function(&self) -> Result<Atom, CookingError> {
        cook_function_view(*self)
    }
    fn wrap_indices(&self, header: Symbol) -> Atom {
        wrap_indices_impl(*self, header)
    }
    fn cook_indices(&self) -> Atom {
        cook_indices_impl(*self)
    }
    fn wrap_dummies<Aind: AbsInd + ParseableAind>(&self, header: Symbol) -> Atom {
        wrap_dummies_impl::<Aind>(*self, header)
    }
    fn list_dangling<Aind: AbsInd + ParseableAind>(&self) -> Vec<Atom> {
        list_dangling_impl::<Aind>(*self)
    }
}

#[cfg(test)]
mod test {
    use spenso::structure::{
        IndexlessNamedStructure,
        abstract_index::AbstractIndex,
        representation::{Minkowski, RepName},
    };
    use symbolica::{
        atom::{Atom, AtomCore, Symbol},
        function, symbol,
    };

    use crate::{
        IndexTooling,
        gamma::{AGS, GammaSimplifier},
        metric::{MetricSimplifier, PermuteWithMetric},
        representations::Bispinor,
    };
    pub fn gamma(
        i: impl Into<AbstractIndex>,
        j: impl Into<AbstractIndex>,
        mu: impl Into<AbstractIndex>,
    ) -> Atom {
        let gamma_strct = IndexlessNamedStructure::<Symbol, ()>::from_iter(
            [
                Bispinor {}.new_rep(4).to_lib(),
                Bispinor {}.new_rep(4).cast(),
                Minkowski {}.new_rep(4).cast(),
            ],
            AGS.gamma,
            None,
        );
        gamma_strct
            .reindex([i.into(), j.into(), mu.into()])
            .unwrap()
            .permute_with_metric()
    }

    pub fn u(i: usize, m: impl Into<AbstractIndex>) -> Atom {
        let m_atom: AbstractIndex = m.into();
        let m_atom: Atom = m_atom.into();
        let mink = Bispinor {}.new_rep(4);
        function!(symbol!("spenso::u"), i, mink.to_symbolic([m_atom]))
    }
    #[test]
    fn gamma_conj() {
        let expr = gamma(1, 2, 3).dirac_adjoint::<AbstractIndex>().unwrap();
        println!("{expr}");

        let ubgu = (u(2, 2) * gamma(1, 2, 3) * (u(1, 1).dirac_adjoint::<AbstractIndex>().unwrap()));
        let expr = ubgu.dirac_adjoint::<AbstractIndex>().unwrap();
        println!("{expr}");

        let mut exp = ubgu.conjugate_transpose(Bispinor {});

        println!("conj trans {exp}");
        exp = exp.simplify_gamma_conj::<AbstractIndex>().unwrap();
        println!("conj gamma simplify {exp}");
        exp = exp.simplify_gamma0();
        println!("gamma0 simplify {exp}");
        exp = exp.simplify_metrics();
        println!("simplify metrics {exp}");

        println!("{exp}")
    }
}

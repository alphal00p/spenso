use color::color_conj_impl;
use gamma::{factor_conj_impl, gamma_conj_impl, pol_conj_impl};
use metric::{
    CookingError, cook_function_view, cook_indices_impl, list_dangling_impl, wrap_dummies_impl,
    wrap_indices_impl,
};

use symbolica::atom::{Atom, AtomView, Symbol};

pub mod color;
pub mod gamma;
pub mod metric;
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
    fn wrap_dummies(&self, header: Symbol) -> Atom;

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
    fn conj(&self) -> Atom;

    /// Identifies and returns a list of dangling (external, uncontracted) indices.
    ///
    /// Analyzes the expression to find indices that are not summed over. Returns them
    /// as `Atom`s. Note that dual indices might be represented wrapped in a `dind` function.
    ///
    /// # Returns
    /// A `Vec<Atom>` where each `Atom` represents a dangling index.
    fn list_dangling(&self) -> Vec<Atom>;
}

impl IndexTooling for Atom {
    fn wrap_indices(&self, header: Symbol) -> Atom {
        self.as_view().wrap_indices(header)
    }
    fn wrap_dummies(&self, header: Symbol) -> Atom {
        self.as_view().wrap_dummies(header)
    }
    fn cook_indices(&self) -> Atom {
        self.as_view().cook_indices()
    }

    fn cook_function(&self) -> Result<Atom, CookingError> {
        self.as_view().cook_function()
    }
    fn conj(&self) -> Atom {
        self.as_view().conj()
    }
    fn list_dangling(&self) -> Vec<Atom> {
        self.as_view().list_dangling()
    }
}

impl<'a> IndexTooling for AtomView<'a> {
    fn conj(&self) -> Atom {
        factor_conj_impl(
            pol_conj_impl(gamma_conj_impl(color_conj_impl(*self).as_view()).as_view()).as_view(),
        )
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
    fn wrap_dummies(&self, header: Symbol) -> Atom {
        wrap_dummies_impl(*self, header)
    }
    fn list_dangling(&self) -> Vec<Atom> {
        list_dangling_impl(*self)
    }
}

use crate::color::{ColorSimplifier, SelectiveExpand};
use crate::gamma::GammaSimplifier;
use crate::metric::MetricSimplifier;

use pyo3::{
    Bound, PyResult,
    exceptions::PyTypeError,
    pyfunction,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};
use symbolica::atom::Atom;

use crate::IndexTooling;
use crate::representations::initialize;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use symbolica::atom::Symbol;

use symbolica::api::python::PythonExpression;

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Calculates the physics-aware conjugate of the expression.
///
/// This considers the conjugation rules for various physics objects:
/// - Complex numbers: `i -> -i`
/// - Polarization vectors: `eps(p) <-> epsbar(p)`
/// - Spinors: `u(p) <-> ubar(p)`, `v(p) <-> vbar(p)`
/// - Gamma matrices: `gamma(mu, a, b) -> -gamma(mu, b, a)` (note the index swap and sign)
/// - Gamma5: `gamma5(a, b) -> gamma5(b, a)`
/// - Color generators: `t(i, a, b) -> t(i, b, a)` (for fundamental `a`, antifundamental `b`)
/// - Color representations: Switches fundamental and anti-fundamental representations.
///
/// # Args:
///     self_ (Expression): The expression to conjugate.
///
/// # Returns:
///     Expression: The conjugated expression.
pub fn conj(self_: &PythonExpression) -> PythonExpression {
    self_.expr.conj().into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
pub fn expand_mink(self_: &PythonExpression) -> PythonExpression {
    self_
        .expr
        .expand_mink()
        .iter()
        .fold(Atom::Zero, |a, (c, s)| a + c * s)
        .into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
pub fn expand_bis(self_: &PythonExpression) -> PythonExpression {
    self_
        .expr
        .expand_bis()
        .iter()
        .fold(Atom::Zero, |a, (c, s)| a + c * s)
        .into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
pub fn expand_mink_bis(self_: &PythonExpression) -> PythonExpression {
    self_
        .expr
        .expand_mink_bis()
        .iter()
        .fold(Atom::Zero, |a, (c, s)| a + c * s)
        .into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
pub fn expand_color(self_: &PythonExpression) -> PythonExpression {
    self_
        .expr
        .expand_color()
        .iter()
        .fold(Atom::Zero, |a, (c, s)| a + c * s)
        .into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
pub fn expand_metrics(self_: &PythonExpression) -> PythonExpression {
    self_
        .expr
        .expand_metrics()
        .iter()
        .fold(Atom::Zero, |a, (c, s)| a + c * s)
        .into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Wraps all abstract indices within the expression using a header symbol.
///
/// This is often used to distinguish indices belonging to different parts
/// of a calculation, e.g., the amplitude and its conjugate in an amplitude squared.
/// For an index `idx`, the transformation is `idx -> header(idx)`.
///
/// # Example:
///     `wrap_indices(expr, symbol("left"))` might turn `p(mink(4, mu))`
///     into `p(mink(4, left(mu)))`.
///
/// # Args:
///     self_ (Expression): The input expression.
///     header (Symbol): The symbol to use as the wrapper function name.
///
/// # Returns:
///     Expression: A new expression with all indices wrapped.
pub fn wrap_indices(self_: &PythonExpression, header: Symbol) -> PythonExpression {
    self_.expr.wrap_indices(header).into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// "Cooks" indices within function arguments into simplified, unique symbols.
///
/// This process takes structured indices like `mink(4, f(g(mu)))` appearing as
/// arguments to functions (but not the top-level function arguments themselves)
/// and replaces them with flattened symbols like `mink(4,f_g_mu)`.
///
/// # Args:
///     self_ (Expression): The expression containing indices to be cooked.
///
/// # Returns:
///     Expression: A new expression with cooked indices inside function arguments.
pub fn cook_indices(self_: &PythonExpression) -> PythonExpression {
    self_.expr.cook_indices().into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Converts a single function atom into a flattened variable symbol.
///
/// Takes an expression that must be a single function call (e.g., `f(a, b)`)
/// and converts it into a variable symbol whose name encodes the original
/// function and its arguments (e.g., `f_a_b`).
///
/// # Args:
///     self_ (Expression): The expression representing the function atom to cook.
///         Must not be a sum, product, power, variable, or number.
///
/// # Returns:
///     Expression: An expression representing the new variable symbol.
///
/// # Raises:
///     TypeError: If the input expression is not a single function atom or
///         if arguments contain types that cannot be cooked (e.g., polynomials).
pub fn cook_function(self_: &PythonExpression) -> PyResult<PythonExpression> {
    self_
        .expr
        .cook_function()
        .map_err(|a| PyTypeError::new_err(format!("cannot cook: {a:?}")))
        .map(|a| a.into())
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Wraps only the dummy (contracted) indices within the expression using a header symbol.
///
/// Similar to `wrap_indices`, but it identifies contracted indices (those appearing
/// once upstairs and once downstairs, or twice in a self-dual representation)
/// and only wraps those, leaving external (dangling) indices untouched.
///
/// # Example:
///     `wrap_dummies(term1 * term2, symbol("internal"))` where `term1` and `term2`
///     share a contracted index `mu`, might wrap `mu` resulting in
///     `internal(mu)` where it appears, but leave other external indices as they are.
///
/// # Args:
///     self_ (Expression): The input expression.
///     header (Symbol): The symbol to use as the wrapper function name for dummy indices.
///
/// # Returns:
///     Expression: A new expression with only dummy indices wrapped.
pub fn wrap_dummies(self_: &PythonExpression, header: Symbol) -> PythonExpression {
    self_.expr.wrap_dummies(header).into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Lists the dangling (external, uncontracted) indices present in the expression.
///
/// Identifies indices that are not summed over (i.e., not dummy indices).
/// For dualizable representations, downstairs indices are represented wrapped
/// in `dind(...)`.
///
/// # Args:
///     self_ (Expression): The expression to analyze.
///
/// # Returns:
///     list[Expression]: A list of expressions, each representing a dangling index.
///
pub fn list_dangling(self_: &PythonExpression) -> Vec<PythonExpression> {
    self_
        .expr
        .list_dangling()
        .into_iter()
        .map(|a| a.into())
        .collect()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Applies Clifford algebra rules and trace identities to simplify gamma matrices.
///
/// Performs simplifications based on the anticommutation relations:
/// `{gamma(mu), gamma(nu)} = 2 * g(mu, nu)`
/// and evaluates traces of gamma matrix chains. Assumes gamma matrices
/// are represented by `alg::gamma(...)` and the metric by `spenso::g(...)`.
/// Uses internal helper symbols like `alg::gamma_chain` and `alg::gamma_trace`.
///
/// # Args:
///     self_ (Expression): The expression containing gamma matrices.
///
/// # Returns:
///     Expression: The simplified expression.
pub fn simplify_gamma(self_: &PythonExpression) -> PythonExpression {
    self_.expr.simplify_gamma().into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Converts contracted Lorentz/Minkowski indices into dot product notation.
///
/// Looks for patterns like `p(mink(D, mu)) * q(mink(D, mu))` and replaces
/// them with `dot(p, q)`. Assumes vectors are represented by functions
/// taking a single Minkowski index.
///
/// # Args:
///     self_ (Expression): The expression with contracted indices.
///
/// # Returns:
///     Expression: The expression with contractions replaced by `dot(...)` calls.
pub fn to_dots(self_: &PythonExpression) -> PythonExpression {
    self_.expr.to_dots().into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Simplifies contractions involving metric tensors and identity tensors.
///
/// Applies rules like:
/// - `g(mu, nu) * p(nu) -> p(mu)`
/// - `g(mu, nu) * g(nu, rho) -> g(mu, rho)` (or `id(mu, rho)`)
/// - `g(mu, mu) -> D` (dimension)
/// - `id(mu, nu) * p(nu) -> p(mu)`
/// Assumes the metric tensor is `g(...)` or `metric(...)` and the identity
/// is `id(...)` or `𝟙(...)`.
///
/// # Args:
///     self_ (Expression): The expression with metric/identity tensors.
///
/// # Returns:
///     Expression: The simplified expression.
pub fn simplify_metrics(self_: &PythonExpression) -> PythonExpression {
    self_.expr.simplify_metrics().into()
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.idenso")
)]
#[pyfunction]
/// Applies SU(N) color algebra rules to simplify color structures.
///
/// Performs simplifications involving:
/// - Structure constants `alg::f(a, b, c)`
/// - Generators `alg::t(a, i, j)`
/// - Traces `alg::TR`
/// - Number of colors `alg::Nc`
/// - Casimir invariants, Fierz identities, etc.
///
/// Note: Simplification might not be complete for all possible color structures.
/// If the result still contains explicit color indices (like `cof(...)`, `coad(...)`),
/// simplification was not fully successful.
///
/// # Args:
///     self_ (Expression): The expression with color factors.
///
/// # Returns:
///     Expression: The simplified expression, potentially containing only color-scalar factors
///                 like `Nc`, `TR`, `CF`, `CA`.
///
/// Raises:
///     RuntimeWarning: If the simplification could not fully eliminate all explicit
///                     color indices, indicating an incomplete simplification. The
///                     partially simplified expression is still returned.
pub fn simplify_color(self_: &PythonExpression) -> PythonExpression {
    self_.expr.simplify_color().into()
}

pub struct IdensoModule;

impl symbolica::api::python::SymbolicaCommunityModule for IdensoModule {
    fn get_name() -> String {
        "idenso".into()
    }

    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        initialize_alg_simp(m)
    }
}

pub(crate) fn initialize_alg_simp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    initialize();

    m.add_function(wrap_pyfunction!(simplify_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(to_dots, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_color, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(cook_indices, m)?)?;
    m.add_function(wrap_pyfunction!(cook_function, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_dummies, m)?)?;
    m.add_function(wrap_pyfunction!(list_dangling, m)?)?;
    m.add_function(wrap_pyfunction!(conj, m)?)?;
    m.add_function(wrap_pyfunction!(expand_bis, m)?)?;
    m.add_function(wrap_pyfunction!(expand_mink_bis, m)?)?;
    m.add_function(wrap_pyfunction!(expand_mink, m)?)?;
    m.add_function(wrap_pyfunction!(expand_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(expand_color, m)?)?;

    Ok(())
}

use std::sync::Arc;

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use dyn_clone::DynClone;
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    coefficient::{Coefficient, CoefficientView, ConvertToRing},
    domains::{
        atom::AtomField,
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
        },
        float::{Real, SingleFloat},
        integer::Z,
        rational::Rational,
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
        EuclideanDomain, InternalOrdering,
    },
    evaluate::{EvaluationFn, ExpressionEvaluator, FunctionMap, OptimizationSettings},
    id::{
        BorrowPatternOrMap, BorrowReplacement, Condition, ConditionResult, Context, MatchSettings,
        Pattern, PatternRestriction,
    },
    poly::{
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, series::Series,
        Exponent, PositiveExponent, Variable,
    },
    tensors::matrix::Matrix,
};

use crate::{
    data::{DataTensor, DenseTensor, SparseTensor, StorageTensor},
    iterators::IteratableTensor,
    structure::{HasStructure, TensorStructure},
};

use super::{EvalTensor, EvalTreeTensor, ParamTensor};

pub trait PatternReplacement {
    fn replace_all_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    );

    fn replace_all_repeat<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self;

    fn replace_all_repeat_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    );

    fn replace_all_multiple_repeat<T: BorrowReplacement>(&self, replacements: &[T]) -> Self;
    fn replace_all_multiple_mut<T: BorrowReplacement>(&mut self, replacements: &[T]);
    fn replace_all_multiple_repeat_mut<T: BorrowReplacement>(&mut self, replacements: &[T]);
    fn replace_map_mut<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&mut self, m: &F);
}

impl PatternReplacement for Atom {
    fn replace_all_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        let a = self.replace_all(pattern, rhs, conditions, settings);
        *self = a;
    }

    fn replace_all_repeat<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Atom {
        let mut out = self.clone();
        let rhs = rhs.borrow();
        while self.replace_all_into(pattern, rhs, conditions, settings, &mut out) {}
        out
    }

    fn replace_all_repeat_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        let atom = self.replace_all_repeat(pattern, rhs, conditions, settings);
        *self = atom;
    }

    fn replace_all_multiple_mut<T: BorrowReplacement>(&mut self, replacements: &[T]) {
        *self = self.as_atom_view().replace_all_multiple(replacements);
    }

    fn replace_all_multiple_repeat<T: BorrowReplacement>(&self, replacements: &[T]) -> Atom {
        let mut out = self.clone();
        while self
            .as_atom_view()
            .replace_all_multiple_into(replacements, &mut out)
        {}
        out
    }

    fn replace_all_multiple_repeat_mut<T: BorrowReplacement>(&mut self, replacements: &[T]) {
        let atom = self.replace_all_multiple_repeat(replacements);
        *self = atom;
    }

    fn replace_map_mut<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&mut self, m: &F) {
        *self = self.replace_map(m);
    }
}

pub trait ClonableAtomMap: Fn(AtomView, &mut Atom) + DynClone {}

pub trait TensorAtomMapsTest {
    type ContainerData<T>;
}

impl<S: StorageTensor<Data = Atom>> TensorAtomMapsTest for S {
    type ContainerData<T> = <S as StorageTensor>::ContainerData<T>;
}
pub trait TensorAtomMaps {
    type ContainerData<T>;

    fn replace_all_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    );

    fn replace_all_repeat<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self;

    fn replace_all_repeat_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    );

    fn replace_all_multiple_repeat<T: BorrowReplacement>(&self, replacements: &[T]) -> Self;
    fn replace_all_multiple_mut<T: BorrowReplacement>(&mut self, replacements: &[T]);
    fn replace_all_multiple_repeat_mut<T: BorrowReplacement>(&mut self, replacements: &[T]);
    fn replace_map_mut<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&mut self, m: &F);
    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    // fn collect<E: Exponent, T: AtomCore>(
    //     &self,
    //     x: T,
    //     key_map: Option<Box<dyn ClonableAtomMap>>,
    //     coeff_map: Option<Box<dyn ClonableAtomMap>>,
    // ) -> Self;

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    // fn collect_multiple<E: Exponent, T: AtomCore>(
    //     &self,
    //     xs: &[T],
    //     key_map: Option<Box<dyn ClonableAtomMap>>,
    //     coeff_map: Option<Box<dyn ClonableAtomMap>>,
    // ) -> Self;

    /// Collect terms involving the literal occurrence of `x`.
    fn coefficient<T: AtomCore>(&self, x: T) -> Self;

    /// Write the expression over a common denominator.
    fn together(&self) -> Self;

    /// Write the expression as a sum of terms with minimal denominators.
    fn apart(&self, x: Symbol) -> Self;

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    fn cancel(&self) -> Self;

    /// Factor the expression over the rationals.
    fn factor(&self) -> Self;

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    fn collect_num(&self) -> Self;

    /// Expand an expression. The function [AtomCore::expand_via_poly] may be faster.
    fn expand(&self) -> Self;

    /// Expand the expression by converting it to a polynomial, optionally
    /// only in the indeterminate `var`. The parameter `E` should be a numerical type
    /// that fits the largest exponent in the expanded expression. Often,
    /// `u8` or `u16` is sufficient.
    fn expand_via_poly<E: Exponent, T: AtomCore>(&self, var: Option<T>) -> Self;

    /// Expand an expression in the variable `var`. The function [AtomCore::expand_via_poly] may be faster.
    fn expand_in<T: AtomCore>(&self, var: T) -> Self;

    /// Expand an expression in the variable `var`.
    fn expand_in_symbol(&self, var: Symbol) -> Self;

    // /// Expand an expression, returning `true` iff the expression changed.
    // fn expand_into(&self, var: Option<AtomView>, out: &mut Atom) -> bool {
    //     self.as_atom_view().expand_into(var, out)
    // }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    fn expand_num(&self) -> Self;

    /// Take a derivative of the expression with respect to `x`.
    fn derivative(&self, x: Symbol) -> Self;

    // /// Take a derivative of the expression with respect to `x` and
    // /// write the result in `out`.
    // /// Returns `true` if the derivative is non-zero.
    // fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
    //     self.as_atom_view().derivative_into(x, out)
    // }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    fn series<T: AtomCore>(
        &self,
        x: Symbol,
        expansion_point: T,
        depth: Rational,
        depth_is_absolute: bool,
    ) -> Result<Self::ContainerData<Series<AtomField>>, &'static str>;

    /// Find the root of a function in `x` numerically over the reals using Newton's method.
    fn nsolve<N: SingleFloat + Real + PartialOrd + Clone>(
        &self,
        x: Symbol,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> Result<Self::ContainerData<N>, String>;

    /// Evaluate a (nested) expression a single time.
    /// For repeated evaluations, use [Self::evaluator()] and convert
    /// to an optimized version or generate a compiled version of your expression.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    fn evaluate<T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomView<'_>, T>,
        function_map: &HashMap<Symbol, EvaluationFn<T>>,
        // cache: &mut HashMap<AtomView<'b>, T>,
    ) -> Result<Self::ContainerData<T>, String>;

    /// Check if the expression could be 0, using (potentially) numerical sampling with
    /// a given tolerance and number of iterations.
    fn zero_test(&self, iterations: usize, tolerance: f64) -> Self::ContainerData<ConditionResult>;

    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Self;

    /// Convert all coefficients to floats with a given precision `decimal_prec``.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    fn coefficients_to_float(&self, decimal_prec: u32) -> Self;

    // /// Convert all coefficients to floats with a given precision `decimal_prec``.
    // /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    // fn coefficients_to_float_into(&self, decimal_prec: u32, out: &mut Atom) {
    //     self.as_atom_view()
    //         .coefficients_to_float_into(decimal_prec, out);
    // }

    /// Map all coefficients using a given function.
    fn map_coefficient<F: Fn(CoefficientView) -> Coefficient + Copy>(&self, f: F) -> Self;

    // /// Map all coefficients using a given function.
    // fn map_coefficient_into<F: Fn(CoefficientView) -> Coefficient + Copy>(
    //     &self,
    //     f: F,
    //     out: &mut Atom,
    // ) {
    //     self.as_atom_view().map_coefficient_into(f, out);
    // }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    fn rationalize_coefficients(&self, relative_error: &Rational) -> Self;

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<MultivariatePolynomial<R, E>>;

    /// Convert the atom to a polynomial in specific variables.
    /// All other parts will be collected into the coefficient, which
    /// is a general expression.
    ///
    /// This routine does not perform expansions.
    fn to_polynomial_in_vars<E: Exponent>(
        &self,
        var_map: &Arc<Vec<Variable>>,
    ) -> Self::ContainerData<MultivariatePolynomial<AtomField, E>>;

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<RationalPolynomial<RO, E>>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>;
    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<FactorizedRationalPolynomial<RO, E>>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize;

    /// Construct a printer for the atom with special options.
    // fn printer<'a>(&'a self, opts: PrintOptions) -> Self::ContainerData<AtomPrinter<'a>>;

    /// Print the atom in a form that is unique and independent of any implementation details.
    ///
    /// Anti-symmetric functions are not supported.
    fn to_canonical_string(&self) -> Self::ContainerData<String>;

    /// Map the function `f` over all terms.
    fn map_terms_single_core(&self, f: impl Fn(AtomView) -> Atom + Clone) -> Self;

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    fn map_terms(&self, f: impl Fn(AtomView) -> Atom + Send + Sync + Clone, n_cores: usize)
        -> Self;

    fn to_pattern(&self) -> Self::ContainerData<Pattern>;

    /// Replace all occurrences of the pattern.
    fn replace_all<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self;

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    fn replace_all_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Self;

    fn replace_map<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&self, m: &F) -> Self;

    // fn replace_iter<'a>(
    //     &'a self,
    //     pattern: &'a Pattern,
    //     rhs: &'a PatternOrMap,
    //     conditions: Option<&'a Condition<PatternRestriction>>,
    //     settings: Option<&'a MatchSettings>,
    // ) -> Self::ContainerData<ReplaceIterator<'a, 'a>>;
}

impl<S: StorageTensor<Data = Atom>> TensorAtomMaps for S {
    type ContainerData<T> = <S as StorageTensor>::ContainerData<T>;

    fn replace_all_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        let rhs = rhs.borrow();
        self.map_data_mut(|a| a.replace_all_mut(pattern, rhs, conditions, settings));
    }

    fn replace_all_repeat<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self {
        let rhs = rhs.borrow();
        self.map_data_ref_self(|a| a.replace_all_repeat(pattern, rhs, conditions, settings))
    }

    fn replace_all_repeat_mut<R: BorrowPatternOrMap>(
        &mut self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        let rhs = rhs.borrow();
        self.map_data_mut(|a| a.replace_all_repeat_mut(pattern, rhs, conditions, settings));
    }

    fn replace_all_multiple_repeat<T: BorrowReplacement>(&self, replacements: &[T]) -> Self {
        self.map_data_ref_self(|a| a.replace_all_multiple_repeat(replacements))
    }
    fn replace_all_multiple_mut<T: BorrowReplacement>(&mut self, replacements: &[T]) {
        self.map_data_mut(|a| a.replace_all_multiple_mut(replacements));
    }
    fn replace_all_multiple_repeat_mut<T: BorrowReplacement>(&mut self, replacements: &[T]) {
        self.map_data_mut(|a| a.replace_all_multiple_repeat_mut(replacements));
    }
    fn replace_map_mut<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&mut self, m: &F) {
        self.map_data_mut(|a| a.replace_map_mut(m));
    }
    // /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    // ///
    // /// ```math
    // /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    // /// ```
    // ///
    // /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    // /// `key_map` and `coeff_map` respectively.
    // fn collect<E: Exponent, T: AtomCore>(
    //     &self,
    //     x: T,
    //     key_map: Option<Box<dyn ClonableAtomMap>>,
    //     coeff_map: Option<Box<dyn ClonableAtomMap>>,
    // ) -> Self {
    //     let x = x.as_atom_view();
    //     match (key_map, coeff_map) {
    //         (Some(key_map), Some(coeff_map)) => self.map_data_ref_self(|a| {
    //             a.collect::<E, AtomView>(x, Some(dyn_clone::clone_box(&*key_map)), Some(coeff_map.()))
    //         }),
    //         (Some(key_map), None) => {
    //             self.map_data_ref_self(|a| a.collect::<E, AtomView>(x, Some(key_map.clone()), None))
    //         }
    //         (None, Some(coeff_map)) => self
    //             .map_data_ref_self(|a| a.collect::<E, AtomView>(x, None, Some(coeff_map.clone()))),
    //         (None, None) => self.map_data_ref_self(|a| a.collect::<E, AtomView>(x, None, None)),
    //     }
    // }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    // fn collect_multiple<E: Exponent, T: AtomCore>(
    //     &self,
    //     xs: &[T],
    //     key_map: Option<Box<dyn ClonableAtomMap>>,
    //     coeff_map: Option<Box<dyn ClonableAtomMap>>,
    // ) -> Self {
    //     self.map_data_ref_self(|a| a.collect_multiple(xs, key_map, coeff_map))
    // }

    /// Collect terms involving the literal occurrence of `x`.
    fn coefficient<T: AtomCore>(&self, x: T) -> Self {
        let x = x.as_atom_view();
        self.map_data_ref_self(|a| a.coefficient(x))
    }

    /// Write the expression over a common denominator.
    fn together(&self) -> Self {
        self.map_data_ref_self(|a| a.together())
    }

    /// Write the expression as a sum of terms with minimal denominators.
    fn apart(&self, x: Symbol) -> Self {
        self.map_data_ref_self(|a| a.apart(x))
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    fn cancel(&self) -> Self {
        self.map_data_ref_self(|a| a.cancel())
    }

    /// Factor the expression over the rationals.
    fn factor(&self) -> Self {
        self.map_data_ref_self(|a| a.factor())
    }

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    fn collect_num(&self) -> Self {
        self.map_data_ref_self(|a| a.collect_num())
    }

    /// Expand an expression. The function [AtomCore::expand_via_poly] may be faster.
    fn expand(&self) -> Self {
        self.map_data_ref_self(|a| a.expand())
    }

    /// Expand the expression by converting it to a polynomial, optionally
    /// only in the indeterminate `var`. The parameter `E` should be a numerical type
    /// that fits the largest exponent in the expanded expression. Often,
    /// `u8` or `u16` is sufficient.
    fn expand_via_poly<E: Exponent, T: AtomCore>(&self, var: Option<T>) -> Self {
        let var = var.as_ref().map(|v| v.as_atom_view());
        self.map_data_ref_self(|a| a.expand_via_poly::<E, AtomView>(var))
    }

    /// Expand an expression in the variable `var`. The function [AtomCore::expand_via_poly] may be faster.
    fn expand_in<T: AtomCore>(&self, var: T) -> Self {
        let var = var.as_atom_view();
        self.map_data_ref_self(|a| a.expand_in(var))
    }

    /// Expand an expression in the variable `var`.
    fn expand_in_symbol(&self, var: Symbol) -> Self {
        self.map_data_ref_self(|a| a.expand_in_symbol(var))
    }

    // /// Expand an expression, returning `true` iff the expression changed.
    // fn expand_into(&self, var: Option<AtomView>, out: &mut Atom) -> bool {
    //     self.as_atom_view().expand_into(var, out)
    // }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    fn expand_num(&self) -> Self {
        self.map_data_ref_self(|a| a.expand_num())
    }

    /// Take a derivative of the expression with respect to `x`.
    fn derivative(&self, x: Symbol) -> Self {
        self.map_data_ref_self(|a| a.derivative(x))
    }

    // /// Take a derivative of the expression with respect to `x` and
    // /// write the result in `out`.
    // /// Returns `true` if the derivative is non-zero.
    // fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
    //     self.as_atom_view().derivative_into(x, out)
    // }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    fn series<T: AtomCore>(
        &self,
        x: Symbol,
        expansion_point: T,
        depth: Rational,
        depth_is_absolute: bool,
    ) -> Result<Self::ContainerData<Series<AtomField>>, &'static str> {
        let expansion_point = expansion_point.as_atom_view();
        self.map_data_ref_result(|a| a.series(x, expansion_point, depth.clone(), depth_is_absolute))
    }

    /// Find the root of a function in `x` numerically over the reals using Newton's method.
    fn nsolve<N: SingleFloat + Real + PartialOrd>(
        &self,
        x: Symbol,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> Result<Self::ContainerData<N>, String> {
        self.map_data_ref_result(|a| a.nsolve::<N>(x, init.clone(), prec.clone(), max_iterations))
    }

    /// Evaluate a (nested) expression a single time.
    /// For repeated evaluations, use [Self::evaluator()] and convert
    /// to an optimized version or generate a compiled version of your expression.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    fn evaluate<T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomView<'_>, T>,
        function_map: &HashMap<Symbol, EvaluationFn<T>>,
    ) -> Result<Self::ContainerData<T>, String> {
        self.map_data_ref_result(|a| {
            let mut cache = HashMap::new();
            a.evaluate(coeff_map, const_map, function_map, &mut cache)
        })
    }

    /// Check if the expression could be 0, using (potentially) numerical sampling with
    /// a given tolerance and number of iterations.
    fn zero_test(&self, iterations: usize, tolerance: f64) -> Self::ContainerData<ConditionResult> {
        self.map_data_ref(|a| a.zero_test(iterations, tolerance))
    }

    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Self {
        self.map_data_ref_self(|a| a.set_coefficient_ring(vars))
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec``.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    fn coefficients_to_float(&self, decimal_prec: u32) -> Self {
        self.map_data_ref_self(|a| a.coefficients_to_float(decimal_prec))
    }

    // /// Convert all coefficients to floats with a given precision `decimal_prec``.
    // /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    // fn coefficients_to_float_into(&self, decimal_prec: u32, out: &mut Atom) {
    //     self.as_atom_view()
    //         .coefficients_to_float_into(decimal_prec, out);
    // }

    /// Map all coefficients using a given function.
    fn map_coefficient<F: Fn(CoefficientView) -> Coefficient + Copy>(&self, f: F) -> Self {
        self.map_data_ref_self(|a| a.map_coefficient(f))
    }

    // /// Map all coefficients using a given function.
    // fn map_coefficient_into<F: Fn(CoefficientView) -> Coefficient + Copy>(
    //     &self,
    //     f: F,
    //     out: &mut Atom,
    // ) {
    //     self.as_atom_view().map_coefficient_into(f, out);
    // }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    fn rationalize_coefficients(&self, relative_error: &Rational) -> Self {
        self.map_data_ref_self(|a| a.rationalize_coefficients(relative_error))
    }

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<MultivariatePolynomial<R, E>> {
        self.map_data_ref(|a| a.to_polynomial(field, var_map.clone()))
    }

    /// Convert the atom to a polynomial in specific variables.
    /// All other parts will be collected into the coefficient, which
    /// is a general expression.
    ///
    /// This routine does not perform expansions.
    fn to_polynomial_in_vars<E: Exponent>(
        &self,
        var_map: &Arc<Vec<Variable>>,
    ) -> Self::ContainerData<MultivariatePolynomial<AtomField, E>> {
        self.map_data_ref(|a| a.to_polynomial_in_vars(var_map))
    }

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<RationalPolynomial<RO, E>>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        self.map_data_ref(|a| a.to_rational_polynomial(field, out_field, var_map.clone()))
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> Self::ContainerData<FactorizedRationalPolynomial<RO, E>>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        self.map_data_ref(|a| {
            a.to_factorized_rational_polynomial(field, out_field, var_map.clone())
        })
    }

    // /// Construct a printer for the atom with special options.
    // fn printer<'a>(&'a self, opts: PrintOptions) -> Self::ContainerData<AtomPrinter<'a>> {
    //     self.map_data_ref(|a| a.printer(opts))
    // }

    /// Print the atom in a form that is unique and independent of any implementation details.
    ///
    /// Anti-symmetric functions are not supported.
    fn to_canonical_string(&self) -> Self::ContainerData<String> {
        self.map_data_ref(|a| a.to_canonical_string())
    }

    /// Map the function `f` over all terms.
    fn map_terms_single_core(&self, f: impl Fn(AtomView) -> Atom + Clone) -> Self {
        self.map_data_ref_self(|a| a.map_terms_single_core(f.clone()))
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    fn map_terms(
        &self,
        f: impl Fn(AtomView) -> Atom + Send + Sync + Clone,
        n_cores: usize,
    ) -> Self {
        self.map_data_ref_self(|a| a.map_terms(f.clone(), n_cores))
    }

    fn to_pattern(&self) -> Self::ContainerData<Pattern> {
        self.map_data_ref(|a| a.to_pattern())
    }

    /// Replace all occurrences of the pattern.
    fn replace_all<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self {
        let rhs = rhs.borrow();
        self.map_data_ref_self(|a| a.replace_all(pattern, rhs, conditions, settings))
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    fn replace_all_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Self {
        self.map_data_ref_self(|a| a.replace_all_multiple(replacements))
    }

    fn replace_map<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&self, m: &F) -> Self {
        self.map_data_ref_self(|a| a.replace_map(m))
    }

    // /// Return an iterator that replaces the pattern in the target once.
    // fn replace_iter<'a>(
    //     &'a self,
    //     pattern: &'a Pattern,
    //     rhs: &'a PatternOrMap,
    //     conditions: Option<&'a Condition<PatternRestriction>>,
    //     settings: Option<&'a MatchSettings>,
    // ) -> Self::ContainerData<ReplaceIterator<'a, 'a>> {
    //     self.map_data_ref(|a| a.replace_iter(pattern, rhs, conditions, settings))
    // }
}

impl<S: TensorStructure> DenseTensor<Atom, S> {
    /// Solve a non-linear system numerically over the reals using Newton's method.
    pub fn nsolve_system<
        N: SingleFloat + Real + PartialOrd + InternalOrdering + Eq + std::hash::Hash,
    >(
        &self,
        vars: &[Symbol],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, String> {
        <Atom as AtomCore>::nsolve_system::<N, Atom>(&self.data, vars, init, prec, max_iterations)
    }

    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    pub fn solve_linear_system<E: PositiveExponent, T: AtomCore>(
        &self,
        vars: &[T],
    ) -> Result<Vec<Atom>, String> {
        <Atom as AtomCore>::solve_linear_system::<E, Atom, T>(&self.data, vars)
    }
    /// Convert a system of linear equations to a matrix representation, returning the matrix
    /// and the right-hand side.
    #[allow(clippy::type_complexity)]
    pub fn system_to_matrix<E: PositiveExponent, T: AtomCore>(
        &self,
        vars: &[T],
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        String,
    > {
        <Atom as AtomCore>::system_to_matrix::<E, Atom, T>(&self.data, vars)
    }
}

pub trait TensorAtomOps: HasStructure {
    /// Collect terms involving the same power of `x` in `xs`, where `xs` is a list of indeterminates.
    /// Return the list of key-coefficient pairs
    fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)>;

    /// Convert nested expressions to a tree suitable for repeated evaluations with
    /// different values for `params`.
    /// All variables and all user functions in the expression must occur in the map.
    fn to_evaluation_tree<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<Rational, Self::Structure>, String>;

    /// Create an efficient evaluator for a (nested) expression.
    /// All free parameters must appear in `params` and all other variables
    /// and user functions in the expression must occur in the function map.
    /// The function map may have nested expressions.
    fn evaluator<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<EvalTensor<ExpressionEvaluator<Rational>, Self::Structure>, String>;

    /// Get all symbols in the expression, optionally including function symbols.
    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol>;

    /// Get all variables and functions in the expression.
    fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView>;

    /// Returns true iff `self` contains the symbol `s`.
    fn contains_symbol(&self, s: Symbol) -> bool;

    /// Returns true iff `self` contains `a` literally.
    fn contains<T: AtomCore>(&self, s: T) -> bool;

    /// Check if the expression can be considered a polynomial in some variables, including
    /// redefinitions. For example `f(x)+y` is considered a polynomial in `f(x)` and `y`, whereas
    /// `f(x)+x` is not a polynomial.
    ///
    /// Rational powers or powers in variables are not rewritten, e.g. `x^(2y)` is not considered
    /// polynomial in `x^y`.
    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>>;
}

impl<S: TensorStructure + Clone> TensorAtomOps for DenseTensor<Atom, S> {
    fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView> {
        let mut indeterminates = HashSet::new();

        for (_, a) in self.iter_flat() {
            indeterminates.extend(a.get_all_indeterminates(enter_functions));
        }

        indeterminates
    }

    fn contains_symbol(&self, s: Symbol) -> bool {
        self.iter_flat().any(|(_, a)| a.contains_symbol(s))
    }

    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        let mut all_symbols = HashSet::new();
        for (_, a) in self.iter_flat() {
            all_symbols.extend(a.get_all_symbols(include_function_symbols));
        }
        all_symbols
    }

    fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)> {
        let mut list = vec![];

        for (_, a) in self.iter_flat() {
            list.extend(a.coefficient_list::<E, _>(xs))
        }
        list
    }

    fn contains<T: AtomCore>(&self, s: T) -> bool {
        let s = s.as_atom_view();
        self.iter_flat().any(|(_, a)| a.contains(s))
    }

    fn to_evaluation_tree<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<Rational, Self::Structure>, String> {
        let atomviews: Vec<AtomView> = self.data.iter().map(|a| a.as_view()).collect();
        let eval = AtomView::to_eval_tree_multiple(&atomviews, fn_map, params)?;

        Ok(EvalTreeTensor {
            eval,
            indexmap: None,
            structure: self.structure.clone(),
        })
    }

    fn evaluator<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<EvalTensor<ExpressionEvaluator<Rational>, Self::Structure>, String> {
        let mut tree = self.to_evaluation_tree(fn_map, params)?;

        Ok(tree.optimize(
            optimization_settings.horner_iterations,
            optimization_settings.n_cores,
            optimization_settings.hot_start.clone(),
            optimization_settings.verbose,
        ))
    }

    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>> {
        let mut is_polynomial = true;
        let mut set = HashSet::new();
        for (_, v) in self.iter_flat() {
            if let Some(v) = v.is_polynomial(allow_not_expanded, allow_negative_powers) {
                set.extend(v);
            } else {
                is_polynomial = false;
                break;
            }
        }

        if is_polynomial {
            Some(set)
        } else {
            None
        }
    }
}

impl<S: TensorStructure + Clone> TensorAtomOps for SparseTensor<Atom, S> {
    fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView> {
        let mut indeterminates = HashSet::new();

        for (_, a) in self.iter_flat() {
            indeterminates.extend(a.get_all_indeterminates(enter_functions));
        }

        indeterminates
    }

    fn contains_symbol(&self, s: Symbol) -> bool {
        self.iter_flat().any(|(_, a)| a.contains_symbol(s))
    }

    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        let mut all_symbols = HashSet::new();
        for (_, a) in self.iter_flat() {
            all_symbols.extend(a.get_all_symbols(include_function_symbols));
        }
        all_symbols
    }

    fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)> {
        let mut list = vec![];

        for (_, a) in self.iter_flat() {
            list.extend(a.coefficient_list::<E, _>(xs))
        }
        list
    }

    fn contains<T: AtomCore>(&self, s: T) -> bool {
        let s = s.as_atom_view();
        self.iter_flat().any(|(_, a)| a.contains(s))
    }

    fn to_evaluation_tree<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<Rational, Self::Structure>, String> {
        let atomviews: Vec<AtomView> = self.iter_flat().map(|(_, a)| a.as_view()).collect();
        let eval = AtomView::to_eval_tree_multiple(&atomviews, fn_map, params)?;

        Ok(EvalTreeTensor {
            eval,
            indexmap: None,
            structure: self.structure.clone(),
        })
    }

    fn evaluator<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<EvalTensor<ExpressionEvaluator<Rational>, Self::Structure>, String> {
        let mut tree = self.to_evaluation_tree(fn_map, params)?;

        Ok(tree.optimize(
            optimization_settings.horner_iterations,
            optimization_settings.n_cores,
            optimization_settings.hot_start.clone(),
            optimization_settings.verbose,
        ))
    }

    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>> {
        let mut is_polynomial = true;
        let mut set = HashSet::new();
        for (_, v) in self.iter_flat() {
            if let Some(v) = v.is_polynomial(allow_not_expanded, allow_negative_powers) {
                set.extend(v);
            } else {
                is_polynomial = false;
                break;
            }
        }

        if is_polynomial {
            Some(set)
        } else {
            None
        }
    }
}

impl<S: TensorStructure + Clone> TensorAtomOps for DataTensor<Atom, S> {
    fn contains<T: AtomCore>(&self, s: T) -> bool {
        let s = s.as_atom_view();
        match self {
            DataTensor::Dense(d) => d.contains(s),
            DataTensor::Sparse(d) => d.contains(s),
        }
    }

    fn evaluator<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<EvalTensor<ExpressionEvaluator<Rational>, Self::Structure>, String> {
        match self {
            DataTensor::Dense(d) => d.evaluator(fn_map, params, optimization_settings),
            DataTensor::Sparse(s) => s.evaluator(fn_map, params, optimization_settings),
        }
    }

    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>> {
        match self {
            DataTensor::Dense(d) => d.is_polynomial(allow_not_expanded, allow_negative_powers),
            DataTensor::Sparse(s) => s.is_polynomial(allow_not_expanded, allow_negative_powers),
        }
    }

    fn contains_symbol(&self, s: Symbol) -> bool {
        match self {
            DataTensor::Dense(d) => d.contains_symbol(s),
            DataTensor::Sparse(d) => d.contains_symbol(s),
        }
    }

    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        match self {
            DataTensor::Dense(d) => d.get_all_symbols(include_function_symbols),
            DataTensor::Sparse(s) => s.get_all_symbols(include_function_symbols),
        }
    }

    fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)> {
        match self {
            DataTensor::Dense(d) => d.coefficient_list::<E, T>(xs),
            DataTensor::Sparse(s) => s.coefficient_list::<E, T>(xs),
        }
    }

    fn to_evaluation_tree<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<Rational, Self::Structure>, String> {
        match self {
            DataTensor::Dense(d) => d.to_evaluation_tree(fn_map, params),
            DataTensor::Sparse(s) => s.to_evaluation_tree(fn_map, params),
        }
    }

    fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView> {
        match self {
            DataTensor::Dense(d) => d.get_all_indeterminates(enter_functions),
            DataTensor::Sparse(s) => s.get_all_indeterminates(enter_functions),
        }
    }
}

impl<S: TensorStructure + Clone> TensorAtomOps for ParamTensor<S> {
    fn contains<T: AtomCore>(&self, s: T) -> bool {
        self.tensor.contains(s)
    }

    fn evaluator<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<EvalTensor<ExpressionEvaluator<Rational>, Self::Structure>, String> {
        self.tensor.evaluator(fn_map, params, optimization_settings)
    }

    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>> {
        self.tensor
            .is_polynomial(allow_not_expanded, allow_negative_powers)
    }

    fn contains_symbol(&self, s: Symbol) -> bool {
        self.tensor.contains_symbol(s)
    }

    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        self.tensor.get_all_symbols(include_function_symbols)
    }

    fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)> {
        self.tensor.coefficient_list::<E, T>(xs)
    }

    fn to_evaluation_tree<'a>(
        &'a self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<Rational, Self::Structure>, String> {
        self.tensor.to_evaluation_tree(fn_map, params)
    }

    fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView> {
        self.tensor.get_all_indeterminates(enter_functions)
    }
}

extern crate derive_more;

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ahash::{AHashMap, HashMap};

use anyhow::anyhow;
use anyhow::{Error, Result};

// use anyhow::Ok;
use enum_try_as_inner::EnumTryAsInner;
use serde::{Deserialize, Serialize};

use crate::{
    complex::{Complex, RealOrComplexTensor},
    contraction::{Contract, ContractableWith, ContractionError, IsZero, RefZero},
    data::{DataIterator, DataTensor, DenseTensor, SetTensorData, SparseTensor},
    iterators::IteratableTensor,
    structure::{
        CastStructure, ExpandedIndex, FlatIndex, HasName, HasStructure, IntoArgs, IntoSymbol,
        NamedStructure, PhysicalSlots, ScalarStructure, ScalarTensor, ShadowMapping, Shadowable,
        StructureContract, TensorStructure, ToSymbolic, TracksCount,
    },
    upgrading_arithmetic::{FallibleAddAssign, FallibleMul, FallibleSubAssign, TrySmallestUpgrade},
};

use symbolica::{
    atom::{representation::FunView, Atom, AtomOrView, AtomView, FunctionBuilder, Symbol},
    coefficient::ConvertToRing,
    domains::{
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
        },
        float::{NumericalFloatLike, Real},
        rational::Rational,
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
        EuclideanDomain,
    },
    evaluate::{
        CompileOptions, CompiledEvaluator, CompiledEvaluatorFloat, EvalTree, EvaluationFn,
        ExpressionEvaluator, FunctionMap,
    },
    id::{Condition, MatchSettings, Pattern, Replacement, WildcardAndRestriction},
    poly::{
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Exponent,
        Variable,
    },
    state::State,
};

use std::hash::Hash;

use symbolica::domains::float::Complex as SymComplex;

// impl RefZero for Atom {
//     fn ref_zero(&self) -> Self {
//         Atom::new_num(0)
//     }
// }

pub trait TensorCoefficient: Display {
    fn cooked_name(&self) -> Option<String>;
    fn name(&self) -> Option<Symbol>;
    fn tags(&self) -> Vec<AtomOrView>;
    fn to_atom(&self) -> Option<Atom>;
    fn to_atom_re(&self) -> Option<Atom>;
    fn to_atom_im(&self) -> Option<Atom>;
    fn add_tagged_function<'c, 'a, 'b: 'c, T>(
        &'c self,
        fn_map: &'b mut FunctionMap<'a, T>,
        body: AtomView<'a>,
    ) -> Result<(), String> {
        if let Some((name, cooked_name)) = self.name().zip(self.cooked_name()) {
            fn_map
                .add_tagged_function(name, self.tags(), cooked_name, vec![], body)
                .map_err(String::from)
        } else {
            Err(format!("unnamed {}", self))
        }
    }
}

#[derive(Debug)]
pub struct FlatCoefficent<Args: IntoArgs> {
    pub name: Option<Symbol>,
    pub index: FlatIndex,
    pub args: Option<Args>,
}

impl<Arg: IntoArgs> Display for FlatCoefficent<Arg> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name {
            write!(f, "{}", name)?
        }
        write!(f, "(")?;
        if let Some(ref args) = self.args {
            let args: Vec<String> = args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }

        write!(f, "{})", self.index)?;
        Result::Ok(())
    }
}

impl<Args: IntoArgs> TensorCoefficient for FlatCoefficent<Args> {
    fn name(&self) -> Option<Symbol> {
        self.name
    }

    fn cooked_name(&self) -> Option<String> {
        let mut name = self.name?.to_string();
        if let Some(ref args) = self.args {
            name += args.cooked_name().as_str();
        }
        Some(name)
    }

    fn tags(&self) -> Vec<AtomOrView> {
        let mut tags: Vec<AtomOrView> = if let Some(ref args) = self.args {
            args.args().into_iter().map(AtomOrView::from).collect()
        } else {
            vec![]
        };
        tags.push(Atom::from(self.index).into());
        tags
    }

    fn to_atom(&self) -> Option<Atom> {
        let mut fn_builder = FunctionBuilder::new(self.name?);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index).as_view());
        Some(fn_builder.finish())
    }

    fn to_atom_re(&self) -> Option<Atom> {
        let name = State::get_symbol(self.name?.to_string() + "_re");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index).as_view());
        Some(fn_builder.finish())
    }

    fn to_atom_im(&self) -> Option<Atom> {
        let name = State::get_symbol(self.name?.to_string() + "_im");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index).as_view());
        Some(fn_builder.finish())
    }
}

pub struct ExpandedCoefficent<Args: IntoArgs> {
    pub name: Option<Symbol>,
    pub index: ExpandedIndex,
    pub args: Option<Args>,
}

impl<Arg: IntoArgs> Display for ExpandedCoefficent<Arg> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name {
            write!(f, "{}", name)?
        }
        write!(f, "(")?;
        if let Some(ref args) = self.args {
            let args: Vec<String> = args.ref_into_args().map(|s| s.to_string()).collect();
            write!(f, "{},", args.join(","))?
        }
        write!(f, "{})", self.index)?;
        Result::Ok(())
    }
}

impl<Args: IntoArgs> TensorCoefficient for ExpandedCoefficent<Args> {
    fn name(&self) -> Option<Symbol> {
        self.name
    }
    fn cooked_name(&self) -> Option<String> {
        let mut name = self.name?.to_string();
        if let Some(ref args) = self.args {
            name += args.cooked_name().as_str();
        }
        Some(name)
    }

    fn tags(&self) -> Vec<AtomOrView> {
        let mut tags: Vec<AtomOrView> = if let Some(ref args) = self.args {
            args.args().into_iter().map(AtomOrView::from).collect()
        } else {
            vec![]
        };
        tags.push(Atom::from(self.index.clone()).into());
        tags
    }

    fn to_atom(&self) -> Option<Atom> {
        let mut fn_builder = FunctionBuilder::new(self.name?);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index.clone()).as_view());
        Some(fn_builder.finish())
    }
    fn to_atom_re(&self) -> Option<Atom> {
        let name = State::get_symbol(self.name?.to_string() + "_re");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index.clone()).as_view());
        Some(fn_builder.finish())
    }

    fn to_atom_im(&self) -> Option<Atom> {
        let name = State::get_symbol(self.name?.to_string() + "_im");

        let mut fn_builder = FunctionBuilder::new(name);
        if let Some(ref args) = self.args {
            for arg in args.ref_into_args() {
                fn_builder = fn_builder.add_arg(arg.as_view());
            }
        }
        fn_builder = fn_builder.add_arg(Atom::from(self.index.clone()).as_view());
        Some(fn_builder.finish())
    }
}

impl<'a> TryFrom<FunView<'a>> for DenseTensor<Atom, NamedStructure<Symbol, Vec<Atom>>> {
    type Error = Error;

    fn try_from(f: FunView<'a>) -> Result<Self> {
        let mut structure: Vec<PhysicalSlots> = vec![];
        let f_id = f.get_symbol();
        let mut args = vec![];

        for arg in f.iter() {
            if let Ok(arg) = arg.try_into() {
                structure.push(arg);
            } else {
                args.push(arg.to_owned());
            }
        }
        let s = NamedStructure::from_iter(structure, f_id, Some(args));
        s.to_dense_expanded_labels()
    }
}

#[derive(Clone, Debug)]
pub struct ParamTensor<S: TensorStructure> {
    pub tensor: DataTensor<Atom, S>,
    pub param_type: ParamOrComposite,
    // Param(DataTensor<Atom, S>),
    // // Concrete(DataTensor<T, S>),
    // Composite(DataTensor<Atom, S>),
}

pub struct ParamTensorSet<S: TensorStructure> {
    pub tensors: Vec<ParamTensor<S>>,
    size: usize,
}

impl<S: TensorStructure + Clone> ParamTensorSet<S> {
    pub fn new(tensors: Vec<ParamTensor<S>>) -> Self {
        let size = tensors
            .iter()
            .map(|t| t.tensor.actual_size())
            .reduce(|acc, a| acc + a)
            .unwrap();

        ParamTensorSet { tensors, size }
    }

    pub fn empty() -> Self {
        ParamTensorSet {
            tensors: vec![],
            size: 0,
        }
    }

    pub fn push(&mut self, tensor: ParamTensor<S>) {
        self.size += tensor.tensor.actual_size();
        self.tensors.push(tensor);
    }
}

impl<S: TensorStructure> ParamTensor<S> {
    pub fn param(tensor: DataTensor<Atom, S>) -> Self {
        ParamTensor {
            tensor,
            param_type: ParamOrComposite::Param,
        }
    }

    pub fn composite(tensor: DataTensor<Atom, S>) -> Self {
        ParamTensor {
            tensor,
            param_type: ParamOrComposite::Composite,
        }
    }

    /// Convert the tensor of atoms to a tensor of polynomials, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    pub fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> DataTensor<MultivariatePolynomial<R, E>, S>
    where
        S: Clone,
    {
        self.tensor
            .map_data_ref(|a| a.as_view().to_polynomial(field, var_map.clone()))
    }

    /// Convert the tensor of atoms to a tensor of rational polynomials, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> DataTensor<RationalPolynomial<RO, E>, S>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
        S: Clone,
    {
        self.tensor.map_data_ref(|a| {
            a.as_view()
                .to_rational_polynomial(field, out_field, var_map.clone())
        })
    }

    /// Convert the tensor of atoms to a tensor of rational polynomials with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> DataTensor<FactorizedRationalPolynomial<RO, E>, S>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
        S: Clone,
    {
        self.tensor.map_data_ref(|a| {
            a.as_view()
                .to_factorized_rational_polynomial(field, out_field, var_map.clone())
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum ParamOrComposite {
    Param,
    Composite,
}

impl<S: TensorStructure, O: From<S> + TensorStructure> CastStructure<ParamTensor<O>>
    for ParamTensor<S>
{
    fn cast(self) -> ParamTensor<O> {
        ParamTensor {
            tensor: self.tensor.cast(),
            param_type: self.param_type,
        }
    }
}

impl<S: TensorStructure> Shadowable for ParamTensor<S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
}

impl<S: TensorStructure, Const> ShadowMapping<Const> for ParamTensor<S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
    // fn shadow_with_map<'a, T>(
    //     &'a self,
    //     fn_map: &mut FunctionMap<'a, Const>,
    //     index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    // ) -> Option<ParamTensor<Self::Structure>>
    // where
    //     T: TensorCoefficient,
    // {
    //     match self {
    //         ParamTensor::Param(_) => return Some(self.clone()),
    //         ParamTensor::Composite(c) => match c {
    //             DataTensor::Dense(d) => {
    //                 let mut data = vec![];
    //                 for (i, a) in d.flat_iter() {
    //                     let labeled_coef = index_to_atom(self.structure(), i);

    //                     labeled_coef
    //                         .add_tagged_function(fn_map, a.as_view())
    //                         .unwrap();
    //                     data.push(labeled_coef.to_atom().unwrap());
    //                 }
    //                 let param = DenseTensor {
    //                     data,
    //                     structure: d.structure.clone(),
    //                 };
    //                 Some(ParamTensor::Param(param.into()))
    //             }
    //             DataTensor::Sparse(d) => {
    //                 let mut data = vec![];
    //                 for (i, a) in d.flat_iter() {
    //                     let labeled_coef = index_to_atom(self.structure(), i);

    //                     labeled_coef
    //                         .add_tagged_function(fn_map, a.as_view())
    //                         .unwrap();
    //                     data.push(labeled_coef.to_atom().unwrap());
    //                 }
    //                 let param = DenseTensor {
    //                     data,
    //                     structure: d.structure.clone(),
    //                 };
    //                 Some(ParamTensor::Param(param.into()))
    //             }
    //         },
    //     }
    // }

    fn append_map<'a, T>(
        &'a self,
        fn_map: &mut FunctionMap<'a, Const>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    ) where
        T: TensorCoefficient,
    {
        match self.param_type {
            ParamOrComposite::Param => {}
            ParamOrComposite::Composite => match &self.tensor {
                DataTensor::Dense(d) => {
                    for (i, a) in d.flat_iter() {
                        let labeled_coef = index_to_atom(self.structure(), i);

                        labeled_coef
                            .add_tagged_function(fn_map, a.as_view())
                            .unwrap();
                    }
                }
                DataTensor::Sparse(d) => {
                    for (i, a) in d.flat_iter() {
                        let labeled_coef = index_to_atom(self.structure(), i);

                        labeled_coef
                            .add_tagged_function(fn_map, a.as_view())
                            .unwrap();
                    }
                }
            },
        }
    }
}

// impl<T: Real> SparseTensor<EvalTree<T>> {
//     pub fn evaluate(&mut self, params: &[T]) -> SparseTensor<T> {
//         let zero = params[0].zero();
//         let mut out_data = SparseTensor::repeat(self.structure.clone(), zero);
//         self.map_data_ref_mut(|e| e.evaluate(params, &mut out_data.data));
//         out_data
//     }
// }

impl DataTensor<EvalTree<Rational>> {
    pub fn horner_scheme(&mut self) {
        self.map_data_mut(|x| x.horner_scheme())
    }
}

pub trait PatternReplacement {
    fn replace_repeat_multiple_atom(expr: &mut Atom, reps: &[Replacement<'_>]) {
        let atom = expr.replace_all_multiple(reps);
        if atom != *expr {
            *expr = atom;
            Self::replace_repeat_multiple_atom(expr, reps)
        }
    }

    fn replace_all(
        &self,
        pattern: &Pattern,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self;

    fn replace_all_mut(
        &mut self,
        pattern: &Pattern,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
    );

    fn replace_all_multiple(&self, replacements: &[Replacement<'_>]) -> Self;
    fn replace_all_multiple_mut(&mut self, replacements: &[Replacement<'_>]);
    fn replace_all_multiple_repeat_mut(&mut self, replacements: &[Replacement<'_>]);
}

impl<S: TensorStructure + Clone> PatternReplacement for ParamTensor<S> {
    fn replace_all(
        &self,
        pattern: &Pattern,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self {
        let tensor = self
            .tensor
            .map_data_ref(|a| a.replace_all(pattern, rhs, conditions, settings));
        ParamTensor {
            tensor,
            param_type: ParamOrComposite::Composite,
        }
    }

    fn replace_all_mut(
        &mut self,
        pattern: &Pattern,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        *self = self.replace_all(pattern, rhs, conditions, settings)
    }

    fn replace_all_multiple(&self, replacements: &[Replacement<'_>]) -> Self {
        let tensor = self
            .tensor
            .map_data_ref(|a| a.replace_all_multiple(replacements));
        ParamTensor {
            tensor,
            param_type: ParamOrComposite::Composite,
        }
    }

    fn replace_all_multiple_mut(&mut self, replacements: &[Replacement<'_>]) {
        *self = self.replace_all_multiple(replacements)
    }

    fn replace_all_multiple_repeat_mut(&mut self, replacements: &[Replacement<'_>]) {
        self.tensor
            .map_data_mut(|a| Self::replace_repeat_multiple_atom(a, replacements));
        self.param_type = ParamOrComposite::Composite;
    }
}

impl<S: TensorStructure + Clone> ParamTensor<S> {
    pub fn eval_tree<'a, T: Clone + Default + Debug + Hash + Ord, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<T, S>, String> {
        self.tensor.eval_tree(coeff_map, fn_map, params)
    }

    pub fn evaluate<'a, 'b, T, F: Fn(&Rational) -> T + Copy, U>(
        &self,
        coeff_map: F,
        const_map: &'b HashMap<AtomView<'a>, T>,
    ) -> DataTensor<U, S>
    where
        T: symbolica::domains::float::Real
            + for<'c> std::convert::From<&'c symbolica::domains::rational::Rational>,
        U: From<T>,

        'a: 'b,
    {
        self.tensor.evaluate(coeff_map, const_map)
    }
}

impl<S: TensorStructure + Clone> ParamTensorSet<S> {
    pub fn eval_tree<'a, T: Clone + Default + Debug + Hash + Ord, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<EvalTreeTensorSet<T, S>> {
        let mut tensors = vec![];

        let mut atoms = vec![];
        let mut id = 0;

        for t in self.tensors.iter() {
            let structure = t.structure().clone();
            let usize_tensor = match &t.tensor {
                DataTensor::Dense(d) => {
                    let oldid = id;
                    id += d.size().unwrap();
                    for (_, a) in d.flat_iter() {
                        atoms.push(a.as_view());
                    }
                    DataTensor::Dense(DenseTensor::from_data(
                        Vec::from_iter(oldid..id),
                        structure,
                    )?)
                }
                DataTensor::Sparse(s) => {
                    let mut t = SparseTensor::empty(structure);
                    for (i, a) in s.flat_iter() {
                        t.set_flat(i, id)?;
                        atoms.push(a.as_view());
                        id += 1;
                    }
                    DataTensor::Sparse(t)
                }
            };
            tensors.push(usize_tensor);
        }

        Ok(EvalTreeTensorSet {
            tensors,
            eval: AtomView::to_eval_tree_multiple(&atoms, coeff_map, fn_map, params)
                .map_err(|s| anyhow!(s))?,
            size: self.size,
        })
    }
}

impl<S: TensorStructure> IteratableTensor for ParamTensor<S> {
    type Data<'a> =  AtomView<'a> where Self: 'a;

    fn iter_expanded(&self) -> impl Iterator<Item = (ExpandedIndex, Self::Data<'_>)> {
        self.tensor.iter_expanded().map(|(i, x)| (i, x.as_view()))
    }

    fn iter_flat(&self) -> impl Iterator<Item = (FlatIndex, Self::Data<'_>)> {
        self.tensor.iter_flat().map(|(i, x)| (i, x.as_view()))
    }
}

impl<S> Display for ParamTensor<S>
where
    S: TensorStructure,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tensor)
    }
}

impl<S: TensorStructure> HasName for ParamTensor<S>
where
    S: HasName,
{
    type Args = S::Args;
    type Name = S::Name;

    fn args(&self) -> Option<Self::Args> {
        self.tensor.args()
    }

    fn name(&self) -> Option<Self::Name> {
        self.tensor.name()
    }

    fn set_name(&mut self, name: Self::Name) {
        if let ParamOrComposite::Composite = self.param_type {
            self.tensor.set_name(name);
        } // never set the name of a param tensor, it is always set by construction
    }
}

#[derive(Debug, Clone)]
// #[derive_err(Debug)]
pub enum ParamOrConcrete<C: HasStructure<Structure = S> + Clone, S: TensorStructure> {
    Concrete(C),
    Param(ParamTensor<S>),
}

impl<C: Display + HasStructure<Structure = S> + Clone, S: TensorStructure> Display
    for ParamOrConcrete<C, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamOrConcrete::Concrete(c) => c.fmt(f),
            ParamOrConcrete::Param(p) => p.fmt(f),
        }
    }
}

impl<
        U: HasStructure<Structure = O> + Clone,
        C: CastStructure<U> + HasStructure<Structure = S> + Clone,
        S: TensorStructure,
        O: From<S> + TensorStructure,
    > CastStructure<ParamOrConcrete<U, O>> for ParamOrConcrete<C, S>
{
    fn cast(self) -> ParamOrConcrete<U, O> {
        match self {
            ParamOrConcrete::Concrete(c) => ParamOrConcrete::Concrete(c.cast()),
            ParamOrConcrete::Param(p) => ParamOrConcrete::Param(p.cast()),
        }
    }
}

impl<
        C: HasStructure<Structure = S> + Clone + Shadowable,
        S: TensorStructure + Clone + HasName<Args: IntoArgs, Name: IntoSymbol>,
    > Shadowable for ParamOrConcrete<C, S>
{
}

impl<
        U,
        C: HasStructure<Structure = S> + Clone + ShadowMapping<U>,
        S: TensorStructure + Clone + HasName<Args: IntoArgs, Name: IntoSymbol>,
    > ShadowMapping<U> for ParamOrConcrete<C, S>
{
    // fn shadow_with_map<'a, T>(
    //     &'a self,
    //     fn_map: &mut FunctionMap<'a, U>,
    //     index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    // ) -> Option<ParamTensor<Self::Structure>>
    // where
    //     T: TensorCoefficient,
    // {
    //     match self {
    //         ParamOrConcrete::Concrete(c) => c.shadow_with_map(fn_map, index_to_atom),
    //         ParamOrConcrete::Param(p) => p.shadow_with_map(fn_map, index_to_atom),
    //     }
    // }

    fn append_map<'a, T>(
        &'a self,
        fn_map: &mut FunctionMap<'a, U>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> T,
    ) where
        T: TensorCoefficient,
    {
        match self {
            ParamOrConcrete::Concrete(c) => c.append_map(fn_map, index_to_atom),
            ParamOrConcrete::Param(p) => p.append_map(fn_map, index_to_atom),
        }
    }
}

pub enum AtomViewOrConcrete<'a, T> {
    Atom(AtomView<'a>),
    Concrete(T),
}

pub enum AtomOrConcrete<T> {
    Atom(Atom),
    Concrete(T),
}

pub trait Concrete {}

impl<T> From<Atom> for AtomOrConcrete<T> {
    fn from(value: Atom) -> Self {
        AtomOrConcrete::Atom(value)
    }
}

impl<T: Concrete> From<T> for AtomOrConcrete<T> {
    fn from(value: T) -> Self {
        AtomOrConcrete::Concrete(value)
    }
}

impl<C: HasStructure<Structure = S> + Clone, S: TensorStructure + Clone> PatternReplacement
    for ParamOrConcrete<C, S>
{
    fn replace_all(
        &self,
        pattern: &Pattern,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Self {
        match self {
            ParamOrConcrete::Param(p) => {
                ParamOrConcrete::Param(p.replace_all(pattern, rhs, conditions, settings))
            }
            _ => self.clone(),
        }
    }

    fn replace_all_multiple(&self, replacements: &[Replacement<'_>]) -> Self {
        match self {
            ParamOrConcrete::Param(p) => {
                ParamOrConcrete::Param(p.replace_all_multiple(replacements))
            }
            _ => self.clone(),
        }
    }

    fn replace_all_mut(
        &mut self,
        pattern: &Pattern,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
    ) {
        if let ParamOrConcrete::Param(p) = self {
            p.replace_all_mut(pattern, rhs, conditions, settings);
        }
    }

    fn replace_all_multiple_mut(&mut self, replacements: &[Replacement<'_>]) {
        if let ParamOrConcrete::Param(p) = self {
            p.replace_all_multiple_mut(replacements);
        }
    }

    fn replace_all_multiple_repeat_mut(&mut self, replacements: &[Replacement<'_>]) {
        if let ParamOrConcrete::Param(p) = self {
            p.replace_all_multiple_repeat_mut(replacements);
        }
    }
}

impl<C: HasStructure<Structure = S> + Clone, S: TensorStructure> ParamOrConcrete<C, S> {
    pub fn is_parametric(&self) -> bool {
        matches!(self, ParamOrConcrete::Param(_))
    }

    pub fn try_into_parametric(self) -> Result<ParamTensor<S>, Self> {
        match self {
            ParamOrConcrete::Param(x) => Ok(x),
            _ => Err(self),
        }
    }

    pub fn try_into_concrete(self) -> Result<C, Self> {
        match self {
            ParamOrConcrete::Concrete(x) => Ok(x),
            _ => Err(self),
        }
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum ConcreteOrParam<C> {
    Concrete(C),
    Param(Atom),
}

impl<D: Display> Display for ConcreteOrParam<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConcreteOrParam::Concrete(c) => c.fmt(f),
            ConcreteOrParam::Param(p) => write!(f, "{}", p),
        }
    }
}

impl<C, S> HasStructure for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + Clone,
    S: TensorStructure,
{
    type Scalar = ConcreteOrParam<C::Scalar>;
    type Structure = S;

    fn structure(&self) -> &Self::Structure {
        match self {
            ParamOrConcrete::Concrete(x) => x.structure(),
            ParamOrConcrete::Param(x) => x.structure(),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            ParamOrConcrete::Concrete(x) => x.mut_structure(),
            ParamOrConcrete::Param(x) => x.mut_structure(),
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        match self {
            ParamOrConcrete::Concrete(x) => x.scalar().map(ConcreteOrParam::Concrete),
            ParamOrConcrete::Param(x) => x.scalar().map(ConcreteOrParam::Param),
        }
    }
}

impl<C, S> ScalarTensor for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + Clone + ScalarTensor,
    S: TensorStructure + ScalarStructure,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        match scalar {
            ConcreteOrParam::Concrete(x) => ParamOrConcrete::Concrete(C::new_scalar(x)),
            ConcreteOrParam::Param(x) => ParamOrConcrete::Param(ParamTensor::new_scalar(x)),
        }
    }
}

impl<C, S> TracksCount for ParamOrConcrete<C, S>
where
    C: TracksCount + HasStructure<Structure = S> + Clone,
    S: TensorStructure + TracksCount,
{
    fn contractions_num(&self) -> usize {
        match self {
            ParamOrConcrete::Concrete(x) => x.contractions_num(),
            ParamOrConcrete::Param(x) => x.contractions_num(),
        }
    }
}

impl<C, S> HasName for ParamOrConcrete<C, S>
where
    C: HasName + HasStructure<Structure = S> + Clone,
    S: TensorStructure + HasName<Name = C::Name, Args = C::Args>,
{
    type Args = C::Args;
    type Name = C::Name;

    fn args(&self) -> Option<Self::Args> {
        match self {
            ParamOrConcrete::Concrete(x) => x.args(),
            ParamOrConcrete::Param(x) => x.args(),
        }
    }

    fn name(&self) -> Option<Self::Name> {
        match self {
            ParamOrConcrete::Concrete(x) => x.name(),
            ParamOrConcrete::Param(x) => x.name(),
        }
    }

    fn set_name(&mut self, name: Self::Name) {
        match self {
            ParamOrConcrete::Concrete(x) => x.set_name(name),
            ParamOrConcrete::Param(x) => x.set_name(name),
        }
    }
}

pub enum IteratorEnum<A, B> {
    A(A),
    B(B),
}

impl<A, B> Iterator for IteratorEnum<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IteratorEnum::A(a) => a.next(),
            IteratorEnum::B(b) => b.next(),
        }
    }
}

impl<C: IteratableTensor + Clone, S: TensorStructure> IteratableTensor for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S>,
{
    type Data<'a> = AtomViewOrConcrete<'a, C::Data<'a>> where Self:'a ;

    fn iter_flat(&self) -> impl Iterator<Item = (FlatIndex, Self::Data<'_>)> {
        match self {
            ParamOrConcrete::Concrete(x) => IteratorEnum::A(
                x.iter_flat()
                    .map(|(i, x)| (i, AtomViewOrConcrete::Concrete(x))),
            ),
            ParamOrConcrete::Param(x) => {
                IteratorEnum::B(x.iter_flat().map(|(i, x)| (i, AtomViewOrConcrete::Atom(x))))
            }
        }
    }

    fn iter_expanded(&self) -> impl Iterator<Item = (ExpandedIndex, Self::Data<'_>)> {
        match self {
            ParamOrConcrete::Concrete(x) => IteratorEnum::A(
                x.iter_expanded()
                    .map(|(i, x)| (i, AtomViewOrConcrete::Concrete(x))),
            ),
            ParamOrConcrete::Param(x) => IteratorEnum::B(
                x.iter_expanded()
                    .map(|(i, x)| (i, AtomViewOrConcrete::Atom(x))),
            ),
        }
    }
}

pub type MixedTensor<T = f64, S = NamedStructure<Symbol, Vec<Atom>>> =
    ParamOrConcrete<RealOrComplexTensor<T, S>, S>;

// #[derive(Clone, Debug, EnumTryAsInner)]
// #[derive_err(Debug)]
// pub enum MixedTensor<T: TensorStructure = VecStructure> {
//     Float(DataTensor<f64, T>),
//     Complex(DataTensor<Complex<f64>, T>),
//     Symbolic(DataTensor<Atom, T>),
// }

impl<'a> TryFrom<FunView<'a>> for MixedTensor {
    type Error = anyhow::Error;

    fn try_from(f: FunView<'a>) -> Result<Self> {
        let mut structure: Vec<PhysicalSlots> = vec![];
        let f_id = f.get_symbol();
        let mut args = vec![];

        for arg in f.iter() {
            if let Ok(arg) = arg.try_into() {
                structure.push(arg);
            } else {
                args.push(arg.to_owned());
            }
        }
        let s = NamedStructure::from_iter(structure, f_id, Some(args));
        s.to_explicit_rep()
    }
}

impl<T: Clone, S: TensorStructure + Clone> PartialEq<MixedTensor<T, S>> for MixedTensor<T, S> {
    fn eq(&self, other: &MixedTensor<T, S>) -> bool {
        matches!(
            (self, other),
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) | (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) | (MixedTensor::Param(_), MixedTensor::Param(_))
        )
    }
}

impl<T: Clone, S: TensorStructure + Clone> Eq for MixedTensor<T, S> {}

impl<T: Clone, S: TensorStructure + Clone> PartialOrd<MixedTensor<T, S>> for MixedTensor<T, S> {
    fn partial_cmp(&self, other: &MixedTensor<T, S>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Clone, S: TensorStructure + Clone> Ord for MixedTensor<T, S> {
    fn cmp(&self, other: &MixedTensor<T, S>) -> std::cmp::Ordering {
        match (self, other) {
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) => std::cmp::Ordering::Equal,
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) => std::cmp::Ordering::Less,
            (MixedTensor::Concrete(RealOrComplexTensor::Real(_)), MixedTensor::Param(_)) => {
                std::cmp::Ordering::Less
            }
            (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) => std::cmp::Ordering::Greater,
            (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) => std::cmp::Ordering::Equal,
            (MixedTensor::Concrete(RealOrComplexTensor::Complex(_)), MixedTensor::Param(_)) => {
                std::cmp::Ordering::Less
            }
            (MixedTensor::Param(_), MixedTensor::Concrete(RealOrComplexTensor::Real(_))) => {
                std::cmp::Ordering::Greater
            }
            (MixedTensor::Param(_), MixedTensor::Concrete(RealOrComplexTensor::Complex(_))) => {
                std::cmp::Ordering::Greater
            }
            (MixedTensor::Param(_), MixedTensor::Param(_)) => std::cmp::Ordering::Equal,
        }
    }
}

impl<'a, I: TensorStructure + Clone + 'a, T: Clone> MixedTensor<T, I> {
    pub fn evaluate_real<'b, F: Fn(&Rational) -> T + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &'b HashMap<AtomView<'a>, T>,
    ) where
        'b: 'a,
        T: Real + for<'c> From<&'c Rational>,
    {
        let content = match self {
            MixedTensor::Param(x) => Some(x),
            _ => None,
        };

        if let Some(x) = content {
            *self =
                MixedTensor::Concrete(RealOrComplexTensor::Real(x.evaluate(coeff_map, const_map)));
        }
    }

    pub fn evaluate_complex<'b, F: Fn(&Rational) -> SymComplex<T> + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &'b HashMap<AtomView<'a>, SymComplex<T>>,
    ) where
        'b: 'a,
        T: Real + for<'c> From<&'c Rational>,
        SymComplex<T>: Real + for<'c> From<&'c Rational>,
    {
        let content = match self {
            MixedTensor::Param(x) => Some(x),
            _ => None,
        };

        if let Some(x) = content {
            *self = MixedTensor::Concrete(RealOrComplexTensor::Complex(
                x.evaluate(coeff_map, const_map),
            ));
        }
    }
}

impl<I> DataTensor<Atom, I>
where
    I: Clone + TensorStructure,
{
    pub fn eval_tree<'a, T: Clone + Default + Debug + Hash + Ord, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<T, I>, String> {
        EvalTreeTensor::from_data(self, coeff_map, fn_map, params)
    }

    pub fn evaluate<'a, 'b, T, F: Fn(&Rational) -> T + Copy, U>(
        &self,
        coeff_map: F,
        const_map: &'b HashMap<AtomView<'a>, T>,
    ) -> DataTensor<U, I>
    where
        T: symbolica::domains::float::Real
            + for<'c> std::convert::From<&'c symbolica::domains::rational::Rational>,
        U: From<T>,

        'a: 'b,
    {
        match self {
            DataTensor::Dense(x) => DataTensor::Dense(x.evaluate(coeff_map, const_map)),
            DataTensor::Sparse(x) => DataTensor::Sparse(x.evaluate(coeff_map, const_map)),
        }
    }
}

impl<I> SparseTensor<Atom, I>
where
    I: Clone + TensorStructure,
{
    pub fn eval_tree<'a, T: Clone + Default + Debug + Hash + Ord, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<T, I>, String> {
        EvalTreeTensor::from_sparse(self, coeff_map, fn_map, params)
    }

    pub fn evaluate<'a, T, F: Fn(&Rational) -> T + Copy, U>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomView<'a>, T>,
    ) -> SparseTensor<U, I>
    where
        T: symbolica::domains::float::Real
            + for<'d> std::convert::From<&'d symbolica::domains::rational::Rational>,
        U: From<T>,
    {
        let fn_map: HashMap<_, EvaluationFn<_>> = HashMap::default();
        let mut cache = HashMap::default();
        let structure = self.structure.clone();
        let data = self
            .elements
            .iter()
            .map(|(idx, x)| {
                (
                    *idx,
                    x.as_view()
                        .evaluate::<T, F>(coeff_map, const_map, &fn_map, &mut cache)
                        .into(),
                )
            })
            .collect::<AHashMap<_, _>>();

        SparseTensor {
            elements: data,
            structure,
        }
    }
}

impl<I> DenseTensor<Atom, I>
where
    I: Clone + TensorStructure,
{
    pub fn eval_tree<'a, T: Clone + Default + Debug + Hash + Ord, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<EvalTreeTensor<T, I>, String> {
        EvalTreeTensor::from_dense(self, coeff_map, fn_map, params)
    }

    pub fn evaluate<'a, T, F: Fn(&Rational) -> T + Copy, U>(
        &'a self,
        coeff_map: F,
        const_map: &HashMap<AtomView<'a>, T>,
    ) -> DenseTensor<U, I>
    where
        T: symbolica::domains::float::Real
            + for<'b> std::convert::From<&'b symbolica::domains::rational::Rational>,
        U: From<T>,
    {
        let fn_map: HashMap<_, EvaluationFn<_>> = HashMap::default();
        let mut cache = HashMap::default();
        let structure = self.structure.clone();
        let data = self
            .data
            .iter()
            .map(|x| {
                x.as_view()
                    .evaluate::<T, F>(coeff_map, const_map, &fn_map, &mut cache)
                    .into()
            })
            .collect::<Vec<_>>();

        DenseTensor { data, structure }
    }

    pub fn append_const_map<'a, 'b, T, U>(
        &'a self,
        data: &DenseTensor<T, I>,
        const_map: &mut HashMap<AtomView<'b>, U>,
    ) where
        I: TensorStructure,
        T: Copy,
        U: From<T>,
        'a: 'b,
    {
        for ((i, a), (j, v)) in self.flat_iter().zip(data.flat_iter()) {
            assert_eq!(i, j);
            const_map.insert(a.as_view(), (*v).into());
        }
    }
}

impl<S> HasStructure for ParamTensor<S>
where
    S: TensorStructure,
{
    type Structure = S;
    type Scalar = Atom;
    fn structure(&self) -> &Self::Structure {
        self.tensor.structure()
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        self.tensor.mut_structure()
    }

    fn scalar(self) -> Option<Self::Scalar> {
        self.tensor.scalar()
    }
}

impl<S> ScalarTensor for ParamTensor<S>
where
    S: TensorStructure + ScalarStructure,
{
    fn new_scalar(scalar: Self::Scalar) -> Self {
        ParamTensor {
            tensor: DataTensor::new_scalar(scalar),
            param_type: ParamOrComposite::Composite,
        }
    }
}

impl<S> TracksCount for ParamTensor<S>
where
    S: TensorStructure + TracksCount,
{
    fn contractions_num(&self) -> usize {
        self.tensor.contractions_num()
    }
}

// pub type MixedTensors = MixedTensor<HistoryStructure<Symbol>>;

impl<I, T: Clone> From<DenseTensor<T, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: DenseTensor<T, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Real(DataTensor::Dense(other)))
    }
}

impl<I, T: Clone> From<SparseTensor<T, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: SparseTensor<T, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Real(DataTensor::Sparse(other)))
    }
}

impl<I, T: Clone> From<DenseTensor<Complex<T>, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: DenseTensor<Complex<T>, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Complex(DataTensor::Dense(other)))
    }
}

impl<I, T: Clone> From<SparseTensor<Complex<T>, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    fn from(other: SparseTensor<Complex<T>, I>) -> Self {
        MixedTensor::Concrete(RealOrComplexTensor::Complex(DataTensor::Sparse(other)))
    }
}

impl<I, T: Clone> MixedTensor<T, I>
where
    I: TensorStructure + Clone,
{
    pub fn param(other: DataTensor<Atom, I>) -> Self {
        MixedTensor::Param(ParamTensor::param(other))
    }

    pub fn composite(other: DataTensor<Atom, I>) -> Self {
        MixedTensor::Param(ParamTensor::composite(other))
    }
}

impl<I> Contract<ParamTensor<I>> for ParamTensor<I>
where
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = ParamTensor<I>;
    fn contract(&self, other: &ParamTensor<I>) -> Result<Self::LCM, ContractionError> {
        let s = self.tensor.contract(&other.tensor)?;

        match (self.param_type, other.param_type) {
            (ParamOrComposite::Param, ParamOrComposite::Param) => Ok(ParamTensor::param(s)),
            (ParamOrComposite::Composite, ParamOrComposite::Composite) => {
                Ok(ParamTensor::composite(s))
            }
            (ParamOrComposite::Param, ParamOrComposite::Composite) => Ok(ParamTensor::composite(s)),
            (ParamOrComposite::Composite, ParamOrComposite::Param) => Ok(ParamTensor::composite(s)),
        }
    }
}

impl<I, T> Contract<ParamOrConcrete<DataTensor<T, I>, I>> for ParamOrConcrete<DataTensor<T, I>, I>
where
    I: TensorStructure + Clone + StructureContract,
    T: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = T>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
    Atom: TrySmallestUpgrade<T, LCM = Atom>, // Atom: ContractableWith<T, Out = Atom> + ContractableWith<Atom, Out = Atom>,
{
    type LCM = ParamOrConcrete<DataTensor<T, I>, I>;
    fn contract(
        &self,
        other: &ParamOrConcrete<DataTensor<T, I>, I>,
    ) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (ParamOrConcrete::Param(s), ParamOrConcrete::Param(o)) => {
                Ok(ParamOrConcrete::Param(s.contract(o)?))
            }
            (ParamOrConcrete::Param(s), ParamOrConcrete::Concrete(o)) => match s.param_type {
                ParamOrComposite::Composite => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.tensor.contract(o)?,
                ))),
                ParamOrComposite::Param => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.tensor.contract(o)?,
                ))),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Param(o)) => match o.param_type {
                ParamOrComposite::Composite => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.contract(&o.tensor)?,
                ))),
                ParamOrComposite::Param => Ok(ParamOrConcrete::Param(ParamTensor::composite(
                    s.contract(&o.tensor)?,
                ))),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Concrete(o)) => {
                Ok(ParamOrConcrete::Concrete(s.contract(o)?))
            }
        }
    }
}

impl<I, T> Contract<ParamOrConcrete<RealOrComplexTensor<T, I>, I>>
    for ParamOrConcrete<RealOrComplexTensor<T, I>, I>
where
    I: TensorStructure + Clone + StructureContract,
    T: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = T>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
    Complex<T>: ContractableWith<Atom, Out = Atom>
        + ContractableWith<T, Out = Complex<T>>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = Complex<T>>
        + FallibleAddAssign<Complex<T>>
        + FallibleSubAssign<Complex<T>>
        + RefZero
        + IsZero,
    Atom: TrySmallestUpgrade<T, LCM = Atom> + TrySmallestUpgrade<Complex<T>, LCM = Atom>,
{
    type LCM = ParamOrConcrete<RealOrComplexTensor<T, I>, I>;
    fn contract(
        &self,
        other: &ParamOrConcrete<RealOrComplexTensor<T, I>, I>,
    ) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (ParamOrConcrete::Param(s), ParamOrConcrete::Param(o)) => {
                Ok(ParamOrConcrete::Param(s.contract(o)?))
            }
            (ParamOrConcrete::Param(s), ParamOrConcrete::Concrete(o)) => match (s.param_type, o) {
                (ParamOrComposite::Composite, RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
                (ParamOrComposite::Composite, RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(s.tensor.contract(o)?)),
                ),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Param(o)) => match (o.param_type, s) {
                (ParamOrComposite::Composite, RealOrComplexTensor::Real(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
                (ParamOrComposite::Composite, RealOrComplexTensor::Complex(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Real(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
                (ParamOrComposite::Param, RealOrComplexTensor::Complex(s)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::composite(o.tensor.contract(s)?)),
                ),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Concrete(o)) => {
                Ok(ParamOrConcrete::Concrete(s.contract(o)?))
            }
        }
    }
}

pub struct EvalTreeTensor<T, S> {
    eval: EvalTree<T>,
    indexmap: Option<Vec<FlatIndex>>,
    structure: S,
}

pub struct EvalTreeTensorSet<T, S: TensorStructure> {
    tensors: Vec<DataTensor<usize, S>>,
    eval: EvalTree<T>,
    size: usize, //
}

impl<S: Clone + TensorStructure> EvalTreeTensorSet<Rational, S> {
    pub fn horner_scheme(&mut self) {
        self.eval.horner_scheme()
    }
}

impl<T, S: TensorStructure> EvalTreeTensorSet<T, S> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTreeTensorSet<T2, S>
    where
        T: Clone + Default + PartialEq,
        S: Clone,
    {
        EvalTreeTensorSet {
            eval: self.eval.map_coeff(f),
            tensors: self.tensors.clone(),
            size: self.size,
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    pub fn linearize(self) -> EvalTensorSet<T, S>
    where
        T: Clone + Default + PartialEq,
    {
        EvalTensorSet {
            eval: self.eval.linearize(),
            tensors: self.tensors,
            size: self.size,
        }
    }

    pub fn common_subexpression_elimination(&mut self, max_subexpr_len: usize)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
    {
        self.eval.common_subexpression_elimination(max_subexpr_len)
    }

    pub fn common_pair_elimination(&mut self)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
    {
        self.eval.common_pair_elimination()
    }

    pub fn evaluate(&mut self, params: &[T]) -> Vec<DataTensor<T, S>>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        let mut out_tensors = Vec::with_capacity(self.tensors.len());
        self.eval.evaluate(params, &mut elements);
        for t in self.tensors.iter() {
            out_tensors.push(t.map_data_ref(|&i| elements[i].clone()));
        }

        out_tensors
    }

    pub fn compile(
        &self,
        filename: &str,
        function_name: &str,
        library_name: &str,
    ) -> CompiledEvalTensorSet<S>
    where
        T: NumericalFloatLike,
        S: Clone,
    {
        CompiledEvalTensorSet {
            eval: self
                .eval
                .export_cpp(filename, function_name, true)
                .unwrap()
                .compile(library_name, CompileOptions::default())
                .unwrap()
                .load()
                .unwrap(),
            size: self.size,
            tensors: self.tensors.clone(),
        }
    }
}

impl<T, S: TensorStructure> HasStructure for EvalTreeTensor<T, S> {
    type Scalar = EvalTree<T>;
    type Structure = S;

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }

    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn scalar(self) -> Option<Self::Scalar> {
        if self.is_scalar() {
            Some(self.eval)
        } else {
            None
        }
    }
}

impl<S: Clone> EvalTreeTensor<Rational, S> {
    pub fn horner_scheme(&mut self) {
        self.eval.horner_scheme()
    }
}

impl<S: Clone, T> EvalTreeTensor<T, S> {
    pub fn from_dense<'a, F: Fn(&Rational) -> T + Copy>(
        dense: &'a DenseTensor<Atom, S>,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<Self, String>
    where
        T: Clone + Default + Debug + Hash + Ord,
    {
        let atomviews: Vec<AtomView> = dense.data.iter().map(|a| a.as_view()).collect();
        let eval = AtomView::to_eval_tree_multiple(&atomviews, coeff_map, fn_map, params)?;

        Ok(EvalTreeTensor {
            eval,
            indexmap: None,
            structure: dense.structure.clone(),
        })
    }

    pub fn from_sparse<'a, F: Fn(&Rational) -> T + Copy>(
        dense: &'a SparseTensor<Atom, S>,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<Self, String>
    where
        T: Clone + Default + Debug + Hash + Ord,
    {
        let atomviews: (Vec<FlatIndex>, Vec<AtomView>) = dense
            .elements
            .iter()
            .map(|(k, a)| (*k, a.as_view()))
            .unzip();
        let eval = AtomView::to_eval_tree_multiple(&atomviews.1, coeff_map, fn_map, params)?;

        Ok(EvalTreeTensor {
            eval,
            indexmap: Some(atomviews.0),
            structure: dense.structure.clone(),
        })
    }

    pub fn from_data<'a, F: Fn(&Rational) -> T + Copy>(
        data: &'a DataTensor<Atom, S>,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<Self, String>
    where
        S: TensorStructure,
        T: Clone + Default + Debug + Hash + Ord,
    {
        match data {
            DataTensor::Dense(d) => Self::from_dense(d, coeff_map, fn_map, params),
            DataTensor::Sparse(s) => Self::from_sparse(s, coeff_map, fn_map, params),
        }
    }

    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTreeTensor<T2, S>
    where
        T: Clone + Default + PartialEq,
    {
        EvalTreeTensor {
            eval: self.eval.map_coeff(f),
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        }
        // self.map_data_ref(|x| x.map_coeff(f))
    }

    pub fn linearize(self) -> EvalTensor<T, S>
    where
        T: Clone + Default + PartialEq,
    {
        EvalTensor {
            eval: self.eval.linearize(),
            structure: self.structure,
            indexmap: self.indexmap,
        }
    }

    pub fn common_subexpression_elimination(&mut self, max_subexpr_len: usize)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
    {
        self.eval.common_subexpression_elimination(max_subexpr_len)
    }

    pub fn common_pair_elimination(&mut self)
    where
        T: Debug + Hash + Eq + Ord + Clone + Default,
    {
        self.eval.common_pair_elimination()
    }

    pub fn evaluate(&mut self, params: &[T]) -> DataTensor<T, S>
    where
        T: Real,
        S: TensorStructure,
    {
        let zero = params[0].zero();
        if let Some(ref indexmap) = self.indexmap {
            let mut elements = vec![zero; indexmap.len()];
            self.eval.evaluate(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), zero.clone());
            self.eval.evaluate(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }

    pub fn compile(
        &self,
        filename: &str,
        function_name: &str,
        library_name: &str,
    ) -> CompiledEvalTensor<S>
    where
        T: NumericalFloatLike,
    {
        CompiledEvalTensor {
            eval: self
                .eval
                .export_cpp(filename, function_name, true)
                .unwrap()
                .compile(library_name, CompileOptions::default())
                .unwrap()
                .load()
                .unwrap(),
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        }
    }
}

pub struct EvalTensor<T, S> {
    eval: ExpressionEvaluator<T>,
    indexmap: Option<Vec<FlatIndex>>,
    structure: S,
}

// #[derive(Debug)]
pub struct EvalTensorSet<T, S: TensorStructure> {
    tensors: Vec<DataTensor<usize, S>>,
    eval: ExpressionEvaluator<T>,
    size: usize, //
}

impl<T, S: TensorStructure> HasStructure for EvalTensor<T, S> {
    type Scalar = ExpressionEvaluator<T>;
    type Structure = S;

    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }

    fn scalar(self) -> Option<Self::Scalar> {
        if self.is_scalar() {
            Some(self.eval)
        } else {
            None
        }
    }
}

impl<T, S> EvalTensor<T, S> {
    pub fn compile(
        &self,
        filename: &str,
        function_name: &str,
        library_name: &str,
    ) -> CompiledEvalTensor<S>
    where
        T: NumericalFloatLike,
        S: Clone,
    {
        CompiledEvalTensor {
            eval: self
                .eval
                .export_cpp(filename, function_name, true)
                .unwrap()
                .compile(library_name, CompileOptions::default())
                .unwrap()
                .load()
                .unwrap(),
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        }
    }

    pub fn compile_asm(
        &self,
        filename: &str,
        function_name: &str,
        library_name: &str,
        include_header: bool,
    ) -> CompiledEvalTensor<S>
    where
        T: NumericalFloatLike,
        S: Clone,
    {
        CompiledEvalTensor {
            eval: self
                .eval
                .export_asm(filename, function_name, include_header)
                .unwrap()
                .compile(library_name, CompileOptions::default())
                .unwrap()
                .load()
                .unwrap(),
            indexmap: self.indexmap.clone(),
            structure: self.structure.clone(),
        }
    }

    pub fn evaluate(&mut self, params: &[T]) -> DataTensor<T, S>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();
        if let Some(ref indexmap) = self.indexmap {
            let mut elements = vec![zero; indexmap.len()];
            self.eval.evaluate_multiple(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), zero.clone());
            self.eval.evaluate_multiple(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }
}

impl<T, S: TensorStructure> EvalTensorSet<T, S> {
    pub fn compile(
        &self,
        filename: &str,
        function_name: &str,
        library_name: &str,
    ) -> CompiledEvalTensorSet<S>
    where
        T: NumericalFloatLike,
        S: Clone,
    {
        CompiledEvalTensorSet {
            eval: self
                .eval
                .export_cpp(filename, function_name, true)
                .unwrap()
                .compile(library_name, CompileOptions::default())
                .unwrap()
                .load()
                .unwrap(),
            tensors: self.tensors.clone(),
            size: self.size,
        }
    }

    pub fn compile_asm(
        &self,
        filename: &str,
        function_name: &str,
        library_name: &str,
        include_header: bool,
    ) -> CompiledEvalTensorSet<S>
    where
        T: NumericalFloatLike,
        S: Clone,
    {
        CompiledEvalTensorSet {
            eval: self
                .eval
                .export_asm(filename, function_name, include_header)
                .unwrap()
                .compile(library_name, CompileOptions::default())
                .unwrap()
                .load()
                .unwrap(),
            tensors: self.tensors.clone(),
            size: self.size,
        }
    }

    pub fn evaluate(&mut self, params: &[T]) -> Vec<DataTensor<T, S>>
    where
        T: Real,
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        let mut out_tensors = Vec::with_capacity(self.tensors.len());
        self.eval.evaluate_multiple(params, &mut elements);
        for t in self.tensors.iter() {
            out_tensors.push(t.map_data_ref(|&i| elements[i].clone()));
        }

        out_tensors
    }
}

#[derive(Debug)]
pub struct CompiledEvalTensor<S> {
    eval: CompiledEvaluator,
    indexmap: Option<Vec<FlatIndex>>,
    structure: S,
}

#[derive(Debug)]
pub struct CompiledEvalTensorSet<S: TensorStructure> {
    eval: CompiledEvaluator,
    tensors: Vec<DataTensor<usize, S>>,
    size: usize,
}

impl<S: TensorStructure> HasStructure for CompiledEvalTensor<S> {
    type Scalar = CompiledEvaluator;
    type Structure = S;

    fn structure(&self) -> &Self::Structure {
        &self.structure
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        &mut self.structure
    }

    fn scalar(self) -> Option<Self::Scalar> {
        if self.is_scalar() {
            Some(self.eval)
        } else {
            None
        }
    }
}

impl<S> CompiledEvalTensor<S> {
    pub fn evaluate_float(&self, params: &[f64]) -> DataTensor<f64, S>
    where
        S: TensorStructure + Clone,
    {
        if let Some(ref indexmap) = self.indexmap {
            let mut elements = vec![0.; indexmap.len()];
            self.eval.evaluate(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), 0.);
            self.eval.evaluate(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }

    pub fn evaluate_complex(&self, params: &[SymComplex<f64>]) -> DataTensor<SymComplex<f64>, S>
    where
        S: TensorStructure + Clone,
    {
        if let Some(ref indexmap) = self.indexmap {
            let mut elements: Vec<SymComplex<f64>> = vec![SymComplex::new_zero(); indexmap.len()];
            self.eval.evaluate_complex(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), SymComplex::new_zero());
            self.eval.evaluate_complex(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }

    pub fn evaluate<T: CompiledEvaluatorFloat + Default + Clone>(
        &self,
        params: &[T],
    ) -> DataTensor<T, S>
    where
        S: TensorStructure + Clone,
    {
        if let Some(ref indexmap) = self.indexmap {
            let mut elements: Vec<T> = vec![T::default(); indexmap.len()];
            self.eval.evaluate(params, &mut elements);
            let s = SparseTensor {
                elements: indexmap.iter().cloned().zip(elements.drain(0..)).collect(),
                structure: self.structure.clone(),
            };
            DataTensor::Sparse(s)
        } else {
            let mut out_data = DenseTensor::repeat(self.structure.clone(), T::default());
            self.eval.evaluate(params, &mut out_data.data);
            DataTensor::Dense(out_data)
        }
    }
}

impl<S: TensorStructure> CompiledEvalTensorSet<S> {
    pub fn evaluate_float(&self, params: &[f64]) -> Vec<DataTensor<f64, S>>
    where
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        let mut out_tensors = Vec::with_capacity(self.tensors.len());
        self.eval.evaluate_double(params, &mut elements);
        for t in self.tensors.iter() {
            out_tensors.push(t.map_data_ref(|&i| elements[i]));
        }

        out_tensors
    }

    pub fn evaluate_complex(
        &self,
        params: &[SymComplex<f64>],
    ) -> Vec<DataTensor<SymComplex<f64>, S>>
    where
        S: TensorStructure + Clone,
    {
        let zero = params[0].zero();

        let mut elements = vec![zero; self.size];
        let mut out_tensors = Vec::with_capacity(self.tensors.len());
        self.eval.evaluate_complex(params, &mut elements);
        for t in self.tensors.iter() {
            out_tensors.push(t.map_data_ref(|&i| elements[i]));
        }

        out_tensors
    }

    pub fn evaluate<T: CompiledEvaluatorFloat + Default + Clone>(
        &self,
        params: &[T],
    ) -> Vec<DataTensor<T, S>>
    where
        S: TensorStructure + Clone,
    {
        let zero = T::default();

        let mut elements = vec![zero; self.size];
        let mut out_tensors = Vec::with_capacity(self.tensors.len());
        self.eval.evaluate(params, &mut elements);
        for t in self.tensors.iter() {
            out_tensors.push(t.map_data_ref(|&i| elements[i].clone()));
        }

        out_tensors
    }
}

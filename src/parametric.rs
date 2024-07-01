extern crate derive_more;
use derive_more::From;
use std::{fmt::Debug, process::Output};

use ahash::{AHashMap, HashMap};
use enum_try_as_inner::EnumTryAsInner;

use crate::{
    Complex, ContractableWith, ContractionError, FallibleAddAssign, FallibleMul, FallibleSubAssign,
    FlatIndex, IsZero, RefZero, TensorStructure, TrySmallestUpgrade,
};
use symbolica::{
    atom::{representation::FunView, Atom, AtomOrView, AtomView, Symbol},
    domains::rational::Rational,
    evaluate::{ConstOrExpr, EvaluationFn, ExpressionEvaluator},
};

use super::{
    Contract, DataIterator, DataTensor, DenseTensor, HasStructure, HistoryStructure, Slot,
    SparseTensor, StructureContract, TracksCount, VecStructure,
};
use symbolica::domains::float::Complex as SymComplex;

impl RefZero for Atom {
    fn ref_zero(&self) -> Self {
        Atom::new_num(0)
    }
}

impl<'a> TryFrom<FunView<'a>> for DenseTensor<Atom> {
    type Error = String;

    fn try_from(f: FunView<'a>) -> Result<Self, Self::Error> {
        let mut structure: Vec<Slot> = vec![];
        let f_id = f.get_symbol();
        let mut args = vec![];

        for arg in f.iter() {
            if let Ok(arg) = arg.try_into() {
                structure.push(arg);
            } else {
                args.push(arg.to_owned());
            }
        }
        let s: VecStructure = structure.into();
        Ok(s.shadow_with(f_id, &args))
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum ParamTensor<S: TensorStructure> {
    Param(DataTensor<Atom, S>),
    // Concrete(DataTensor<T, S>),
    Composite(DataTensor<Atom, S>),
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum ParamOrConcrete<S: TensorStructure, T: ContractableWith<Atom>> {
    Concrete(DataTensor<T, S>),
    Param(DataTensor<Atom, S>),
    Composite(DataTensor<Atom, S>),
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum MixedTensor<T: TensorStructure = VecStructure> {
    Float(DataTensor<f64, T>),
    Complex(DataTensor<Complex<f64>, T>),
    Symbolic(DataTensor<Atom, T>),
}

impl<'a> TryFrom<FunView<'a>> for MixedTensor {
    type Error = String;

    fn try_from(f: FunView<'a>) -> Result<Self, Self::Error> {
        let mut structure: Vec<Slot> = vec![];
        let f_id = f.get_symbol();
        let mut args = vec![];

        for arg in f.iter() {
            if let Ok(arg) = arg.try_into() {
                structure.push(arg);
            } else {
                args.push(arg.to_owned());
            }
        }
        let s: VecStructure = structure.into();
        Ok(s.to_explicit_rep(f_id, &args))
    }
}

impl<T: TensorStructure> PartialEq<MixedTensor<T>> for MixedTensor<T> {
    fn eq(&self, other: &MixedTensor<T>) -> bool {
        match (self, other) {
            (MixedTensor::Float(_), MixedTensor::Float(_)) => true,
            (MixedTensor::Complex(_), MixedTensor::Complex(_)) => true,
            (MixedTensor::Symbolic(_), MixedTensor::Symbolic(_)) => true,
            _ => false,
        }
    }
}

impl<T: TensorStructure> Eq for MixedTensor<T> {}

impl<T: TensorStructure> PartialOrd<MixedTensor<T>> for MixedTensor<T> {
    fn partial_cmp(&self, other: &MixedTensor<T>) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (MixedTensor::Float(_), MixedTensor::Float(_)) => Some(std::cmp::Ordering::Equal),
            (MixedTensor::Float(_), MixedTensor::Complex(_)) => Some(std::cmp::Ordering::Less),
            (MixedTensor::Float(_), MixedTensor::Symbolic(_)) => Some(std::cmp::Ordering::Less),
            (MixedTensor::Complex(_), MixedTensor::Float(_)) => Some(std::cmp::Ordering::Greater),
            (MixedTensor::Complex(_), MixedTensor::Complex(_)) => Some(std::cmp::Ordering::Equal),
            (MixedTensor::Complex(_), MixedTensor::Symbolic(_)) => Some(std::cmp::Ordering::Less),
            (MixedTensor::Symbolic(_), MixedTensor::Float(_)) => Some(std::cmp::Ordering::Greater),
            (MixedTensor::Symbolic(_), MixedTensor::Complex(_)) => {
                Some(std::cmp::Ordering::Greater)
            }
            (MixedTensor::Symbolic(_), MixedTensor::Symbolic(_)) => Some(std::cmp::Ordering::Equal),
        }
    }
}

impl<T: TensorStructure> Ord for MixedTensor<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a, I: TensorStructure + Clone + 'a> MixedTensor<I> {
    pub fn evaluate_float<'b, F: Fn(&Rational) -> f64 + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &'b HashMap<AtomView<'a>, f64>,
    ) where
        'b: 'a,
    {
        let content = match self {
            MixedTensor::Symbolic(x) => Some(x),
            _ => None,
        };

        if let Some(x) = content {
            *self = MixedTensor::Float(x.evaluate(coeff_map, const_map));
        }
    }

    pub fn evaluate_complex<'b, F: Fn(&Rational) -> SymComplex<f64> + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &'b HashMap<AtomView<'a>, SymComplex<f64>>,
    ) where
        'b: 'a,
    {
        let content = match self {
            MixedTensor::Symbolic(x) => Some(x),
            _ => None,
        };

        if let Some(x) = content {
            *self = MixedTensor::Complex(x.evaluate(coeff_map, const_map));
        }
    }
}

impl<I> DataTensor<Atom, I>
where
    I: Clone + TensorStructure,
{
    pub fn evaluator<'a, T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
    ) -> DataTensor<ExpressionEvaluator<T>, I> {
        match self {
            DataTensor::Dense(x) => DataTensor::Dense(x.evaluator(coeff_map, const_map, params)),
            DataTensor::Sparse(x) => DataTensor::Sparse(x.evaluator(coeff_map, const_map, params)),
        }
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
    pub fn evaluator<'a, T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
    ) -> SparseTensor<ExpressionEvaluator<T>, I> {
        let eval_data: AHashMap<FlatIndex, ExpressionEvaluator<_>> = AHashMap::from_iter(
            self.elements
                .iter()
                .map(|(&i, a)| (i, a.as_view().evaluator(coeff_map, const_map, params))),
        );
        SparseTensor {
            elements: eval_data,
            structure: self.structure.clone(),
        }
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
    pub fn evaluator<'a, T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
    ) -> DenseTensor<ExpressionEvaluator<T>, I> {
        let eval_data: Vec<ExpressionEvaluator<_>> = self
            .data
            .iter()
            .map(|x| x.as_view().evaluator(coeff_map, const_map, params))
            .collect();
        DenseTensor {
            data: eval_data,
            structure: self.structure.clone(),
        }
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
        match self {
            ParamTensor::Param(t) => t.structure(),
            ParamTensor::Composite(t) => t.structure(),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            ParamTensor::Param(t) => t.mut_structure(),
            ParamTensor::Composite(t) => t.mut_structure(),
        }
    }
}

impl<T> HasStructure for MixedTensor<T>
where
    T: TensorStructure,
{
    type Scalar = Atom;
    type Structure = T;

    fn structure(&self) -> &Self::Structure {
        match self {
            MixedTensor::Float(t) => t.structure(),
            MixedTensor::Complex(t) => t.structure(),
            MixedTensor::Symbolic(t) => t.structure(),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            MixedTensor::Float(t) => t.mut_structure(),
            MixedTensor::Complex(t) => t.mut_structure(),
            MixedTensor::Symbolic(t) => t.mut_structure(),
        }
    }
}

impl<S> TracksCount for ParamTensor<S>
where
    S: TensorStructure + TracksCount,
{
    fn contractions_num(&self) -> usize {
        match self {
            ParamTensor::Param(t) => t.contractions_num(),
            ParamTensor::Composite(t) => t.contractions_num(),
        }
    }
}

impl<T> TracksCount for MixedTensor<T>
where
    T: TracksCount + TensorStructure,
{
    fn contractions_num(&self) -> usize {
        match self {
            MixedTensor::Float(t) => t.contractions_num(),
            MixedTensor::Complex(t) => t.contractions_num(),
            MixedTensor::Symbolic(t) => t.contractions_num(),
        }
    }
}

// pub type MixedTensors = MixedTensor<HistoryStructure<Symbol>>;

impl<I> From<DenseTensor<f64, I>> for MixedTensor<I>
where
    I: TensorStructure,
{
    fn from(other: DenseTensor<f64, I>) -> Self {
        MixedTensor::<I>::Float(DataTensor::Dense(other))
    }
}

impl<I> From<SparseTensor<f64, I>> for MixedTensor<I>
where
    I: TensorStructure,
{
    fn from(other: SparseTensor<f64, I>) -> Self {
        MixedTensor::<I>::Float(DataTensor::Sparse(other))
    }
}

impl<I> From<DenseTensor<Complex<f64>, I>> for MixedTensor<I>
where
    I: TensorStructure,
{
    fn from(other: DenseTensor<Complex<f64>, I>) -> Self {
        MixedTensor::<I>::Complex(DataTensor::Dense(other))
    }
}

impl<I> From<SparseTensor<Complex<f64>, I>> for MixedTensor<I>
where
    I: TensorStructure,
{
    fn from(other: SparseTensor<Complex<f64>, I>) -> Self {
        MixedTensor::<I>::Complex(DataTensor::Sparse(other))
    }
}

impl<I> From<DenseTensor<Atom, I>> for MixedTensor<I>
where
    I: TensorStructure,
{
    fn from(other: DenseTensor<Atom, I>) -> Self {
        MixedTensor::<I>::Symbolic(DataTensor::Dense(other))
    }
}

impl<I> From<SparseTensor<Atom, I>> for MixedTensor<I>
where
    I: TensorStructure,
{
    fn from(other: SparseTensor<Atom, I>) -> Self {
        MixedTensor::<I>::Symbolic(DataTensor::Sparse(other))
    }
}

impl<I> Contract<ParamTensor<I>> for ParamTensor<I>
where
    I: TensorStructure + Clone + StructureContract,
{
    type LCM = ParamTensor<I>;
    fn contract(&self, other: &ParamTensor<I>) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (ParamTensor::<I>::Param(s), ParamTensor::<I>::Param(o)) => {
                Ok(ParamTensor::<I>::Composite(s.contract(o)?))
            }
            (ParamTensor::<I>::Param(s), ParamTensor::<I>::Composite(o)) => {
                Ok(ParamTensor::<I>::Composite(s.contract(o)?))
            }
            (ParamTensor::<I>::Composite(s), ParamTensor::<I>::Param(o)) => {
                Ok(ParamTensor::<I>::Composite(s.contract(o)?))
            }
            (ParamTensor::<I>::Composite(s), ParamTensor::<I>::Composite(o)) => {
                Ok(ParamTensor::<I>::Composite(s.contract(o)?))
            }
        }
    }
}

impl<I, T> Contract<ParamOrConcrete<I, T>> for ParamOrConcrete<I, T>
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
    Atom: ContractableWith<T, Out = Atom> + ContractableWith<Atom, Out = Atom>,
{
    type LCM = ParamOrConcrete<I, T>;
    fn contract(&self, other: &ParamOrConcrete<I, T>) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (ParamOrConcrete::<I, T>::Param(s), ParamOrConcrete::<I, T>::Param(o)) => {
                Ok(ParamOrConcrete::<I, T>::Param(s.contract(o)?))
            }
            (ParamOrConcrete::<I, T>::Param(s), ParamOrConcrete::<I, T>::Concrete(o)) => {
                Ok(ParamOrConcrete::<I, T>::Composite(s.contract(o)?))
            }
            (ParamOrConcrete::<I, T>::Param(s), ParamOrConcrete::<I, T>::Composite(o)) => {
                Ok(ParamOrConcrete::<I, T>::Composite(s.contract(o)?))
            }
            (ParamOrConcrete::<I, T>::Composite(s), ParamOrConcrete::<I, T>::Param(o)) => {
                Ok(ParamOrConcrete::<I, T>::Composite(s.contract(o)?))
            }
            (ParamOrConcrete::<I, T>::Composite(s), ParamOrConcrete::<I, T>::Composite(o)) => {
                Ok(ParamOrConcrete::<I, T>::Composite(s.contract(o)?))
            }
            (ParamOrConcrete::<I, T>::Composite(s), ParamOrConcrete::<I, T>::Concrete(o)) => {
                Ok(ParamOrConcrete::<I, T>::Composite(o.contract(s)?))
            }
            (ParamOrConcrete::<I, T>::Concrete(s), ParamOrConcrete::<I, T>::Param(o)) => {
                Ok(ParamOrConcrete::<I, T>::Param(s.contract(o)?))
            }
            (ParamOrConcrete::<I, T>::Concrete(s), ParamOrConcrete::<I, T>::Concrete(o)) => {
                Ok(ParamOrConcrete::<I, T>::Concrete(s.contract(o)?))
            }
            (ParamOrConcrete::<I, T>::Concrete(s), ParamOrConcrete::<I, T>::Composite(o)) => {
                Ok(ParamOrConcrete::<I, T>::Composite(s.contract(o)?))
            }
        }
    }
}

impl<I> Contract<MixedTensor<I>> for MixedTensor<I>
where
    I: TensorStructure + Clone + StructureContract + Debug,
{
    type LCM = MixedTensor<I>;
    fn contract(&self, other: &MixedTensor<I>) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (MixedTensor::<I>::Float(s), MixedTensor::<I>::Float(o)) => {
                Ok(MixedTensor::<I>::Float(s.contract(o)?))
            }
            (MixedTensor::<I>::Float(s), MixedTensor::<I>::Complex(o)) => {
                Ok(MixedTensor::<I>::Complex(s.contract(o)?))
            }
            (MixedTensor::<I>::Float(s), MixedTensor::<I>::Symbolic(o)) => {
                Ok(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Complex(s), MixedTensor::<I>::Float(o)) => {
                Ok(MixedTensor::<I>::Complex(s.contract(o)?))
            }
            (MixedTensor::<I>::Complex(s), MixedTensor::<I>::Complex(o)) => {
                Ok(MixedTensor::<I>::Complex(s.contract(o)?))
            }
            (MixedTensor::<I>::Complex(s), MixedTensor::<I>::Symbolic(o)) => {
                Ok(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Symbolic(s), MixedTensor::<I>::Float(o)) => {
                Ok(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Symbolic(s), MixedTensor::<I>::Complex(o)) => {
                Ok(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Symbolic(s), MixedTensor::<I>::Symbolic(o)) => {
                Ok(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
        }
    }
}

extern crate derive_more;

use std::fmt::{Debug, Display};

use ahash::{AHashMap, HashMap};
use enum_try_as_inner::EnumTryAsInner;

use crate::{
    Complex, ContractableWith, ContractionError, FallibleAddAssign, FallibleMul, FallibleSubAssign,
    FlatIndex, HasName, IsZero, IteratableTensor, RefZero, TensorStructure, ToSymbolic,
};
use symbolica::{
    atom::{representation::FunView, Atom, AtomOrView, AtomView},
    domains::{float::Real, rational::Rational},
    evaluate::{ConstOrExpr, EvaluationFn, ExpressionEvaluator},
};

use super::{
    Contract, DataIterator, DataTensor, DenseTensor, HasStructure, Slot, SparseTensor,
    StructureContract, TracksCount, VecStructure,
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

impl<S: TensorStructure + Clone> ParamTensor<S> {
    pub fn evaluator<'a, T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &'a self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
    ) -> DataTensor<ExpressionEvaluator<T>, S> {
        match self {
            ParamTensor::Composite(x) => x.evaluator(coeff_map, const_map, params),
            ParamTensor::Param(x) => x.evaluator(coeff_map, const_map, params),
        }
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
        match self {
            ParamTensor::Composite(x) => x.evaluate(coeff_map, const_map),
            ParamTensor::Param(x) => x.evaluate(coeff_map, const_map),
        }
    }
}

impl<S: TensorStructure> IteratableTensor for ParamTensor<S> {
    type Data<'a> =  AtomView<'a> where Self: 'a;

    fn iter_expanded<'a>(&'a self) -> impl Iterator<Item = (crate::ExpandedIndex, Self::Data<'a>)> {
        match self {
            ParamTensor::Composite(x) => {
                IteratorEnum::A(x.iter_expanded().map(|(i, x)| (i, x.as_view())))
            }
            ParamTensor::Param(x) => {
                IteratorEnum::B(x.iter_expanded().map(|(i, x)| (i, x.as_view())))
            }
        }
    }

    fn iter_flat<'a>(&'a self) -> impl Iterator<Item = (FlatIndex, Self::Data<'a>)> {
        match self {
            ParamTensor::Composite(x) => {
                IteratorEnum::A(x.iter_flat().map(|(i, x)| (i, x.as_view())))
            }
            ParamTensor::Param(x) => IteratorEnum::B(x.iter_flat().map(|(i, x)| (i, x.as_view()))),
        }
    }
}

impl<S> Display for ParamTensor<S>
where
    S: TensorStructure,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamTensor::Composite(x) => write!(f, "{}", x),
            ParamTensor::Param(x) => write!(f, "{}", x),
        }
    }
}

impl<S: TensorStructure> HasName for ParamTensor<S>
where
    S: HasName,
{
    type Args = S::Args;
    type Name = S::Name;

    fn id(&self) -> Option<Self::Args> {
        match self {
            ParamTensor::Composite(x) => x.id(),
            ParamTensor::Param(x) => x.id(),
        }
    }

    fn name(&self) -> Option<Self::Name> {
        match self {
            ParamTensor::Composite(x) => x.name(),
            ParamTensor::Param(x) => x.name(),
        }
    }

    fn set_name(&mut self, name: Self::Name) {
        match self {
            ParamTensor::Composite(x) => x.set_name(name),
            ParamTensor::Param(x) => x.set_name(name),
        }
    }
}

#[derive(Debug, Clone)]
// #[derive_err(Debug)]
pub enum ParamOrConcrete<C: HasStructure + Clone, S: TensorStructure> {
    Concrete(C),
    Param(ParamTensor<S>),
}

pub enum AtomViewOrConcrete<'a, T> {
    Atom(AtomView<'a>),
    Concrete(T),
}

impl<C: HasStructure + Clone, S: TensorStructure> ParamOrConcrete<C, S> {
    pub fn is_parametric(&self) -> bool {
        match self {
            ParamOrConcrete::Param(_) => true,
            _ => false,
        }
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

impl<C, S> HasStructure for ParamOrConcrete<C, S>
where
    C: HasStructure<Structure = S> + Clone,
    S: TensorStructure,
{
    type Scalar = Atom;
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
}

impl<C, S> TracksCount for ParamOrConcrete<C, S>
where
    C: TracksCount + HasStructure + Clone,
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
    C: HasName + HasStructure + Clone,
    S: TensorStructure + HasName<Name = C::Name, Args = C::Args>,
{
    type Args = C::Args;
    type Name = C::Name;

    fn id(&self) -> Option<Self::Args> {
        match self {
            ParamOrConcrete::Concrete(x) => x.id(),
            ParamOrConcrete::Param(x) => x.id(),
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

    fn iter_flat<'a>(&'a self) -> impl Iterator<Item = (FlatIndex, Self::Data<'a>)> {
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

    fn iter_expanded<'a>(&'a self) -> impl Iterator<Item = (crate::ExpandedIndex, Self::Data<'a>)> {
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

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum RealOrComplexTensor<T, S: TensorStructure> {
    Real(DataTensor<T, S>),
    Complex(DataTensor<Complex<T>, S>),
}

pub enum RealOrComplexRef<'a, T> {
    Real(&'a T),
    Complex(&'a Complex<T>),
}

pub type MixedTensor<T = f64, S = VecStructure> = ParamOrConcrete<RealOrComplexTensor<T, S>, S>;

impl<T: Clone, S: TensorStructure> HasStructure for RealOrComplexTensor<T, S> {
    type Scalar = Complex<T>;
    type Structure = S;
    fn structure(&self) -> &Self::Structure {
        match self {
            RealOrComplexTensor::Real(r) => r.structure(),
            RealOrComplexTensor::Complex(r) => r.structure(),
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            RealOrComplexTensor::Real(r) => r.mut_structure(),
            RealOrComplexTensor::Complex(r) => r.mut_structure(),
        }
    }
}

impl<T, S> TracksCount for RealOrComplexTensor<T, S>
where
    S: TensorStructure + TracksCount,
    T: Clone,
{
    fn contractions_num(&self) -> usize {
        match self {
            RealOrComplexTensor::Real(r) => r.contractions_num(),
            RealOrComplexTensor::Complex(r) => r.contractions_num(),
        }
    }
}

impl<T, S> HasName for RealOrComplexTensor<T, S>
where
    S: TensorStructure + HasName,
    T: Clone,
{
    type Args = S::Args;
    type Name = S::Name;

    fn name(&self) -> Option<S::Name> {
        match self {
            RealOrComplexTensor::Real(r) => r.name(),
            RealOrComplexTensor::Complex(r) => r.name(),
        }
    }

    fn set_name(&mut self, name: Self::Name) {
        match self {
            RealOrComplexTensor::Real(r) => r.set_name(name),
            RealOrComplexTensor::Complex(r) => r.set_name(name),
        }
    }

    fn id(&self) -> Option<S::Args> {
        match self {
            RealOrComplexTensor::Real(r) => r.id(),
            RealOrComplexTensor::Complex(r) => r.id(),
        }
    }
}

impl<T: Clone, S: TensorStructure> IteratableTensor for RealOrComplexTensor<T, S> {
    type Data<'a>=  RealOrComplexRef<'a,T>
        where
            Self: 'a;

    fn iter_expanded<'a>(&'a self) -> impl Iterator<Item = (crate::ExpandedIndex, Self::Data<'a>)> {
        match self {
            RealOrComplexTensor::Real(x) => IteratorEnum::A(
                x.iter_expanded()
                    .map(|(i, x)| (i, RealOrComplexRef::Real(x))),
            ),
            RealOrComplexTensor::Complex(x) => IteratorEnum::B(
                x.iter_expanded()
                    .map(|(i, x)| (i, RealOrComplexRef::Complex(x))),
            ),
        }
    }

    fn iter_flat<'a>(&'a self) -> impl Iterator<Item = (FlatIndex, Self::Data<'a>)> {
        match self {
            RealOrComplexTensor::Real(x) => {
                IteratorEnum::A(x.iter_flat().map(|(i, x)| (i, RealOrComplexRef::Real(x))))
            }
            RealOrComplexTensor::Complex(x) => IteratorEnum::B(
                x.iter_flat()
                    .map(|(i, x)| (i, RealOrComplexRef::Complex(x))),
            ),
        }
    }
}

// #[derive(Clone, Debug, EnumTryAsInner)]
// #[derive_err(Debug)]
// pub enum MixedTensor<T: TensorStructure = VecStructure> {
//     Float(DataTensor<f64, T>),
//     Complex(DataTensor<Complex<f64>, T>),
//     Symbolic(DataTensor<Atom, T>),
// }

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

impl<T: Clone, S: TensorStructure + Clone> PartialEq<MixedTensor<T, S>> for MixedTensor<T, S> {
    fn eq(&self, other: &MixedTensor<T, S>) -> bool {
        match (self, other) {
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) => true,
            (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) => true,
            (MixedTensor::Param(_), MixedTensor::Param(_)) => true,
            _ => false,
        }
    }
}

impl<T: Clone, S: TensorStructure + Clone> Eq for MixedTensor<T, S> {}

impl<T: Clone, S: TensorStructure + Clone> PartialOrd<MixedTensor<T, S>> for MixedTensor<T, S> {
    fn partial_cmp(&self, other: &MixedTensor<T, S>) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) => Some(std::cmp::Ordering::Equal),
            (
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) => Some(std::cmp::Ordering::Less),
            (MixedTensor::Concrete(RealOrComplexTensor::Real(_)), MixedTensor::Param(_)) => {
                Some(std::cmp::Ordering::Less)
            }
            (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Real(_)),
            ) => Some(std::cmp::Ordering::Greater),
            (
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
                MixedTensor::Concrete(RealOrComplexTensor::Complex(_)),
            ) => Some(std::cmp::Ordering::Equal),
            (MixedTensor::Concrete(RealOrComplexTensor::Complex(_)), MixedTensor::Param(_)) => {
                Some(std::cmp::Ordering::Less)
            }
            (MixedTensor::Param(_), MixedTensor::Concrete(RealOrComplexTensor::Real(_))) => {
                Some(std::cmp::Ordering::Greater)
            }
            (MixedTensor::Param(_), MixedTensor::Concrete(RealOrComplexTensor::Complex(_))) => {
                Some(std::cmp::Ordering::Greater)
            }
            (MixedTensor::Param(_), MixedTensor::Param(_)) => Some(std::cmp::Ordering::Equal),
        }
    }
}

impl<T: Clone, S: TensorStructure + Clone> Ord for MixedTensor<T, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
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
        MixedTensor::Param(ParamTensor::Param(other))
    }

    pub fn composite(other: DataTensor<Atom, I>) -> Self {
        MixedTensor::Param(ParamTensor::Composite(other))
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
    Atom: ContractableWith<T, Out = Atom> + ContractableWith<Atom, Out = Atom>,
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
            (ParamOrConcrete::Param(s), ParamOrConcrete::Concrete(o)) => match s {
                ParamTensor::Composite(s) => Ok(ParamOrConcrete::Param(ParamTensor::Composite(
                    s.contract(o)?,
                ))),
                ParamTensor::Param(s) => Ok(ParamOrConcrete::Param(ParamTensor::Composite(
                    s.contract(o)?,
                ))),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Param(o)) => match o {
                ParamTensor::Composite(o) => Ok(ParamOrConcrete::Param(ParamTensor::Composite(
                    s.contract(o)?,
                ))),
                ParamTensor::Param(o) => Ok(ParamOrConcrete::Param(ParamTensor::Composite(
                    s.contract(o)?,
                ))),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Concrete(o)) => {
                Ok(ParamOrConcrete::Concrete(s.contract(o)?))
            }
        }
    }
}

impl<S, T> Contract<RealOrComplexTensor<T, S>> for RealOrComplexTensor<T, S>
where
    S: TensorStructure + Clone + StructureContract,
    T: ContractableWith<T, Out = T>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
    Complex<T>: ContractableWith<T, Out = Complex<T>>
        + ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = Complex<T>>
        + FallibleAddAssign<Complex<T>>
        + FallibleSubAssign<Complex<T>>
        + RefZero
        + IsZero,
{
    type LCM = RealOrComplexTensor<T, S>;
    fn contract(&self, other: &RealOrComplexTensor<T, S>) -> Result<Self::LCM, ContractionError> {
        match (self, other) {
            (RealOrComplexTensor::Real(s), RealOrComplexTensor::Real(o)) => {
                Ok(RealOrComplexTensor::Real(s.contract(o)?))
            }
            (RealOrComplexTensor::Real(s), RealOrComplexTensor::Complex(o)) => {
                Ok(RealOrComplexTensor::Complex(s.contract(o)?))
            }
            (RealOrComplexTensor::Complex(s), RealOrComplexTensor::Real(o)) => {
                Ok(RealOrComplexTensor::Complex(s.contract(o)?))
            }
            (RealOrComplexTensor::Complex(s), RealOrComplexTensor::Complex(o)) => {
                Ok(RealOrComplexTensor::Complex(s.contract(o)?))
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
    Atom: ContractableWith<T, Out = Atom>
        + ContractableWith<Atom, Out = Atom>
        + ContractableWith<Complex<T>, Out = Atom>,
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
            (ParamOrConcrete::Param(s), ParamOrConcrete::Concrete(o)) => match (s, o) {
                (ParamTensor::Composite(s), RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
                (ParamTensor::Composite(s), RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Param(o)) => match (o, s) {
                (ParamTensor::Composite(s), RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
                (ParamTensor::Composite(s), RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Real(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Complex(o)) => Ok(
                    ParamOrConcrete::Param(ParamTensor::Composite(s.contract(o)?)),
                ),
            },
            (ParamOrConcrete::Concrete(s), ParamOrConcrete::Concrete(o)) => {
                Ok(ParamOrConcrete::Concrete(s.contract(o)?))
            }
        }
    }
}

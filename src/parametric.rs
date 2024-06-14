use std::fmt::Debug;

use ahash::{AHashMap, HashMap};
use enum_try_as_inner::EnumTryAsInner;

use crate::{Complex, RefZero};
use symbolica::{
    atom::{Atom, AtomView, Symbol},
    domains::rational::Rational,
    evaluate::EvaluationFn,
};

use super::{
    Contract, DataIterator, DataTensor, DenseTensor, HasName, HasStructure, HistoryStructure, Slot,
    SparseTensor, StructureContract, TracksCount, VecStructure,
};
use symbolica::domains::float::Complex as SymComplex;

impl RefZero for Atom {
    fn ref_zero(&self) -> Self {
        Atom::new_num(0)
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum MixedTensor<T: HasStructure = VecStructure> {
    Float(DataTensor<f64, T>),
    Complex(DataTensor<Complex<f64>, T>),
    Symbolic(DataTensor<Atom, T>),
}

impl<T: HasStructure> PartialEq<MixedTensor<T>> for MixedTensor<T> {
    fn eq(&self, other: &MixedTensor<T>) -> bool {
        match (self, other) {
            (MixedTensor::Float(_), MixedTensor::Float(_)) => true,
            (MixedTensor::Complex(_), MixedTensor::Complex(_)) => true,
            (MixedTensor::Symbolic(_), MixedTensor::Symbolic(_)) => true,
            _ => false,
        }
    }
}

impl<T: HasStructure> Eq for MixedTensor<T> {}

impl<T: HasStructure> PartialOrd<MixedTensor<T>> for MixedTensor<T> {
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

impl<T: HasStructure> Ord for MixedTensor<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a, I: HasStructure + Clone + 'a> MixedTensor<I> {
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
    I: Clone + HasStructure,
{
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
    I: Clone,
{
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
    I: Clone,
{
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
        I: HasStructure,
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

impl<T> HasStructure for MixedTensor<T>
where
    T: HasStructure,
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
    fn external_structure(&self) -> &[Slot] {
        match self {
            MixedTensor::Float(t) => t.external_structure(),
            MixedTensor::Complex(t) => t.external_structure(),
            MixedTensor::Symbolic(t) => t.external_structure(),
        }
    }
}

impl<T> HasName for MixedTensor<T>
where
    T: HasName + HasStructure,
{
    type Name = T::Name;

    fn name(&self) -> Option<std::borrow::Cow<'_, <T as HasName>::Name>> {
        match self {
            MixedTensor::Float(t) => t.name(),
            MixedTensor::Complex(t) => t.name(),
            MixedTensor::Symbolic(t) => t.name(),
        }
    }

    fn set_name(&mut self, name: &Self::Name) {
        match self {
            MixedTensor::Float(t) => t.set_name(name),
            MixedTensor::Complex(t) => t.set_name(name),
            MixedTensor::Symbolic(t) => t.set_name(name),
        }
    }
}

impl<T> TracksCount for MixedTensor<T>
where
    T: TracksCount + HasStructure,
{
    fn contractions_num(&self) -> usize {
        match self {
            MixedTensor::Float(t) => t.contractions_num(),
            MixedTensor::Complex(t) => t.contractions_num(),
            MixedTensor::Symbolic(t) => t.contractions_num(),
        }
    }
}

pub type MixedTensors = MixedTensor<HistoryStructure<Symbol>>;

impl<I> From<DenseTensor<f64, I>> for MixedTensor<I>
where
    I: HasStructure,
{
    fn from(other: DenseTensor<f64, I>) -> Self {
        MixedTensor::<I>::Float(DataTensor::Dense(other))
    }
}

impl<I> From<SparseTensor<f64, I>> for MixedTensor<I>
where
    I: HasStructure,
{
    fn from(other: SparseTensor<f64, I>) -> Self {
        MixedTensor::<I>::Float(DataTensor::Sparse(other))
    }
}

impl<I> From<DenseTensor<Complex<f64>, I>> for MixedTensor<I>
where
    I: HasStructure,
{
    fn from(other: DenseTensor<Complex<f64>, I>) -> Self {
        MixedTensor::<I>::Complex(DataTensor::Dense(other))
    }
}

impl<I> From<SparseTensor<Complex<f64>, I>> for MixedTensor<I>
where
    I: HasStructure,
{
    fn from(other: SparseTensor<Complex<f64>, I>) -> Self {
        MixedTensor::<I>::Complex(DataTensor::Sparse(other))
    }
}

impl<I> From<DenseTensor<Atom, I>> for MixedTensor<I>
where
    I: HasStructure,
{
    fn from(other: DenseTensor<Atom, I>) -> Self {
        MixedTensor::<I>::Symbolic(DataTensor::Dense(other))
    }
}

impl<I> From<SparseTensor<Atom, I>> for MixedTensor<I>
where
    I: HasStructure,
{
    fn from(other: SparseTensor<Atom, I>) -> Self {
        MixedTensor::<I>::Symbolic(DataTensor::Sparse(other))
    }
}

impl<I> Contract<MixedTensor<I>> for MixedTensor<I>
where
    I: HasStructure + Clone + StructureContract + Debug,
{
    type LCM = MixedTensor<I>;
    fn contract(&self, other: &MixedTensor<I>) -> Option<Self::LCM> {
        match (self, other) {
            (MixedTensor::<I>::Float(s), MixedTensor::<I>::Float(o)) => {
                Some(MixedTensor::<I>::Float(s.contract(o)?))
            }
            (MixedTensor::<I>::Float(s), MixedTensor::<I>::Complex(o)) => {
                Some(MixedTensor::<I>::Complex(s.contract(o)?))
            }
            (MixedTensor::<I>::Float(s), MixedTensor::<I>::Symbolic(o)) => {
                Some(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Complex(s), MixedTensor::<I>::Float(o)) => {
                Some(MixedTensor::<I>::Complex(s.contract(o)?))
            }
            (MixedTensor::<I>::Complex(s), MixedTensor::<I>::Complex(o)) => {
                Some(MixedTensor::<I>::Complex(s.contract(o)?))
            }
            (MixedTensor::<I>::Complex(s), MixedTensor::<I>::Symbolic(o)) => {
                Some(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Symbolic(s), MixedTensor::<I>::Float(o)) => {
                Some(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Symbolic(s), MixedTensor::<I>::Complex(o)) => {
                Some(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
            (MixedTensor::<I>::Symbolic(s), MixedTensor::<I>::Symbolic(o)) => {
                Some(MixedTensor::<I>::Symbolic(s.contract(o)?))
            }
        }
    }
}

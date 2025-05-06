use symbolica::atom::Atom;

use crate::{
    arithmetic::ScalarMul,
    complex::{Complex, RealOrComplex, RealOrComplexTensor},
    data::DataTensor,
    structure::TensorStructure,
    symbolica_utils::SerializableAtom,
    upgrading_arithmetic::FallibleMul,
};

use super::{ConcreteOrParam, MixedTensor, ParamOrComposite, ParamOrConcrete, ParamTensor};

impl<S: TensorStructure + Clone> ScalarMul<SerializableAtom> for ParamTensor<S> {
    type Output = ParamTensor<S>;
    fn scalar_mul(&self, rhs: &SerializableAtom) -> Option<Self::Output> {
        Some(ParamTensor {
            tensor: self.tensor.scalar_mul(&rhs.0)?,
            param_type: ParamOrComposite::Composite,
        })
    }
}

impl<S: TensorStructure + Clone> ScalarMul<f64> for ParamTensor<S> {
    type Output = ParamTensor<S>;
    fn scalar_mul(&self, rhs: &f64) -> Option<Self::Output> {
        Some(ParamTensor::composite(self.tensor.scalar_mul(rhs)?))
    }
}

impl<S: TensorStructure + Clone> ScalarMul<Complex<f64>> for ParamTensor<S> {
    type Output = ParamTensor<S>;
    fn scalar_mul(&self, rhs: &Complex<f64>) -> Option<Self::Output> {
        Some(ParamTensor::composite(self.tensor.scalar_mul(rhs)?))
    }
}

impl<S: TensorStructure + Clone, T> ScalarMul<RealOrComplex<T>> for ParamTensor<S>
where
    ParamTensor<S>:
        ScalarMul<T, Output = ParamTensor<S>> + ScalarMul<Complex<T>, Output = ParamTensor<S>>,
{
    type Output = ParamTensor<S>;
    fn scalar_mul(&self, rhs: &RealOrComplex<T>) -> Option<Self::Output> {
        Some(match rhs {
            RealOrComplex::Complex(c) => self.scalar_mul(c)?,
            RealOrComplex::Real(c) => self.scalar_mul(c)?,
        })
    }
}

impl<S: TensorStructure + Clone, T> ScalarMul<ConcreteOrParam<T>> for ParamTensor<S>
where
    ParamTensor<S>: ScalarMul<T, Output = ParamTensor<S>>,
{
    type Output = ParamTensor<S>;
    fn scalar_mul(&self, rhs: &ConcreteOrParam<T>) -> Option<Self::Output> {
        Some(match rhs {
            ConcreteOrParam::Param(a) => ParamTensor::composite(self.tensor.scalar_mul(a)?),
            ConcreteOrParam::Concrete(c) => self.scalar_mul(c)?,
        })
    }
}

impl<S: TensorStructure + Clone> ScalarMul<Atom> for ParamTensor<S> {
    type Output = ParamTensor<S>;
    fn scalar_mul(&self, rhs: &Atom) -> Option<Self::Output> {
        Some(ParamTensor {
            tensor: self.tensor.scalar_mul(rhs)?,
            param_type: ParamOrComposite::Composite,
        })
    }
}

impl<T, I> ScalarMul<Atom> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
    T: Clone + FallibleMul<Atom, Output = Atom>,
    Complex<T>: FallibleMul<Atom, Output = Atom>,
{
    type Output = MixedTensor<T, I>;
    fn scalar_mul(&self, rhs: &Atom) -> Option<Self::Output> {
        match self {
            MixedTensor::Param(a) => Some(MixedTensor::Param(a.scalar_mul(rhs)?)),

            MixedTensor::Concrete(RealOrComplexTensor::Real(a)) => Some(MixedTensor::Param(
                ParamTensor::composite(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Complex(a)) => Some(MixedTensor::Param(
                ParamTensor::composite(a.scalar_mul(rhs)?),
            )),
        }
    }
}

impl<U, S> ScalarMul<Atom> for RealOrComplexTensor<U, S>
where
    DataTensor<U, S>: ScalarMul<Atom, Output = DataTensor<Atom, S>>,

    DataTensor<Complex<U>, S>: ScalarMul<Atom, Output = DataTensor<Atom, S>>,
    S: TensorStructure + Clone,
{
    type Output = ParamTensor<S>;

    fn scalar_mul(&self, rhs: &Atom) -> Option<Self::Output> {
        Some(match self {
            RealOrComplexTensor::Real(rt) => ParamTensor::composite(rt.scalar_mul(rhs)?),
            RealOrComplexTensor::Complex(rt) => ParamTensor::composite(rt.scalar_mul(rhs)?),
        })
    }
}

impl<T, U, S, Out> ScalarMul<ConcreteOrParam<T>> for ParamOrConcrete<U, S>
where
    ParamTensor<S>:
        ScalarMul<T, Output = ParamTensor<S>> + ScalarMul<Atom, Output = ParamTensor<S>>,
    U: ScalarMul<T, Output = Out> + ScalarMul<Atom, Output = ParamTensor<S>>,
    S: TensorStructure + Clone,
{
    type Output = ParamOrConcrete<Out, S>;

    fn scalar_mul(&self, rhs: &ConcreteOrParam<T>) -> Option<Self::Output> {
        Some(match (self, rhs) {
            (ParamOrConcrete::Param(rt), ConcreteOrParam::Param(rs)) => {
                ParamOrConcrete::Param(rt.scalar_mul(rs)?)
            }
            (ParamOrConcrete::Concrete(rt), ConcreteOrParam::Param(rs)) => {
                ParamOrConcrete::Param(rt.scalar_mul(rs)?)
            }
            (ParamOrConcrete::Param(rt), ConcreteOrParam::Concrete(rs)) => {
                ParamOrConcrete::Param(rt.scalar_mul(rs)?)
            }
            (ParamOrConcrete::Concrete(rt), ConcreteOrParam::Concrete(rs)) => {
                ParamOrConcrete::Concrete(rt.scalar_mul(rs)?)
            }
        })
    }
}

impl<T, I> ScalarMul<SerializableAtom> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
    T: Clone + FallibleMul<Atom, Output = Atom>,
    Complex<T>: FallibleMul<Atom, Output = Atom>,
{
    type Output = MixedTensor<T, I>;
    fn scalar_mul(&self, rhs: &SerializableAtom) -> Option<Self::Output> {
        self.scalar_mul(&rhs.0)
    }
}

impl<I, T> ScalarMul<T> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
    T: FallibleMul<T, Output = T> + Clone,
    Atom: FallibleMul<T, Output = Atom>,
    Complex<T>: FallibleMul<T, Output = Complex<T>>,
{
    type Output = MixedTensor<T, I>;
    fn scalar_mul(&self, rhs: &T) -> Option<Self::Output> {
        match self {
            MixedTensor::Param(a) => Some(MixedTensor::Param(ParamTensor::composite(
                a.tensor.scalar_mul(rhs)?,
            ))),
            MixedTensor::Concrete(RealOrComplexTensor::Real(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Real(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Complex(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(a.scalar_mul(rhs)?),
            )),
        }
    }
}

impl<I, T> ScalarMul<Complex<T>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
    T: FallibleMul<Complex<T>, Output = Complex<T>> + Clone,
    Atom: FallibleMul<Complex<T>, Output = Atom>,
    Complex<T>: FallibleMul<Complex<T>, Output = Complex<T>>,
{
    type Output = MixedTensor<T, I>;
    fn scalar_mul(&self, rhs: &Complex<T>) -> Option<Self::Output> {
        match self {
            MixedTensor::Param(a) => Some(MixedTensor::Param(ParamTensor::composite(
                a.tensor.scalar_mul(rhs)?,
            ))),
            MixedTensor::Concrete(RealOrComplexTensor::Real(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Complex(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(a.scalar_mul(rhs)?),
            )),
        }
    }
}

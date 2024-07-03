use crate::{
    ConcreteIndex, GetTensorData, IsZero, IteratableTensor, RefZero, SetTensorData,
    TensorStructure, TrySmallestUpgrade,
};
use std::ops::Neg;

use super::{
    DataTensor, DenseTensor, FallibleAdd, FallibleMul, FallibleSub, HasStructure, SparseTensor,
};

#[cfg(feature = "shadowing")]
use super::{Complex, MixedTensor, ParamTensor, RealOrComplexTensor};
#[cfg(feature = "shadowing")]
use symbolica::atom::Atom;

impl<T, S> RefZero for DenseTensor<T, S>
where
    T: RefZero + Clone,
    S: TensorStructure + Clone,
{
    fn ref_zero(&self) -> Self {
        let zero = self.data[0].ref_zero();
        DenseTensor {
            structure: self.structure.clone(),
            data: vec![zero; self.data.len()],
        }
    }
}

impl<T, U, S> std::ops::Neg for DenseTensor<T, S>
where
    T: std::ops::Neg<Output = U>,
    S: TensorStructure + Clone,
{
    type Output = DenseTensor<U, S>;
    fn neg(self) -> Self::Output {
        DenseTensor {
            structure: self.structure.clone(),
            data: self.data.into_iter().map(|x| x.neg()).collect(),
        }
    }
}

impl<T, U, I, Out> FallibleAdd<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: FallibleAdd<T, Output = Out>,
    I: TensorStructure + Clone,
{
    type Output = DenseTensor<Out, I>;
    fn add_fallible(&self, rhs: &DenseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        // Makes rhs into self ,when applied.
        let permutation = self.structure().find_permutation(rhs.structure()).unwrap();
        let structure = self.structure().clone();

        let data: Option<Vec<Out>> = self
            .iter_expanded()
            .map(|(indices, u)| {
                let permuted_indices: Vec<ConcreteIndex> =
                    permutation.iter().map(|&index| indices[index]).collect();
                let t = rhs.get(&permuted_indices).unwrap();
                u.add_fallible(t)
            })
            .collect();

        data.map(|data| DenseTensor { structure, data })
    }
}

impl<'a, T, U, I, Out> FallibleAdd<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: FallibleAdd<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    I: TensorStructure + Clone,
    T: Clone,
    Out: Clone,
{
    type Output = DenseTensor<Out, I>;
    fn add_fallible(&self, rhs: &SparseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        let permutation = self.structure().find_permutation(rhs.structure()).unwrap();
        let structure = self.structure().clone();

        let data: Option<Vec<Out>> = self
            .iter_expanded()
            .map(|(indices, u)| {
                let permuted_indices: Vec<ConcreteIndex> =
                    permutation.iter().map(|&index| indices[index]).collect();
                let t = rhs.get(&permuted_indices);
                if let Ok(t) = t {
                    u.add_fallible(&t)
                } else {
                    Some(u.try_upgrade().unwrap().into_owned())
                }
            })
            .collect();

        data.map(|data| DenseTensor { structure, data })
    }
}

impl<'a, T, U, I, Out> FallibleAdd<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: FallibleAdd<T, Output = Out>,
    I: TensorStructure + Clone,
    T: TrySmallestUpgrade<U, LCM = Out>,

    Out: Clone,
{
    type Output = DenseTensor<Out, I>;
    fn add_fallible(&self, rhs: &DenseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        let permutation = rhs.structure().find_permutation(self.structure()).unwrap();
        let structure = rhs.structure().clone();

        let data: Option<Vec<Out>> = rhs
            .iter_expanded()
            .map(|(indices, t)| {
                let permuted_indices: Vec<ConcreteIndex> =
                    permutation.iter().map(|&index| indices[index]).collect();
                let u = self.get(&permuted_indices);
                if let Ok(u) = u {
                    u.add_fallible(t)
                } else {
                    Some(t.try_upgrade().unwrap().into_owned())
                }
            })
            .collect();

        data.map(|data| DenseTensor { structure, data })
    }
}

impl<'a, T, U, I, Out> FallibleAdd<SparseTensor<T, I>> for SparseTensor<U, I>
where
    I: TensorStructure + Clone,
    U: FallibleAdd<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    T: Clone + TrySmallestUpgrade<U, LCM = Out>,
    Out: IsZero + Clone,
{
    type Output = SparseTensor<Out, I>;
    fn add_fallible(&self, rhs: &SparseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        let permutation = self.structure().find_permutation(rhs.structure()).unwrap();
        let structure = self.structure().clone();
        let mut data = SparseTensor::empty(structure);
        for (indices, u) in self.iter_expanded() {
            let permuted_indices: Vec<ConcreteIndex> =
                permutation.iter().map(|&index| indices[index]).collect();
            let t = rhs.get(&permuted_indices);
            if let Ok(t) = t {
                data.smart_set(&indices, u.add_fallible(t)?).unwrap();
            } else {
                data.smart_set(&indices, u.try_upgrade().unwrap().into_owned())
                    .unwrap();
            }
            // println!("{:?}", t);
            // data.smart_set(&indices, u.add_fallible(&t)?).unwrap();
        }

        let permutation: Vec<usize> = rhs.structure().find_permutation(self.structure()).unwrap();
        for (i, t) in rhs.iter_expanded() {
            let permuted_indices: Vec<ConcreteIndex> =
                permutation.iter().map(|&index| i[index]).collect();

            if self.get(&permuted_indices).is_err() {
                data.smart_set(&i, t.try_upgrade().unwrap().into_owned())
                    .unwrap();
            }
        }

        Some(data)
    }
}

impl<T, U, Out, I> FallibleAdd<DataTensor<T, I>> for DataTensor<U, I>
where
    U: FallibleAdd<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    T: Clone + TrySmallestUpgrade<U, LCM = Out>,
    Out: IsZero + Clone,
    I: TensorStructure + Clone,
{
    type Output = DataTensor<Out, I>;
    fn add_fallible(&self, rhs: &DataTensor<T, I>) -> Option<Self::Output> {
        match (self, rhs) {
            (DataTensor::Dense(a), DataTensor::Dense(b)) => {
                Some(DataTensor::Dense(a.add_fallible(b)?))
            }
            (DataTensor::Sparse(a), DataTensor::Sparse(b)) => {
                Some(DataTensor::Sparse(a.add_fallible(b)?))
            }
            (DataTensor::Dense(a), DataTensor::Sparse(b)) => {
                Some(DataTensor::Dense(a.add_fallible(b)?))
            }
            (DataTensor::Sparse(a), DataTensor::Dense(b)) => {
                Some(DataTensor::Dense(a.add_fallible(b)?))
            }
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure + Clone> FallibleAdd<RealOrComplexTensor<T, S>>
    for RealOrComplexTensor<T, S>
where
    T: FallibleAdd<T, Output = T>
        + TrySmallestUpgrade<T, LCM = T>
        + IsZero
        + Clone
        + FallibleAdd<Complex<T>, Output = Complex<T>>
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>,
    Complex<T>: FallibleAdd<T, Output = Complex<T>>
        + FallibleAdd<Complex<T>, Output = Complex<T>>
        + IsZero
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>
        + TrySmallestUpgrade<T, LCM = Complex<T>>
        + Clone,
{
    type Output = RealOrComplexTensor<T, S>;

    fn add_fallible(&self, rhs: &RealOrComplexTensor<T, S>) -> Option<Self::Output> {
        match (self, rhs) {
            (RealOrComplexTensor::Real(s), RealOrComplexTensor::Real(r)) => {
                Some(RealOrComplexTensor::Real(s.add_fallible(r)?))
            }
            (RealOrComplexTensor::Real(s), RealOrComplexTensor::Complex(r)) => {
                Some(RealOrComplexTensor::Complex(s.add_fallible(r)?))
            }
            (RealOrComplexTensor::Complex(s), RealOrComplexTensor::Real(r)) => {
                Some(RealOrComplexTensor::Complex(s.add_fallible(r)?))
            }
            (RealOrComplexTensor::Complex(s), RealOrComplexTensor::Complex(r)) => {
                Some(RealOrComplexTensor::Complex(s.add_fallible(r)?))
            }
        }
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure + Clone> FallibleAdd<ParamTensor<S>> for ParamTensor<S> {
    type Output = ParamTensor<S>;
    fn add_fallible(&self, rhs: &ParamTensor<S>) -> Option<Self::Output> {
        match (self, rhs) {
            (ParamTensor::Composite(s), ParamTensor::Composite(r)) => {
                Some(ParamTensor::Composite(s.add_fallible(r)?))
            }
            (ParamTensor::Composite(s), ParamTensor::Param(r)) => {
                Some(ParamTensor::Composite(s.add_fallible(r)?))
            }
            (ParamTensor::Param(s), ParamTensor::Composite(r)) => {
                Some(ParamTensor::Composite(s.add_fallible(r)?))
            }
            (ParamTensor::Param(s), ParamTensor::Param(r)) => {
                Some(ParamTensor::Composite(s.add_fallible(r)?))
            }
        }
    }
}

#[cfg(feature = "shadowing")]
impl<I, T> FallibleAdd<MixedTensor<T, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
    T: FallibleAdd<T, Output = T>
        + FallibleAdd<Atom, Output = Atom>
        + TrySmallestUpgrade<T, LCM = T>
        + IsZero
        + Clone
        + FallibleAdd<Complex<T>, Output = Complex<T>>
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>
        + TrySmallestUpgrade<Atom, LCM = Atom>,
    Complex<T>: FallibleAdd<T, Output = Complex<T>>
        + FallibleAdd<Complex<T>, Output = Complex<T>>
        + FallibleAdd<Atom, Output = Atom>
        + IsZero
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>
        + TrySmallestUpgrade<T, LCM = Complex<T>>
        + TrySmallestUpgrade<Atom, LCM = Atom>
        + Clone,
    Atom: TrySmallestUpgrade<T, LCM = Atom>
        + TrySmallestUpgrade<Complex<T>, LCM = Atom>
        + FallibleAdd<T, Output = Atom>
        + FallibleAdd<Complex<T>, Output = Atom>,
{
    type Output = MixedTensor<T, I>;
    fn add_fallible(&self, rhs: &MixedTensor<T, I>) -> Option<Self::Output> {
        match (self, rhs) {
            (MixedTensor::Param(a), MixedTensor::Param(b)) => {
                Some(MixedTensor::Param(a.add_fallible(b)?))
            }
            (MixedTensor::Param(s), MixedTensor::Concrete(o)) => match (s, o) {
                (ParamTensor::Composite(s), RealOrComplexTensor::Real(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.add_fallible(o)?)),
                ),
                (ParamTensor::Composite(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.add_fallible(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Real(o)) => Some(MixedTensor::Param(
                    ParamTensor::Composite(s.add_fallible(o)?),
                )),
                (ParamTensor::Param(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.add_fallible(o)?)),
                ),
            },
            (MixedTensor::Concrete(s), MixedTensor::Param(o)) => match (o, s) {
                (ParamTensor::Composite(s), RealOrComplexTensor::Real(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.add_fallible(o)?)),
                ),
                (ParamTensor::Composite(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.add_fallible(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Real(o)) => Some(MixedTensor::Param(
                    ParamTensor::Composite(s.add_fallible(o)?),
                )),
                (ParamTensor::Param(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.add_fallible(o)?)),
                ),
            },
            (MixedTensor::Concrete(s), MixedTensor::Concrete(o)) => {
                Some(MixedTensor::Concrete(s.add_fallible(o)?))
            }
        }
    }
}

impl<T, U, I, Out> FallibleSub<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: FallibleSub<T, Output = Out>,
    I: TensorStructure + Clone,
{
    type Output = DenseTensor<Out, I>;
    fn sub_fallible(&self, rhs: &DenseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        let permutation = self.structure().find_permutation(rhs.structure()).unwrap();
        let structure = self.structure().clone();

        let data: Option<Vec<Out>> = self
            .iter_expanded()
            .map(|(indices, u)| {
                let permuted_indices: Vec<ConcreteIndex> =
                    permutation.iter().map(|&index| indices[index]).collect();
                let t = rhs.get(&permuted_indices).unwrap();
                u.sub_fallible(t)
            })
            .collect();

        data.map(|data| DenseTensor { structure, data })
    }
}

impl<T, U, I, Out> FallibleSub<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: FallibleSub<T, Output = Out>,
    I: TensorStructure + Clone,
    T: TrySmallestUpgrade<U, LCM = Out>,
    Out: Neg<Output = Out> + Clone,
{
    type Output = DenseTensor<Out, I>;
    fn sub_fallible(&self, rhs: &DenseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        let permutation = rhs.structure().find_permutation(self.structure()).unwrap();
        let structure = rhs.structure().clone();

        let data: Option<Vec<Out>> = rhs
            .iter_expanded()
            .map(|(indices, t)| {
                let permuted_indices: Vec<ConcreteIndex> =
                    permutation.iter().map(|&index| indices[index]).collect();
                let u = self.get(&permuted_indices);
                if let Ok(u) = u {
                    u.sub_fallible(t)
                } else {
                    Some(t.try_upgrade().unwrap().into_owned().neg())
                }
            })
            .collect();

        data.map(|data| DenseTensor { structure, data })
    }
}

impl<T, U, I, Out> FallibleSub<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: FallibleSub<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    I: TensorStructure + Clone,
    T: Clone,
    Out: Clone,
{
    type Output = DenseTensor<Out, I>;
    fn sub_fallible(&self, rhs: &SparseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        let permutation = self.structure().find_permutation(rhs.structure()).unwrap();
        let structure = self.structure().clone();

        let data: Option<Vec<Out>> = self
            .iter_expanded()
            .map(|(indices, u)| {
                let permuted_indices: Vec<ConcreteIndex> =
                    permutation.iter().map(|&index| indices[index]).collect();
                let t = rhs.get(&permuted_indices);
                if let Ok(t) = t {
                    u.sub_fallible(&t)
                } else {
                    Some(u.try_upgrade().unwrap().into_owned())
                }
            })
            .collect();

        data.map(|data| DenseTensor { structure, data })
    }
}

impl<T, U, I, Out> FallibleSub<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: FallibleSub<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    I: TensorStructure + Clone,
    T: Clone + TrySmallestUpgrade<U, LCM = Out>,
    Out: IsZero + Clone + Neg<Output = Out>,
{
    type Output = SparseTensor<Out, I>;
    fn sub_fallible(&self, rhs: &SparseTensor<T, I>) -> Option<Self::Output> {
        assert!(self.structure().same_external(rhs.structure()));
        let permutation = self.structure().find_permutation(rhs.structure()).unwrap();
        let structure = self.structure().clone();
        let mut data = SparseTensor::empty(structure);
        for (indices, u) in self.iter_expanded() {
            let permuted_indices: Vec<ConcreteIndex> =
                permutation.iter().map(|&index| indices[index]).collect();
            let t = rhs.get(&permuted_indices);
            if let Ok(t) = t {
                data.smart_set(&indices, u.sub_fallible(t)?).unwrap();
            } else {
                data.smart_set(&indices, u.try_upgrade().unwrap().into_owned())
                    .unwrap();
            }
        }
        let permutation: Vec<usize> = rhs.structure().find_permutation(self.structure()).unwrap();
        for (i, t) in rhs.iter_expanded() {
            let permuted_indices: Vec<ConcreteIndex> =
                permutation.iter().map(|&index| i[index]).collect();

            if self.get(&permuted_indices).is_err() {
                data.smart_set(&i, t.try_upgrade().unwrap().into_owned().neg())
                    .unwrap();
            }
        }

        Some(data)
    }
}

impl<T, U, Out, I> FallibleSub<DataTensor<T, I>> for DataTensor<U, I>
where
    U: FallibleSub<T, Output = Out> + TrySmallestUpgrade<T, LCM = Out>,
    T: Clone + TrySmallestUpgrade<U, LCM = Out>,
    Out: IsZero + Clone + Neg<Output = Out>,
    I: TensorStructure + Clone,
{
    type Output = DataTensor<Out, I>;
    fn sub_fallible(&self, rhs: &DataTensor<T, I>) -> Option<Self::Output> {
        match (self, rhs) {
            (DataTensor::Dense(a), DataTensor::Dense(b)) => {
                Some(DataTensor::Dense(a.sub_fallible(b)?))
            }
            (DataTensor::Sparse(a), DataTensor::Sparse(b)) => {
                Some(DataTensor::Sparse(a.sub_fallible(b)?))
            }
            (DataTensor::Dense(a), DataTensor::Sparse(b)) => {
                Some(DataTensor::Dense(a.sub_fallible(b)?))
            }
            (DataTensor::Sparse(a), DataTensor::Dense(b)) => {
                Some(DataTensor::Dense(a.sub_fallible(b)?))
            }
        }
    }
}
#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure + Clone> FallibleSub<RealOrComplexTensor<T, S>>
    for RealOrComplexTensor<T, S>
where
    T: FallibleSub<T, Output = T>
        + TrySmallestUpgrade<T, LCM = T>
        + IsZero
        + Neg<Output = T>
        + Clone
        + FallibleSub<Complex<T>, Output = Complex<T>>
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>,
    Complex<T>: FallibleSub<T, Output = Complex<T>>
        + FallibleSub<Complex<T>, Output = Complex<T>>
        + IsZero
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>
        + TrySmallestUpgrade<T, LCM = Complex<T>>
        + Neg<Output = Complex<T>>
        + Clone,
{
    type Output = RealOrComplexTensor<T, S>;

    fn sub_fallible(&self, rhs: &RealOrComplexTensor<T, S>) -> Option<Self::Output> {
        match (self, rhs) {
            (RealOrComplexTensor::Real(s), RealOrComplexTensor::Real(r)) => {
                Some(RealOrComplexTensor::Real(s.sub_fallible(r)?))
            }
            (RealOrComplexTensor::Real(s), RealOrComplexTensor::Complex(r)) => {
                Some(RealOrComplexTensor::Complex(s.sub_fallible(r)?))
            }
            (RealOrComplexTensor::Complex(s), RealOrComplexTensor::Real(r)) => {
                Some(RealOrComplexTensor::Complex(s.sub_fallible(r)?))
            }
            (RealOrComplexTensor::Complex(s), RealOrComplexTensor::Complex(r)) => {
                Some(RealOrComplexTensor::Complex(s.sub_fallible(r)?))
            }
        }
    }
}

#[cfg(feature = "shadowing")]
impl<S: TensorStructure + Clone> FallibleSub<ParamTensor<S>> for ParamTensor<S> {
    type Output = ParamTensor<S>;
    fn sub_fallible(&self, rhs: &ParamTensor<S>) -> Option<Self::Output> {
        match (self, rhs) {
            (ParamTensor::Composite(s), ParamTensor::Composite(r)) => {
                Some(ParamTensor::Composite(s.sub_fallible(r)?))
            }
            (ParamTensor::Composite(s), ParamTensor::Param(r)) => {
                Some(ParamTensor::Composite(s.sub_fallible(r)?))
            }
            (ParamTensor::Param(s), ParamTensor::Composite(r)) => {
                Some(ParamTensor::Composite(s.sub_fallible(r)?))
            }
            (ParamTensor::Param(s), ParamTensor::Param(r)) => {
                Some(ParamTensor::Composite(s.sub_fallible(r)?))
            }
        }
    }
}

#[cfg(feature = "shadowing")]
impl<I, T> FallibleSub<MixedTensor<T, I>> for MixedTensor<T, I>
where
    I: TensorStructure + Clone,
    T: FallibleSub<T, Output = T>
        + FallibleSub<Atom, Output = Atom>
        + TrySmallestUpgrade<T, LCM = T>
        + IsZero
        + Clone
        + Neg<Output = T>
        + FallibleSub<Complex<T>, Output = Complex<T>>
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>
        + TrySmallestUpgrade<Atom, LCM = Atom>,
    Complex<T>: FallibleSub<T, Output = Complex<T>>
        + FallibleSub<Complex<T>, Output = Complex<T>>
        + FallibleSub<Atom, Output = Atom>
        + IsZero
        + Neg<Output = Complex<T>>
        + TrySmallestUpgrade<Complex<T>, LCM = Complex<T>>
        + TrySmallestUpgrade<T, LCM = Complex<T>>
        + TrySmallestUpgrade<Atom, LCM = Atom>
        + Clone,
    Atom: TrySmallestUpgrade<T, LCM = Atom>
        + TrySmallestUpgrade<Complex<T>, LCM = Atom>
        + FallibleSub<T, Output = Atom>
        + FallibleSub<Complex<T>, Output = Atom>,
{
    type Output = MixedTensor<T, I>;
    fn sub_fallible(&self, rhs: &MixedTensor<T, I>) -> Option<Self::Output> {
        match (self, rhs) {
            (MixedTensor::Param(a), MixedTensor::Param(b)) => {
                Some(MixedTensor::Param(a.sub_fallible(b)?))
            }
            (MixedTensor::Param(s), MixedTensor::Concrete(o)) => match (s, o) {
                (ParamTensor::Composite(s), RealOrComplexTensor::Real(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.sub_fallible(o)?)),
                ),
                (ParamTensor::Composite(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.sub_fallible(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Real(o)) => Some(MixedTensor::Param(
                    ParamTensor::Composite(s.sub_fallible(o)?),
                )),
                (ParamTensor::Param(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.sub_fallible(o)?)),
                ),
            },
            (MixedTensor::Concrete(s), MixedTensor::Param(o)) => match (o, s) {
                (ParamTensor::Composite(s), RealOrComplexTensor::Real(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.sub_fallible(o)?)),
                ),
                (ParamTensor::Composite(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.sub_fallible(o)?)),
                ),
                (ParamTensor::Param(s), RealOrComplexTensor::Real(o)) => Some(MixedTensor::Param(
                    ParamTensor::Composite(s.sub_fallible(o)?),
                )),
                (ParamTensor::Param(s), RealOrComplexTensor::Complex(o)) => Some(
                    MixedTensor::Param(ParamTensor::Composite(s.sub_fallible(o)?)),
                ),
            },
            (MixedTensor::Concrete(s), MixedTensor::Concrete(o)) => {
                Some(MixedTensor::Concrete(s.sub_fallible(o)?))
            }
        }
    }
}

pub trait ScalarMul<T> {
    type Output;
    fn scalar_mul(&self, rhs: &T) -> Option<Self::Output>;
}

impl<T, U, I, Out> ScalarMul<T> for DenseTensor<U, I>
where
    U: FallibleMul<T, Output = Out>,
    I: TensorStructure + Clone,
{
    type Output = DenseTensor<Out, I>;
    fn scalar_mul(&self, rhs: &T) -> Option<Self::Output> {
        let data: Option<Vec<Out>> = self.iter_flat().map(|(_, u)| u.mul_fallible(rhs)).collect();

        data.map(|data| DenseTensor {
            structure: self.structure().clone(),
            data,
        })
    }
}

impl<T, U, I, Out> ScalarMul<T> for SparseTensor<U, I>
where
    U: FallibleMul<T, Output = Out>,
    I: TensorStructure + Clone,
{
    type Output = SparseTensor<Out, I>;
    fn scalar_mul(&self, rhs: &T) -> Option<Self::Output> {
        let mut data = SparseTensor::empty(self.structure().clone());
        for (indices, u) in self.iter_flat() {
            data.set_flat(indices, u.mul_fallible(rhs)?).unwrap();
        }
        Some(data)
    }
}

impl<T, U, I, Out> ScalarMul<T> for DataTensor<U, I>
where
    U: FallibleMul<T, Output = Out>,
    I: TensorStructure + Clone,
{
    type Output = DataTensor<Out, I>;
    fn scalar_mul(&self, rhs: &T) -> Option<Self::Output> {
        match self {
            DataTensor::Dense(a) => Some(DataTensor::Dense(a.scalar_mul(rhs)?)),
            DataTensor::Sparse(a) => Some(DataTensor::Sparse(a.scalar_mul(rhs)?)),
        }
    }
}

#[cfg(feature = "shadowing")]
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
            MixedTensor::Param(ParamTensor::Composite(a)) => Some(MixedTensor::Param(
                ParamTensor::Composite(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Param(ParamTensor::Param(a)) => Some(MixedTensor::Param(
                ParamTensor::Composite(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Real(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Real(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Complex(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(a.scalar_mul(rhs)?),
            )),
        }
    }
}

#[cfg(feature = "shadowing")]
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
            MixedTensor::Param(ParamTensor::Composite(a)) => Some(MixedTensor::Param(
                ParamTensor::Composite(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Param(ParamTensor::Param(a)) => Some(MixedTensor::Param(
                ParamTensor::Composite(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Real(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Complex(a)) => Some(MixedTensor::Concrete(
                RealOrComplexTensor::Complex(a.scalar_mul(rhs)?),
            )),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<I> ScalarMul<Atom> for ParamTensor<I>
where
    I: TensorStructure + Clone,
{
    type Output = ParamTensor<I>;
    fn scalar_mul(&self, rhs: &Atom) -> Option<Self::Output> {
        match self {
            ParamTensor::Composite(a) => Some(ParamTensor::Composite(a.scalar_mul(rhs)?)),
            ParamTensor::Param(a) => Some(ParamTensor::Composite(a.scalar_mul(rhs)?)),
        }
    }
}

#[cfg(feature = "shadowing")]
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
                ParamTensor::Composite(a.scalar_mul(rhs)?),
            )),
            MixedTensor::Concrete(RealOrComplexTensor::Complex(a)) => Some(MixedTensor::Param(
                ParamTensor::Composite(a.scalar_mul(rhs)?),
            )),
        }
    }
}

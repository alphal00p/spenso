use std::ops::AddAssign;

use crate::{
    complex::{Complex, RealOrComplexTensor},
    contraction::RefZero,
    data::{DataTensor, DenseTensor, GetTensorData, SparseTensor},
    structure::{ScalarStructure, TensorStructure},
};

impl<T, U, I> AddAssign<DenseTensor<T, I>> for DenseTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: DenseTensor<T, I>) {
        for (u, t) in self.data.iter_mut().zip(rhs.data.iter()) {
            *u += t;
        }
    }
}

impl<T, U, I> AddAssign<DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: DenseTensor<T, I>) {
        for (i, u) in self.elements.iter_mut() {
            *u += &rhs[*i];
        }
    }
}

impl<T, U, I> AddAssign<&DenseTensor<T, I>> for SparseTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: &DenseTensor<T, I>) {
        for (i, u) in self.elements.iter_mut() {
            *u += &rhs[*i];
        }
    }
}

impl<T, U, I> AddAssign<SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: SparseTensor<T, I>) {
        for (i, u) in self.elements.iter_mut() {
            if let Some(t) = rhs.get_ref_linear(*i) {
                *u += t;
            }
        }
    }
}

impl<T, U, I> AddAssign<&SparseTensor<T, I>> for SparseTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: &SparseTensor<T, I>) {
        for (i, u) in self.elements.iter_mut() {
            if let Some(t) = rhs.get_ref_linear(*i) {
                *u += t;
            }
        }
    }
}

impl<T, U, I> AddAssign<SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: SparseTensor<T, I>) {
        for (i, u) in rhs.elements.iter() {
            self[*i] += u;
        }
    }
}

impl<T, U, I> AddAssign<&SparseTensor<T, I>> for DenseTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: &SparseTensor<T, I>) {
        for (i, u) in rhs.elements.iter() {
            self[*i] += u;
        }
    }
}

impl<T, U, I> AddAssign<DataTensor<T, I>> for DataTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: DataTensor<T, I>) {
        match (self, rhs) {
            (DataTensor::Dense(a), DataTensor::Dense(b)) => {
                *a += b;
            }
            (DataTensor::Sparse(a), DataTensor::Sparse(b)) => {
                *a += b;
            }
            (DataTensor::Dense(a), DataTensor::Sparse(b)) => {
                *a += b;
            }
            (DataTensor::Sparse(a), DataTensor::Dense(b)) => {
                *a += b;
            }
        }
    }
}

impl<T, U, I> AddAssign<&DataTensor<T, I>> for DataTensor<U, I>
where
    U: for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone,
{
    fn add_assign(&mut self, rhs: &DataTensor<T, I>) {
        match (self, rhs) {
            (DataTensor::Dense(a), DataTensor::Dense(b)) => {
                *a += b;
            }
            (DataTensor::Sparse(a), DataTensor::Sparse(b)) => {
                *a += b;
            }
            (DataTensor::Dense(a), DataTensor::Sparse(b)) => {
                *a += b;
            }
            (DataTensor::Sparse(a), DataTensor::Dense(b)) => {
                *a += b;
            }
        }
    }
}

impl<T, U, I> AddAssign<RealOrComplexTensor<T, I>> for RealOrComplexTensor<U, I>
where
    U: for<'a> AddAssign<&'a T> + RefZero,
    Complex<U>: for<'a> AddAssign<&'a Complex<T>> + for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone + ScalarStructure,
{
    fn add_assign(&mut self, rhs: RealOrComplexTensor<T, I>) {
        match (self, rhs) {
            (RealOrComplexTensor::Complex(a), RealOrComplexTensor::Complex(b)) => {
                *a += b;
            }
            (RealOrComplexTensor::Real(a), RealOrComplexTensor::Real(b)) => {
                *a += b;
            }
            (RealOrComplexTensor::Complex(a), RealOrComplexTensor::Real(b)) => {
                *a += b;
            }
            (a, b) => {
                a.to_complex();
                *a += b;
            }
        }
    }
}

impl<T, U, I> AddAssign<&RealOrComplexTensor<T, I>> for RealOrComplexTensor<U, I>
where
    U: for<'a> AddAssign<&'a T> + RefZero,
    Complex<U>: for<'a> AddAssign<&'a Complex<T>> + for<'a> AddAssign<&'a T>,
    I: TensorStructure + Clone + ScalarStructure,
{
    fn add_assign(&mut self, rhs: &RealOrComplexTensor<T, I>) {
        match (self, rhs) {
            (RealOrComplexTensor::Complex(a), RealOrComplexTensor::Complex(b)) => {
                *a += b;
            }
            (RealOrComplexTensor::Real(a), RealOrComplexTensor::Real(b)) => {
                *a += b;
            }
            (RealOrComplexTensor::Complex(a), RealOrComplexTensor::Real(b)) => {
                *a += b;
            }
            (a, b) => {
                a.to_complex();
                *a += b;
            }
        }
    }
}

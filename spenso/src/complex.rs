use std::{
    fmt::{Debug, Display, LowerExp},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    data::{SparseTensor, StorageTensor},
    network::Ref,
    structure::concrete_index::ConcreteIndex,
};
use anyhow::{anyhow, Result};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use duplicate::duplicate;
use enum_try_as_inner::EnumTryAsInner;
use num::{Float, One, Zero};
use ref_ops::{RefAdd, RefDiv, RefMul, RefSub};
use serde::{Deserialize, Serialize};
#[cfg(feature = "shadowing")]
use symbolica::{
    atom::Atom,
    domains::{
        float::{Complex as SymComplex, ConstructibleFloat, NumericalFloatLike, Real, SingleFloat},
        rational::Rational,
    },
    evaluate::{CompiledEvaluatorFloat, FunctionMap},
};

use crate::structure::abstract_index::AbstractIndex;
use crate::structure::dimension::Dimension;
use crate::structure::representation::Representation;
use crate::structure::slot::IsAbstractSlot;
use crate::structure::StructureError;
use delegate::delegate;

#[cfg(feature = "shadowing")]
use crate::{
    data::{DataIterator, DenseTensor},
    parametric::TensorCoefficient,
    shadowing::{ShadowMapping, Shadowable},
    structure::ToSymbolic,
    symbolica_utils::{IntoArgs, IntoSymbol},
};

use crate::{
    contraction::{Contract, ContractableWith, ContractionError, IsZero, RefOne, RefZero, Trace},
    data::{DataTensor, GetTensorData, HasTensorData, SetTensorData, SparseOrDense},
    iterators::{IteratableTensor, IteratorEnum},
    structure::{
        concrete_index::{ExpandedIndex, FlatIndex},
        CastStructure, HasName, HasStructure, ScalarStructure, ScalarTensor, StructureContract,
        TensorStructure, TracksCount,
    },
    upgrading_arithmetic::{FallibleAddAssign, FallibleMul, FallibleSubAssign},
};

pub trait R {}
duplicate! {
    [t;
    [f32];
    [f64];
    [i8];
    [i16];
    [i32];
    [i64];
    [i128];
    [u8];
    [u16];
    [u32];
    [u64];
    [u128];
    ]
    impl R for t {}
}
#[derive(Copy, Clone, PartialEq, Serialize, Deserialize, Hash, Eq, PartialOrd, Ord)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T> Complex<T> {
    pub fn as_ref(&self) -> Complex<&T> {
        Complex {
            re: &self.re,
            im: &self.im,
        }
    }
}

pub mod add;
pub mod add_assign;
pub mod div;
pub mod div_assign;
pub mod mul;
pub mod mul_assign;
pub mod neg;
pub mod sub;
pub mod sub_assign;

#[cfg(feature = "shadowing")]
pub mod symbolica_traits;

impl<T: AbsDiffEq> AbsDiffEq for Complex<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        T::abs_diff_eq(&self.re, &other.re, epsilon) && T::abs_diff_eq(&self.im, &other.im, epsilon)
    }
}

impl<T: RelativeEq> RelativeEq for Complex<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        T::relative_eq(&self.re, &other.re, epsilon, max_relative)
            && T::relative_eq(&self.im, &other.im, epsilon, max_relative)
    }
}

impl<T: UlpsEq> UlpsEq for Complex<T>
where
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.re, &other.re, epsilon, max_ulps)
            && T::ulps_eq(&self.im, &other.im, epsilon, max_ulps)
    }
}

#[cfg(feature = "shadowing")]
impl RefZero for Rational {
    fn ref_zero(&self) -> Self {
        self.zero()
    }
}

#[cfg(feature = "shadowing")]
impl From<f64> for Complex<Rational> {
    fn from(re: f64) -> Self {
        Complex {
            re: Rational::from(re),
            im: Rational::zero(),
        }
    }
}

#[cfg(feature = "shadowing")]
impl From<Complex<f64>> for Complex<Rational> {
    fn from(value: Complex<f64>) -> Self {
        Complex {
            re: Rational::from(value.re),
            im: Rational::from(value.im),
        }
    }
}

impl<T: RefZero> RefZero for Complex<T> {
    fn ref_zero(&self) -> Self {
        Complex::new(self.re.ref_zero(), self.im.ref_zero())
    }
}

impl<T: RefZero + RefOne> RefOne for Complex<T> {
    fn ref_one(&self) -> Self {
        Complex::new(self.re.ref_one(), self.im.ref_zero())
    }
}

impl<T> From<T> for Complex<T>
where
    T: RefZero,
{
    fn from(re: T) -> Self {
        Complex {
            im: re.ref_zero(),
            re,
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Real> From<Complex<T>> for SymComplex<T> {
    fn from(complex: Complex<T>) -> Self {
        SymComplex::new(complex.re, complex.im)
    }
}

#[cfg(feature = "shadowing")]
impl<T: Real> From<SymComplex<T>> for Complex<T> {
    fn from(complex: SymComplex<T>) -> Self {
        Complex::new(complex.re, complex.im)
    }
}

impl<T: Default> Default for Complex<T> {
    fn default() -> Self {
        Complex {
            re: T::default(),
            im: T::default(),
        }
    }
}

pub trait SymbolicaComplex {
    type R;
    fn arg(&self) -> Self::R;
}

pub trait NumTraitComplex {
    type R;
    fn arg(&self) -> Self::R;
}

#[cfg(feature = "shadowing")]
impl<T: Real> SymbolicaComplex for Complex<T> {
    type R = T;
    fn arg(&self) -> T {
        self.im.atan2(&self.re)
    }
}

impl<T: Float> NumTraitComplex for Complex<T> {
    type R = T;
    fn arg(&self) -> T {
        self.im.atan2(self.re)
    }
}

impl<T: Zero> Zero for Complex<T> {
    fn zero() -> Self {
        Complex {
            re: T::zero(),
            im: T::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    fn set_zero(&mut self) {
        self.re.set_zero();
        self.im.set_zero();
    }
}

impl<T: Zero + One + PartialEq + Sub<Output = T> + Clone> One for Complex<T> {
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.re.is_one() && self.im.is_zero()
    }

    fn one() -> Self {
        Complex {
            re: T::one(),
            im: T::zero(),
        }
    }

    fn set_one(&mut self) {
        self.re.set_one();
        self.im.set_zero();
    }
}

pub trait FloatDerived<T: Float> {
    fn norm(&self) -> T;

    fn to_polar_coordinates(self) -> (T, T);
    fn from_polar_coordinates(r: T, phi: T) -> Self;
}

impl<T: Float + for<'a> RefMul<&'a T, Output = T> + Add<T, Output = T>> FloatDerived<T>
    for Complex<T>
{
    fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    #[inline]
    fn to_polar_coordinates(self) -> (T, T)
    where
        T: num::Float,
    {
        (self.norm_squared().sqrt(), self.arg())
    }

    #[inline]
    fn from_polar_coordinates(r: T, phi: T) -> Complex<T> {
        Complex::new(r * phi.cos(), r * phi.sin())
    }
}

impl<T> Complex<T> {
    pub fn map_ref<U>(&self, f: impl Fn(&T) -> U) -> Complex<U> {
        Complex {
            re: f(&self.re),
            im: f(&self.im),
        }
    }

    pub fn map<U>(self, f: impl Fn(T) -> U) -> Complex<U> {
        Complex {
            re: f(self.re),
            im: f(self.im),
        }
    }

    pub fn map_mut(&mut self, mut f: impl FnMut(&mut T)) {
        f(&mut self.re);
        f(&mut self.im);
    }

    pub fn map_ref_mut<U>(&mut self, mut f: impl FnMut(&mut T) -> U) -> Complex<U> {
        Complex {
            re: f(&mut self.re),
            im: f(&mut self.im),
        }
    }
    #[inline]
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }

    pub fn new_re(re: T) -> Complex<T>
    where
        T: RefZero,
    {
        Complex {
            im: re.ref_zero(),
            re,
        }
    }

    pub fn new_im(im: T) -> Complex<T>
    where
        T: RefZero,
    {
        Complex {
            re: im.ref_zero(),
            im,
        }
    }

    #[cfg(feature = "shadowing")]
    pub fn new_zero() -> Self
    where
        T: ConstructibleFloat,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

    #[cfg(feature = "shadowing")]
    #[inline]
    pub fn new_i() -> Self
    where
        T: ConstructibleFloat,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_one(),
        }
    }

    pub fn ref_i(&self) -> Complex<T>
    where
        T: RefOne + RefZero,
    {
        Complex {
            re: self.re.ref_zero(),
            im: self.im.ref_one(),
        }
    }

    pub fn pow<'a>(&'a self, e: u64) -> Self
    where
        T: RefOne
            + RefZero
            + for<'c> RefMul<&'c T, Output = T>
            + for<'c> RefAdd<&'c T, Output = T>
            + for<'c> RefSub<&'c T, Output = T>,
    {
        // TODO: use binary exponentiation
        let mut r = self.ref_one();
        for _ in 0..e {
            r *= self;
        }
        r
    }

    pub fn conj(&self) -> Complex<T>
    where
        T: Clone + Neg<Output = T>,
    {
        Complex::new(self.re.clone(), -self.im.clone())
    }

    #[inline]
    pub fn i() -> Complex<T>
    where
        T: num::Zero + num::One,
    {
        Complex {
            re: T::zero(),
            im: T::one(),
        }
    }

    #[inline]
    pub fn norm_squared(&self) -> T
    where
        T: for<'a> RefMul<&'a T, Output = T> + Add<T, Output = T>,
    {
        (self.re.ref_mul(&self.re)) + (self.im.ref_mul(&self.im))
    }

    pub fn inv(&self) -> Self
    where
        T: for<'a> RefMul<&'a T, Output = T>
            + Add<T, Output = T>
            + for<'a> RefDiv<&'a T, Output = T>
            + Neg<Output = T>,
    {
        let n = self.norm_squared();
        Complex::new(self.re.ref_div(&n), -self.im.ref_div(&n))
    }
}

impl<T: Display> std::fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}+{}i)", self.re, self.im))
    }
}

impl<T: Debug> std::fmt::Debug for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({:?}+{:?}i)", self.re, self.im))
    }
}

impl<T: LowerExp> std::fmt::LowerExp for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({:e}+{:e}i)", self.re, self.im))
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum RealOrComplexTensor<T, S: TensorStructure> {
    Real(DataTensor<T, S>),
    Complex(DataTensor<Complex<T>, S>),
}

impl<T, S: TensorStructure> crate::network::Ref for RealOrComplexTensor<T, S> {
    type Ref<'a>
        = &'a RealOrComplexTensor<T, S>
    where
        Self: 'a;

    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        self
    }
}

impl<T: RefZero, S: TensorStructure + ScalarStructure + Clone> RealOrComplexTensor<T, S> {
    pub fn to_complex(&mut self) {
        if self.is_real() {
            let old = std::mem::replace(
                self,
                RealOrComplexTensor::Real(DataTensor::Sparse(SparseTensor::empty(
                    S::scalar_structure(),
                ))),
            );

            if let RealOrComplexTensor::Real(r) = old {
                *self = RealOrComplexTensor::Complex(r.map_data(|a| Complex::new_re(a)));
            }
        }
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum RealOrComplexRef<'a, T> {
    Real(&'a T),
    Complex(&'a Complex<T>),
}

#[derive(Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum RealOrComplexMut<'a, T> {
    Real(&'a mut T),
    Complex(&'a mut Complex<T>),
}

impl<T: Clone, S: TensorStructure> SetTensorData for RealOrComplexTensor<T, S> {
    type SetData = RealOrComplex<T>;

    fn set(
        &mut self,
        indices: &[crate::structure::concrete_index::ConcreteIndex],
        value: Self::SetData,
    ) -> Result<()> {
        match self {
            RealOrComplexTensor::Real(d) => d.set(
                indices,
                value.try_into_real().map_err(|r| anyhow!(r.to_string()))?,
            )?,
            RealOrComplexTensor::Complex(d) => d.set(
                indices,
                value
                    .try_into_complex()
                    .map_err(|r| anyhow!(r.to_string()))?,
            )?,
        }
        Ok(())
    }

    fn set_flat(&mut self, index: FlatIndex, value: Self::SetData) -> Result<()> {
        match self {
            RealOrComplexTensor::Real(d) => d.set_flat(
                index,
                value.try_into_real().map_err(|r| anyhow!(r.to_string()))?,
            )?,
            RealOrComplexTensor::Complex(d) => d.set_flat(
                index,
                value
                    .try_into_complex()
                    .map_err(|r| anyhow!(r.to_string()))?,
            )?,
        }
        Ok(())
    }
}

impl<T: Clone, S: TensorStructure> GetTensorData for RealOrComplexTensor<T, S> {
    type GetDataRef<'a>
        = RealOrComplexRef<'a, T>
    where
        Self: 'a;

    type GetDataRefMut<'a>
        = RealOrComplexMut<'a, T>
    where
        Self: 'a;

    type GetDataOwned = RealOrComplex<T>;

    fn get_ref<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataRef<'_>> {
        match self {
            RealOrComplexTensor::Real(d) => Ok(RealOrComplexRef::Real(d.get_ref(indices)?)),
            RealOrComplexTensor::Complex(d) => Ok(RealOrComplexRef::Complex(d.get_ref(indices)?)),
        }
    }

    fn get_ref_linear(&self, index: FlatIndex) -> Option<Self::GetDataRef<'_>> {
        match self {
            RealOrComplexTensor::Real(d) => d.get_ref_linear(index).map(RealOrComplexRef::Real),
            RealOrComplexTensor::Complex(d) => {
                d.get_ref_linear(index).map(RealOrComplexRef::Complex)
            }
        }
    }

    fn get_mut_linear(&mut self, index: FlatIndex) -> Option<Self::GetDataRefMut<'_>> {
        match self {
            RealOrComplexTensor::Real(d) => d.get_mut_linear(index).map(RealOrComplexMut::Real),
            RealOrComplexTensor::Complex(d) => {
                d.get_mut_linear(index).map(RealOrComplexMut::Complex)
            }
        }
    }

    fn get_owned<C: AsRef<[ConcreteIndex]>>(&self, indices: C) -> Result<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        match self {
            RealOrComplexTensor::Real(d) => Ok(RealOrComplex::Real(d.get_owned(indices)?)),
            RealOrComplexTensor::Complex(d) => Ok(RealOrComplex::Complex(d.get_owned(indices)?)),
        }
    }

    fn get_owned_linear(&self, index: FlatIndex) -> Option<Self::GetDataOwned>
    where
        Self::GetDataOwned: Clone,
    {
        match self {
            RealOrComplexTensor::Real(d) => Some(RealOrComplex::Real(d.get_owned_linear(index)?)),
            RealOrComplexTensor::Complex(d) => {
                Some(RealOrComplex::Complex(d.get_owned_linear(index)?))
            }
        }
    }
}

impl<T: Clone, S: TensorStructure + Clone> HasTensorData for RealOrComplexTensor<T, S> {
    type Data = RealOrComplex<T>;

    fn data(&self) -> Vec<Self::Data> {
        match self {
            RealOrComplexTensor::Real(d) => d.data().into_iter().map(RealOrComplex::Real).collect(),
            RealOrComplexTensor::Complex(d) => {
                d.data().into_iter().map(RealOrComplex::Complex).collect()
            }
        }
    }

    fn hashmap(&self) -> indexmap::IndexMap<ExpandedIndex, Self::Data> {
        match self {
            RealOrComplexTensor::Real(d) => d
                .hashmap()
                .into_iter()
                .map(|(k, v)| (k, RealOrComplex::Real(v)))
                .collect(),
            RealOrComplexTensor::Complex(d) => d
                .hashmap()
                .into_iter()
                .map(|(k, v)| (k, RealOrComplex::Complex(v)))
                .collect(),
        }
    }

    fn indices(&self) -> Vec<ExpandedIndex> {
        match self {
            RealOrComplexTensor::Real(d) => d.indices(),
            RealOrComplexTensor::Complex(d) => d.indices(),
        }
    }

    #[cfg(feature = "shadowing")]
    fn symhashmap(
        &self,
        name: symbolica::atom::Symbol,
        args: &[Atom],
    ) -> std::collections::HashMap<Atom, Self::Data> {
        match self {
            RealOrComplexTensor::Real(d) => d
                .symhashmap(name, args)
                .into_iter()
                .map(|(k, v)| (k, RealOrComplex::Real(v)))
                .collect(),
            RealOrComplexTensor::Complex(d) => d
                .symhashmap(name, args)
                .into_iter()
                .map(|(k, v)| (k, RealOrComplex::Complex(v)))
                .collect(),
        }
    }
}

// use crate::data::StorageTensor;

impl<T: Display, S: TensorStructure> Display for RealOrComplexTensor<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealOrComplexTensor::Real(d) => d.fmt(f),
            RealOrComplexTensor::Complex(d) => d.fmt(f),
        }
    }
}

impl<T: Clone, S: TensorStructure, O: From<S> + TensorStructure>
    CastStructure<RealOrComplexTensor<T, O>> for RealOrComplexTensor<T, S>
{
    fn cast_structure(self) -> RealOrComplexTensor<T, O> {
        match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.cast_structure()),
            RealOrComplexTensor::Complex(d) => RealOrComplexTensor::Complex(d.cast_structure()),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Clone, S: TensorStructure> Shadowable for RealOrComplexTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
{
    fn shadow<C>(
        &self,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> C,
    ) -> Result<DenseTensor<Atom, Self::Structure>>
    where
        C: TensorCoefficient,
    {
        match self {
            RealOrComplexTensor::Real(r) => r.shadow(index_to_atom),
            RealOrComplexTensor::Complex(r) => Ok(r
                .structure()
                .clone()
                .to_dense_labeled_complex(index_to_atom)?),
        }
        // Some(self.structure().clone().to_dense_labeled(index_to_atom))
    }
}

impl<T: Default + Clone + PartialEq, S: TensorStructure + Clone> SparseOrDense
    for RealOrComplexTensor<T, S>
{
    fn to_sparse(self) -> Self {
        match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.to_sparse()),
            RealOrComplexTensor::Complex(d) => RealOrComplexTensor::Complex(d.to_sparse()),
        }
    }

    fn to_dense(self) -> Self {
        match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.to_dense()),
            RealOrComplexTensor::Complex(d) => RealOrComplexTensor::Complex(d.to_dense()),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Clone + RefZero, S: TensorStructure, R> ShadowMapping<R> for RealOrComplexTensor<T, S>
where
    S: HasName + Clone,
    S::Name: IntoSymbol,
    S::Args: IntoArgs,
    R: From<T>,
{
    fn append_map<C>(
        &self,
        fn_map: &mut FunctionMap<R>,
        index_to_atom: impl Fn(&Self::Structure, FlatIndex) -> C,
    ) where
        C: TensorCoefficient,
    {
        match self {
            RealOrComplexTensor::Real(c) => c.append_map(fn_map, index_to_atom),
            RealOrComplexTensor::Complex(p) => match p {
                DataTensor::Dense(d) => {
                    for (i, c) in d.flat_iter() {
                        let labeled_coef_re =
                            index_to_atom(self.structure(), i).to_atom_re().unwrap();
                        let labeled_coef_im =
                            index_to_atom(self.structure(), i).to_atom_im().unwrap();
                        fn_map.add_constant(labeled_coef_re.clone(), c.re.clone().into());
                        fn_map.add_constant(labeled_coef_im.clone(), c.re.clone().into());
                    }
                }
                DataTensor::Sparse(d) => {
                    for (i, c) in d.flat_iter() {
                        let labeled_coef_re =
                            index_to_atom(self.structure(), i).to_atom_re().unwrap();
                        let labeled_coef_im =
                            index_to_atom(self.structure(), i).to_atom_im().unwrap();
                        fn_map.add_constant(labeled_coef_re.clone(), c.re.clone().into());
                        fn_map.add_constant(labeled_coef_im.clone(), c.re.clone().into());
                    }
                }
            }, // p.append_map(fn_map, index_to_atom),
        }
    }
}

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum RealOrComplex<T> {
    Real(T),
    Complex(Complex<T>),
}

impl<T> Ref for RealOrComplex<T> {
    type Ref<'a>
        = RealOrComplexRef<'a, T>
    where
        Self: 'a;
    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        match self {
            RealOrComplex::Real(r) => RealOrComplexRef::Real(r),
            RealOrComplex::Complex(c) => RealOrComplexRef::Complex(c),
        }
    }
}

impl<T: Default> RealOrComplex<T> {
    pub fn to_complex(self) -> Complex<T> {
        match self {
            RealOrComplex::Real(r) => Complex::new(r, T::default()),
            RealOrComplex::Complex(c) => c,
        }
    }
}

impl<T: RefZero> RealOrComplex<T> {
    pub fn zero(&self) -> Self {
        match self {
            RealOrComplex::Real(r) => RealOrComplex::Real(r.ref_zero()),
            RealOrComplex::Complex(c) => RealOrComplex::Complex(c.ref_zero()),
        }
    }

    pub fn to_complex_mut(&mut self) {
        if self.is_real() {
            let old = std::mem::replace(self, self.zero());

            if let RealOrComplex::Real(re) = old {
                *self = RealOrComplex::Complex(Complex::new_re(re));
            }
        }
    }
}

impl<T: Display> Display for RealOrComplex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealOrComplex::Complex(c) => c.fmt(f),
            RealOrComplex::Real(r) => r.fmt(f),
        }
    }
}

impl<T: Clone, S> TensorStructure for RealOrComplexTensor<T, S>
where
    S: TensorStructure,
{
    // type R = <T::Structure as TensorStructure>::R;
    type Indexed = RealOrComplexTensor<T, S::Indexed>;
    type Slot = S::Slot;

    fn reindex(self, indices: &[AbstractIndex]) -> Result<Self::Indexed, StructureError> {
        self.map_structure_result(|s| s.reindex(indices))
    }

    fn dual(self) -> Self {
        self.map_same_structure(|s| s.dual())
    }

    delegate! {
        to self.structure() {
            fn external_reps_iter(&self)-> impl Iterator<Item = Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn external_indices_iter(&self)-> impl Iterator<Item = AbstractIndex>;
            fn external_dims_iter(&self)-> impl Iterator<Item = Dimension>;
            fn external_structure_iter(&self)-> impl Iterator<Item = Self::Slot>;
            fn get_slot(&self, i: usize)-> Option<Self::Slot>;
            fn get_rep(&self, i: usize)-> Option<Representation<<Self::Slot as IsAbstractSlot>::R>>;
            fn get_dim(&self, i: usize)-> Option<Dimension>;
            fn get_aind(&self, i: usize)-> Option<AbstractIndex>;
            fn order(&self)-> usize;
        }
    }
}

impl<T: Clone, S: TensorStructure> HasStructure for RealOrComplexTensor<T, S> {
    type Scalar = RealOrComplex<T>;
    type ScalarRef<'a>
        = RealOrComplexRef<'a, T>
    where
        Self: 'a;
    type Structure = S;
    type Store<U>
        = RealOrComplexTensor<T, U>
    where
        U: TensorStructure;

    fn map_structure<S2: TensorStructure>(self, f: impl Fn(S) -> S2) -> RealOrComplexTensor<T, S2> {
        match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.map_structure(f)),
            RealOrComplexTensor::Complex(d) => RealOrComplexTensor::Complex(d.map_structure(f)),
        }
    }

    fn map_structure_result<S2: TensorStructure, E>(
        self,
        f: impl Fn(S) -> Result<S2, E>,
    ) -> Result<RealOrComplexTensor<T, S2>, E> {
        Ok(match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.map_structure_result(f)?),
            RealOrComplexTensor::Complex(d) => {
                RealOrComplexTensor::Complex(d.map_structure_result(f)?)
            }
        })
    }

    fn structure(&self) -> &Self::Structure {
        match self {
            RealOrComplexTensor::Real(r) => r.structure(),
            RealOrComplexTensor::Complex(r) => r.structure(),
        }
    }

    fn map_same_structure(self, f: impl FnOnce(Self::Structure) -> Self::Structure) -> Self {
        match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.map_same_structure(f)),
            RealOrComplexTensor::Complex(d) => {
                RealOrComplexTensor::Complex(d.map_same_structure(f))
            }
        }
    }

    fn mut_structure(&mut self) -> &mut Self::Structure {
        match self {
            RealOrComplexTensor::Real(r) => r.mut_structure(),
            RealOrComplexTensor::Complex(r) => r.mut_structure(),
        }
    }

    fn scalar(self) -> Option<Self::Scalar> {
        match self {
            RealOrComplexTensor::Real(r) => r.scalar().map(|x| RealOrComplex::Real(x)),
            RealOrComplexTensor::Complex(r) => r.scalar().map(|x| RealOrComplex::Complex(x)),
        }
    }

    fn scalar_ref(&self) -> Option<Self::ScalarRef<'_>> {
        match self {
            RealOrComplexTensor::Real(r) => r.scalar_ref().map(|x| RealOrComplexRef::Real(x)),
            RealOrComplexTensor::Complex(r) => r.scalar_ref().map(|x| RealOrComplexRef::Complex(x)),
        }
    }
}

impl<T: Clone, S: TensorStructure + ScalarStructure> ScalarTensor for RealOrComplexTensor<T, S> {
    fn new_scalar(scalar: Self::Scalar) -> Self {
        match scalar {
            RealOrComplex::Real(r) => RealOrComplexTensor::Real(DataTensor::new_scalar(r)),
            RealOrComplex::Complex(r) => RealOrComplexTensor::Complex(DataTensor::new_scalar(r)),
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

    fn args(&self) -> Option<S::Args> {
        match self {
            RealOrComplexTensor::Real(r) => r.args(),
            RealOrComplexTensor::Complex(r) => r.args(),
        }
    }
}

impl<T: Clone, S: TensorStructure> IteratableTensor for RealOrComplexTensor<T, S> {
    type Data<'a>
        = RealOrComplexRef<'a, T>
    where
        Self: 'a;

    fn iter_expanded(&self) -> impl Iterator<Item = (ExpandedIndex, Self::Data<'_>)> {
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

    fn iter_flat(&self) -> impl Iterator<Item = (FlatIndex, Self::Data<'_>)> {
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

impl<S, T> Trace for RealOrComplexTensor<T, S>
where
    S: TensorStructure + Clone + StructureContract,
    T: ContractableWith<T, Out = T>
        + Clone
        + FallibleMul<Output = T>
        + FallibleAddAssign<T>
        + FallibleSubAssign<T>
        + RefZero
        + IsZero,
    Complex<T>: ContractableWith<Complex<T>, Out = Complex<T>>
        + Clone
        + FallibleMul<Output = Complex<T>>
        + FallibleAddAssign<Complex<T>>
        + FallibleSubAssign<Complex<T>>
        + RefZero
        + IsZero,
{
    fn internal_contract(&self) -> Self {
        match self {
            RealOrComplexTensor::Real(x) => RealOrComplexTensor::Real(x.internal_contract()),
            RealOrComplexTensor::Complex(x) => RealOrComplexTensor::Complex(x.internal_contract()),
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

use std::{
    fmt::{Debug, Display, LowerExp},
    ops::{Add, Neg, Sub},
};

use crate::{
    iterators::IteratorEnum,
    network::Ref,
    structure::concrete_index::ConcreteIndex,
    tensors::data::{SparseTensor, StorageTensor},
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
        float::{Complex as SymComplex, ConstructibleFloat, NumericalFloatLike, Real},
        rational::Rational,
    },
    evaluate::FunctionMap,
};

use crate::structure::abstract_index::AbstractIndex;
use crate::structure::dimension::Dimension;
use crate::structure::representation::Representation;
use crate::structure::slot::IsAbstractSlot;
use crate::structure::StructureError;
use delegate::delegate;

#[cfg(feature = "shadowing")]
use crate::{
    shadowing::symbolica_utils::{IntoArgs, IntoSymbol},
    shadowing::{ShadowMapping, Shadowable},
    structure::ToSymbolic,
    tensors::data::{DataIterator, DenseTensor},
    tensors::parametric::TensorCoefficient,
};

use crate::{
    algebra::algebraic_traits::{IsZero, RefOne, RefZero},
    algebra::upgrading_arithmetic::{FallibleAddAssign, FallibleMul, FallibleSubAssign},
    contraction::{Contract, ContractableWith, ContractionError, Trace},
    iterators::IteratableTensor,
    structure::{
        concrete_index::{ExpandedIndex, FlatIndex},
        CastStructure, HasName, HasStructure, ScalarStructure, ScalarTensor, StructureContract,
        TensorStructure, TracksCount,
    },
    tensors::data::{DataTensor, GetTensorData, HasTensorData, SetTensorData, SparseOrDense},
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
#[derive(
    Copy,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    Hash,
    Eq,
    PartialOrd,
    Ord,
    bincode_trait_derive::Encode,
    bincode_trait_derive::Decode,
)]
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

impl<T: Clone> From<&Complex<T>> for Complex<T> {
    fn from(value: &Complex<T>) -> Self {
        value.clone()
    }
}

impl<T> Ref for Complex<T> {
    type Ref<'a>
        = &'a Complex<T>
    where
        Self: 'a;
    fn refer<'a>(&'a self) -> Self::Ref<'a> {
        self
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

#[derive(Clone, Debug, EnumTryAsInner)]
#[derive_err(Debug)]
pub enum RealOrComplex<T> {
    Real(T),
    Complex(Complex<T>),
}

impl<T: RefZero> From<RealOrComplex<T>> for Complex<T> {
    fn from(value: RealOrComplex<T>) -> Self {
        match value {
            RealOrComplex::Real(r) => {
                let z = r.ref_zero();
                Complex::new(r, z)
            }
            RealOrComplex::Complex(c) => c,
        }
    }
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

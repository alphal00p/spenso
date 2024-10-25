use std::{
    fmt::{Debug, Display, LowerExp},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
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

#[cfg(feature = "shadowing")]
use crate::{
    data::{DataIterator, DenseTensor},
    parametric::TensorCoefficient,
    shadowing::{ShadowMapping, Shadowable},
    structure::ToSymbolic,
    symbolica_utils::{IntoArgs, IntoSymbol},
};

#[cfg(feature = "shadowing")]
use rand::Rng;

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

#[cfg(feature = "shadowing")]
impl<T: SingleFloat> SingleFloat for Complex<T>
where
    T: for<'a> RefMul<&'a T, Output = T>
        + for<'a> RefAdd<&'a T, Output = T>
        + for<'a> RefSub<&'a T, Output = T>
        + for<'a> RefDiv<&'a T, Output = T>,
{
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn is_one(&self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }

    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    fn from_rational(&self, rat: &Rational) -> Self {
        Complex {
            re: self.re.from_rational(rat),
            im: self.im.zero(),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: NumericalFloatLike> NumericalFloatLike for Complex<T>
where
    T: for<'a> RefMul<&'a T, Output = T>
        + for<'a> RefAdd<&'a T, Output = T>
        + for<'a> RefSub<&'a T, Output = T>
        + for<'a> RefDiv<&'a T, Output = T>,
{
    #[inline]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b.clone()
    }

    #[inline]
    fn neg(&self) -> Self {
        Complex {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }

    #[inline]
    fn zero(&self) -> Self {
        Complex {
            re: self.re.zero(),
            im: self.im.zero(),
        }
    }

    fn new_zero() -> Self {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

    fn one(&self) -> Self {
        Complex {
            re: self.re.one(),
            im: self.im.zero(),
        }
    }

    fn pow(&self, e: u64) -> Self {
        // TODO: use binary exponentiation
        let mut r = self.one();
        for _ in 0..e {
            r *= self;
        }
        r
    }

    fn inv(&self) -> Self {
        self.inv()
    }

    fn from_usize(&self, a: usize) -> Self {
        Complex {
            re: self.re.from_usize(a),
            im: self.im.zero(),
        }
    }

    fn from_i64(&self, a: i64) -> Self {
        Complex {
            re: self.re.from_i64(a),
            im: self.im.zero(),
        }
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        self.re.get_precision().min(self.im.get_precision())
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        (2.0f64).powi(-(self.get_precision() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.re.fixed_precision() || self.im.fixed_precision()
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        Complex {
            re: self.re.sample_unit(rng),
            im: self.im.zero(),
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Real> Real for Complex<T>
where
    T: for<'a> RefMul<&'a T, Output = T>
        + for<'a> RefAdd<&'a T, Output = T>
        + for<'a> RefSub<&'a T, Output = T>
        + for<'a> RefDiv<&'a T, Output = T>
        + RefOne
        + RefZero,
{
    fn e(&self) -> Self {
        Complex {
            re: self.re.e(),
            im: self.im.zero(),
        }
    }

    fn pi(&self) -> Self {
        Complex {
            re: self.re.pi(),
            im: self.im.zero(),
        }
    }

    fn phi(&self) -> Self {
        Complex {
            re: self.re.phi(),
            im: self.im.zero(),
        }
    }

    fn euler(&self) -> Self {
        Complex {
            re: self.re.euler(),
            im: self.im.zero(),
        }
    }

    #[inline]
    fn norm(&self) -> Self {
        Complex::new(self.norm_squared().sqrt(), self.im.zero())
    }

    #[inline]
    fn sqrt(&self) -> Self {
        let (r, phi) = (self.norm_squared().sqrt(), self.im.atan2(&self.re));
        let phi = phi.ref_div(&self.re.from_usize(2));
        let r = r.sqrt();
        Complex::new(r.ref_mul(&phi.cos()), r.ref_mul(&phi.sin()))
    }

    #[inline]
    fn log(&self) -> Self {
        Complex::new(self.norm_squared().sqrt().log(), self.im.atan2(&self.re))
    }

    #[inline]
    fn exp(&self) -> Self {
        let r = self.re.exp();
        Complex::new(r.clone() * self.im.cos(), r * self.im.sin())
    }

    #[inline]
    fn sin(&self) -> Self {
        Complex::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    #[inline]
    fn cos(&self) -> Self {
        Complex::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    #[inline]
    fn tan(&self) -> Self {
        let (r, i) = (self.re.clone() + &self.re, self.im.clone() + &self.im);
        let m = r.cos() + i.cosh();
        Self::new(r.sin() / &m, i.sinh() / m)
    }

    #[inline]
    fn asin(&self) -> Self {
        let i = self.ref_i();
        -i.clone() * ((self.one() - self.clone() * self).sqrt() + i * self).log()
    }

    #[inline]
    fn acos(&self) -> Self {
        let i = self.ref_i();
        -i.clone() * (i * (self.one() - self.clone() * self).sqrt() + self).log()
    }

    #[inline]
    fn atan2(&self, x: &Self) -> Self {
        // TODO: pick proper branch
        let r = self.clone() / x;
        let i = self.ref_i();
        let one = self.one();
        let two = one.clone() + &one;
        // TODO: add edge cases
        ((&one + &i * &r).log() - (&one - &i * r).log()) / (two * i)
    }

    #[inline]
    fn sinh(&self) -> Self {
        Complex::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }

    #[inline]
    fn cosh(&self) -> Self {
        Complex::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }

    #[inline]
    fn tanh(&self) -> Self {
        let (two_re, two_im) = (self.re.clone() + &self.re, self.im.clone() + &self.im);
        let m = two_re.cosh() + two_im.cos();
        Self::new(two_re.sinh() / &m, two_im.sin() / m)
    }

    #[inline]
    fn asinh(&self) -> Self {
        let one = self.one();
        (self.clone() + (one + self.clone() * self).sqrt()).log()
    }

    #[inline]
    fn acosh(&self) -> Self {
        let one = self.one();
        let two = one.clone() + &one;
        &two * (((self.clone() + &one) / &two).sqrt() + ((self.clone() - one) / &two).sqrt()).log()
    }

    #[inline]
    fn atanh(&self) -> Self {
        let one = self.one();
        let two = one.clone() + &one;
        // TODO: add edge cases
        ((&one + self).log() - (one - self).log()) / two
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        if e.re == self.re.zero() && e.im == self.im.zero() {
            self.one()
        } else if e.im == self.im.zero() {
            let (r, phi) = (self.norm_squared().sqrt(), self.im.atan2(&self.re));
            let r = r.powf(&e.re);
            let phi = phi * &e.re;
            Complex::new(r.ref_mul(&phi.cos()), r.ref_mul(&phi.sin()))
        } else {
            (e * self.log()).exp()
        }
    }
}

#[cfg(feature = "shadowing")]
impl<T> CompiledEvaluatorFloat for Complex<T>
where
    f64: From<T>,
    T: From<f64> + Copy,
{
    #[allow(clippy::useless_conversion)]
    fn evaluate(
        eval: &mut symbolica::evaluate::CompiledEvaluator,
        args: &[Self],
        out: &mut [Self],
    ) {
        let args: Vec<SymComplex<f64>> = args
            .iter()
            .map(|&c| SymComplex {
                re: c.re.into(),
                im: c.im.into(),
            })
            .collect();
        let mut castout: Vec<SymComplex<f64>> = out
            .iter()
            .map(|&c| SymComplex {
                re: c.re.into(),
                im: c.im.into(),
            })
            .collect();

        eval.evaluate_complex(&args, &mut castout);

        for (o, &c) in out.iter_mut().zip(castout.iter()) {
            *o = Complex {
                re: c.re.into(),
                im: c.im.into(),
            };
        }
    }
}

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

impl<'a, 'b, T> Add<&'a Complex<T>> for &'b Complex<T>
where
    T: Add<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl<'a, 'b, T> Add<&'a T> for &'b Complex<T>
where
    T: for<'c> RefAdd<&'c T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a T) -> Self::Output {
        Complex {
            re: self.re.ref_add(rhs),
            im: self.im.clone(),
        }
    }
}

impl<'a, T> Add<&'a Complex<T>> for Complex<T>
where
    T: Add<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a Complex<T>) -> Self::Output {
        self + rhs.clone()
    }
}

impl<'a, T> Add<&'a T> for Complex<T>
where
    T: for<'c> RefAdd<&'c T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a T) -> Self::Output {
        Complex {
            re: self.re.ref_add(rhs),
            im: self.im.clone(),
        }
    }
}

impl<'a, T> Add<Complex<T>> for &'a Complex<T>
where
    T: Add<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'b, T> Add<T> for &'b Complex<T>
where
    T: RefAdd<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Complex {
            re: self.re.ref_add(rhs),
            im: self.im.clone(),
        }
    }
}

impl<T> Add for Complex<T>
where
    T: Add<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T> Add<T> for Complex<T>
where
    T: Add<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Complex {
            re: self.re.add(rhs),
            im: self.im,
        }
    }
}

impl<T> AddAssign for Complex<T>
where
    for<'a> T: AddAssign<&'a T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl<T> AddAssign<T> for Complex<T>
where
    T: AddAssign<T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.re += rhs;
    }
}

impl<T> AddAssign<&Complex<T>> for Complex<T>
where
    for<'a> T: AddAssign<&'a T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.re += &rhs.re;
        self.im += &rhs.im;
    }
}

impl<T> AddAssign<&T> for Complex<T>
where
    T: for<'a> RefAdd<&'a T, Output = T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &T) {
        self.re = self.re.ref_add(rhs);
    }
}

impl<'a, 'b, T> Sub<&'a Complex<T>> for &'b Complex<T>
where
    T: Sub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl<'a, 'b, T> Sub<&'a T> for &'b Complex<T>
where
    T: for<'c> RefSub<&'c T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a T) -> Self::Output {
        Complex {
            re: self.re.ref_sub(rhs),
            im: self.im.clone(),
        }
    }
}

impl<'a, T> Sub<&'a Complex<T>> for Complex<T>
where
    T: Sub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a Complex<T>) -> Self::Output {
        self - rhs.clone()
    }
}

impl<'a, T> Sub<&'a T> for Complex<T>
where
    T: for<'c> RefSub<&'c T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a T) -> Self::Output {
        Complex {
            re: self.re.ref_sub(rhs),
            im: self.im,
        }
    }
}

impl<'a, T> Sub<Complex<T>> for &'a Complex<T>
where
    T: Sub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'b, T> Sub<T> for &'b Complex<T>
where
    T: RefSub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Complex {
            re: self.re.ref_sub(rhs),
            im: self.im.clone(),
        }
    }
}

impl<T> Sub for Complex<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T> Sub<T> for Complex<T>
where
    T: Sub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Complex {
            re: self.re - rhs,
            im: self.im,
        }
    }
}

impl<T> SubAssign for Complex<T>
where
    for<'a> T: SubAssign<&'a T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<T> SubAssign<T> for Complex<T>
where
    T: SubAssign<T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.re -= rhs;
    }
}

impl<T> SubAssign<&Complex<T>> for Complex<T>
where
    for<'a> T: SubAssign<&'a T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.re -= &rhs.re;
        self.im -= &rhs.im;
    }
}

impl<T> SubAssign<&T> for Complex<T>
where
    T: for<'a> RefSub<&'a T, Output = T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &T) {
        self.re = self.re.ref_sub(rhs);
    }
}

impl<'a, 'b, T> Mul<&'a Complex<T>> for &'b Complex<T>
where
    T: for<'c> RefMul<&'c T, Output = T>
        + for<'c> RefAdd<&'c T, Output = T>
        + for<'c> RefSub<&'c T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        Complex::new(
            self.re.ref_mul(&rhs.re).ref_sub(&self.im.ref_mul(&rhs.im)),
            self.re.ref_mul(&rhs.im).ref_add(&self.im.ref_mul(&rhs.re)),
        )
    }
}

impl<'a, 'b, T> Mul<&'a T> for &'b Complex<T>
where
    T: for<'c> RefMul<&'c T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a T) -> Self::Output {
        Complex::new(self.re.ref_mul(rhs), self.im.ref_mul(rhs))
    }
}

impl<'a, T> Mul<&'a Complex<T>> for Complex<T>
where
    for<'b> T: Mul<&'b T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        Complex::new(
            self.re.clone() * &rhs.re - self.im.clone() * &rhs.im,
            self.re.clone() * &rhs.im + self.im.clone() * &rhs.re,
        )
    }
}

impl<'a, T> Mul<&'a T> for Complex<T>
where
    T: for<'c> RefMul<&'c T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a T) -> Self::Output {
        Complex::new(self.re.ref_mul(rhs), self.im.ref_mul(rhs))
    }
}

impl<'a, T> Mul<Complex<T>> for &'a Complex<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'b, T> Mul<T> for &'b Complex<T>
where
    T: RefMul<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Complex::new(self.re.ref_mul(rhs.clone()), self.im.ref_mul(rhs))
    }
}

impl<T> Mul for Complex<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Clone,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Complex::new(
            self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone(),
            self.re.clone() * rhs.im.clone() + self.im.clone() * rhs.re.clone(),
        )
    }
}

impl<T> Mul<T> for Complex<T>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Complex::new(self.re * rhs.clone(), self.im * rhs)
    }
}

impl<T> MulAssign for Complex<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Clone,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T> MulAssign<T> for Complex<T>
where
    T: MulAssign<T> + Clone,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.re *= rhs.clone();
        self.im *= rhs;
    }
}

impl<T> MulAssign<&Complex<T>> for Complex<T>
where
    T: for<'c> RefMul<&'c T, Output = T>
        + for<'c> RefAdd<&'c T, Output = T>
        + for<'c> RefSub<&'c T, Output = T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = &*self * rhs;
    }
}

impl<T> MulAssign<&T> for Complex<T>
where
    T: for<'c> RefMul<&'c T, Output = T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        self.re = self.re.ref_mul(rhs);
        self.im = self.im.ref_mul(rhs);
    }
}

impl<'a, 'b, T> Div<&'a Complex<T>> for &'b Complex<T>
where
    T: Clone
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + for<'r> RefMul<&'r T, Output = T>
        + Add<T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        let n = rhs.norm_squared();
        let re = self.re.ref_mul(&rhs.re) + self.im.ref_mul(&rhs.im);
        let im = self.im.ref_mul(&rhs.re) - self.re.ref_mul(&rhs.im);
        Complex::new(re / n.clone(), im / n)
    }
}

impl<'a, 'b, T> Div<&'a T> for &'b Complex<T>
where
    T: for<'c> RefDiv<&'c T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a T) -> Self::Output {
        Complex::new(self.re.ref_div(rhs), self.im.ref_div(rhs))
    }
}

impl<'a, T> Div<&'a Complex<T>> for Complex<T>
where
    T: Clone
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + for<'b> RefMul<&'b T, Output = T>
        + Add<T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        &self / rhs
    }
}

impl<'a, T> Div<&'a T> for Complex<T>
where
    T: for<'c> Div<&'c T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a T) -> Self::Output {
        Complex::new(self.re.div(rhs), self.im.div(rhs))
    }
}

impl<'a, T> Div<Complex<T>> for &'a Complex<T>
where
    T: Clone
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + for<'b> RefMul<&'b T, Output = T>
        + Add<T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        self / &rhs
    }
}

impl<'b, T> Div<T> for &'b Complex<T>
where
    T: RefDiv<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Complex::new(self.re.ref_div(rhs.clone()), self.im.ref_div(rhs))
    }
}

impl<T> Div for Complex<T>
where
    T: Clone
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + for<'a> RefMul<&'a T, Output = T>
        + Add<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<T> Div<T> for Complex<T>
where
    T: Div<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Complex::new(self.re.div(rhs.clone()), self.im.div(rhs))
    }
}

impl<T> DivAssign for Complex<T>
where
    T: Clone
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + for<'a> RefMul<&'a T, Output = T>
        + Add<T, Output = T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.div_assign(&rhs)
    }
}

impl<T> DivAssign<T> for Complex<T>
where
    T: DivAssign<T> + Clone,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.re /= rhs.clone();
        self.im /= rhs;
    }
}

impl<T> DivAssign<&Complex<T>> for Complex<T>
where
    T: Clone
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + for<'a> RefMul<&'a T, Output = T>
        + Add<T, Output = T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &Self) {
        *self = &*self / rhs;
    }
}

impl<T> DivAssign<&T> for Complex<T>
where
    T: for<'a> RefDiv<&'a T, Output = T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &T) {
        self.re = self.re.ref_div(rhs);
        self.im = self.im.ref_div(rhs);
    }
}

impl<T> Neg for Complex<T>
where
    T: Neg<Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn neg(self) -> Complex<T> {
        Complex::new(-self.re, -self.im)
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
    type GetDataRef<'a> = RealOrComplexRef<'a, T>
    where
        Self: 'a;

    type GetDataRefMut<'a> = RealOrComplexMut<'a, T>
    where
        Self: 'a;

    type GetDataOwned = RealOrComplex<T>;

    fn get_ref<'a>(
        &'a self,
        indices: &[crate::structure::concrete_index::ConcreteIndex],
    ) -> Result<Self::GetDataRef<'a>> {
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

    fn get_owned(
        &self,
        indices: &[crate::structure::concrete_index::ConcreteIndex],
    ) -> Result<Self::GetDataOwned>
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

impl<T, S: TensorStructure> RealOrComplexTensor<T, S> {
    pub fn map_structure<S2: TensorStructure>(
        self,
        f: impl Fn(S) -> S2,
    ) -> RealOrComplexTensor<T, S2> {
        match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.map_structure(f)),
            RealOrComplexTensor::Complex(d) => RealOrComplexTensor::Complex(d.map_structure(f)),
        }
    }

    pub fn map_structure_fallible<S2: TensorStructure, E>(
        self,
        f: impl Fn(S) -> Result<S2, E>,
    ) -> Result<RealOrComplexTensor<T, S2>, E> {
        Ok(match self {
            RealOrComplexTensor::Real(d) => RealOrComplexTensor::Real(d.map_structure_fallible(f)?),
            RealOrComplexTensor::Complex(d) => {
                RealOrComplexTensor::Complex(d.map_structure_fallible(f)?)
            }
        })
    }
}

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
    fn append_map<'a, C>(
        &'a self,
        fn_map: &mut FunctionMap<'a, R>,
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

impl<T: Display> Display for RealOrComplex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealOrComplex::Complex(c) => c.fmt(f),
            RealOrComplex::Real(r) => r.fmt(f),
        }
    }
}

impl<T: Clone, S: TensorStructure> HasStructure for RealOrComplexTensor<T, S> {
    type Scalar = RealOrComplex<T>;
    type Structure = S;
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
    type Data<'a>=  RealOrComplexRef<'a,T>
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

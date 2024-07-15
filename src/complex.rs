use std::{
    fmt::{Debug, Display, LowerExp},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{RefOne, RefZero};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use duplicate::duplicate;
use ref_ops::{RefAdd, RefDiv, RefMul, RefSub};
use serde::{Deserialize, Serialize};
#[cfg(feature = "shadowing")]
use symbolica::domains::float::Complex as SymComplex;
#[cfg(feature = "shadowing")]
use symbolica::domains::float::ConstructibleFloat;
#[cfg(feature = "shadowing")]
use symbolica::domains::{float::NumericalFloatLike, float::Real, rational::Rational};

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
            re: Rational::from_f64(re),
            im: Rational::zero(),
        }
    }
}

#[cfg(feature = "shadowing")]
impl From<Complex<f64>> for Complex<Rational> {
    fn from(value: Complex<f64>) -> Self {
        Complex {
            re: Rational::from_f64(value.re),
            im: Rational::from_f64(value.im),
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

    pub fn zero() -> Complex<T>
    where
        T: num::Zero,
    {
        Complex {
            re: T::zero(),
            im: T::zero(),
        }
    }

    pub fn one() -> Complex<T>
    where
        T: num::One + num::Zero,
    {
        Complex {
            re: T::one(),
            im: T::zero(),
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

    pub fn norm(&self) -> T
    where
        T: num::Float + for<'a> RefMul<&'a T, Output = T> + Add<T, Output = T>,
    {
        self.norm_squared().sqrt()
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

    #[inline]
    pub fn arg(&self) -> T
    where
        T: num::Float,
    {
        self.im.atan2(self.re)
    }

    #[inline]
    pub fn to_polar_coordinates(self) -> (T, T)
    where
        T: num::Float,
        T: for<'a> RefMul<&'a T, Output = T> + Add<T, Output = T>,
    {
        (self.norm_squared().sqrt(), self.arg())
    }

    #[inline]
    pub fn from_polar_coordinates(r: T, phi: T) -> Complex<T>
    where
        T: num::Float,
    {
        Complex::new(r * phi.cos(), r * phi.sin())
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

impl<'a, 'b, T> Add<T> for &'b Complex<T>
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

use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ref_ops::{RefAdd, RefMul, RefSub};
use serde::{Deserialize, Serialize};
use symbolica::domains::float::ConstructibleFloat;
#[cfg(feature = "shadowing")]
use symbolica::domains::float::Real;

use crate::{RefOne, RefZero};

#[derive(Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
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
    T: num::Zero,
{
    fn from(re: T) -> Self {
        Complex { re, im: T::zero() }
    }
}

#[cfg(feature = "shadowing")]
impl<T: Real> From<Complex<T>> for symbolica::domains::float::Complex<T> {
    fn from(complex: Complex<T>) -> Self {
        symbolica::domains::float::Complex::new(complex.re, complex.im)
    }
}

#[cfg(feature = "shadowing")]
impl<T: Real> From<symbolica::domains::float::Complex<T>> for Complex<T> {
    fn from(complex: symbolica::domains::float::Complex<T>) -> Self {
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
    #[inline]
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }

    pub fn new_zero() -> Self
    where
        T: ConstructibleFloat,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

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
        T: num::Float,
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
        T: Mul<T, Output = T> + Add<T, Output = T> + Clone,
    {
        (self.re.clone() * self.re.clone()) + (self.im.clone() * self.im.clone())
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
        T: Mul<T, Output = T> + Add<T, Output = T>,
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

impl<T> AddAssign for Complex<T>
where
    for<'a> T: AddAssign<&'a T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
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

impl<T> SubAssign for Complex<T>
where
    for<'a> T: SubAssign<&'a T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
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

impl<'a, T> Mul<&'a Complex<T>> for Complex<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Clone,
{
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        self * rhs.clone()
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

impl<T> MulAssign for Complex<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Clone,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        self.im = self.re.clone() * rhs.im.clone() + self.im.clone() * rhs.re.clone();
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
        self.re = self.re.ref_mul(&rhs.re).ref_sub(&self.im.ref_mul(&rhs.im));
        self.im = self.re.ref_mul(&rhs.im).ref_add(&self.im.ref_mul(&rhs.re));
    }
}

impl<'a, 'b, T> Div<&'a Complex<T>> for &'b Complex<T>
where
    T: Clone + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() / rhs.clone()
    }
}

impl<'a, T> Div<&'a Complex<T>> for Complex<T>
where
    T: Clone + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        self / rhs.clone()
    }
}

impl<'a, T> Div<Complex<T>> for &'a Complex<T>
where
    T: Clone + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>,
{
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T> Div for Complex<T>
where
    T: Clone + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let n = rhs.norm_squared();
        let re = self.re.clone() * rhs.re.clone() + self.im.clone() * rhs.im.clone();
        let im = self.im.clone() * rhs.re.clone() - self.re.clone() * rhs.im.clone();
        Complex::new(re / n.clone(), im / n)
    }
}

impl<T> DivAssign for Complex<T>
where
    for<'a> T: DivAssign<&'a T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.div_assign(&rhs)
    }
}

impl<T> DivAssign<&Complex<T>> for Complex<T>
where
    for<'a> T: DivAssign<&'a T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &Self) {
        self.re /= &rhs.re;
        self.im /= &rhs.im;
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

use std::ops::{Add, Div, Sub};

use ref_ops::{RefDiv, RefMul};

use super::Complex;

impl<'a, T> Div<&'a Complex<T>> for &Complex<T>
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

impl<'a, T> Div<&'a T> for &Complex<T>
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

impl<T> Div<Complex<T>> for &Complex<T>
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

impl<T> Div<T> for &Complex<T>
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

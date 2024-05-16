use std::ops::{Add, Div, Mul, Sub};

use symbolica::atom::Atom;

pub trait FieldOps<Lhs, Rhs>
where
    for<'c, 'd> &'c Lhs: Add<&'d Rhs, Output = Lhs>
        + Mul<&'d Rhs, Output = Lhs>
        + Div<&'d Rhs, Output = Lhs>
        + Sub<&'d Rhs, Output = Lhs>,
{
}

pub trait Scalar: Clone
where
    for<'c, 'd> &'c Self: Add<&'d Self, Output = Self>
        + Mul<&'d Self, Output = Self>
        + Div<&'d Self, Output = Self>
        + Sub<&'d Self, Output = Self>,
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

impl Scalar for Atom {
    fn zero() -> Self {
        Atom::new_num(0)
    }

    fn one() -> Self {
        Atom::new_num(1)
    }
}

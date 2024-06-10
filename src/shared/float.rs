use std::{fmt::Debug, ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub}};

pub trait Float:
    Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + AddAssign
    + Div<Output = Self>
    + Clone
    + Copy
    + PartialOrd
    + Sized
    + Sync
    + Send
    + Debug
{
    //constants
    fn pi() -> Self;
    fn zero() -> Self;
    fn maximum() -> Self;
    fn minimum() -> Self;
    fn small() -> Self;
    fn epsilon() -> Self;
    //conversion functions
    fn usize(u: usize) -> Self;
    fn isize(i: isize) -> Self;
    fn float(f: f32) -> Self;
    fn double(f: f64) -> Self;

    //utility
    fn sin_cos(&self) -> (Self, Self);
    fn exp(&self) -> Self;
    fn recip(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn norm(&self) -> Self;
    fn square_norm(&self) -> Self;

    //shared
    fn greater(&self, other: Self) -> Self {
        if *self > other {
            *self
        } else {
            other
        }
    }

    fn lesser(&self, other: Self) -> Self {
        if *self < other {
            *self
        } else {
            other
        }
    }
}

impl Float for f32 {
    #[inline(always)]
    fn pi() -> Self {
        std::f32::consts::PI
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn maximum() -> Self {
        f32::MAX
    }

    #[inline(always)]
    fn minimum() -> Self {
        f32::MIN
    }

    #[inline(always)]
    fn small() -> Self {
        f32::MIN_POSITIVE
    }

    #[inline(always)]
    fn epsilon() -> Self {
        f32::EPSILON
    }

    #[inline(always)]
    fn sin_cos(&self) -> (Self, Self) {
        (*self as f32).sin_cos()
    }

    #[inline(always)]
    fn usize(u: usize) -> Self {
        u as f32
    }

    #[inline(always)]
    fn isize(i: isize) -> Self {
        i as f32
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        (*self as f32).exp()
    }

    #[inline(always)]
    fn float(f: f32) -> Self {
        f
    }

    #[inline(always)]
    fn double(f: f64) -> Self {
        f as f32
    }

    #[inline(always)]
    fn recip(&self) -> Self {
        (*self as f32).recip()
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        (*self as f32).sqrt()
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        (*self as f32).abs()
    }

    #[inline(always)]
    fn square_norm(&self) -> Self {
        (*self as f32).powi(2)
    }
}
impl Float for f64 {
    #[inline(always)]
    fn pi() -> Self {
        std::f64::consts::PI
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn maximum() -> Self {
        f64::MAX
    }

    #[inline(always)]
    fn minimum() -> Self {
        f64::MIN
    }

    #[inline(always)]
    fn small() -> Self {
        f64::MIN_POSITIVE
    }

    #[inline(always)]
    fn epsilon() -> Self {
        f64::EPSILON
    }

    #[inline(always)]
    fn usize(u: usize) -> Self {
        u as f64
    }
    #[inline(always)]
    fn isize(i: isize) -> Self {
        i as f64
    }

    #[inline(always)]
    fn float(f: f32) -> Self {
        f as f64
    }

    #[inline(always)]
    fn double(f: f64) -> Self {
        f
    }

    #[inline(always)]
    fn sin_cos(&self) -> (Self, Self) {
        (*self as f64).sin_cos()
    }
    #[inline(always)]
    fn exp(&self) -> Self {
        (*self as f64).exp()
    }
    #[inline(always)]
    fn recip(&self) -> Self {
        (*self as f64).recip()
    }
    #[inline(always)]
    fn sqrt(&self) -> Self {
        (*self as f64).sqrt()
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        (*self as f64).abs()
    }

    #[inline(always)]
    fn square_norm(&self) -> Self {
        (*self as f64).powi(2)
    }
}

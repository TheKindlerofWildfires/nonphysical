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
    fn one() -> Self;
    fn maximum() -> Self;
    fn minimum() -> Self;
    fn small() -> Self;
    fn epsilon() -> Self;
    //conversion functions
    fn usize(u: usize) -> Self;
    fn isize(i: isize) -> Self;
    fn float(f: f32) -> Self;
    fn double(f: f64) -> Self;
    fn to_usize(&self) -> usize;
    fn is_nan(&self) -> bool;

    //utility
    fn sin_cos(&self) -> (Self, Self);
    fn atan2(&self, other: Self) -> Self;
    fn exp(&self) -> Self;
    fn recip(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn ln(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn norm(&self) -> Self;
    fn square_norm(&self) -> Self;

    fn tan(&self) -> Self;

    fn tanh(&self) -> Self;

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
    fn one() -> Self {
        1.0
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
    #[inline(always)]
    fn atan2(&self, other: Self) -> Self{
        (*self as f32).atan2(other as f32)
    }

    #[inline(always)]
    fn ln(&self) -> Self {
        (*self as f32).ln()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        (*self as f32).sin()
    }
    #[inline(always)]
    fn cos(&self) -> Self {
        (*self as f32).cos()
    }
    fn tan(&self) -> Self{
        (*self as f32).tan()
    }

    fn to_usize(&self) -> usize{
        *self as usize 
    }
    fn is_nan(&self) -> bool{
        (*self as f32).is_nan()
    }

    fn sinh(&self) -> Self{
        (*self as f32).sinh()
    }
    fn cosh(&self) -> Self{
        (*self as f32).cosh()
    }
    fn tanh(&self) -> Self{
        (*self as f32).tanh()
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
    fn one() -> Self {
        1.0
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

    #[inline(always)]
    fn atan2(&self, other: Self) -> Self{
        (*self as f64).atan2(other as f64)
    }
    #[inline(always)]
    fn ln(&self) -> Self {
        (*self as f64).ln()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        (*self as f64).sin()
    }
    #[inline(always)]
    fn cos(&self) -> Self {
        (*self as f64).cos()
    }

    fn tan(&self) -> Self{
        (*self as f64).tan()
    }

    fn to_usize(&self) -> usize{
        *self as usize 
    }
    fn is_nan(&self) -> bool{
        (*self as f64).is_nan()
    }

    fn sinh(&self) -> Self{
        (*self as f64).sinh()
    }
    fn cosh(&self) -> Self{
        (*self as f64).cosh()
    }

    fn tanh(&self) -> Self{
        (*self as f64).tanh()
    }
}

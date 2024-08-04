
use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::primitive::Primitive;
use alloc::boxed::Box;


/*
    This Trait is shared by both real and complex numbers and defines functions that can be expected from any float in a field

*/

pub trait Float:
    Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + MulAssign
    + AddAssign
    + DivAssign
    + SubAssign
    + Clone
    + Copy
    + Sized
    + Sync
    + Send
    + Debug
    + PartialEq
{
    type Primitive: Primitive;
    const ZERO: Self;
    const IDENTITY: Self;
    fn l1_norm(self) -> Self::Primitive;
    fn l2_norm(self) -> Self::Primitive;
    fn fma(self, mul: Self, add: Self) -> Self;
    fn powf(self, other: Self) -> Self;
    fn powi(self, other: i32) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn recip(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn to_be_bytes(self) -> Box<[u8]>;
    fn to_le_bytes(self) -> Box<[u8]>;
}


use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::primitive::Primitive;
use alloc::{boxed::Box, string::String};


/*
    This Trait is shared by both real and complex numbers and defines functions that can be expected from any float in a field

*/

pub trait Float:
    Neg<Output = Self>
    + Add<Self,Output = Self>
    + Sub<Self,Output = Self>
    + Mul<Self,Output = Self>
    + Div<Self,Output = Self>
    + Mul<Self::Primitive,Output = Self>
    + Div<Self::Primitive,Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + MulAssign<Self::Primitive>
    + DivAssign<Self::Primitive>
    + Clone
    + Copy
    + Sized
    + Sync
    + Send
    + Debug
    + PartialEq
    + PartialOrd
{
    type Primitive: Primitive;
    const ZERO: Self;
    const IDENTITY: Self;
    const MIN: Self;
    const MAX: Self;
    const NAN: Self;
    const INFINITY: Self;
    const NEGATIVE_INFINITY: Self;
    fn l1_norm(self) -> Self::Primitive;
    fn l2_norm(self) -> Self::Primitive;
    fn fma(self, mul: Self, add: Self) -> Self;
    fn powf(self, other: Self) -> Self;
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
    fn type_id()->String;
    fn greater(self,other: Self)->Self;
    fn lesser(self,other: Self)->Self;
    fn finite(self)->bool;
    fn is_nan(self)->bool;

}

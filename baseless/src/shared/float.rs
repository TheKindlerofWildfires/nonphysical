#[cfg(not(feature = "std"))]
use core::intrinsics;

#[cfg(not(feature = "std"))]
use super::primitive::Scaler;

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

#[cfg(feature = "std")]
impl Float for f32 {
    const ZERO: Self = 0.0;
    const IDENTITY: Self = 1.0;
    type Primitive = f32;
    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        (self).abs()
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        (self).powi(2)
    }

    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        (self).mul_add(mul, add)
    }

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        (self).powf(other)
    }

    #[inline(always)]
    fn powi(self, other: i32) -> Self {
        (self).powi(other)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        (self).sqrt()
    }

    #[inline(always)]
    fn ln(self) -> Self {
        (self).ln()
    }

    #[inline(always)]
    fn log2(self) -> Self {
        (self).log2()
    }

    #[inline(always)]
    fn exp(self) -> Self {
        (self).exp()
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        (self).exp2()
    }

    #[inline(always)]
    fn recip(self) -> Self {
        (self).recip()
    }

    #[inline(always)]
    fn sin(self) -> Self {
        (self).sin()
    }

    #[inline(always)]
    fn cos(self) -> Self {
        (self).cos()
    }

    #[inline(always)]
    fn tan(self) -> Self {
        (self).tan()
    }

    #[inline(always)]
    fn asin(self) -> Self {
        (self).asin()
    }

    #[inline(always)]
    fn acos(self) -> Self {
        (self).acos()
    }

    #[inline(always)]
    fn atan(self) -> Self {
        (self).atan()
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        (self).sinh()
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        (self).cosh()
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        (self).tanh()
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        (self).asinh()
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        (self).acosh()
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        (self).atanh()
    }

    #[inline(always)]
    fn to_be_bytes(self) -> Box<[u8]> {
        Box::new((self).to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(self) -> Box<[u8]> {
        Box::new((self).to_le_bytes())
    }
}

#[cfg(feature = "std")]
impl Float for f64 {
    const ZERO: Self = 0.0;
    const IDENTITY: Self = 1.0;
    type Primitive = f64;
    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        (self).abs()
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        (self).powi(2)
    }
    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        (self).mul_add(mul, add)
    }

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        (self).powf(other)
    }

    #[inline(always)]
    fn powi(self, other: i32) -> Self {
        (self).powi(other)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        (self).sqrt()
    }

    #[inline(always)]
    fn ln(self) -> Self {
        (self).ln()
    }

    #[inline(always)]
    fn log2(self) -> Self {
        (self).log2()
    }

    #[inline(always)]
    fn exp(self) -> Self {
        (self).exp()
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        (self).exp2()
    }

    #[inline(always)]
    fn recip(self) -> Self {
        (self).recip()
    }

    #[inline(always)]
    fn sin(self) -> Self {
        (self).sin()
    }

    #[inline(always)]
    fn cos(self) -> Self {
        (self).cos()
    }

    #[inline(always)]
    fn tan(self) -> Self {
        (self).tan()
    }

    #[inline(always)]
    fn asin(self) -> Self {
        (self).asin()
    }

    #[inline(always)]
    fn acos(self) -> Self {
        (self).acos()
    }

    #[inline(always)]
    fn atan(self) -> Self {
        (self).atan()
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        (self).sinh()
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        (self).cosh()
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        (self).tanh()
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        (self).asinh()
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        (self).acosh()
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        (self).atanh()
    }

    #[inline(always)]
    fn to_be_bytes(self) -> Box<[u8]> {
        Box::new((self).to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(self) -> Box<[u8]> {
        Box::new((self).to_le_bytes())
    }
}

#[cfg(not(feature = "std"))]
impl Float for f32 {
    const ZERO: Self = 0.0;
    const IDENTITY: Self = 1.0;
    type Primitive = f32;
    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        unsafe { intrinsics::fabsf32(self) }
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        (self).powi(2)
    }

    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        unsafe { intrinsics::fmaf32(self, mul, add) }
    }

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        unsafe { intrinsics::powf32(self, other) }
    }

    #[inline(always)]
    fn powi(self, other: i32) -> Self {
        unsafe { intrinsics::powif32(self, other) }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe { intrinsics::sqrtf32(self) }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        unsafe { intrinsics::logf32(self) }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        unsafe { intrinsics::log2f32(self) }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        unsafe { intrinsics::expf32(self) }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        unsafe { intrinsics::exp2f32(self) }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        (self).recip()
    }

    #[inline(always)]
    fn sin(self) -> Self {
        unsafe { intrinsics::sinf32(self) }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        unsafe { intrinsics::cosf32(self) }
    }

    #[inline(always)]
    fn tan(self) -> Self {
        self.sin() / self.cos()
    }

    #[inline(always)]
    fn asin(self) -> Self {
        //lib c port
        let mut y = 0.0;
        loop {
            let (ys, yc) = y.sin_cos();
            if y > Self::FRAC_PI_2 || y < -Self::FRAC_PI_2 {
                y %= Self::PI;
            }
            if ys + Self::EPSILON >= self && ys - Self::EPSILON <= self {
                break;
            }
            y -= (ys - self) / yc
        }
        y
    }

    #[inline(always)]
    fn acos(self) -> Self {
        Self::FRAC_PI_2 - self.sin()
    }

    #[inline(always)]
    fn atan(self) -> Self {
        (self / (self * self + 1.0).sqrt()).asin()
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        let tmp = self.exp();
        (tmp - tmp.recip()) / 2.0
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        let tmp = self.exp();
        (tmp + tmp.recip()) / 2.0
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        let tmp1 = self.exp();
        let tmp2 = tmp1.recip();
        (tmp1 + tmp2) / (tmp1 - tmp2)
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn to_be_bytes(self) -> Box<[u8]> {
        Box::new((self).to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(self) -> Box<[u8]> {
        Box::new((self).to_le_bytes())
    }
}

#[cfg(not(feature = "std"))]
impl Float for f64 {
    const ZERO: Self = 0.0;
    const IDENTITY: Self = 1.0;
    type Primitive = f64;
    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        unsafe { intrinsics::fabsf64(self) }
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        (self).powi(2)
    }

    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        unsafe { intrinsics::fmaf64(self, mul, add) }
    }

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        unsafe { intrinsics::powf64(self, other) }
    }

    #[inline(always)]
    fn powi(self, other: i32) -> Self {
        unsafe { intrinsics::powif64(self, other) }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        unsafe { intrinsics::sqrtf64(self) }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        unsafe { intrinsics::logf64(self) }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        unsafe { intrinsics::log2f64(self) }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        unsafe { intrinsics::expf64(self) }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        unsafe { intrinsics::exp2f64(self) }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        (self).recip()
    }

    #[inline(always)]
    fn sin(self) -> Self {
        unsafe { intrinsics::sinf64(self) }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        unsafe { intrinsics::cosf64(self) }
    }

    #[inline(always)]
    fn tan(self) -> Self {
        self.sin() / self.cos()
    }

    #[inline(always)]
    fn asin(self) -> Self {
        //lib c port
        let mut y = 0.0;
        loop {
            let (ys, yc) = y.sin_cos();
            if y > Self::FRAC_PI_2 || y < -Self::FRAC_PI_2 {
                y %= Self::PI;
            }
            if ys + Self::EPSILON >= self && ys - Self::EPSILON <= self {
                break;
            }
            y -= (ys - self) / yc
        }
        y
    }

    #[inline(always)]
    fn acos(self) -> Self {
        Self::FRAC_PI_2 - self.sin()
    }

    #[inline(always)]
    fn atan(self) -> Self {
        (self / (self * self + 1.0).sqrt()).asin()
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        let tmp = self.exp();
        (tmp - tmp.recip()) / 2.0
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        let tmp = self.exp();
        (tmp + tmp.recip()) / 2.0
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        let tmp1 = self.exp();
        let tmp2 = tmp1.recip();
        (tmp1 + tmp2) / (tmp1 - tmp2)
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn to_be_bytes(self) -> Box<[u8]> {
        Box::new((self).to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(self) -> Box<[u8]> {
        Box::new((self).to_le_bytes())
    }
}

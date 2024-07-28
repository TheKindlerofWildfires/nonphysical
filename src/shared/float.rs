use core::{fmt::Debug, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};

use super::real::Real;

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
    type Primitive: Real;
    const ZERO: Self;
    const IDENTITY: Self;
    fn l1_norm(&self) -> Self::Primitive;
    fn l2_norm(&self) -> Self::Primitive;
    fn fma(&self, mul: Self, add: Self) -> Self;
    fn powf(&self, other: Self) -> Self;
    fn powi(&self, other: i32) ->Self;
    fn sqrt(&self) -> Self;
    fn cbrt(&self) -> Self;
    fn ln(&self) -> Self;
    fn log2(&self) -> Self;
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;
    fn recip(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn to_be_bytes(&self) -> Box<[u8]>;
    fn to_le_bytes(&self) -> Box<[u8]>;
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const IDENTITY: Self = 1.0;
    type Primitive = f32;
    #[inline(always)]
    fn l1_norm(&self) -> Self::Primitive {
        (*self).abs()
    }

    #[inline(always)]
    fn l2_norm(&self) -> Self::Primitive {
        (*self).powi(2)
    }

    #[inline(always)]
    fn fma(&self, mul: Self, add: Self) -> Self{
        (*self).mul_add(mul,add)
    }

    #[inline(always)]
    fn powf(&self, other: Self) -> Self{
        (*self).powf(other)
    }

    #[inline(always)]
    fn powi(&self, other: i32) ->Self{
        (*self).powi(other)
    }

    #[inline(always)]
    fn sqrt(&self) -> Self{
        (*self).sqrt()
    }

    #[inline(always)]
    fn cbrt(&self) -> Self{
        (*self).cbrt()
    }

    #[inline(always)]
    fn ln(&self) -> Self{
        (*self).ln()
    }

    #[inline(always)]
    fn log2(&self) -> Self{
        (*self).log2()
    }

    #[inline(always)]
    fn exp(&self) -> Self{
        (*self).exp() 
    }

    #[inline(always)]
    fn exp2(&self) -> Self{
        (*self).exp2() 
    }

    #[inline(always)]
    fn recip(&self) -> Self{
        (*self).recip()
    }

    #[inline(always)]
    fn sin(&self) -> Self{
        (*self).sin()
    }

    #[inline(always)]
    fn cos(&self) -> Self{
        (*self).cos()
    }

    #[inline(always)]
    fn tan(&self) -> Self{
        (*self).cos()
    }

    #[inline(always)]
    fn asin(&self) -> Self{
        (*self).asin()
    }

    #[inline(always)]
    fn acos(&self) -> Self{
        (*self).acos()
    }

    #[inline(always)]
    fn atan(&self) -> Self{
        (*self).atan()
    }

    #[inline(always)]
    fn sinh(&self) -> Self{
        (*self).sinh()
    }

    #[inline(always)]
    fn cosh(&self) -> Self{
        (*self).cosh()
    }

    #[inline(always)]
    fn tanh(&self) -> Self{
        (*self).tanh() 
    }

    #[inline(always)]
    fn asinh(&self) -> Self{
        (*self).asinh()
    }

    #[inline(always)]
    fn acosh(&self) -> Self{
        (*self).acosh()
    }

    #[inline(always)]
    fn atanh(&self) -> Self{
        (*self).atanh()
    }

    #[inline(always)]
    fn to_be_bytes(&self) -> Box<[u8]>{
        Box::new((*self).to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(&self) ->  Box<[u8]>{
        Box::new((*self).to_le_bytes())
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const IDENTITY: Self = 1.0;
    type Primitive = f64;
    #[inline(always)]
    fn l1_norm(&self) -> Self::Primitive {
        (*self).abs()
    }

    #[inline(always)]
    fn l2_norm(&self) -> Self::Primitive {
        (*self).powi(2)
    }
    #[inline(always)]
    fn fma(&self, mul: Self, add: Self) -> Self{
        (*self).mul_add(mul,add)
    }

    #[inline(always)]
    fn powf(&self, other: Self) -> Self{
        (*self).powf(other)
    }

    #[inline(always)]
    fn powi(&self, other: i32) ->Self{
        (*self).powi(other)
    }

    #[inline(always)]
    fn sqrt(&self) -> Self{
        (*self).sqrt()
    }

    #[inline(always)]
    fn cbrt(&self) -> Self{
        (*self).cbrt()
    }

    #[inline(always)]
    fn ln(&self) -> Self{
        (*self).ln()
    }

    #[inline(always)]
    fn log2(&self) -> Self{
        (*self).log2()
    }

    #[inline(always)]
    fn exp(&self) -> Self{
        (*self).exp() 
    }

    #[inline(always)]
    fn exp2(&self) -> Self{
        (*self).exp2() 
    }

    #[inline(always)]
    fn recip(&self) -> Self{
        (*self).recip()
    }

    #[inline(always)]
    fn sin(&self) -> Self{
        (*self).sin()
    }

    #[inline(always)]
    fn cos(&self) -> Self{
        (*self).cos()
    }

    #[inline(always)]
    fn tan(&self) -> Self{
        (*self).cos()
    }

    #[inline(always)]
    fn asin(&self) -> Self{
        (*self).asin()
    }

    #[inline(always)]
    fn acos(&self) -> Self{
        (*self).acos()
    }

    #[inline(always)]
    fn atan(&self) -> Self{
        (*self).atan()
    }

    #[inline(always)]
    fn sinh(&self) -> Self{
        (*self).sinh()
    }

    #[inline(always)]
    fn cosh(&self) -> Self{
        (*self).cosh()
    }

    #[inline(always)]
    fn tanh(&self) -> Self{
        (*self).tanh() 
    }

    #[inline(always)]
    fn asinh(&self) -> Self{
        (*self).asinh()
    }

    #[inline(always)]
    fn acosh(&self) -> Self{
        (*self).acosh()
    }

    #[inline(always)]
    fn atanh(&self) -> Self{
        (*self).atanh()
    }

    #[inline(always)]
    fn to_be_bytes(&self) -> Box<[u8]>{
        Box::new((*self).to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(&self) ->  Box<[u8]>{
        Box::new((*self).to_le_bytes())
    }
}


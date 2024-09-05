use core::f64;

use nonphysical_core::shared::float::Float;

use super::primitive::{F32, F64};

impl Float for F32 {
    const ZERO: Self = F32(0.0);
    const IDENTITY: Self = F32(1.0);
    const MAX: Self = F32(f32::MAX);
    const MIN: Self = F32(f32::MIN);
    const NAN: Self = F32(f32::NAN);
    const INFINITY: Self = F32(f32::INFINITY);
    const NEGATIVE_INFINITY: Self = F32(f32::NEG_INFINITY);
    type Primitive = F32;
    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        F32(self.0.abs())
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        F32(self.0.powi(2))
    }

    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        F32(self.0.mul_add(mul.0, add.0))
    }

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        F32(self.0.powf(other.0))
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        F32(self.0.sqrt())
    }

    #[inline(always)]
    fn ln(self) -> Self {
        F32(self.0.ln())
    }

    #[inline(always)]
    fn log2(self) -> Self {
        F32(self.0.log2())
    }

    #[inline(always)]
    fn exp(self) -> Self {
        F32(self.0.exp())
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        F32(self.0.exp2())
    }

    #[inline(always)]
    fn recip(self) -> Self {
        F32(self.0.recip())
    }

    #[inline(always)]
    fn sin(self) -> Self {
        F32(self.0.sin())
    }

    #[inline(always)]
    fn cos(self) -> Self {
        F32(self.0.cos())
    }

    #[inline(always)]
    fn tan(self) -> Self {
        F32(self.0.tan())
    }

    #[inline(always)]
    fn asin(self) -> Self {
        F32(self.0.asin())
    }

    #[inline(always)]
    fn acos(self) -> Self {
        F32(self.0.acos())
    }

    #[inline(always)]
    fn atan(self) -> Self {
        F32(self.0.atan())
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        F32(self.0.sinh())
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        F32(self.0.cosh())
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        F32(self.0.tanh())
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        F32(self.0.asinh())
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        F32(self.0.acosh())
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        F32(self.0.atanh())
    }

    #[inline(always)]
    fn to_be_bytes(self) -> Box<[u8]> {
        Box::new(self.0.to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(self) -> Box<[u8]> {
        Box::new(self.0.to_le_bytes())
    }
    
    fn type_id()->String {
        "f32".to_string()
    }

    
    fn greater(self,other: Self)->Self {
        if self>other{
            self
        }else{
            other
        }
    }
    
    fn lesser(self,other: Self)->Self {
        if self<other{
            self
        }else{
            other
        }
    }
    fn finite(self)->bool {
        self.0.is_finite()
    }

    fn is_nan(self)->bool {
        self.0.is_nan()
    }
}

impl Float for F64 {
    const ZERO: Self = F64(0.0);
    const IDENTITY: Self = F64(1.0);
    const MAX: Self = F64(f64::MAX);
    const MIN: Self = F64(f64::MIN);
    const NAN: Self = F64(f64::NAN);
    const INFINITY: Self = F64(f64::INFINITY);
    const NEGATIVE_INFINITY: Self = F64(f64::NEG_INFINITY);
    type Primitive = F64;
    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        F64(self.0.abs())
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        F64(self.0.powi(2))
    }

    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        F64(self.0.mul_add(mul.0, add.0))
    }

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        F64(self.0.powf(other.0))
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        F64(self.0.sqrt())
    }

    #[inline(always)]
    fn ln(self) -> Self {
        F64(self.0.ln())
    }

    #[inline(always)]
    fn log2(self) -> Self {
        F64(self.0.log2())
    }

    #[inline(always)]
    fn exp(self) -> Self {
        F64(self.0.exp())
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        F64(self.0.exp2())
    }

    #[inline(always)]
    fn recip(self) -> Self {
        F64(self.0.recip())
    }

    #[inline(always)]
    fn sin(self) -> Self {
        F64(self.0.sin())
    }

    #[inline(always)]
    fn cos(self) -> Self {
        F64(self.0.cos())
    }

    #[inline(always)]
    fn tan(self) -> Self {
        F64(self.0.tan())
    }

    #[inline(always)]
    fn asin(self) -> Self {
        F64(self.0.asin())
    }

    #[inline(always)]
    fn acos(self) -> Self {
        F64(self.0.acos())
    }

    #[inline(always)]
    fn atan(self) -> Self {
        F64(self.0.atan())
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        F64(self.0.sinh())
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        F64(self.0.cosh())
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        F64(self.0.tanh())
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        F64(self.0.asinh())
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        F64(self.0.acosh())
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        F64(self.0.atanh())
    }

    #[inline(always)]
    fn to_be_bytes(self) -> Box<[u8]> {
        Box::new(self.0.to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(self) -> Box<[u8]> {
        Box::new(self.0.to_le_bytes())
    }
    
    fn type_id()->String {
        "f64".to_string()
    }

    fn greater(self,other: Self)->Self {
        if self>other{
            self
        }else{
            other
        }
    }
    
    fn lesser(self,other: Self)->Self {
        if self<other{
            self
        }else{
            other
        }
    }
    fn finite(self)->bool {
        self.0.is_finite()
    }
    fn is_nan(self)->bool {
        self.0.is_nan()
    }
}


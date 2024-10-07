use super::primitive::Primitive;
use super::float::Float;
use core::cmp::Ordering;
use core::fmt::{Result,Formatter,Debug};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use alloc::boxed::Box;
use alloc::format;
/*
    This Struct defines complex numbers and functions which are unique to them
*/
#[derive(Copy, Clone, PartialEq,Eq)]
pub struct ComplexScaler<P: Primitive> {
    pub real: P,
    pub imag: P,
}

impl<P: Primitive<Primitive = P>> Float for ComplexScaler<P> {
    const IDENTITY: Self = Self::ONE;
    const ZERO: Self = ComplexScaler {
        real: P::ZERO,
        imag: P::ZERO,
    };
    const MAX: Self = ComplexScaler{
        real: P::MAX,
        imag: P::MAX,
    };
    const MIN: Self = Self::ZERO;
    type Primitive = P;

    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        (self.real.l2_norm() + self.imag.l2_norm()).sqrt()
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        self.real.l2_norm() + self.imag.l2_norm()
    }

    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        let real = self
            .real
            .fma(mul.real, (-self.imag).fma(mul.imag, add.real));
        let imag = self.real.fma(mul.imag, self.imag.fma(mul.real, add.imag));
        Self { real, imag }
    }
    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        assert!(other.imag == P::ZERO); //not handling this yet
        let magnitude = self.l1_norm().powf(other.real);
        let phase = self.phase() * other.real;
        let (st, ct) = phase.sin_cos();
        let real = ct * magnitude;
        let imag = st * magnitude;
        Self { real, imag }
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.powf(Self::real(P::u8(2).recip()))
    }

    #[inline(always)]
    fn ln(self) -> Self {
        let real = self.l1_norm().ln();
        let imag = self.phase();
        Self { real, imag }
    }
    #[inline(always)]
    fn log2(self) -> Self {
        self.ln() / (P::u8(2).ln())
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let exp = self.real.exp();
        let (st, ct) = self.imag.sin_cos();
        let real = ct * exp;
        let imag = st * exp;
        Self { real, imag }
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn recip(self) -> Self {
        self.conjugate() / self.l2_norm()
    }

    #[inline(always)]
    fn sin(self) -> Self {
        let real = self.real.sin() * self.imag.cosh();
        let imag = self.real.cos() * self.imag.sinh();
        Self { real, imag }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        let real = self.real.cos() * self.imag.cosh();
        let imag = -self.real.sin() * self.imag.sinh();
        Self { real, imag }
    }

    #[inline(always)]
    fn tan(self) -> Self {
        let num = Self::new(self.real.tan(), self.imag.tanh());
        let denom = Self::new(P::ONE, -self.real.tan() * self.imag.tan());
        num / denom
    }

    #[inline(always)]
    fn asin(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acos(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atan(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        let real = self.real.sinh() * self.imag.cos();
        let imag = self.real.cosh() * self.imag.sin();
        Self { real, imag }
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        let real = self.real.cosh() * self.imag.cos();
        let imag = self.real.sinh() * self.imag.sin();
        Self { real, imag }
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        let num = Self::new(-self.real.tan(), self.imag.tanh());
        let denom = Self::new(self.real.tan() * self.imag.tan(), P::ONE);
        num / denom
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

    fn to_be_bytes(self) -> Box<[u8]> {
        let mut real_bytes = self.real.to_be_bytes().to_vec();
        let imag_bytes = self.imag.to_be_bytes().to_vec();
        real_bytes.extend(imag_bytes);
        real_bytes.into_boxed_slice()
    }

    fn to_le_bytes(self) -> Box<[u8]> {
        let mut real_bytes = self.real.to_le_bytes().to_vec();
        let imag_bytes = self.imag.to_le_bytes().to_vec();
        real_bytes.extend(imag_bytes);
        real_bytes.into_boxed_slice()
    }
    
    fn type_id()->alloc::string::String {
        format!("complex_{}",P::type_id())
    }
    
    fn greater(self,other: Self)->Self {
        if self.l2_norm()>other.l2_norm(){
            self
        }else{
            other
        }
    }
    
    fn lesser(self,other: Self)->Self {
        if self.l2_norm()<other.l2_norm(){
            self
        }else{
            other
        }
    }
    
    fn finite(self)->bool {
        self.real().finite()&&self.imag().finite()
    }

    fn is_nan(self)->bool {
        self.real().is_nan()||self.imag().is_nan()
    }
    
    const NAN: Self = Self{real: Self::Primitive::NAN, imag:Self::Primitive::NAN};
    
    const INFINITY: Self = Self{real: Self::Primitive::INFINITY, imag:Self::Primitive::INFINITY};
    
    const NEGATIVE_INFINITY: Self = Self{real: Self::Primitive::NEGATIVE_INFINITY, imag:Self::Primitive::NEGATIVE_INFINITY};


}

impl<P: Primitive<Primitive = P>> Neg for ComplexScaler<P> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let real = -self.real;
        let imag = -self.imag;
        Self { real, imag }
    }
}
impl<P: Primitive<Primitive = P>> Add for ComplexScaler<P> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        let real = self.real + other.real;
        let imag = self.imag + other.imag;
        Self { real, imag }
    }
}

impl<P: Primitive<Primitive = P>> Sub for ComplexScaler<P> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        let real = self.real - other.real;
        let imag = self.imag - other.imag;
        Self { real, imag }
    }
}

impl<P: Primitive<Primitive = P>> Mul for ComplexScaler<P> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        let real = self.real * other.real - self.imag * other.imag;
        let imag = self.real * other.imag + self.imag * other.real;
        Self { real, imag }
    }
}

impl<P: Primitive<Primitive = P>> Div for ComplexScaler<P> {
    type Output = Self;
    fn div(self, other: Self) -> Self::Output {
        let l2 = other.l2_norm();
        let real = (self.real * other.real + self.imag * other.imag) / l2;
        let imag = (-self.real * other.imag + self.imag * other.real) / l2;
        Self { real, imag }
    }
}

impl<P: Primitive<Primitive = P>> Mul<P> for ComplexScaler<P> {
    type Output = Self;
    fn mul(self, other: P) -> Self::Output {
        let real = self.real * other;
        let imag = self.imag * other;
        Self { real, imag }
    }
}

impl<P: Primitive<Primitive = P>> Div<P> for ComplexScaler<P> {
    type Output = Self;
    fn div(self, other: P) -> Self::Output {
        let div = other.recip();
        let real = self.real * div;
        let imag = self.imag * div;
        Self { real, imag }
    }
}

impl<P: Primitive<Primitive = P>> AddAssign for ComplexScaler<P> {
    fn add_assign(&mut self, other: Self) {
        self.real += other.real;
        self.imag += other.imag
    }
}

impl<P: Primitive<Primitive = P>> SubAssign for ComplexScaler<P> {
    fn sub_assign(&mut self, other: Self) {
        self.real -= other.real;
        self.imag -= other.imag;
    }
}

impl<P: Primitive<Primitive = P>> MulAssign for ComplexScaler<P> {
    fn mul_assign(&mut self, other: Self) {
        let tmp = self.real * other.imag + self.imag * other.real;
        self.real = self.real * other.real - self.imag * other.imag;
        self.imag = tmp;
    }
}

impl<P: Primitive<Primitive = P>> DivAssign for ComplexScaler<P> {
    fn div_assign(&mut self, other: Self) {
        let l2 = other.l2_norm();
        let tmp = (-self.real * other.imag + self.imag * other.real) / l2;
        self.real = (self.real * other.real + self.imag * other.imag) / l2;
        self.imag = tmp;
    }
}

impl<P: Primitive<Primitive = P>> MulAssign<P> for ComplexScaler<P> {
    fn mul_assign(&mut self, other: P) {
        self.real *= other;
        self.imag *= other
    }
}

impl<P: Primitive<Primitive = P>> DivAssign<P> for ComplexScaler<P> {
    fn div_assign(&mut self, other: P) {
        let div = other.recip();
        self.real *= div;
        self.imag *= div
    }
}

impl<P: Primitive> Debug for ComplexScaler<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?} + {:?}i", self.real, self.imag)
    }
}

impl<P: Primitive<Primitive = P>> PartialOrd for ComplexScaler<P>{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.l1_norm().partial_cmp(&other.l1_norm())
    }
}

pub trait Complex:
    Float
    + Mul<Self::Primitive, Output = Self>
    + Div<Self::Primitive, Output = Self>
    + MulAssign<Self::Primitive>
    + DivAssign<Self::Primitive>
{
    const ONE: Self;
    const NEGATIVE_ONE: Self;
    const I: Self;
    const NEGATIVE_I: Self;

    fn new(real: Self::Primitive, imag: Self::Primitive) -> Self;
    fn new_phase(norm: Self::Primitive, phase: Self::Primitive) -> Self;
    fn real(self) -> Self::Primitive;
    fn imag(self) -> Self::Primitive;
    fn real_ref(&mut self) -> &mut Self::Primitive;
    fn imag_ref(&mut self) -> &mut Self::Primitive;
    fn swap(self) -> Self;
    fn conjugate(self) -> Self;
    fn phase(self) -> Self::Primitive;
    fn mul_i(self) -> Self;
    fn mul_ni(self) -> Self;
}


impl<P: Primitive<Primitive = P>> Complex for ComplexScaler<P> {
    const ONE: Self = Self {
        real: P::ONE,
        imag: P::ZERO,
    };
    const NEGATIVE_ONE: Self = Self {
        real: P::NEGATIVE_ONE,
        imag: P::ZERO,
    };
    const I: Self = Self {
        real: P::ZERO,
        imag: P::ONE,
    };
    const NEGATIVE_I: Self = Self {
        real: P::ZERO,
        imag: P::NEGATIVE_ONE,
    };

    #[inline(always)]
    fn new(real: Self::Primitive, imag: Self::Primitive) -> Self {
        Self { real, imag }
    }

    #[inline(always)]
    fn new_phase(norm: Self::Primitive, phase: Self::Primitive) -> Self {
        let (st, ct) = phase.sin_cos();
        let real = ct * norm;
        let imag = st * norm;
        Self { real, imag }
    }

    #[inline(always)]
    fn real_ref(&mut self) -> &mut Self::Primitive {
        &mut self.real
    }

    #[inline(always)]
    fn imag_ref(&mut self) -> &mut Self::Primitive {
        &mut self.imag
    }

    #[inline(always)]
    fn real(self) -> Self::Primitive {
        self.real
    }
    #[inline(always)]
    fn imag(self) -> Self::Primitive {
        self.imag
    }
    #[inline(always)]
    fn swap(self) -> Self {
        Self {
            real: self.imag,
            imag: self.real,
        }
    }
    #[inline(always)]
    fn conjugate(self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }

    #[inline(always)]
    fn phase(self) -> P {
        self.imag.atan2(self.real)
    }

    #[inline(always)]
    fn mul_i(self) -> Self {
        Self {
            real: -self.imag,
            imag: self.real,
        }
    }

    #[inline(always)]
    fn mul_ni(self) -> Self {
        Self {
            real: self.imag,
            imag: -self.real,
        }
    }
}

impl<P: Primitive> ComplexScaler<P> {
    #[inline(always)]
    pub fn real(real: P) -> Self {
        let imag = P::ZERO;
        Self { real, imag }
    }

    #[inline(always)]
    pub fn imag(imag: P) -> Self {
        let real = P::ZERO;
        Self { real, imag }
    }
}
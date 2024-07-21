use core::{fmt::Debug,ops::{Add, AddAssign, Div, Mul, Neg, Sub}};
use std::ops::{DivAssign, MulAssign};
use super::float::Float;

#[derive(Copy, Clone, PartialEq)]
pub struct Complex<T: Float> {
    pub real: T,
    pub imag: T,
}

impl<T: Float> Complex<T> {

    pub const ZERO: Complex<T> = Self{real: T::ZERO, imag: T::ZERO};
    pub const ONE: Complex<T> = Self{real: T::ONE, imag: T::ZERO};
    pub const N_ONE: Complex<T> = Self{real: T::N_ONE, imag: T::ZERO};
    pub const I: Complex<T> = Self{real: T::ZERO, imag: T::ONE};
    pub const N_I: Complex<T> = Self{real: T::ZERO, imag: T::N_ONE};
    #[inline(always)]
    pub fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }

    #[inline(always)]
    pub fn new_phase(norm: T, phase: T) -> Self {
        let (st, ct) = phase.sin_cos();
        Self {
            real: ct * norm,
            imag: st * norm,
        }
    }

    #[inline(always)]
    pub fn swap(&self) -> Self {
        Self {
            real: self.imag,
            imag: self.real,
        }
    }
    #[inline(always)]
    pub fn conj(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
    #[inline(always)]
    pub fn norm(&self) -> T {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
    #[inline(always)]
    pub fn square_norm(&self) -> T {
        self.real * self.real + self.imag * self.imag
    }
    #[inline(always)]
    pub fn phase(&self)-> T{
        self.imag.atan2(&self.real)
    }
    #[inline(always)]
    pub fn recip(&self) -> Self {
        self.conj() / self.square_norm()
    }
    #[inline(always)]
    pub fn mul_i(&self) -> Self{
        Self { real:-self.imag, imag:self.real }
    }
    #[inline(always)]
    pub fn mul_ni(&self) -> Self{
        Self { real:self.imag, imag:-self.real }
    }
    #[inline(always)]
    pub fn ln(&self) -> Self{
        Self { real: self.norm().ln(), imag: self.phase() }
    }
    #[inline(always)]
    pub fn exp(&self) -> Self{
        let ea = self.real.exp();
        let (st,ct) = self.imag.sin_cos();

        Self { real: ea*ct, imag: ea*st }
    }
    #[inline(always)]
    pub fn sin(&self) -> Self{
        Self::new(self.real.sin()*self.imag.cosh(), self.real.cos()*self.imag.sinh())
    }
    #[inline(always)]
    pub fn cos(&self) -> Self{
        Self::new(self.real.cos()*self.imag.cosh(), -self.real.sin()*self.imag.sinh())
    }
    #[inline(always)]
    pub fn tan(&self) -> Self{
        Self::new(self.real.tan(), self.imag.tanh())/Self::new(T::ONE,-self.real.tan()*self.imag.tan())
    }
    #[inline(always)]
    pub fn sinh(&self) -> Self{
        Self::new(self.real.sinh()*self.imag.cos(),self.real.cosh()*self.imag.sin())
    }
    #[inline(always)]
    pub fn cosh(&self) -> Self{
        Self::new(self.real.cosh()*self.imag.cos(),self.real.sinh()*self.imag.sin())
    }
    #[inline(always)]
    pub fn tanh(&self) -> Self{
        Self::new(-self.imag.tan(), self.real.tanh())/Self::new(self.real.tan()*self.imag.tan(),T::ONE)
    }
    #[inline(always)]
    pub fn fma(&self, mul: Self, add: Self) -> Self{
        let real = self.real.fma(mul.real, (-self.imag).fma(mul.imag, add.real));
        let imag = self.real.fma(mul.imag, self.imag.fma(mul.real, add.imag));
        Self::new(real,imag)

    }
    #[inline(always)]
    pub fn sqrt(&self) -> Self{
        if self.imag == T::ZERO{
            Self::new(self.real.sqrt(),T::ZERO)
        }else{
            let real = ((self.real+(self.real.square_norm()+self.imag.square_norm()).sqrt())/T::usize(2)).sqrt();
            let imag = self.imag/self.imag.norm()*((-self.real+(self.real.square_norm()+self.imag.square_norm()).sqrt())/T::usize(2)).sqrt();
            Self::new(real,imag)
        }


    }

}

impl<T: Float> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

impl<T: Float> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self{ real: self.real + rhs.real, imag: self.imag + rhs.imag}
    }
}
impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self{real: self.real - rhs.real, imag: self.imag - rhs.imag}
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self{
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self, scaler: T) -> Self {
        Self{
            real:self.real * scaler, imag: self.imag * scaler}
    }
}

impl<T: Float> Div for Complex<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.conj() / rhs.square_norm()
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;
    fn div(self, scaler: T) -> Self {
        Self{
            real:self.real / scaler, imag : self.imag / scaler}
    }
}

impl <T:Float> AddAssign for Complex<T>{
    fn add_assign(&mut self, rhs: Self) {
        self.real+=rhs.real;
        self.imag+=rhs.imag;
    }
}

impl <T:Float> MulAssign for Complex<T>{
    fn mul_assign(&mut self, rhs: Self) {
        let real = self.real * rhs.real - self.imag * rhs.imag;
        self.imag = self.real * rhs.imag + self.imag * rhs.real;
        self.real = real;
    }
}

impl <T:Float> DivAssign<T> for Complex<T>{
    fn div_assign(&mut self, rhs: T) {
        self.real /= rhs;
        self.imag /= rhs;
    }
}

impl <T:Float> Debug for Complex<T>{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} + {:?}i",self.real, self.imag)
    }
}
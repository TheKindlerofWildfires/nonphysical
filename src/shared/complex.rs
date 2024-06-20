use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::fmt::Debug;

use super::float::Float;

#[derive(Copy, Clone, PartialEq)]
pub struct Complex<T: Float> {
    pub real: T,
    pub imag: T,
}

impl<T: Float> Complex<T> {
    pub fn zero() -> Self {
        Self {
            real: T::zero(),
            imag: T::zero(),
        }
    }
    pub fn one() -> Self {
        Self {
            real: T::one(),
            imag: T::zero(),
        }
    }

    pub fn none() -> Self {
        Self {
            real: -T::one(),
            imag: T::zero(),
        }
    }

    pub fn i() -> Self {
        Self {
            real: T::zero(),
            imag: T::one(),
        }
    }

    pub fn ni() -> Self {
        Self {
            real: T::zero(),
            imag: -T::one(),
        }
    }

    pub fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }
    pub fn new_phase(norm: T, phase: T) -> Self {
        let (st, ct) = phase.sin_cos();
        Self {
            real: ct * norm,
            imag: st * norm,
        }
    }
    pub fn swap(&self) -> Self {
        Self {
            real: self.imag,
            imag: self.real,
        }
    }
    pub fn conj(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
    pub fn norm(&self) -> T {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
    pub fn square_norm(&self) -> T {
        self.real * self.real + self.imag * self.imag
    }

    pub fn phase(&self)-> T{
        self.imag.atan2(self.real)
    }
    pub fn recip(&self) -> Self {
        self.conj() / self.square_norm()
    }
    pub fn mul_i(&self) -> Self{
        Self { real:-self.imag, imag:self.real }
    }
    pub fn mul_ni(&self) -> Self{
        Self { real:self.imag, imag:-self.real }
    }
    pub fn ln(&self) -> Self{
        Self { real: self.norm().ln(), imag: self.phase() }
    }

    pub fn exp(&self) -> Self{
        let ea = self.real.exp();
        let (st,ct) = self.imag.sin_cos();

        Self { real: ea*ct, imag: ea*st }
    }
    pub fn sin(&self) -> Self{
        Self::new(self.real.sin()*self.imag.cosh(), self.real.cos()*self.imag.sinh())
    }

    pub fn cos(&self) -> Self{
        Self::new(self.real.cos()*self.imag.cosh(), -self.real.sin()*self.imag.sinh())
    }

    pub fn tan(&self) -> Self{
        Self::new(self.real.tan(), self.imag.tanh())/Self::new(T::one(),-self.real.tan()*self.imag.tan())
    }

    pub fn sinh(&self) -> Self{
        Self::new(self.real.sinh()*self.imag.cos(),self.real.cosh()*self.imag.sin())
    }

    pub fn cosh(&self) -> Self{
        Self::new(self.real.cosh()*self.imag.cos(),self.real.sinh()*self.imag.sin())
    }

    pub fn tanh(&self) -> Self{
        Self::new(-self.imag.tan(), self.real.tanh())/Self::new(self.real.tan()*self.imag.tan(),T::one())
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

impl <T:Float> Debug for Complex<T>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} + {:?}i",self.real, self.imag)
    }
}
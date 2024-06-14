//the result of doing numerical algebra in polar space, the addition penalty is too significant

use std::ops::{Add, Div, Mul, Neg, Sub};

use super::float::Float;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Complex<T: Float> {
    pub magnitude: T,
    pub phase: T,
}

impl<T: Float> Complex<T> {
    pub fn zero() -> Self {
        Self {
            magnitude: T::zero(),
            phase: T::zero(),
        }
    }
    pub fn new(real: T, imag: T) -> Self {
        let magnitude = (real*real+imag*imag).sqrt();
        let phase = imag.atan2(real);
        Self { magnitude, phase }
    }

    pub fn real(&self)->T{
        let (st,ct) = self.phase.sin_cos();
        self.magnitude*ct
    }

    pub fn imag(&self)->T{
        let (st,ct) = self.phase.sin_cos();
        self.magnitude*st
    }

    pub fn components(&self) -> (T,T){
        let (st,ct) = self.phase.sin_cos();
        (self.magnitude*ct,self.magnitude*st)
    }
    pub fn new_phase(magnitude: T, phase: T) -> Self {
        Self { magnitude, phase }
    }
    pub fn mul_i(&self) -> Self{
        Self { magnitude: self.magnitude, phase: self.phase+T::pi()/T::usize(2) }
    }
    pub fn mul_ni(&self) -> Self{
        Self { magnitude: self.magnitude, phase: self.phase-T::pi()/T::usize(2) }
    }
    pub fn conj(&self) -> Self {
        Self {
            magnitude: self.magnitude,
            phase: -self.phase,
        }
    }
    pub fn norm(&self) -> T {
        self.magnitude
    }
    pub fn square_norm(&self) -> T {
        self.magnitude*self.magnitude
    }
    pub fn recip(&self) -> Self {
        self.conj() / self.square_norm()
    }
}

impl<T: Float> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            magnitude: self.magnitude,
            phase: self.phase+T::pi()
        }
    }
}

impl<T: Float> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        //is this the most effective way to do this?
        let (self_real,self_imag) = self.components();
        let (other_real, other_imag) = rhs.components();
        Self::new(self_real+other_real, self_imag+other_imag)
    }
}
impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        //is this the most effective way to do this?
        let (self_real,self_imag) = self.components();
        let (other_real, other_imag) = rhs.components();
        Self::new(self_real-other_real, self_imag-other_imag)
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self{
            magnitude: self.magnitude*rhs.magnitude,
            phase: self.phase+rhs.phase,
        }
    }
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self, scaler: T) -> Self {
        Self { magnitude: self.magnitude*scaler, phase: self.phase }
    }
}

impl<T: Float> Div for Complex<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self{
            magnitude: self.magnitude/rhs.magnitude,
            phase: self.phase-rhs.phase,
        }
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;
    fn div(self, scaler: T) -> Self {
        Self { magnitude: self.magnitude/scaler, phase: self.phase }
    }
}

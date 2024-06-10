use std::ops::{Add, Div, Mul, Neg, Sub};

use super::float::Float;

#[derive(Copy, Clone, Debug, PartialEq)]
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
    pub fn recip(&self) -> Self {
        self.conj() / self.square_norm()
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
        Self::new(self.real + rhs.real, self.imag + rhs.imag)
    }
}
impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.real - rhs.real, self.imag - rhs.imag)
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real * rhs.real - self.imag * rhs.imag,
            self.real * rhs.imag + self.imag * rhs.real,
        )
    }
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self, scaler: T) -> Self {
        Self::new(self.real * scaler, self.imag * scaler)
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
        Self::new(self.real / scaler, self.imag / scaler)
    }
}

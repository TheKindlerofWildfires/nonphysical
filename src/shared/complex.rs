use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy)]
pub struct Complex64 {
    pub real: f32,
    pub imag: f32,
}
impl Complex64 {
    pub fn new(real: f32, imag: f32) -> Self {
        Complex64 { real, imag }
    }
    pub fn new_phase(magnitude: f32, phase: f32) -> Self {
        let (st, ct) = phase.sin_cos();
        Complex64 {
            real: ct * magnitude,
            imag: st * magnitude,
        }
    }
    pub fn swap(&self) -> Self {
        Complex64 {
            real: self.imag,
            imag: self.real,
        }
    }
    pub fn conj(&self) -> Self {
        Complex64 {
            real: self.real,
            imag: -self.imag,
        }
    }
}
impl Add for Complex64 {
    fn add(self, rhs: Self) -> Self::Output {
        Complex64::new(self.real + rhs.real, self.imag + rhs.imag)
    }
    type Output = Complex64;
}
impl Sub for Complex64 {
    fn sub(self, rhs: Self) -> Self::Output {
        Complex64::new(self.real - rhs.real, self.imag - rhs.imag)
    }
    type Output = Complex64;
}

impl Mul for Complex64 {
    fn mul(self, rhs: Self) -> Self::Output {
        Complex64::new(
            self.real * rhs.real - self.imag * rhs.imag,
            self.real * rhs.imag + self.imag * rhs.real,
        )
    }
    type Output = Complex64;
}

impl Mul<f32> for Complex64 {
    type Output = Self;
    fn mul(self, scaler: f32) -> Self {
        Complex64 {
            real: self.real * scaler,
            imag: self.imag * scaler,
        }
    }
}

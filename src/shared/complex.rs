use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Float: Neg<Output = Self>+Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Clone+ Copy + Sized {
    //constants
    fn pi() -> Self;
    fn zero() -> Self;
    //conversion functions
    fn usize(u:usize) -> Self;
    fn float(f:f32) ->Self;
    fn double(f:f64) -> Self;

    //utility
    fn sin_cos(&self) -> (Self,Self);
    fn exp(&self) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn sin_cos(&self) -> (Self,Self) {
        (*self as f32).sin_cos()
    }
    #[inline(always)]
    fn pi() -> Self{
        std::f32::consts::PI
    }
    #[inline(always)]
    fn usize(u:usize) -> Self {
        u as f32
    }
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn exp(&self) -> Self {
        self.exp2()
    }

    #[inline(always)]
    fn float(f:f32) -> Self {
        f
    }

    #[inline(always)]
    fn double(f:f64) -> Self {
        f as f32
    }


}
impl Float for f64 {
    #[inline(always)]
    fn sin_cos(&self) -> (Self,Self) {
        (*self as f64).sin_cos()
    }
    #[inline(always)]
    fn pi() -> Self{
        std::f64::consts::PI
    }
    #[inline(always)]
    fn usize(u:usize) -> Self {
        u as f64
    }
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn exp(&self) -> Self {
        self.exp2()
    }

    #[inline(always)]
    fn float(f:f32) -> Self {
        f as f64
    }

    #[inline(always)]
    fn double(f:f64) -> Self {
        f
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Complex<T: Float> {
    pub real: T,
    pub imag: T,
}

impl<T: Float> Complex<T> {
    pub fn new(real: T, imag: T) -> Self {
        Self { real, imag }
    }
    pub fn new_phase(magnitude: T, phase: T) -> Self {
        let (st, ct) = phase.sin_cos();
        Self {
            real: ct * magnitude,
            imag: st * magnitude,
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

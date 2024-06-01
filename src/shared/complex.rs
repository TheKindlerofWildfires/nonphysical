use std::ops::{Add, Sub};


#[derive(Debug,Clone,Copy)]
pub struct Complex64{
    pub real: f32,
    pub imaginary: f32,
}
impl Complex64{
    pub fn new(real:f32,imaginary:f32) -> Self{
        Complex64{real,imaginary}
    }
    pub fn new_phase(magnitude: f32, phase: f32) -> Self{

        let (st,ct) = phase.sin_cos();
        Complex64{
            real: ct*magnitude,
            imaginary: st*magnitude,
        }
    }
    pub fn swap(&self) -> Self{
        Complex64{
            real:self.imaginary,
            imaginary: self.real,
        }
    }
    pub fn conj(&self) -> Self{
        Complex64{
            real: self.real,
            imaginary: -self.imaginary,
        }
    }
}
impl Add for Complex64{
    fn add(self, rhs: Self) -> Self::Output {
        Complex64::new(self.real+rhs.real, self.imaginary+rhs.imaginary)
    }
    type Output = Complex64;
}
impl Sub for Complex64{
    fn sub(self, rhs: Self) -> Self::Output {
        Complex64::new(self.real-rhs.real, self.imaginary-rhs.imaginary)
    }
    type Output = Complex64;
}
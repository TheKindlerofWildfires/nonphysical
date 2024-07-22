use core::ops::Mul;
use std::ops::Range;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix};

pub struct Jacobi<T: Float> {
    pub s: Complex<T>,
    pub c: Complex<T>,
}

impl<'a, T: Float + 'a> Jacobi<T> {
    pub fn new(s: Complex<T>, c: Complex<T>) -> Self {
        Self { s, c }
    }

    pub fn make_jacobi(matrix: &mut Matrix<T>, p: usize, q: usize) -> Self {
        let x = matrix.coeff(p, p).real;
        let y = matrix.coeff(p, q);
        let z = matrix.coeff(q, q).real;
        let denominator = T::usize(2) * y.norm();
        match denominator < T::EPSILON {
            true => Self::new(Complex::<T>::ONE, Complex::<T>::ZERO),
            false => {
                let tau = (x - z) / denominator;
                let w = (tau.square_norm() + T::ONE).sqrt();
                let t = match tau > T::ZERO {
                    true => (tau + w).recip(),
                    false => (tau - w).recip(),
                };
                let sign = match t > T::ZERO {
                    true => T::ONE,
                    false => -T::ONE,
                };
                let n = (t.square_norm() + T::ONE).sqrt().recip();
                let s: Complex<T> = -(y.conj() / y.norm()) * sign * t.norm() * n;

                Self::new(s, Complex::<T>::new(n, T::ZERO))
            }
        }
    }
    //not complex friendly
    pub fn make_givens(p: Complex<T>, q: Complex<T>, r: &mut Complex<T>) -> Self {
        let (c, s) = if q == Complex::ZERO {
            let c = Complex::new(-p.real.sign(),T::ZERO);
            let s = Complex::ZERO;
            *r = c * p;
            (c, s)
        } else if p == Complex::ZERO {
            let s = Complex::new(-q.real.sign(),T::ZERO);
            let c = Complex::ZERO;
            *r = s * q;
            (c, s)
        } else {
            let p1: T = p.norm();
            let q1 = q.norm();
            let (c, s) = if p1 > q1 {
                let ps = p / p1;
                let p2 = ps.square_norm(); //probably 1
                let qs = q / p1;
                let q2 = qs.square_norm();

                let mut u = (T::ONE + q2 / p2).sqrt();
                if p.real < T::ZERO {
                    u = -u;
                }
                let c = Complex::new(u.recip(), T::ZERO);
                let s = -qs * ps.conj() * (c / p2);
                *r = p * u;
                (c, s)
            } else {
                let ps = p / q1;
                let p2 = ps.square_norm();
                let qs = q / q1;
                let q2 = qs.square_norm();

                let mut u = q1 * (p2 + q2).sqrt();
                if p.real < T::ZERO {
                    u = -u;
                }
                let p1 = p.norm();
                let ps = p / p1;
                let c = Complex::new(p1 / u, T::ZERO);
                let s = -ps.conj() * (q / u);
                *r = ps * u;
                (c, s)
            };
            (c, s)
        };
        Self { s, c }
    }

    pub fn apply_left(&self, matrix: &mut Matrix<T>, p: usize, q: usize, range: Range<usize>) {
        if self.c == Complex::<T>::ONE && self.s == Complex::<T>::ZERO {
            return;
        }
        range.for_each(|i| {
            let tmp_p = matrix.coeff(i, p);
            let tmp_q = matrix.coeff(i, q); 
            *matrix.coeff_ref(i, p) = self.c.fma(tmp_p,self.s.conj() * tmp_q);
            *matrix.coeff_ref(i, q) = (-self.s).fma(tmp_p,self.c.conj() * tmp_q);
        });
    }

    pub fn apply_right(&self, matrix: &mut Matrix<T>, p: usize, q: usize, range: Range<usize>) {
        if self.c == Complex::<T>::ONE && self.s == Complex::<T>::ZERO {
            return;
        }
        range.for_each(|i| {
            let tmp_p = matrix.coeff(p,i);
            let tmp_q = matrix.coeff(q,i); 
            *matrix.coeff_ref(p, i) = self.c.fma(tmp_p, self.s.conj() * tmp_q);
            *matrix.coeff_ref(q, i) =  (-self.s).fma(tmp_p, self.c.conj() * tmp_q);

        });
    }

    pub fn transpose(&self) -> Self {
        Self {
            s: -self.s.conj(),
            c: self.c,
        }
    }

    pub fn adjoint(&self) -> Self {
        Self {
            s: -self.s,
            c: self.c.conj(),
        }
    }
}

impl<T: Float> Mul for Jacobi<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let c = self.c * rhs.c - self.s.conj() * rhs.s;
        let s = (self.c * rhs.s.conj() + self.s.conj() * rhs.c.conj()).conj();
        Self { c, s }
    }
}

#[cfg(test)]
mod jacobi_tests {
    use super::*;

    #[test]
    fn test_left_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(-9.96223e-7, 0.0),
            Complex::new(-2.8054e-7, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.76837e-7, 0.0),
            Complex::new(1.96297e-7, 0.0),
            Complex::new(3.40572e-7, 0.0),
        ];
        let mut m = Matrix::new(4, data);

        let jacobi = Jacobi {
            c: Complex::new(-7.964988e-8, 0.0),
            s: Complex::new(1.0, 0.0),
        };
        jacobi.apply_left(&mut m, 1, 0,0..4);
        let known_data = vec![
            Complex::new(3.74166, 0.0),
            Complex::new(2.9802277e-7, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-29.999999, 0.0),
            Complex::new(-14.966603, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.44949, 0.0),
            Complex::new(1.9510159e-7, 0.0),
            Complex::new(-9.96223e-7, 0.0),
            Complex::new(-2.8054e-7, 0.0),
            Complex::new(-4.76837e-7, 0.0),
            Complex::new(-3.7980009829560004e-14, 0.0),
            Complex::new(1.96297e-7, 0.0),
            Complex::new(3.40572e-7, 0.0),
        ];
        known_data
            .iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((*k - *c).square_norm() < f32::EPSILON));
    }

    #[test]
    fn test_right_1() {
        let data = vec![
            Complex::new(3.74166, 0.0),
            Complex::new(2.9802277e-7, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-29.999998, 0.0),
            Complex::new(-14.966603, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.44949, 0.0),
            Complex::new(1.9510159e-7, 0.0),
            Complex::new(-9.96223e-7, 0.0),
            Complex::new(-2.8054e-7, 0.0),
            Complex::new(-4.76837e-7, 0.0),
            Complex::new(-3.798001e-14, 0.0),
            Complex::new(1.96297e-7, 0.0),
            Complex::new(3.40572e-7, 0.0),
        ];
        let mut m = Matrix::new(4, data);

        let jacobi = Jacobi {
            c: Complex::new(-7.964988e-8, 0.0),
            s: Complex::new(1.0, 0.0),
        };
        jacobi.apply_right(&mut m, 1, 0,0..4);
        let known_data = vec![
            Complex::new(29.999998, 0.0),
            Complex::new(14.966603, 0.0),
            Complex::new(9.79796, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.7416625, 0.0),
            Complex::new(1.4901109e-6, 0.0),
            Complex::new(7.8040637e-7, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.44949, 0.0),
            Complex::new(1.9510159e-7, 0.0),
            Complex::new(-9.96223e-7, 0.0),
            Complex::new(-2.8054e-7, 0.0),
            Complex::new(-4.76837e-7, 0.0),
            Complex::new(-3.798001e-14, 0.0),
            Complex::new(1.96297e-7, 0.0),
            Complex::new(3.40572e-7, 0.0),
        ];
        known_data
            .iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((*k - *c).square_norm() < f32::EPSILON));
    }
}

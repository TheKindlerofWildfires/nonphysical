use std::ops::Mul;

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
        let y = matrix.coeff(q, p);
        let z = matrix.coeff(q, q).real;
        let deno = T::usize(2) * y.norm();
        match deno < T::small() {
            true => Self::new(
                Complex::<T>::new(T::usize(1), T::zero()),
                Complex::<T>::new(T::zero(), T::zero()),
            ),
            false => {
                let tau = (x - z) / deno;
                let w = (tau.square_norm() + T::usize(1)).sqrt();
                let t = match tau > T::zero() {
                    true => T::usize(1) / (tau + w),
                    false => T::usize(1) / (tau - w),
                };
                let sign = match t > T::zero() {
                    true => T::usize(1),
                    false => T::isize(-1),
                };
                let n = (t.square_norm() + T::usize(1)).sqrt().recip();
                let s: Complex<T> = -(y.conj() / y.norm()) * sign * t.norm() * n;

                Self::new(s, Complex::<T>::new(n, T::zero()))
            }
        }
    }

    //this method does n extra copy operations to be safe incase p == q or there is a bug in data_row
    pub fn apply_left(&self, matrix: &mut Matrix<T>, p: usize, q: usize) {
        //safety check
        let j = self;
        if j.c == Complex::<T>::new(T::usize(1), T::zero()) && j.s == Complex::<T>::zero() {
            return;
        }
        
        //Get a temp copy of x and y
        let x: Vec<_> = matrix.data_row(p).map(|c| c.clone()).collect();
        let y: Vec<_> = matrix.data_row(q).map(|c| c.clone()).collect();

        //apply the transform to a mutable copy of x
        let xm = matrix.data_row_ref(p);
        xm.zip(y).for_each(|(xp, yp)| {
            *xp = j.c * *xp + j.s.conj() * yp;
            
        });

        //apply the transform to a mutable copy of y
        let ym = matrix.data_row_ref(q);
        ym.zip(x).for_each(|(yp, xp)| {
            *yp = -j.s * xp + j.c.conj() * *yp;
            
        });
    }

    //this method does n extra copy operations to be safe incase p == q or there is a bug in data_column
    pub fn apply_right(&self, matrix: &mut Matrix<T>, p: usize, q: usize) {
        //safety check
        let j = self.transpose();
        if j.c == Complex::<T>::new(T::usize(1), T::zero()) && j.s == Complex::<T>::zero() {
            return;
        }
        //Get a temp copy of x and y
        let x: Vec<_> = matrix.data_column(p).map(|c| c.clone()).collect();
        let y: Vec<_> = matrix.data_column(q).map(|c| c.clone()).collect();
        //apply the transform to a mutable copy of x
        let xm = matrix.data_column_ref(p);
        xm.zip(y).for_each(|(xp, yp)| {
            *xp = j.c * *xp + j.s.conj() * yp;
        });

        //apply the transform to a mutable copy of y
        let ym = matrix.data_column_ref(q);
        ym.zip(x).for_each(|(yp, xp)| {
            *yp = -j.s * xp + j.c.conj() * *yp;
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

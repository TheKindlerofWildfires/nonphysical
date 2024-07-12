use core::ops::Mul;

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
        match denominator < T::SMALL {
            true => Self::new(
                Complex::<T>::ONE,
                Complex::<T>::ZERO,
            ),
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

    //this method does n extra copy operations to be safe incase p == q or there is a bug in data_row
    pub fn apply_left(&self, matrix: &mut Matrix<T>, p: usize, q: usize) {
        //safety check could be removed to reduce branching if necessary 
        let j = self;
        if j.c == Complex::<T>::new(T::ONE, T::ZERO) && j.s == Complex::<T>::ZERO {
            return;
        }
        matrix.data_rows_ref().for_each(|row| {
            let tmp = row[p];
            row[p] = self.c * row[p] + self.s.conj() * row[q];
            row[q] = -self.s * tmp + self.c.conj() * row[q];
        });
        
    }

    pub fn apply_right(&self, matrix: &mut Matrix<T>, p: usize, q: usize) {
        //safety check could be removed to reduce branching if necessary 
        let j = self.transpose();
        if j.c == Complex::<T>::ONE && j.s == Complex::<T>::ZERO {
            return;
        }
        let mut rows = matrix.data_rows_ref();
        let (row_p, row_q) = match q>p{
            true => {
                let rp = rows.nth(p).unwrap();
                let rq = rows.nth(q-p-1).unwrap();
                (rp, rq)
            },
            false => {
                let rq = rows.nth(q).unwrap();
                let rp = rows.nth(p-q-1).unwrap();
 
                (rp, rq)
            }
        };
        
        row_p.iter_mut().zip(row_q.iter_mut()).for_each(|(pp,qp)|{
            let tmp = *pp;
            *pp = j.c* *pp +j.s.conj() * *qp; 
            *qp = -j.s*tmp +j.c.conj() * *qp;
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
    fn make_jacobi_static() {todo!()}

    #[test]
    fn apply_left_static() {todo!()}

    #[test]
    fn apply_right_static() {todo!()}
}
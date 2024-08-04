use core::ops::Range;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix, real::Real};

pub struct RealJacobi<R: Real> {
    pub s: R,
    pub c: R,
}
pub struct ComplexJacobi<C:Complex>{
    pub s: C,
    pub c: C,
}

pub trait Jacobian<F:Float>{
    fn new(s: F, c: F) -> Self;
    fn make_jacobi(matrix: &mut Matrix<F>, p: usize, q: usize) -> Self;
    fn make_givens(p: F, q: F, r: &mut F) -> Self;
    fn apply_left(&self, matrix: &mut Matrix<F>, p: usize, q: usize, range: Range<usize>);
    fn apply_right(&self, matrix: &mut Matrix<F>, p: usize, q: usize, range: Range<usize>);
    fn transpose(&self) -> Self;
    fn adjoint(&self) -> Self;
    fn dot(&self, other:Self)->Self;

}

impl<R:Real<Primitive=R>> Jacobian<R> for RealJacobi<R>{
    fn new(s: R, c: R) -> Self {
        Self{s,c}
    }

    fn make_jacobi(matrix: &mut Matrix<R>, p: usize, q: usize) -> Self {
        let x = matrix.coeff(p, p);
        let y = matrix.coeff(p, q);
        let z = matrix.coeff(q, q);
        let denominator = R::usize(2) * y.l1_norm();
        match denominator < R::EPSILON {
            true => Self{c:R::ONE, s:R::ZERO},
            false => {
                let tau = (x - z) / denominator;
                let w = (tau.l2_norm() + R::Primitive::ONE).sqrt();
                let t = match tau > R::ZERO {
                    true => (tau + w).recip(),
                    false => (tau - w).recip(),
                };
                let sign = match t > R::ZERO {
                    true => R::ONE,
                    false => R::NEGATIVE_ONE,
                };
                let n = (t.l2_norm() +  R::Primitive::ONE).sqrt().recip();
                let s = -sign*(y/y.l1_norm())*t.l1_norm()*n;
                let  c = n;
                Self::new(s, c)
            }
        }
        
    }

    //Undertested
    fn make_givens(p: R, q: R, r: &mut R) -> Self {
        let (c, s) = if q == R::ZERO {
            let c = -p.sign();
            let s = R::ZERO;
            *r = c * p;
            (c, s)
        } else if p == R::ZERO {
            let s = -q.sign();
            let c = R::ZERO;
            *r = s * q;
            (c, s)
        } else {
            let p1  = p.l1_norm();
            let q1 = q.l1_norm();
            let (c, s) = if p1 > q1 {
                let ps = p / p1;
                let p2 = ps.l2_norm(); //probably 1
                let qs = q / p1;
                let q2 = qs.l2_norm();

                let mut u = (R::ONE + q2 / p2).sqrt();
                if p < R::ZERO {
                    u = -u;
                }
                let c = u.recip();
                let s = -qs * ps * (c / p2);
                *r = p * u;
                (c, s)
            } else {
                let ps = p / q1;
                let p2 = ps.l2_norm();
                let qs = q / q1;
                let q2 = qs.l2_norm();

                let mut u = q1 * (p2 + q2).sqrt();
                if p < R::ZERO {
                    u = -u;
                }
                let p1 = p.l1_norm();
                let ps = p / p1;
                let c = p1 / u;
                let s = -ps * (q / u);
                *r = ps * u;
                (c, s)
            };
            (c, s)
        };
        Self { s, c }
    }

    fn apply_left(&self, matrix: &mut Matrix<R>, p: usize, q: usize, range: Range<usize>) {
        if self.c == R::ONE && self.s == R::ZERO {
            return;
        }
        range.for_each(|i| {
            let tmp_p = matrix.coeff(i, p);
            let tmp_q = matrix.coeff(i, q); 
            *matrix.coeff_ref(i, p) = self.c.fma(tmp_p,self.s * tmp_q);
            *matrix.coeff_ref(i, q) = (-self.s).fma(tmp_p,self.c * tmp_q);
        });
    }

    fn apply_right(&self, matrix: &mut Matrix<R>, p: usize, q: usize, range: Range<usize>) {
        if self.c == R::ONE && self.s == R::ZERO {
            return;
        }
        range.for_each(|i| {
            let tmp_p = matrix.coeff(p,i);
            let tmp_q = matrix.coeff(q,i); 
            *matrix.coeff_ref(p, i) = self.c.fma(tmp_p, self.s * tmp_q);
            *matrix.coeff_ref(q, i) =  (-self.s).fma(tmp_p, self.c* tmp_q);

        });
    }
    
    fn transpose(&self) -> Self {
        Self {
            s: -self.s,
            c: self.c,
        }
    }
    
    fn adjoint(&self) -> Self {
        Self {
            s: -self.s,
            c: self.c,
        }
    }

    fn dot(&self, other: Self) -> Self {
        let c = self.c.fma(other.c, -self.s * other.s);
        let s = self.c.fma( other.s, self.s * other.c);
        Self { c, s }
    }
}

impl<R:Real<Primitive = R>,C:Complex<Primitive=R>> Jacobian<C> for ComplexJacobi<C>{
    fn new(s: C, c: C) -> Self {
        Self{s,c}
    }

    fn make_jacobi(matrix: &mut Matrix<C>, p: usize, q: usize) -> Self {
        let x = matrix.coeff(p, p).real();
        let y = matrix.coeff(p, q);
        let z = matrix.coeff(q, q).real();
        let denominator = C::Primitive::usize(2) * y.l1_norm();
        match denominator < C::Primitive::EPSILON {
            true => Self{c:C::ONE, s:C::ZERO},
            false => {
                let tau = (x - z) / denominator;
                let w = (tau.l2_norm() + C::Primitive::ONE).sqrt();
                let t = match tau > C::Primitive::ZERO {
                    true => (tau + w).recip(),
                    false => (tau - w).recip(),
                };
                let sign = match t > C::Primitive::ZERO {
                    true => C::Primitive::ONE,
                    false => C::Primitive::NEGATIVE_ONE,
                };
                let n = (t.l2_norm() + C::Primitive::ONE).sqrt().recip();
                let s = -(y.conjugate()/y.l1_norm())*t.l1_norm()*n*sign;
                let c = C::new(n,C::Primitive::ZERO);


                Self::new(s, c)
            }
        }
    }

    //Undertested
    fn make_givens(p: C, q: C, r: &mut C) -> Self {
        let (c, s) = if q == C::ZERO {
            let c = C::new(-p.real().sign(),C::Primitive::ZERO);
            let s = C::ZERO;
            *r = c * p;
            (c, s)
        } else if p == C::ZERO {
            let s = C::new(-q.real().sign(),C::Primitive::ZERO);
            let c = C::ZERO;
            *r = s * q;
            (c, s)
        } else {
            let p1  = p.l1_norm();
            let q1 = q.l1_norm();
            let (c, s) = if p1 > q1 {
                let ps = p / p1;
                let p2 = ps.l2_norm(); //probably 1
                let qs = q / p1;
                let q2 = qs.l2_norm();

                let mut u = (C::Primitive::ONE + q2 / p2).sqrt();
                if p.real() < C::Primitive::ZERO {
                    u = -u;
                }
                let c = C::new(u.recip(), C::Primitive::ZERO);
                let s = -qs * ps.conjugate() * (c / p2);
                *r = p * u;
                (c, s)
            } else {
                let ps = p / q1;
                let p2 = ps.l2_norm();
                let qs = q / q1;
                let q2 = qs.l2_norm();

                let mut u = q1 * (p2 + q2).sqrt();
                if p.real() < C::Primitive::ZERO {
                    u = -u;
                }
                let p1 = p.l1_norm();
                let ps = p / p1;
                let c = C::new(p1 / u, C::Primitive::ZERO);
                let s = -ps.conjugate() * (q / u);
                *r = ps * u;
                (c, s)
            };
            (c, s)
        };
        Self { s, c }
    }
    

    fn apply_left(&self, matrix: &mut Matrix<C>, p: usize, q: usize, range: Range<usize>) {
        if self.c == C::ONE && self.s == C::ZERO {
            return;
        }
        range.for_each(|i| {
            let tmp_p = matrix.coeff(i, p);
            let tmp_q = matrix.coeff(i, q); 
            *matrix.coeff_ref(i, p) = self.c.fma(tmp_p,self.s.conjugate() * tmp_q);
            *matrix.coeff_ref(i, q) = (-self.s).fma(tmp_p,self.c.conjugate() * tmp_q);
        });
    }

    fn apply_right(&self, matrix: &mut Matrix<C>, p: usize, q: usize, range: Range<usize>) {
        if self.c == C::ONE && self.s == C::ZERO {
            return;
        }
        range.for_each(|i| {
            let tmp_p = matrix.coeff(p,i);
            let tmp_q = matrix.coeff(q,i); 
            *matrix.coeff_ref(p, i) = self.c.fma(tmp_p, self.s.conjugate() * tmp_q);
            *matrix.coeff_ref(q, i) =  (-self.s).fma(tmp_p, self.c.conjugate() * tmp_q);

        });
    }
    
    fn transpose(&self) -> Self {
        Self {
            s: -self.s.conjugate(),
            c: self.c,
        }
    }
    
    fn adjoint(&self) -> Self {
        Self {
            s: -self.s,
            c: self.c.conjugate(),
        }
    }

    fn dot(&self, other: Self) -> Self {
        let c = self.c * other.c - self.s.conjugate() * other.s;
        let s = (self.c * other.s.conjugate() + self.s.conjugate() * other.c.conjugate()).conjugate();
        Self { c, s }
    }
}
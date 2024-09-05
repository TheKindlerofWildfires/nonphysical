use core::cmp::min;

use crate::{
    linear::{hessenberg::Hessenberg, jacobi::Jacobi},
    shared::{complex::Complex, float::Float, matrix::Matrix},
};

pub trait Schur<T: Float> {
    fn schur(&mut self) -> Self;

    fn reduce_triangular(t: &mut Self, u: &mut Self);

    fn small_sub_diagonal(t: &mut Matrix<T>, i: usize) -> bool;

    fn compute_shift(t: &mut Matrix<T>, i: usize, iter: usize) -> Complex<T>;
}
impl<T: Float> Schur<T> for Matrix<T> {
    fn schur(&mut self) -> Self {
        let h_coefficients = <Matrix<T> as Hessenberg<T>>::hessenberg(self);
        let mut u = <Matrix<T> as Hessenberg<T>>::sequence(self, &h_coefficients);
        Self::reduce_triangular(self, &mut u);
        todo!()
    }
    fn reduce_triangular(t: &mut Self, u: &mut Self) {
        let max_iter = 100;
        let mut iu = t.rows - 1;
        let mut iter = 0;
        let mut total_iter = 0;
        loop {
            while iu > 0 {
                if !Self::small_sub_diagonal(t, iu - 1) {
                    break;
                }
                iter = 0;
                iu -= 1;
            }
            if iu == 0 {
                break;
            }
            iter += 1;
            total_iter += 1;
            if total_iter > max_iter {
                break;
            }
            let mut il = iu - 1;
            while il > 0 && !Self::small_sub_diagonal(t, il - 1) {
                il -= 1;
            }

            let shift = Self::compute_shift(t, iu, iter);

            let mut temp = Complex::ZERO;
            let rot = Jacobi::make_givens(t.coeff(il, il) - shift, t.coeff(il, il + 1), &mut temp);

            rot.apply_left(t, il + 1, il, il..t.rows);
            let edge = min(il+2,iu)+1;
            rot.apply_right(t, il + 1, il, 0..edge);
            rot.apply_right(u, il + 1, il, 0..t.rows);
            
            for i in il + 1..iu {
                let rot = Jacobi::make_givens(
                    t.coeff(i - 1, i),
                    t.coeff(i - 1, i + 1),
                    t.coeff_ref(i - 1, i),
                );
                *t.coeff_ref(i - 1, i + 1) = Complex::ZERO;
                rot.apply_left(t, i + 1, i, i..t.rows); //can introduce bugs
                let edge = min(i+2,iu)+1;
                rot.apply_right(t, i + 1, i, 0..edge); //can introduce bugs
                rot.apply_right(u, i + 1, i, 0..t.rows);
            }
        }
    }

    fn small_sub_diagonal(t: &mut Matrix<T>, i: usize) -> bool {
        let d = t.coeff(i, i).norm() + t.coeff(i + 1, i + 1).norm();
        let sd = t.coeff(i, i + 1).norm();
        if sd.norm() <= d.norm() * T::EPSILON {
            *t.coeff_ref(i, i + 1) = Complex::ZERO;
            true
        } else {
            false
        }
    }

    fn compute_shift(t: &mut Matrix<T>, i: usize, iter: usize) -> Complex<T> {
        if iter == 10 || iter == 20 {
            Complex::new(
                t.coeff(i, i - 1).real.norm() + t.coeff(i - 1, i - 2).real.norm(),
                T::ZERO,
            )
        } else {
            let mut t00 = t.coeff(i - 1, i - 1);
            let mut t01 = t.coeff(i - 1, i);
            let mut t10 = t.coeff(i, i - 1);
            let mut t11 = t.coeff(i, i);

            let norm_t = t00.norm() + t01.norm() + t10.norm() + t11.norm();
            t00 /= norm_t;
            t01 /= norm_t;
            t10 /= norm_t;
            t11 /= norm_t;
            let b = t01 * t10;
            let c = t00 - t11;
            let disc = (c.fma(c,b*T::usize(4))).sqrt();
            let det = t00.fma(t11,-b);
            let trace = t00 + t11;
            let mut eival1 = (trace + disc) / T::usize(2);
            let mut eival2 = (trace - disc) / T::usize(2);

            if eival1.norm() > eival2.norm() {
                eival2 = det / eival1;
            } else {
                eival1 = det / eival2;
            }

            if (eival1 - t11).norm() < (eival2 - t11).norm() {
                eival1 * norm_t
            } else {
                eival2 * norm_t
            }
        }
    }
}

#[cfg(test)]
mod schur_tests {
    use super::*;

    #[test]
    fn pre_work_3x3(){
        let mut m = Matrix::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Hessenberg<f32>>::hessenberg(&mut m);
        let u = <Matrix<f32> as Hessenberg<f32>>::sequence(&mut m, &coefficients);
        let known_coefficients = vec![Complex::new(1.44721,0.0), Complex::new(0.0,0.0)];
        coefficients.iter().zip(known_coefficients.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        let known_m = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-2.23607, 0.0),
            Complex::new(0.61803395, 0.0),
            Complex::new(-6.7082, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
        ];
        m.data().zip(known_m.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });

        let known_u = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.447214, 0.0),
            Complex::new(-0.894427, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.894427, 0.0),
            Complex::new(0.447214, 0.0),
        ];
        u.data().zip(known_u.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
    #[test]
    fn pre_work_4x4(){
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Hessenberg<f32>>::hessenberg(&mut m);
        let u = <Matrix<f32> as Hessenberg<f32>>::sequence(&mut m, &coefficients);
        let known_coefficients = vec![Complex::new(1.26726,0.0), Complex::new(1.14995,0.0), Complex::new(0.0,0.0)];
        coefficients.iter().zip(known_coefficients.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        let known_m = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        m.data().zip(known_m.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });

        let known_u = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.267261, 0.0),
            Complex::new(-0.534522, 0.0),
            Complex::new(-0.801784, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.872871, 0.0),
            Complex::new(0.218218, 0.0),
            Complex::new(-0.436436, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.408248, 0.0),
            Complex::new(-0.816497, 0.0),
            Complex::new(0.408248, 0.0),
        ];
        u.data().zip(known_u.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }

    #[test]
    fn pre_work_5x5(){
        let mut m = Matrix::new(5, (0..25).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Hessenberg<f32>>::hessenberg(&mut m);
        let u = <Matrix<f32> as Hessenberg<f32>>::sequence(&mut m, &coefficients);
        let known_coefficients =  vec![Complex::new(1.18257,0.0), Complex::new(1.15614,0.0), Complex::new(1.13359,0.0), Complex::new(0.0,0.0)];
        coefficients.iter().zip(known_coefficients.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        let known_m = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(-27.3861, 0.0),
            Complex::new(60.0, 0.0),
            Complex::new(22.3607, 0.0),
            Complex::new(-0.327098, 0.0),
            Complex::new(-0.789245, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.47213, 0.0),
            Complex::new(-1.19158e-06, 0.0),
            Complex::new(-2.17911e-07, 0.0),
            Complex::new(-0.874244, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-7.2217e-07, 0.0),
            Complex::new(-1.39336e-07, 0.0),
            Complex::new(-2.62754e-07, 0.0),
            Complex::new(4.07865e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.78137e-06, 0.0),
            Complex::new(-1.01109e-06, 0.0),
            Complex::new(7.8637e-08, 0.0),
            Complex::new(-4.53019e-07, 0.0),
        ];
        m.data().zip(known_m.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });

        let known_u = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.182574, 0.0),
            Complex::new(-0.365148, 0.0),
            Complex::new(-0.547723, 0.0),
            Complex::new(-0.730297, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.816496, 0.0),
            Complex::new(-0.408248, 0.0),
            Complex::new(-5.96046e-08, 0.0),
            Complex::new(0.408248, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.317271, 0.0),
            Complex::new(0.75581, 0.0),
            Complex::new(-0.559808, 0.0),
            Complex::new(0.121268, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.446474, 0.0),
            Complex::new(0.358819, 0.0),
            Complex::new(0.621784, 0.0),
            Complex::new(-0.534129, 0.0),
        ];
        u.data().zip(known_u.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
    #[test]
    fn triangular_3x3(){
        let data_m = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        let data_t = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        let mut m = Matrix::new(3, data_m);
        let mut t = Matrix::new(3, data_t);
        //this is closer to right, but bugs still emerge when passing through schur
        <Matrix<f32> as Schur<f32>>::reduce_triangular(&mut m, &mut t);
    }
    #[test]
    fn basic_reduce_triangular() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data).transposed();
        let mut t = Matrix::zero(4, 4);
        //this is closer to right, but bugs still emerge when passing through schur
        <Matrix<f32> as Schur<f32>>::reduce_triangular(&mut m, &mut t);
        //plan: Pick a low issue matrix and go back through, make sure everything is right right (pick one at 3x3, 4x4 and 5x5 )
    }
    #[test]
    fn triangular_2() {
        let data: Vec<_> = (0..4).map(|i| Complex::new(i as f32, 0.0)).collect();
        let mut t = Matrix::new(2, data.clone());
        let mut u = Matrix::new(2, data.clone());
        <Matrix<f32> as Schur<f32>>::reduce_triangular(&mut t, &mut u);
        let known_t = vec![
            Complex::new(-0.5615529, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0000002, 0.0),
            Complex::new(3.5615532, 0.0),
        ];
        let known_u = vec![
            Complex::new(0.540646, 0.0),
            Complex::new(-0.15180072, 0.0),
            Complex::new(-1.9255395, 0.0),
            Complex::new(-3.1586323, 0.0),
        ];
        known_t.iter().zip(t.data()).for_each(|(k, c)| {
            assert!((*k - *c).square_norm() < f32::EPSILON);
        });
        known_u.iter().zip(u.data()).for_each(|(k, c)| {
            assert!((*k - *c).square_norm() < f32::EPSILON);
        });
    }
    #[test]
    fn triangular_3() {
        let data: Vec<_> = (0..9).map(|i| Complex::new(i as f32, 0.0)).collect();
        let mut t = Matrix::new(3, data.clone());
        let mut u = Matrix::new(3, data.clone());
        <Matrix<f32> as Schur<f32>>::reduce_triangular(&mut t, &mut u);
        let known_t = vec![
            Complex::new(13.348473, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.8989806, 0.0),
            Complex::new(-1.3484696, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-8.639932e-7, 0.0),
            Complex::new(1.4571746e-7, 0.0),
            Complex::new(2.4523172e-7, 0.0),
        ];
        let known_u = vec![
            Complex::new(-5.8765697, 0.0),
            Complex::new(-7.5801744, 0.0),
            Complex::new(-9.283779, 0.0),
            Complex::new(-3.2351115, 0.0),
            Complex::new(-2.9224942, 0.0),
            Complex::new(-2.609877, 0.0),
            Complex::new(1.9156505e-7, 0.0),
            Complex::new(2.609513e-7, 0.0),
            Complex::new(2.1139476e-7, 0.0),
        ];
        known_t.iter().zip(t.data()).for_each(|(k, c)| {
            assert!((*k - *c).square_norm() < f32::EPSILON);
        });
        known_u.iter().zip(u.data()).for_each(|(k, c)| {
            assert!((*k - *c).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn triangular_4() {
        let data: Vec<_> = (0..16).map(|i| Complex::new((i - 1) as f32, 0.0)).collect();
        let mut t = Matrix::new(4, data.clone());
        let mut u = Matrix::new(4, data.clone());
        <Matrix<f32> as Schur<f32>>::reduce_triangular(&mut t, &mut u);
        let known_t = vec![
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(-12.573192, 0.0),
            Complex::new(27.861404, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.5732784, 0.0),
            Complex::new(-7.348468, 0.0),
            Complex::new(-0.8614069, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-8.587281e-7, 0.0),
            Complex::new(-6.31451e-7, 0.0),
            Complex::new(2.5106584e-7, 0.0),
            Complex::new(4.8962005e-7, 0.0),
        ];
        for i in t.data() {
            print!("Complex::new({:?},0.0),", i.real)
        }
        let known_u = vec![
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(-12.573192, 0.0),
            Complex::new(27.861404, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.5732784, 0.0),
            Complex::new(-7.348468, 0.0),
            Complex::new(-0.8614069, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-8.587281e-7, 0.0),
            Complex::new(-6.31451e-7, 0.0),
            Complex::new(2.5106584e-7, 0.0),
            Complex::new(4.8962005e-7, 0.0),
        ];
        known_t.iter().zip(t.data()).for_each(|(k, c)| {
            assert!((*k - *c).square_norm() < f32::EPSILON);
        });
        known_u.iter().zip(u.data()).for_each(|(k, c)| {
            assert!((*k - *c).square_norm() < f32::EPSILON);
        });
    }
}

fn non() {
    /*
    for i in t.data(){
        print!("Complex::new({:?},0.0),",i.real)
    }*/
}

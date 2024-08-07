use core::cmp::min;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix, real::Real, vector::Vector};

use super::jacobi::{ComplexJacobi, Jacobian, RealJacobi};

pub struct RealSingularValueDecomposition {}
pub struct ComplexSingularValueDecomposition {}
pub trait SingularValueDecomposition<F: Float, R: Real> {
    type J: Jacobian<F>;
    fn jacobi_svd_full(data: &mut Matrix<F>) -> (Matrix<F>, Vec<R>, Matrix<F>);
    fn jacobi_svd(data: &mut Matrix<F>) -> Vec<R>;
    fn jacobi_2x2(data: &mut Matrix<F>, p: usize, q: usize) -> (Self::J, Self::J);

    fn bidiagonal_svd(data:&mut Matrix<F>)-> (Matrix<F>, Vec<R>, Matrix<F>);
}

impl<R: Real<Primitive = R>> SingularValueDecomposition<R, R> for RealSingularValueDecomposition {
    type J = RealJacobi<R>;

    fn jacobi_svd_full(data: &mut Matrix<R>) -> (Matrix<R>, Vec<R>, Matrix<R>) {
        //Doesn't handle different matrix sizes
        assert!(data.rows == data.cols);
        let precision = R::Primitive::EPSILON * R::Primitive::float(2.0);
        let small = R::Primitive::SMALL;
        let diag_size = min(data.cols, data.rows);
        let mut scale = <Vec<&'_ R> as Vector<R>>::l1_max(data.data());
        if scale == R::Primitive::ZERO {
            scale = R::Primitive::float(1.0);
        }

        let mut u = Matrix::<R>::identity(data.rows, data.rows);
        let mut v = Matrix::<R>::identity(data.cols, data.cols);

        <Vec<&'_ R> as Vector<R>>::mul(data.data_ref(), scale.recip());

        let mut max_diag = <Vec<&'_ R> as Vector<R>>::l1_max(data.data_diag());
        //step 2. with improvement options
        let mut finished = false;

        while !finished {
            finished = true;
            //note: these l1_norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if data.coeff(q, p).l1_norm() > threshold
                        || data.coeff(p, q).l1_norm() > threshold
                    {
                        finished = false;
                        let (j_left, j_right) = Self::jacobi_2x2(data, p, q);
                        j_left.apply_left(data, p, q, 0..data.rows);
                        j_left.apply_right(&mut u, p, q, 0..data.rows);
                        j_right.transpose().apply_right(data, p, q, 0..data.rows);
                        j_right.transpose().apply_right(&mut v, p, q, 0..data.rows);
                        max_diag = max_diag.greater(
                            data.coeff(p, p)
                                .l1_norm()
                                .greater(data.coeff(q, q).l1_norm()),
                        );
                    }
                })
            });
        }
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = data.data_diag().enumerate().map(|(i,r)|{
            if *r<R::Primitive::ZERO{
                <Vec<&'_ R> as Vector<R>>::mul(u.data_row_ref(i), -R::Primitive::ONE);
                -*r
            }else{
                *r
            } 
        }).collect::<Vec<_>>();

        singular.iter_mut().for_each(|s| *s *= scale);
        let mut indices = (0..singular.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&a, &b| singular[b].partial_cmp(&singular[a]).unwrap());
        //don't need to do last swap
        (0..singular.len() - 1).for_each(|i| {
            let new_idx = i;
            let old_idx = indices[i];
            if new_idx != old_idx {
                singular.swap(new_idx, old_idx);
                u.row_swap(new_idx, old_idx);
                v.row_swap(new_idx, old_idx);

                for i in indices.iter_mut() {
                    if *i == old_idx {
                        *i = new_idx;
                    } else if *i == new_idx {
                        *i = old_idx;
                    }
                }
            }
        });
        //step 4 sort the singular values

        (u, singular, v)
    }

    fn jacobi_svd(data: &mut Matrix<R>) -> Vec<R> {
        //Doesn't handle different matrix sizes
        assert!(data.rows == data.cols);
        let precision = R::Primitive::EPSILON * R::Primitive::float(2.0);
        let small = R::Primitive::SMALL;
        let diag_size = min(data.cols, data.rows);
        let mut scale = <Vec<&'_ R> as Vector<R>>::l1_max(data.data());
        if scale == R::Primitive::ZERO {
            scale = R::Primitive::float(1.0);
        }

        <Vec<&'_ R> as Vector<R>>::mul(data.data_ref(), scale.recip());

        let mut max_diag = <Vec<&'_ R> as Vector<R>>::l1_max(data.data_diag());
        //step 2. with improvement options
        let mut finished = false;

        while !finished {
            finished = true;
            //note: these l1_norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if data.coeff(q, p).l1_norm() > threshold
                        || data.coeff(p, q).l1_norm() > threshold
                    {
                        finished = false;
                        let (j_left, j_right) = Self::jacobi_2x2(data, p, q);
                        j_left.apply_left(data, p, q, 0..data.rows);
                        j_right.transpose().apply_right(data, p, q, 0..data.rows);
                        max_diag = max_diag.greater(
                            data.coeff(p, p)
                                .l1_norm()
                                .greater(data.coeff(q, q).l1_norm()),
                        );
                    }
                })
            });
        }
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = data.data_diag().map(|r|{
            r.l1_norm()*scale
        }).collect::<Vec<_>>();
        //step 4 sort the singular values
        singular.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        singular
    }

    //under tested
    fn jacobi_2x2(data: &mut Matrix<R>, p: usize, q: usize) -> (Self::J, Self::J) {
        let sub_data = vec![
            data.coeff(p, p),
            data.coeff(p, q),
            data.coeff(q, p),
            data.coeff(q, q),
        ];
        let mut sub_matrix = Matrix::new(2, sub_data);
        let t = sub_matrix.coeff(0, 0) + sub_matrix.coeff(1, 1);
        let d = sub_matrix.coeff(0, 1) - sub_matrix.coeff(1, 0);

        let rot1 = match d.l1_norm() < R::Primitive::EPSILON {
            true => Self::J::new(R::ZERO, R::ONE),
            false => {
                let u = t / d;
                let tmp = (R::Primitive::ONE + u.l2_norm()).sqrt();
                Self::J::new(tmp.recip(), u / tmp)
            }
        };
        rot1.apply_left(&mut sub_matrix, 0, 1, 0..2);
        let j_right = Self::J::make_jacobi(&mut sub_matrix, 0, 1);
        let j_left = rot1.dot(j_right.transpose());
        (j_left, j_right)
    }
    
    fn bidiagonal_svd(data:&mut Matrix<R>)-> (Matrix<R>, Vec<R>, Matrix<R>) {
        let mut scale = <Vec<&'_ R> as Vector<R>>::l1_max(data.data());
        if scale == R::Primitive::ZERO {
            scale = R::Primitive::float(1.0);
        }

        todo!();
    }
}

impl<R: Real<Primitive = R>, C: Complex<Primitive = R>> SingularValueDecomposition<C, R>
    for ComplexSingularValueDecomposition
{
    type J = ComplexJacobi<C>;

    fn jacobi_svd_full(_data: &mut Matrix<C>) -> (Matrix<C>, Vec<R>, Matrix<C>) {
        todo!();
    }

    fn jacobi_svd(_data: &mut Matrix<C>) -> Vec<R> {
        todo!();
    }

    //under tested
    fn jacobi_2x2(_data: &mut Matrix<C>, _p: usize, _q: usize) -> (Self::J, Self::J) {
        todo!();
    }
    
    fn bidiagonal_svd(_data:&mut Matrix<C>)-> (Matrix<C>, Vec<R>, Matrix<C>) {
        todo!()
    }
}

#[cfg(test)]
pub mod svd_tests {
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn jacobi_svd_square_r() {
        let mut in_mat = Matrix::new(3, (0..9).map(|i| i as f32).collect());
        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([14.2267, 1.26523, 7.16572e-8])
            .for_each(|(si, ki)| {
                assert!((si - ki).l2_norm() < f32::EPSILON);
            });

        let mut in_mat = Matrix::new(5, (0..25).map(|i| i as f32).collect());

        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([69.9086, 3.5761, 1.4977e-6, 1.0282e-6, 2.22847e-7])
            .for_each(|(si, ki)| {
                assert!((si - ki).l2_norm() < f32::EPSILON);
            });
    }

    #[test]
    fn jacobi_svd_full_square_r() {
        let mut in_mat = Matrix::new(3, (0..9).map(|i| i as f32).collect());
        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter()
            .zip([14.2267, 1.26523, 9.010921e-8])
            .for_each(|(si, ki)| {
                assert!((si - ki).l2_norm() < f32::EPSILON);
            });

        let kv = vec![
            0.46632808,
            0.5709908,
            0.6756534,
            -0.7847747,
            -0.085456595,
            0.6138613,
            -0.40824822,
            0.8164965,
            -0.40824825,
        ];
        u.data()
            .zip(kv.into_iter())
            .for_each(|(up, k)| assert!((up - k).l2_norm() < f32::EPSILON));

        let ku = vec![
            0.13511896,
            0.49633512,
            0.85755134,
            0.9028158,
            0.29493162,
            -0.31295204,
            -0.40824804,
            0.81649673,
            -0.40824836,
        ];

        v.data()
            .zip(ku.into_iter())
            .for_each(|(up, k)| assert!((up - k).l2_norm() < f32::EPSILON));
    }
    #[test]
    fn jacobi_svd_full_square_r_pca() {
        let mut in_mat: Matrix<f32> = Matrix::zero(3, 3);
        in_mat.data_row_ref(0).for_each(|rp| *rp = -3.0);
        in_mat.data_row_ref(2).for_each(|rp| *rp = 3.0);

        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter().zip([7.348469, 0.0, 0.0]).for_each(|(si, ki)| {
            assert!((si - ki).l2_norm() < f32::EPSILON);
        });

        let ku = vec![
            -0.57735026,
            -0.57735026,
            -0.57735026,
            0.70710677,
            -0.70710677,
            0.0,
            0.40824828,
            0.40824828,
            -0.8164966,
        ];
        let kv = vec![
            0.70710677,
            0.0,
            -0.70710677,
            0.0,
            1.0,
            0.0,
            0.70710677,
            0.0,
            0.70710677,
        ];

        u.data()
            .zip(ku.into_iter())
            .for_each(|(up, k)| assert!((up - k).l2_norm() < f32::EPSILON));

        v.data()
            .zip(kv.into_iter())
            .for_each(|(up, k)| assert!((up - k).l2_norm() < f32::EPSILON));
    }

    #[test]
    fn jacobi_svd_full_square_r_pca2() {
        let mut in_mat: Matrix<f32> = Matrix::zero(3, 3);
        in_mat.data_row_ref(0).for_each(|rp| *rp = -3.0);
        in_mat.data_row_ref(2).for_each(|rp| *rp = 3.0);
        in_mat = in_mat.transposed();

        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter().zip([7.348469, 0.0, 0.0]).for_each(|(si, ki)| {
            assert!((si - ki).l2_norm() < f32::EPSILON);
        });

        let ku = vec![
            -0.70710677,
            0.0,
            0.70710677,
            -0.70710677,
            0.0,
            -0.70710677,
            0.0,
            1.0,
            0.0,
        ];
        let kv = vec![
            0.57735026,
            0.57735026,
            0.57735026,
            -0.40824828,
            -0.40824828,
            0.8164966,
            0.70710677,
            -0.70710677,
            0.0,
        ];
        u.data()
            .zip(ku.into_iter())
            .for_each(|(up, k)| assert!((up - k).l2_norm() < f32::EPSILON));

        v.data()
            .zip(kv.into_iter())
            .for_each(|(up, k)| assert!((up - k).l2_norm() < f32::EPSILON));
    }
    #[test]
    fn jacobi_svd_speed_square() {
        let mut in_mat = Matrix::new(512, (0..512 * 512).map(|i| i as f32).collect());
        let now = SystemTime::now();
        let (_, _, _) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        let _ = println!("{:?}", now.elapsed());
    }
}

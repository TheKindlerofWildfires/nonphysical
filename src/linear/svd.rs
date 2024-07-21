use core::cmp::min;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector};

use super::jacobi::Jacobi;

pub trait SingularValueDecomposition<T: Float> {
    fn jacobi_svd_full(&mut self) -> (Self, Vec<T>, Self) where Self: Sized;
    fn jacobi_svd(&mut self) -> Vec<T>;
    fn real_jacobi_2x2(&mut self, p: usize, q: usize) -> (Jacobi<T>, Jacobi<T>);
}

impl<T: Float> SingularValueDecomposition<T> for Matrix<T> {
       //current u and v are slightly swapped.. this is a non problem for the square version but may come again
       fn jacobi_svd_full(&mut self) -> (Self, Vec<T>, Self) where Self: Sized {
        let precision = T::EPSILON * T::float(2.0);
        let small = T::SMALL;
        let diag_size = min(self.columns, self.rows);
        let mut scale = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(self.data());
        if scale == T::ZERO {
            scale = T::float(1.0);
        }

        let mut u = Matrix::<T>::identity(self.rows, self.rows);
        let mut v = Matrix::<T>::identity(self.columns, self.columns);
        //Poorly implemented step 1, doesn't handle miss sized matrices
        if self.rows != self.columns {
            panic!("not ready for these kind of matrices");
        }
        *self *= scale.recip();

        let mut max_diag = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(self.data_diag());
        //step 2. with improvement options
        let mut finished = false;

        while !finished {
            finished = true;
            //note: these norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if self.coeff(q, p).norm() > threshold
                        || self.coeff(p, q).norm() > threshold
                    {
                        finished = false;
                        let (j_left, j_right) = Self::real_jacobi_2x2(self, p, q);
                        j_left.apply_left(self, p, q,0..self.rows);
                        j_left.apply_right(&mut u, p, q,0..self.rows);
                        j_right.transpose().apply_right(self, p, q,0..self.rows);
                        j_right.transpose().apply_right(&mut v, p, q,0..self.rows);
                        max_diag = max_diag
                            .greater(self.coeff(p, p).norm().greater(self.coeff(q, q).norm()));
                    }
                })
            });
        }
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = Vec::with_capacity(diag_size);
        (0..diag_size).for_each(|i| {
            if self.coeff(i, i).imag.norm() > small {
                let a = self.coeff(i, i).norm();
                singular.push(a.norm());

                <Vec<&'_ Complex<T>> as Vector<T>>::mul(
                    u.data_column_ref(i),
                    self.coeff(i, i) / a,
                );
            } else {
                let a = self.coeff(i, i).real;
                singular.push(a.norm());

                if a < T::ZERO {
                    <Vec<&'_ Complex<T>> as Vector<T>>::scale(u.data_column_ref(i), -T::ONE);
                }
            }
        });

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

    fn jacobi_svd(&mut self) -> Vec<T>{
        let precision = T::EPSILON * T::float(2.0);
        let small = T::SMALL;
        let diag_size = min(self.columns, self.rows);
        let mut scale = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(self.data());
        if scale == T::ZERO {
            scale = T::float(1.0);
        }
        //Poorly implemented step 1, doesn't handle miss sized matrices
        if self.rows != self.columns {
            panic!("not ready for these kind of matrices");
        }
        *self *= scale.recip();

        let mut max_diag = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(self.data_diag());
        //step 2. with improvement options
        let mut finished = false;
        while !finished {
            finished = true;
            //note: these norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if self.coeff(q, p).norm() > threshold
                        || self.coeff(p, q).norm() > threshold
                    {
                        finished = false;
                        if true {
                            //Self::real_precondition_2x2(self, p, q, &mut max_diag) {
                            let (j_left, j_right) = Self::real_jacobi_2x2(self, p, q);
                            //this loaded in the right vectors (or close, x and y may be swapped)
                            j_left.apply_left(self, p, q,0..self.rows); //failed here, was it the jacobi or the apply?
                            j_right.transpose().apply_right(self, p, q,0..self.rows);
                            max_diag = max_diag.greater(
                                self.coeff(p, p).norm().greater(self.coeff(q, q).norm()),
                            );
                        }
                    }
                })
            });
        }
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = Vec::with_capacity(diag_size);
        (0..diag_size).for_each(|i| {
            if self.coeff(i, i).imag.norm() > small {
                let a = self.coeff(i, i).norm();
                singular.push(a.norm());
            } else {
                let a = self.coeff(i, i).real;
                singular.push(a.norm());
            }
        });
        singular.iter_mut().for_each(|s| *s *= scale);
        //step 4 sort the singular values
        singular.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        singular
    }
    /*sub problem related */

    fn real_jacobi_2x2(&mut self, p: usize, q: usize) -> (Jacobi<T>, Jacobi<T>){
        let mut sub_data = vec![
            self.coeff(p, p),
            self.coeff(p, q),
            self.coeff(q, p),
            self.coeff(q, q),
        ];
        sub_data.iter_mut().for_each(|c| c.imag = T::ZERO);
        let mut sub_matrix = Matrix::new(2, sub_data);
        let t = (sub_matrix.coeff(0, 0) + sub_matrix.coeff(1, 1)).real;
        let d = (sub_matrix.coeff(0, 1) - sub_matrix.coeff(1, 0)).real;

        let rot1 = match d.norm() < T::EPSILON {
            true => Jacobi::<T>::new(Complex::<T>::ZERO, Complex::<T>::ONE),
            false => {
                let u = t / d;
                let tmp = (T::ONE + u.square_norm()).sqrt();
                Jacobi::<T>::new(
                    Complex::<T>::new(tmp.recip(), T::ZERO),
                    Complex::<T>::new(u / tmp, T::ZERO),
                )
            }
        };
        rot1.apply_left(&mut sub_matrix, 0, 1,0..self.rows);
        let j_right = Jacobi::<T>::make_jacobi(&mut sub_matrix, 0, 1);
        let j_left = rot1 * j_right.transpose();
        (j_left, j_right)
    }
}

#[cfg(test)]
pub mod svd_tests {
    use super::*;

    #[test]
    fn jacobi_svd_square() {
        let mut in_mat =
            Matrix::<f32>::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let s = <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd(&mut in_mat);

        s.iter()
            .zip([14.2267, 1.26523, 7.16572e-8])
            .for_each(|(si, ki)| {
                assert!((si - ki).square_norm() < f32::EPSILON);
            });

        let mut in_mat =
            Matrix::<f32>::new(5, (0..25).map(|i| Complex::new(i as f32, 0.0)).collect());
        let s = <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd(&mut in_mat);
        s.iter()
            .zip([69.9086, 3.5761, 1.4977e-6, 1.0282e-6, 2.22847e-7])
            .for_each(|(si, ki)| {
                assert!((si - ki).square_norm() < f32::EPSILON);
            });
    }

    #[test]
    fn jacobi_svd_full_square() {
        let mut in_mat =
            Matrix::<f32>::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let (u, s, v) =
            <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd_full(&mut in_mat);

        s.iter()
            .zip([14.2267, 1.26523, 7.16572e-8])
            .for_each(|(si, ki)| {
                assert!((si - ki).square_norm() < f32::EPSILON);
            });

        let kv = vec![
            -0.46632808,
            0.5709908,
            0.6756534,
            -0.7847747,
            0.085456595,
            -0.6138613,
            0.40824822,
            0.8164965,
            -0.40824825,

        ];
        u.data()
            .zip(kv.iter())
            .for_each(|(up, k)| assert!((up.real - k).square_norm() < f32::EPSILON));

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
            .zip(ku.iter())
            .for_each(|(up, k)| assert!((up.real - k).square_norm() < f32::EPSILON));

        let mut in_mat =
            Matrix::<f32>::new(5, (0..25).map(|i| Complex::new(i as f32, 0.0)).collect());
        let s = <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd(&mut in_mat);
        s.iter()
            .zip([69.9086, 3.5761, 1.4977e-6, 1.0282e-6, 2.22847e-7])
            .for_each(|(si, ki)| {
                assert!((si - ki).square_norm() < f32::EPSILON);
            });

    }
}

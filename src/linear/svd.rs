use std::cmp::min;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector};

use super::jacobi::Jacobi;

pub trait SingularValueDecomposition<T: Float> {
    //partially broken, gives right values but some swapping occurs (like u and vt and swapped and one of the matrixs is transposed)
    fn jacobi_svd_full(matrix: &mut Matrix<T>) -> (Matrix<T>, Vec<T>, Matrix<T>) {
        let precision = T::epsilon() * T::float(2.0);
        let small = T::small();
        let diag_size = min(matrix.columns, matrix.rows);
        let mut scale = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(matrix.data());
        if scale == T::zero() {
            scale = T::float(1.0);
        }

        let mut u = Matrix::<T>::identity(matrix.rows, matrix.rows);
        let mut v = Matrix::<T>::identity(matrix.columns, matrix.columns);
        //Poorly implemented step 1, doesn't handle miss sized matrices
        if matrix.rows != matrix.columns {
            panic!("not ready for these kind of matrices");
        }
        *matrix *= scale.recip();

        let mut max_diag = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(matrix.data_diag());
        //step 2. with improvement options
        let mut finished = false;

        (0..4).for_each(|i| {
            finished = true;
            //note: these norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if matrix.coeff(p, q).norm() > threshold
                        || matrix.coeff(q, p).norm() > threshold
                    {
                        finished = false;
                        let (j_left, j_right) = Self::real_jacobi_2x2(matrix, p, q);
                        j_left.apply_left(matrix, p, q);
                        j_left.transpose().apply_right(&mut u, p, q);
                        j_right.apply_right(matrix, p, q);
                        j_right.apply_right(&mut v, p, q);
                        max_diag = max_diag
                            .greater(matrix.coeff(p, p).norm().greater(matrix.coeff(q, q).norm()));
                    }
                })
            });
        });
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = Vec::with_capacity(diag_size);
        (0..diag_size).for_each(|i| {
            if matrix.coeff(i, i).imag.norm() > small {
                let a = matrix.coeff(i, i).norm();
                singular.push(a.norm());

                
                <Vec<&'_ Complex<T>> as Vector<T>>::mul(
                    u.data_column_ref(i),
                    matrix.coeff(i, i) / a,
                );
            } else {
                let a = matrix.coeff(i, i).real;
                singular.push(a.norm());
                
                if a < T::zero() {
                    <Vec<&'_ Complex<T>> as Vector<T>>::scale(u.data_column_ref(i), T::isize(-1));
                }
            }
        });
        
        singular.iter_mut().for_each(|s| *s *= scale);
        let mut indices: Vec<_> = (0..singular.len()).collect();
        indices.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        indices.iter().enumerate().for_each(|(new_idx, &old_idx)| {
            let tmp = singular[new_idx];
            singular[new_idx] = singular[old_idx];
            singular[old_idx]= tmp;
            u.col_swap(new_idx, old_idx);
            v.col_swap(new_idx, old_idx);
        });
        //step 4 sort the singular values
        singular.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        (u, singular, v)
    }

    fn jacobi_svd(matrix: &mut Matrix<T>) -> Vec<T> {
        let precision = T::epsilon() * T::float(2.0);
        let small = T::small();
        let diag_size = min(matrix.columns, matrix.rows);
        let mut scale = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(matrix.data());
        if scale == T::zero() {
            scale = T::float(1.0);
        }

        //Poorly implemented step 1, doesn't handle miss sized matrices
        if matrix.rows != matrix.columns {
            panic!("not ready for these kind of matrices");
        }
        *matrix *= scale.recip();

        let mut max_diag = <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(matrix.data_diag());
        //step 2. with improvement options
        let mut finished = false;

        (0..4).for_each(|i| {
            finished = true;
            //note: these norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if matrix.coeff(p, q).norm() > threshold
                        || matrix.coeff(q, p).norm() > threshold
                    {
                        finished = false;
                        if true {
                            //Self::real_precondition_2x2(matrix, p, q, &mut max_diag) {
                            let (j_left, j_right) = Self::real_jacobi_2x2(matrix, p, q);
                            //this loaded in the right vectors (or close, x and y may be swapped)
                            j_left.apply_left(matrix, p, q); //failed here, was it the jacobi or the apply?
                            j_right.apply_right(matrix, p, q);
                            max_diag = max_diag.greater(
                                matrix.coeff(p, p).norm().greater(matrix.coeff(q, q).norm()),
                            );
                        }
                    }
                })
            });
        });
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = Vec::with_capacity(diag_size);
        (0..diag_size).for_each(|i| {
            if matrix.coeff(i, i).imag.norm() > small {
                let a = matrix.coeff(i, i).norm();
                singular.push(a.norm());
            } else {
                let a = matrix.coeff(i, i).real;
                singular.push(a.norm());
            }
        });
        singular.iter_mut().for_each(|s| *s *= scale);
        //step 4 sort the singular values
        singular.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        singular
    }
    /*sub problem related */

    fn real_jacobi_2x2(matrix: &Matrix<T>, p: usize, q: usize) -> (Jacobi<T>, Jacobi<T>) {
        let mut sub_data = vec![
            matrix.coeff(p, p),
            matrix.coeff(q, p),
            matrix.coeff(p, q),
            matrix.coeff(q, q),
        ];
        sub_data.iter_mut().for_each(|c| c.imag = T::zero());
        let mut sub_matrix = Matrix::new(2, sub_data);
        let t = (sub_matrix.coeff(0, 0) + sub_matrix.coeff(1, 1)).real;
        let d = (sub_matrix.coeff(1, 0) - sub_matrix.coeff(0, 1)).real;

        let rot1 = match d.norm() < T::small() {
            true => Jacobi::<T>::new(
                Complex::<T>::new(T::zero(), T::zero()),
                Complex::<T>::new(T::usize(1), T::zero()),
            ),
            false => {
                let u = t / d;
                let tmp = (T::usize(1) + u.square_norm()).sqrt();
                Jacobi::<T>::new(
                    Complex::<T>::new(T::usize(1) / tmp, T::zero()),
                    Complex::<T>::new(u / tmp, T::zero()),
                )
            }
        };
        rot1.apply_left(&mut sub_matrix, 0, 1);
        let j_right = Jacobi::<T>::make_jacobi(&mut sub_matrix, 0, 1);
        let j_left = rot1 * j_right.transpose();
        (j_left, j_right)
    }
}

impl<T: Float> SingularValueDecomposition<T> for Matrix<T> {}

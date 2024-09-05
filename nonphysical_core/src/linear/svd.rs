use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use std::time::SystemTime;

use crate::
    shared::{
        float::Float,
        matrix::{heap::MatrixHeap, Matrix},
        real::Real,
        vector::Vector,
    }
;

use super::{
    jacobi::{Jacobian, RealJacobi},
    qr_decomposition::{QRDecomposition, RealQRDecomposition},
};

pub struct RealSingularValueDecomposition {}
pub trait SingularValueDecomposition<F: Float, R: Real> {
    type J: Jacobian<F>;
    type Matrix: Matrix<F>;
    type QRDecomposition: QRDecomposition<F>;
    fn jacobi_svd_full(matrix: &mut Self::Matrix) -> (Self::Matrix, Vec<R>, Self::Matrix);
    fn jacobi_svd(matrix: &mut Self::Matrix) -> Vec<R>;
    fn jacobi_2x2(matrix: &mut Self::Matrix, p: usize, q: usize) -> (Self::J, Self::J);

    fn col_precondition(matrix: &mut Self::Matrix);
    fn row_precondition(matrix: &mut Self::Matrix);
    fn col_precondition_full(matrix: &mut Self::Matrix) -> (Self::Matrix, Self::Matrix);
    fn row_precondition_full(matrix: &mut Self::Matrix) -> (Self::Matrix, Self::Matrix);

    //fn bidiagonal_svd(data:&mut Self::Matrix)-> (Self::Matrix, Vec<R>, Self::Matrix);
}

impl<R: Real<Primitive = R>> SingularValueDecomposition<R, R> for RealSingularValueDecomposition {
    type J = RealJacobi<R>;
    type Matrix = MatrixHeap<R>;
    type QRDecomposition = RealQRDecomposition<R>;

    fn jacobi_svd_full(matrix: &mut Self::Matrix) -> (Self::Matrix, Vec<R>, Self::Matrix) {
        let precision = R::Primitive::EPSILON * R::Primitive::float(2.0);
        let small = R::Primitive::SMALL;
        let diag_size = min(matrix.cols, matrix.rows);
        let mut scale = Vector::l1_max(matrix.data());
        if scale == R::Primitive::ZERO {
            scale = R::Primitive::float(1.0);
        }

        Vector::mul(matrix.data_ref(), scale.recip());
        //There is unresolved disagreement between the U/V matrices between square and non square
        //Best guess is one of them is tilted, but I don't know which
        //I'm guessing it happened in the adjoint call
        let (mut u, mut v) = if matrix.n_cols() > matrix.n_rows() {
            Self::col_precondition_full(matrix)
        } else if matrix.n_rows()>matrix.n_cols(){
            Self::row_precondition_full(matrix)
        }else {
            (
                
                Matrix::<R>::identity(matrix.rows, matrix.rows),
                Matrix::<R>::identity(matrix.cols, matrix.cols),
            )
        };

        
        let mut max_diag = Vector::l1_max(matrix.data_diag());
        //step 2. with improvement options
        let mut finished = false;

        while !finished {
            finished = true;
            //note: these l1_norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if matrix.coeff(q, p).l1_norm() > threshold
                        || matrix.coeff(p, q).l1_norm() > threshold
                    {
                        finished = false;
                        let (j_left, j_right) = Self::jacobi_2x2(matrix, p, q);
                        j_left.apply_left(matrix, p, q, 0..matrix.rows);
                        j_left.apply_right(&mut u, p, q, 0..matrix.rows);
                        j_right
                            .transpose()
                            .apply_right(matrix, p, q, 0..matrix.rows);

                        j_right
                            .transpose()
                            .apply_right(&mut v, p, q, 0..matrix.rows);
                        max_diag = max_diag.greater(
                            matrix
                                .coeff(p, p)
                                .l1_norm()
                                .greater(matrix.coeff(q, q).l1_norm()),
                        );
                    }
                })
            });
        }
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = matrix
            .data_diag()
            .enumerate()
            .map(|(i, r)| {
                if *r < R::Primitive::ZERO {
                    Vector::mul(u.data_row_ref(i), -R::Primitive::ONE);
                    -*r
                } else {
                    *r
                }
            })
            .collect::<Vec<_>>();

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

    fn jacobi_svd(matrix: &mut Self::Matrix) -> Vec<R> {
        let precision = R::Primitive::EPSILON * R::Primitive::float(2.0);
        let small = R::Primitive::SMALL;
        let diag_size = min(matrix.cols, matrix.rows);
        let mut scale = Vector::l1_max(matrix.data());
        if scale == R::Primitive::ZERO {
            scale = R::Primitive::float(1.0);
        }

        Vector::mul(matrix.data_ref(), scale.recip());
        if matrix.n_cols() > matrix.n_rows() {
            Self::col_precondition(matrix);
        }else if matrix.n_rows()>matrix.n_cols(){
            Self::row_precondition(matrix);
        }

        let mut max_diag = Vector::l1_max(matrix.data_diag());
        //step 2. with improvement options
        let mut finished = false;

        while !finished {
            finished = true;
            //note: these l1_norms involve sqrt, but so far as just comparisons...
            (1..diag_size).for_each(|p| {
                (0..p).for_each(|q| {
                    let threshold = small.greater(precision * max_diag);
                    if matrix.coeff(q, p).l1_norm() > threshold
                        || matrix.coeff(p, q).l1_norm() > threshold
                    {
                        finished = false;
                        let (j_left, j_right) = Self::jacobi_2x2(matrix, p, q);
                        j_left.apply_left(matrix, p, q, 0..matrix.rows);

                        j_right
                            .transpose()
                            .apply_right(matrix, p, q, 0..matrix.rows);
                        max_diag = max_diag.greater(
                            matrix
                                .coeff(p, p)
                                .l1_norm()
                                .greater(matrix.coeff(q, q).l1_norm()),
                        );
                    }
                })
            });
        }
        //step3 recover the singular values -> essentially get the positive real numbers
        let mut singular = matrix
            .data_diag()
            .map(|r| r.l1_norm() * scale)
            .collect::<Vec<_>>();
        //step 4 sort the singular values
        singular.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        singular
    }

    //under tested
    fn jacobi_2x2(matrix: &mut Self::Matrix, p: usize, q: usize) -> (Self::J, Self::J) {
        let sub_data = vec![
            matrix.coeff(p, p),
            matrix.coeff(p, q),
            matrix.coeff(q, p),
            matrix.coeff(q, q),
        ];
        let mut sub_matrix = Self::Matrix::new((2, sub_data));
        let t = sub_matrix.coeff(0, 0) + sub_matrix.coeff(1, 1);
        let d = sub_matrix.coeff(0, 1)-sub_matrix.coeff(1, 0);

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

    fn col_precondition(matrix: &mut Self::Matrix) {
        Self::QRDecomposition::col_pivot(matrix);
        //could have done this in place, but did a lazy malloc
        let new_matrix = Self::Matrix::new((
            matrix.n_rows(),
            matrix
                .data_north_west(matrix.n_rows(), matrix.n_rows())
                .cloned()
                .collect::<Vec<_>>(),
        ));
        *matrix = new_matrix.transposed();

        matrix
            .data_lower_triangular_ref()
            .for_each(|mp| *mp = R::ZERO);
    }

    fn row_precondition(matrix: &mut Self::Matrix) {
        Self::QRDecomposition::col_pivot(matrix);
        let new_matrix = Self::Matrix::new((
            matrix.n_rows(),
            matrix
                .data_north_west(matrix.n_cols(), matrix.n_cols())
                .cloned()
                .collect::<Vec<_>>(),
        ));


        *matrix = new_matrix.transposed();

        matrix
            .data_lower_triangular_ref()
            .for_each(|mp| *mp = R::ZERO);
    }

    fn col_precondition_full(matrix: &mut Self::Matrix) -> (Self::Matrix, Self::Matrix) {
        let qr = Self::QRDecomposition::col_pivot(matrix);
        let u = qr.householder_sequence(matrix);
        let v = qr.col_permutations();
        //could have done this in place, but did a lazy malloc
        let new_matrix = Self::Matrix::new((
            matrix.n_rows(),
            matrix
                .data_north_west(matrix.n_rows(), matrix.n_rows())
                .cloned()
                .collect::<Vec<_>>(),
        ));
        *matrix = new_matrix.transposed();

        matrix
            .data_lower_triangular_ref()
            .for_each(|mp| *mp = R::ZERO);
        (u, v)
    }

    fn row_precondition_full(matrix: &mut Self::Matrix) -> (Self::Matrix, Self::Matrix) {
        let qr = Self::QRDecomposition::col_pivot(matrix);
        let u = qr.col_permutations();
        let v = qr.householder_sequence(matrix);
        let new_matrix = Self::Matrix::new((
            matrix.n_rows(),
            matrix
                .data_north_west(matrix.n_cols(), matrix.n_cols())
                .cloned()
                .collect::<Vec<_>>(),
        ));


        *matrix = new_matrix.transposed();

        matrix
            .data_lower_triangular_ref()
            .for_each(|mp| *mp = R::ZERO);
        (u,v)
    }

    /*
    fn bidiagonal_svd(matrix:&mut Self::Matrix)-> (Self::Matrix, Vec<R>, Self::Matrix) {
        let diag = min(matrix.rows, matrix.cols);
        let mut scale = Vector::l1_max(matrix.data());
        if scale == R::Primitive::ZERO {
            scale = R::Primitive::float(1.0);
        }
        Vector::mul(matrix.data_ref(), scale);
        let bidiagonal = RealBidiagonal::new(matrix);
        let mut computed = MatrixHeap::zero(diag+1, diag);
        computed.data_ref().zip(bidiagonal.bidiagonal.data()).for_each(|(cp,bp)|{
            *cp=*bp
        });

        todo!()
    }*/
}

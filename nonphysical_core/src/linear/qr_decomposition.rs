use core::cmp::min;

use crate::shared::{
    float::Float,
    matrix::{heap::MatrixHeap, Matrix},
    real::Real,
    vector::Vector,
};

use super::householder::{Householder, RealHouseholder};


pub trait QRDecomposition<F: Float> {
    type Matrix: Matrix<F>;
    type Householder: Householder<F>;
    fn row_pivot(matrix: &mut Self::Matrix) -> Self;
    fn row_permutations(&self) -> Self::Matrix;
    fn householder_sequence(&self,matrix: &Self::Matrix) -> Self::Matrix;

}
pub struct RealQRDecomposition<R:Real> {
    pub permutations:MatrixHeap<R>,
    pub tau: Vec<R>,
    pub det: i32,
    pub cols: usize,
}
impl<R: Real<Primitive = R>> QRDecomposition<R> for RealQRDecomposition<R> {
    type Matrix = MatrixHeap<R>;
    type Householder = RealHouseholder<R>;

    fn row_pivot(matrix: &mut Self::Matrix) -> Self {
        let rows = matrix.n_rows();
        let cols = matrix.n_cols();
        let size = min(rows, cols);

        let mut row_norms_update = Vec::with_capacity(rows);
        let mut row_norms_direct = Vec::with_capacity(rows);
        let mut tau = Vec::with_capacity(rows);
        let mut row_transpositions = vec![0; rows];
        let mut number_of_transpositions = 0;
        (0..rows).for_each(|i| {
            let row_norm = Vector::l2_sum(matrix.data_row(i)).sqrt();
            row_norms_direct.push(row_norm);
            row_norms_update.push(row_norm);
        });

        let mut non_zero_pivots = size;
        let mut max_pivot = R::ZERO;
        let threshold_helper =
            Vector::l1_max(row_norms_update.iter()) * R::EPSILON / R::usize(rows);
        let norm_threshold = R::EPSILON.sqrt();
        (0..size).for_each(|k| {
            let (mut biggest_idx, mut biggest_row_norm) = row_norms_update
                .iter()
                .skip(k)
                .enumerate()
                .fold(
                    (0, R::MIN),
                    |acc, (i, p)| {
                        if *p > acc.1 {
                            (i, *p)
                        } else {
                            acc
                        }
                    },
                );
            biggest_row_norm = biggest_row_norm.l2_norm();
            biggest_idx += k;
            if non_zero_pivots == size && biggest_row_norm < threshold_helper * R::usize(rows - k)
            {
                non_zero_pivots = k;
            }
            row_transpositions[k] = biggest_idx;
            if k != biggest_idx {
                matrix.row_swap(k, biggest_idx);
                row_norms_update.swap(k, biggest_idx);
                row_norms_direct.swap(k,biggest_idx);
                number_of_transpositions += 1;
            }
            let house = Self::Householder::make_householder_col(matrix, k, k);
            *matrix.coeff_ref(k, k) = house.beta;
            tau.push(house.tau);
            if house.beta.l1_norm() > max_pivot {
                max_pivot = house.beta.l1_norm();
            }
            let essential = matrix.data_row(k).skip(k + 1).cloned().collect::<Vec<_>>();
            house.apply_left(matrix, &essential, [k, cols], [k + 1, rows]);
            (k + 1..rows).for_each(|j| {
                if row_norms_update[j] != R::ZERO {
                    let mut tmp = matrix.coeff(j, k).l1_norm() / row_norms_update[j];
                    tmp = (R::ONE + tmp) * (R::ONE - tmp);
                    if tmp < R::ZERO {
                        tmp = R::ZERO;
                    }
                    let tmp2 = tmp * (row_norms_update[j] / row_norms_direct[j]).l2_norm();

                    if tmp2 <= norm_threshold {
                        let norm = Vector::l2_sum(matrix.data_col(j).skip(k + 1)).sqrt();
                        row_norms_direct[j] = norm;
                        row_norms_update[j] = norm;
                    } else {
                        row_norms_update[j] *= tmp.sqrt();
                    }
                }
            });
        });
        let mut permutations = Self::Matrix::identity(rows, rows);
        (0..size).for_each(|k|{
            if k!=row_transpositions[k]{
                permutations.row_swap(k, row_transpositions[k]);
            }
        });

        let det = if number_of_transpositions & 1 == 0 {
            -1
        } else {
            1
        };
        Self{permutations,det,tau,cols}
    }
    
    fn row_permutations(&self) -> Self::Matrix {
        self.permutations.explicit_copy()
    }
    
    fn householder_sequence(&self,matrix: &Self::Matrix) -> Self::Matrix {
        let mut output = Self::Matrix::identity(self.cols, self.cols);
        for i in (0..self.tau.len()).rev() {
            let householder = Self::Householder {
                tau: self.tau[i],
                beta: R::ZERO,
            };
            
            let vec = matrix.data_row(i).skip(i+1).cloned().collect::<Vec<_>>();
            householder.apply_left(&mut output, &vec, [i,self.cols], [i,self.cols]);
        }
        output
    }
}

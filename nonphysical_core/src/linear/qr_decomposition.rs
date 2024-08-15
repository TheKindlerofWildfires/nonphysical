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
    fn col_pivot(matrix: &mut Self::Matrix) -> Self;
    fn col_permutations(&self) -> Self::Matrix;
    fn householder_sequence(&self,matrix: &Self::Matrix) -> Self::Matrix;

}
pub struct RealQRDecomposition<R:Real> {
    pub permutations:MatrixHeap<R>,
    pub tau: Vec<R>,
    pub det: i32,
    pub rows: usize,
}
impl<R: Real<Primitive = R>> QRDecomposition<R> for RealQRDecomposition<R> {
    type Matrix = MatrixHeap<R>;
    type Householder = RealHouseholder<R>;

    fn col_pivot(matrix: &mut Self::Matrix) -> Self {
        let rows = matrix.n_rows();
        let cols = matrix.n_cols();
        let size = min(rows, cols);

        let mut col_norms_update = Vec::with_capacity(cols);
        let mut col_norms_direct = Vec::with_capacity(cols);
        let mut tau = Vec::with_capacity(cols);
        let mut col_transpositions = vec![0; cols];
        let mut number_of_transpositions = 0;
        (0..cols).for_each(|i| {
            let col_norm = Vector::l2_sum(matrix.data_col(i)).sqrt();
            col_norms_direct.push(col_norm);
            col_norms_update.push(col_norm);
        });

        let mut non_zero_pivots = size;
        let mut max_pivot = R::ZERO;
        let threshold_helper =
            (Vector::l1_max(col_norms_update.iter()) * R::EPSILON).l2_norm() / R::usize(rows);
        let norm_threshold = R::EPSILON.sqrt();
        (0..size).for_each(|k| {
            let (mut biggest_idx, mut biggest_col_norm) = col_norms_update
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
            biggest_col_norm = biggest_col_norm.l2_norm();
            biggest_idx += k;

            if non_zero_pivots == size && biggest_col_norm < threshold_helper * R::usize(cols - k)
            {
                non_zero_pivots = k;
            }
            col_transpositions[k] = biggest_idx;
            if k != biggest_idx {
                matrix.col_swap(k, biggest_idx);
                col_norms_update.swap(k, biggest_idx);
                col_norms_direct.swap(k,biggest_idx);
                number_of_transpositions += 1;
            }
            let house = Self::Householder::make_householder_row(matrix, k, k);
            *matrix.coeff_ref(k, k) = house.beta;

            tau.push(house.tau);
            if house.beta.l1_norm() > max_pivot {
                max_pivot = house.beta.l1_norm();
            }
            let essential = matrix.data_col(k).skip(k+1).cloned().collect::<Vec<_>>();
            house.apply_left(matrix, &essential, [k+1, cols], [k, cols]);

            (k + 1..cols).for_each(|j| {
                if col_norms_update[j] != R::ZERO {
                    let mut tmp = matrix.coeff(k, j).l1_norm() / col_norms_update[j];
                    tmp = (R::ONE + tmp) * (R::ONE - tmp);
                    if tmp < R::ZERO {
                        tmp = R::ZERO;
                    }
                    let tmp2 = tmp * (col_norms_update[j] / col_norms_direct[j]).l2_norm();

                    if tmp2 <= norm_threshold {
                        let norm = Vector::l2_sum(matrix.data_row(j).skip(k + 1)).sqrt();
                        col_norms_direct[j] = norm;
                        col_norms_update[j] = norm;
                    } else {
                        col_norms_update[j] *= tmp.sqrt();
                    }
                }
            });

        });
        let mut permutations = Self::Matrix::identity(cols, cols);
        (0..size).for_each(|k|{
            if k!=col_transpositions[k]{
                permutations.col_swap(k, col_transpositions[k]);
            }
        });

        let det = if number_of_transpositions & 1 == 0 {
            -1
        } else {
            1
        };
        Self{permutations,det,tau,rows}
    }
    
    fn col_permutations(&self) -> Self::Matrix {
        self.permutations.explicit_copy()
    }
    
    fn householder_sequence(&self,matrix: &Self::Matrix) -> Self::Matrix {
        let mut output = Self::Matrix::identity(self.rows, self.rows);
        for i in (0..self.tau.len()).rev() {
            let householder = Self::Householder {
                tau: self.tau[i],
                beta: R::ZERO,
            };
            
            let vec = matrix.data_col(i).skip(i+1).cloned().collect::<Vec<_>>();
            householder.apply_left(&mut output, &vec, [i,self.rows], [i,self.rows]);
        }
        output
    }
}

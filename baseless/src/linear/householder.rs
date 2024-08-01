use core::ops::Range;

use alloc::vec::Vec;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix, real::Real, vector::Vector};

pub trait Householder<'a, F: Float+'a> {
    fn make_householder_prep(data: &mut impl Iterator<Item = &'a F>) -> (F,F);

    fn make_householder_local(data: &mut impl Iterator<Item = &'a mut F>,prep:(F,F)) -> Self;

    //probably some desire to jump to iterators here too
    fn apply_left_local(
        &self,
        matrix: &mut Matrix<F>,
        offset: usize,
        row_range: Range<usize>,
        col_range: Range<usize>,
    );
    fn apply_right_local(
        &self,
        matrix: &mut Matrix<F>,
        offset: usize,
        row_range: Range<usize>,
        col_range: Range<usize>,
    );
    fn apply_left_local_vec(
        &self,
        matrix: &mut Matrix<F>,
        vec: &[F],
        row_range: Range<usize>,
        col_range: Range<usize>,
    );

    fn apply_right_local_vec(
        &self,
        matrix: &mut Matrix<F>,
        vec: &[F],
        row_range: Range<usize>,
        col_range: Range<usize>,
    );
}

pub struct RealHouseholder<R: Real> {
    pub tau: R,
    pub beta: R,
}

pub struct ComplexHouseholder<C: Complex> {
    pub tau: C,
    pub beta: C,
}

//matrix.data_row(row).skip(column + 1)
impl<'a, R: Real<Primitive = R>+'a> Householder<'a,R> for RealHouseholder<R> {
    fn make_householder_prep(data: &mut impl Iterator<Item = &'a R>) -> (R,R) {
        let first = data.next().unwrap();
        let sns = <Vec<&'_ R> as Vector<R>>::l2_sum(data);
        (*first,sns)
    }
    fn make_householder_local(data: &mut impl Iterator<Item = &'a mut R>, prep: (R,R)) -> Self {
        let first = prep.0;
        let squared_norm_sum =prep.1;
        data.next();

        let (beta, tau) = if squared_norm_sum <= R::SMALL {
            data.for_each(|r| *r = R::ZERO);
            (first, R::ZERO)
        } else {
            let beta = (first.l2_norm() + squared_norm_sum).sqrt() * -first.sign();
            let bmc = beta - first;
            let inv_bmc = -bmc.recip();
            data.for_each(|r| *r *= inv_bmc);

            let tau = bmc / beta;
            (beta, tau)
        };

        Self { tau, beta }
    }

    fn apply_left_local(
        &self,
        matrix: &mut Matrix<R>,
        offset: usize,
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        let mut vec = matrix.data_row(offset).copied().collect::<Vec<_>>();
        vec[row_range.clone().nth(0).unwrap()] = R::ONE;
        self.apply_left_local_vec(matrix, &vec, row_range, col_range);
    }

    fn apply_right_local(
        &self,
        matrix: &mut Matrix<R>,
        offset: usize,
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        let mut vec = matrix.data_row(offset).copied().collect::<Vec<_>>();
        vec[row_range.clone().nth(0).unwrap()] = R::ONE;
        self.apply_right_local_vec(matrix, &vec, row_range, col_range);
    }

    fn apply_left_local_vec(
        &self,
        matrix: &mut Matrix<R>,
        vec: &[R],
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        if self.tau == R::ZERO {
            return;
        }
        let mut tmp = R::ZERO;
        col_range.for_each(|i| {
            tmp = R::ZERO;
            row_range.clone().for_each(|j| {
                tmp = matrix.coeff(i, j).fma(vec[j], tmp);
            });
            tmp *= -self.tau;
            row_range
                .clone()
                .for_each(|j| *matrix.coeff_ref(i, j) = vec[j].fma(tmp, matrix.coeff(i, j)));
        })
    }

    fn apply_right_local_vec(
        &self,
        matrix: &mut Matrix<R>,
        vec: &[R],
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        if self.tau == R::ZERO {
            return;
        }
        let mut tmp = R::ZERO;
        col_range.for_each(|i| {
            tmp = R::ZERO;
            row_range.clone().for_each(|j| {
                tmp = matrix.coeff(j, i).fma(vec[j], tmp);
            });
            tmp *= -self.tau;
            row_range
                .clone()
                .for_each(|j| *matrix.coeff_ref(j, i) = vec[j].fma(tmp, matrix.coeff(j, i)));
        })
    }
    

}
use crate::shared::{
    complex::Complex,
    float::Float,
    matrix::{heap::MatrixHeap, Matrix},
    real::Real,
    vector::Vector,
};

pub trait Householder<F: Float> {
    type Matrix: Matrix<F>;
    fn make_householder_row(matrix: &mut Self::Matrix, row: usize, col: usize) -> Self
    where
        Self: Sized,
    {
        let prep = Self::make_householder_prep(&mut matrix.data_col(col).skip(row));
        Self::make_householder_local(&mut matrix.data_col_ref(col).skip(row), prep)
    }

    fn make_householder_col(matrix: &mut Self::Matrix, row: usize, col: usize) -> Self
    where
        Self: Sized,
    {
        let prep = Self::make_householder_prep(&mut matrix.data_row(row).skip(col));
        Self::make_householder_local(&mut matrix.data_row_ref(row).skip(col), prep)
    }

    fn make_householder_prep<'a>(data: &mut impl Iterator<Item = &'a F>) -> (F, F)
    where
        F: 'a;

    fn make_householder_local<'a>(data: &mut impl Iterator<Item = &'a mut F>, prep: (F, F)) -> Self
    where
        F: 'a;

    fn apply_left(
        &self,
        data: &mut Self::Matrix,
        essential: &[F],
        columns: [usize; 2],
        rows: [usize; 2],
    );

    fn apply_right(
        &self,
        data: &mut Self::Matrix,
        essential: &[F],
        columns: [usize; 2],
        rows: [usize; 2],
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
impl<R: Real<Primitive = R>> Householder<R> for RealHouseholder<R> {
    type Matrix = MatrixHeap<R>;
    fn make_householder_prep<'a>(data: &mut impl Iterator<Item = &'a R>) -> (R, R)
    where
        R: 'a,
    {
        let first = data.next().unwrap();
        let sns = Vector::l2_sum(data);
        (*first, sns)
    }
    fn make_householder_local<'a>(data: &mut impl Iterator<Item = &'a mut R>, prep: (R, R)) -> Self
    where
        R: 'a,
    {
        let first = prep.0;
        let squared_norm_sum = prep.1;
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

    fn apply_left(
        &self,
        data: &mut Self::Matrix,
        essential: &[R],
        cols: [usize; 2],
        rows: [usize; 2],
    ) {
        if self.tau == R::ZERO {
            return;
        }
        let mut pre_essential = Vec::with_capacity(1 + essential.len());
        pre_essential.push(R::ONE);
        pre_essential.extend_from_slice(&essential);
        (rows[0]..rows[1]).for_each(|i| {
            let tmp = data
                .data_row(i)
                .skip(cols[0])
                .zip(pre_essential.iter())
                .map(|(mp, ep)| *mp * *ep)
                .fold(R::ZERO, |acc, p| acc + p)
                * -self.tau;

            data.data_row_ref(i)
                .skip(cols[0])
                .zip(pre_essential.iter())
                .for_each(|(mp, ep)| {
                    *mp = tmp.fma(*ep, *mp);
                });
        })
    }

    fn apply_right(
        &self,
        data: &mut Self::Matrix,
        essential: &[R],
        cols: [usize; 2],
        rows: [usize; 2],
    ) {
        if self.tau == R::ZERO {
            return;
        }
        let mut pre_essential = Vec::with_capacity(1 + essential.len());
        pre_essential.push(R::ONE);
        pre_essential.extend_from_slice(&essential);
        (cols[0]..cols[1]).for_each(|i| {
            let tmp = data
                .data_col(i)
                .skip(rows[0])
                .zip(pre_essential.iter())
                .map(|(mp, ep)| *mp * *ep)
                .fold(R::ZERO, |acc, p| acc + p)
                * -self.tau;
            data.data_col_ref(i)
                .skip(rows[0])
                .zip(pre_essential.iter())
                .for_each(|(mp, ep)| {
                    *mp = ep.fma(tmp, *mp);
                });
        })
    }
}

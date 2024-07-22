use crate::shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector};

#[derive(Debug)]
pub struct Householder<T: Float> {
    pub tau: Complex<T>,
    pub beta: Complex<T>,
}

impl<'a, T: Float + 'a> Householder<T> {
    //modifies vector to become householder

    pub fn make_householder_local(matrix: &mut Matrix<T>, row: usize, column: usize) -> Self {
        let squared_norm_sum =
            <Vec<&Complex<T>> as Vector<T>>::square_norm_sum(matrix.data_row(column).skip(row + 1));
        let first = matrix.coeff(column, row);

        let (beta, tau) = if squared_norm_sum < T::EPSILON && first.imag.norm() < T::EPSILON {
            matrix
                .data_column_ref(column)
                .skip(row)
                .for_each(|c| *c = Complex::ZERO);
            (Complex::ZERO, first)
        } else {
            let beta = Complex::new((first.square_norm() + squared_norm_sum).sqrt(), T::ZERO)
                * -first.real.sign();
            let bmc = beta - first;
            let inv_bmc = -bmc.recip();

            matrix
                .data_column_ref(column)
                .skip(row)
                .for_each(|c| *c *= inv_bmc);

            let tau = (bmc / beta).conj();
            (beta, tau)
        };

        Self { tau, beta }
    }

    //I want to redo this with range vectors (and vec as an iterator), I just don't know how to yet
    //This is a cache optimized version written originally for cuda and may not be optimal here (not tested for real complex numbers)
    pub fn apply_left_local(
        &self,
        matrix: &mut Matrix<T>,
        offset: usize,
        row_range: [usize; 2],
        col_range: [usize; 2],
    ) {
        let mut vec = matrix
            .data_row(offset)
            .map(|c| c.conj())
            .collect::<Vec<_>>();
        vec[row_range[0]] = Complex::ONE;
        self.apply_left_local_vec(matrix, &vec, row_range, col_range);
    }

    pub fn apply_left_local_vec(
        &self,
        matrix: &mut Matrix<T>,
        vec: &[Complex<T>],
        row_range: [usize; 2],
        col_range: [usize; 2],
    ) {
        if self.tau == Complex::ZERO {
            return;
        }
        let mut tmp = Complex::ZERO;
        (col_range[0]..col_range[1]).for_each(|i| {
            tmp = Complex::ZERO;
            (row_range[0]..row_range[1]).for_each(|j| {
                tmp = matrix.coeff(i, j).fma(vec[j], tmp);
            });
            tmp *= -self.tau;
            (row_range[0]..row_range[1])
                .for_each(|j| *matrix.coeff_ref(i, j) = vec[j].fma(tmp, matrix.coeff(i, j)));
        })
    }

    //This is a cache optimized version written originally for cuda and may not be optimal here (not tested for real complex numbers)
    pub fn apply_right_local(
        &self,
        matrix: &mut Matrix<T>,
        offset: usize,
        row_range: [usize; 2],
        col_range: [usize; 2],
    ) {
        let mut vec = matrix
            .data_row(offset)
            .map(|c| c.conj())
            .collect::<Vec<_>>();
        vec[row_range[0]] = Complex::ONE;
        self.apply_right_local_vec(matrix, &vec, row_range, col_range);
    }

    pub fn apply_right_local_vec(
        &self,
        matrix: &mut Matrix<T>,
        vec: &[Complex<T>],
        row_range: [usize; 2],
        col_range: [usize; 2],
    ) {
        if self.tau == Complex::ZERO {
            return;
        }
        let mut tmp = Complex::ZERO;
        (col_range[0]..col_range[1]).for_each(|i| {
            tmp = Complex::ZERO;
            (row_range[0]..row_range[1]).for_each(|j| {
                tmp = matrix.coeff(j, i).fma(vec[j], tmp);
            });
            tmp *= -self.tau;
            (row_range[0]..row_range[1])
                .for_each(|j| *matrix.coeff_ref(j, i) = vec[j].fma(tmp, matrix.coeff(j, i)));
        })
    }
}

#[cfg(test)]
mod householder_tests {
    use super::*;
    #[test]
    fn make_local_1() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let house = Householder::make_householder_local(&mut m, 1, 0);
        assert!((house.beta.real + 3.7416575).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(1.2672611, 0.0)).square_norm() < f32::EPSILON);
    }
    #[test]
    fn left_local_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(15.0, 0.0),
        ];
        let mut m = Matrix::new(4, data).transposed();
        let house = Householder {
            tau: Complex::new(1.26726, 0.0),
            beta: Complex::new(-3.74166, 0.0),
        };
        house.apply_left_local(&mut m, 0, [1, 4], [1, 4]);
        let know_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(-10.155907, 0.0),
            Complex::new(-16.570163995199998, 0.0),
            Complex::new(-22.98442, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-0.39265555, 0.0),
            Complex::new(-0.7853164, 0.0),
            Complex::new(-1.1779773, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-2.5889907, 0.0),
            Complex::new(-5.177987, 0.0),
            Complex::new(-7.7669835, 0.0),
        ];
        m.transposed()
            .data()
            .zip(know_data.iter())
            .for_each(|(a, b)| assert!((*a - *b).square_norm() < f32::EPSILON));
    }

    #[test]
    fn left_local_2() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(2.98023e-07, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data).transposed();
        let house = Householder {
            tau: Complex::new(1.14995, 0.0),
            beta: Complex::new(-9.79796, 0.0),
        };
        house.apply_left_local(&mut m, 1, [2, 4], [2, 4]);

        let know_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-0.000000044688562, 0.0),
            Complex::new(0.0000010143897, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-0.0000002946524, 0.0),
            Complex::new(0.000000328435, 0.0),
        ];
        m.transposed()
            .data()
            .zip(know_data.iter())
            .for_each(|(a, b)| assert!((*a - *b).square_norm() < f32::EPSILON));
    }

    #[test]
    fn left_local_3() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data).transposed();
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new(-2.8054e-07, 0.0),
        };
        house.apply_left_local(&mut m, 2, [3, 4], [3, 4]);

        let know_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        m.transposed()
            .data()
            .zip(know_data.iter())
            .for_each(|(a, b)| assert!((*a - *b).square_norm() < f32::EPSILON));
    }
    #[test]
    fn right_local_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(-3.7416573867739413, 0.0),
            Complex::new(-10.155927192672129, 0.0),
            Complex::new(-16.5701969985703, 0.0),
            Complex::new(-22.984466804468497, 0.0),
            Complex::new(0.42179344411906794, 0.0),
            Complex::new(-0.3926707294150136, 0.0),
            Complex::new(-0.7853414588300254, 0.0),
            Complex::new(-1.178012188245038, 0.0),
            Complex::new(0.632690166178602, 0.0),
            Complex::new(-2.5890060941225226, 0.0),
            Complex::new(-5.178012188245038, 0.0),
            Complex::new(-7.767018282367559, 0.0),
        ];
        let mut m = Matrix::<f32>::new(4, data).transposed();
        let house = Householder {
            tau: Complex::new(1.2672612419124243, 0.0),
            beta: Complex::new(-3.7416573867739413, 0.0),
        };
        house.apply_right_local(&mut m, 0, [1, 4], [0, 4]);

        let know_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.966629547095767, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(1.46924, 0.0),
            Complex::new(2.98023e-07, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(9.68717, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        m.transposed()
            .data()
            .zip(know_data.iter())
            .for_each(|(a, b)| assert!((*a - *b).square_norm() < f32::EPSILON));
    }

    #[test]
    fn right_local_2() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-4.46897e-08, 0.0),
            Complex::new(1.01439e-06, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-2.94653e-07, 0.0),
            Complex::new(3.28438e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data).transposed();
        let house = Householder {
            tau: Complex::new(1.14995, 0.0),
            beta: Complex::new(-9.79796, 0.0),
        };
        house.apply_right_local(&mut m, 1, [2, 4], [0, 4]);

        let know_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.4494781, 0.0),
            Complex::new(5.605246e-6, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.962163e-7, 0.0),
            Complex::new(1.9629792e-7, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-2.805402e-7, 0.0),
            Complex::new(3.4057175e-7, 0.0),
        ];
        m.transposed()
            .data()
            .zip(know_data.iter())
            .for_each(|(a, b)| assert!((*a - *b).square_norm() < f32::EPSILON));
    }

    #[test]
    fn right_local_3() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data).transposed();
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new(-2.8054e-07, 0.0),
        };
        house.apply_right_local(&mut m, 2, [3, 4], [0, 4]);

        let know_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        m.transposed()
            .data()
            .zip(know_data.iter())
            .for_each(|(a, b)| assert!((*a - *b).square_norm() < f32::EPSILON));
    }
}

use std::ops::Range;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector};

#[derive(Debug)]
pub struct Householder<T: Float> {
    pub tau: Complex<T>,
    pub beta: Complex<T>,
}

impl<'a, T: Float + 'a> Householder<T> {
    //modifies vector to become householder

    pub fn make_householder_local(matrix: &mut Matrix<T>, column: usize, row: usize) -> Self {
        let squared_norm_sum =
            <Vec<&Complex<T>> as Vector<T>>::square_norm_sum(matrix.data_row(row).skip(column + 1));
        let first = matrix.coeff(row, column);
        let (beta, tau) = if squared_norm_sum <= T::SMALL && first.imag.square_norm() <= T::SMALL {
            matrix
                .data_row_ref(row)
                .skip(column + 1)
                .for_each(|c| *c = Complex::ZERO);
            (first, Complex::ZERO)
        } else {
            let beta = Complex::new((first.square_norm() + squared_norm_sum).sqrt(), T::ZERO)
                * -first.real.sign();
            let bmc = beta - first;
            let inv_bmc = -bmc.recip();

            matrix
                .data_row_ref(row)
                .skip(column + 1)
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
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        let mut vec = matrix
            .data_row(offset)
            .map(|c| c.conj())
            .collect::<Vec<_>>();
        vec[row_range.clone().nth(0).unwrap()] = Complex::ONE;
        self.apply_left_local_vec(matrix, &vec, row_range, col_range);
    }

    pub fn apply_left_local_vec(
        &self,
        matrix: &mut Matrix<T>,
        vec: &[Complex<T>],
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        if self.tau == Complex::ZERO {
            return;
        }
        let mut tmp = Complex::ZERO;
        col_range.for_each(|i| {
            tmp = Complex::ZERO;
            row_range.clone().for_each(|j| {
                tmp = matrix.coeff(i, j).fma(vec[j], tmp);
            });
            tmp *= -self.tau;
            row_range
                .clone()
                .for_each(|j| *matrix.coeff_ref(i, j) = vec[j].fma(tmp, matrix.coeff(i, j)));
        })
    }

    //This is a cache optimized version written originally for cuda and may not be optimal here (not tested for real complex numbers)
    pub fn apply_right_local(
        &self,
        matrix: &mut Matrix<T>,
        offset: usize,
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        let mut vec = matrix
            .data_row(offset)
            .map(|c| c.conj())
            .collect::<Vec<_>>();
        vec[row_range.clone().nth(0).unwrap()] = Complex::ONE;
        self.apply_right_local_vec(matrix, &vec, row_range, col_range);
    }

    pub fn apply_right_local_vec(
        &self,
        matrix: &mut Matrix<T>,
        vec: &[Complex<T>],
        row_range: Range<usize>,
        col_range: Range<usize>,
    ) {
        if self.tau == Complex::ZERO {
            return;
        }
        let mut tmp = Complex::ZERO;
        col_range.for_each(|i| {
            tmp = Complex::ZERO;
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

#[cfg(test)]
mod householder_tests {
    use super::*;

    #[test]
    fn make_local_3x3_real_1() {
        let mut m = Matrix::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let house = Householder::make_householder_local(&mut m, 1, 0);
        assert!((house.beta.real + 2.23607).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(1.44721, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.61803395, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_3x3_real_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-2.23607, 0.0),
            Complex::new(0.61803395, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];
        let mut m = Matrix::new(3, data);
        let house = Householder {
            tau: Complex::new(1.44721, 0.0),
            beta: Complex::new(-2.23607, 0.0),
        };
        house.apply_left_local(&mut m, 0, 1..3, 1..3);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-2.23607, 0.0),
            Complex::new(0.61803395, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(-6.26099, 0.0),
            Complex::new(-1.34164, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(-10.2859, 0.0),
            Complex::new(-2.68328, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_3x3_real_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-2.23607, 0.0),
            Complex::new(0.61803395, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(-6.26099, 0.0),
            Complex::new(-1.34164, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(-10.2859, 0.0),
            Complex::new(-2.68328, 0.0),
        ];
        let mut m = Matrix::new(3, data);
        let house = Householder {
            tau: Complex::new(1.44721, 0.0),
            beta: Complex::new(-2.23607, 0.0),
        };
        house.apply_right_local(&mut m, 0, 1..3, 0..3);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_3x3_real_2() {
        let data = vec![
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
        let mut m = Matrix::new(3, data);
        let house = Householder::make_householder_local(&mut m, 2, 1);
        assert!((house.beta.real - 3.0).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(0.0, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_3x3_real_2() {
        let data = vec![
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
        let mut m = Matrix::new(3, data);
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new(3.0, 0.0),
        };
        house.apply_left_local(&mut m, 0, 2..3, 2..3);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_3x3_real_2() {
        let data = vec![
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
        let mut m = Matrix::new(3, data);
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new(3.0, 0.0),
        };
        house.apply_right_local(&mut m, 0, 2..3, 0..3);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_4x4_real_1() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let house = Householder::make_householder_local(&mut m, 1, 0);
        assert!((house.beta.real + 3.74166).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(1.26726, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(15.0, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_4x4_real_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(15.0, 0.0),
        ];
        let mut m = Matrix::new(4, data);
        let house = Householder {
            tau: Complex::new(1.26726, 0.0),
            beta: Complex::new(-3.74166, 0.0),
        };
        house.apply_left_local(&mut m, 0, 1..4, 1..4);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(-10.1559, 0.0),
            Complex::new(-0.392671, 0.0),
            Complex::new(-2.58901, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(-16.5702, 0.0),
            Complex::new(-0.785341, 0.0),
            Complex::new(-5.17801, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(-22.9845, 0.0),
            Complex::new(-1.17801, 0.0),
            Complex::new(-7.76702, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_4x4_real_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(-10.1559, 0.0),
            Complex::new(-0.392671, 0.0),
            Complex::new(-2.58901, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(-16.5702, 0.0),
            Complex::new(-0.785341, 0.0),
            Complex::new(-5.17801, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(-22.9845, 0.0),
            Complex::new(-1.17801, 0.0),
            Complex::new(-7.76702, 0.0),
        ];
        let mut m = Matrix::new(4, data);
        let house = Householder {
            tau: Complex::new(1.26726, 0.0),
            beta: Complex::new(-3.74166, 0.0),
        };
        house.apply_right_local(&mut m, 0, 1..4, 0..4);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(1.46924, 0.0),
            Complex::new(9.68717, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.98023e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_4x4_real_2() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(1.46924, 0.0),
            Complex::new(9.68717, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.98023e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data);
        let house = Householder::make_householder_local(&mut m, 2, 1);
        assert!((house.beta.real + 9.79796).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(1.14995, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(1.46924, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.98023e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_4x4_real_2() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(2.98023e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data);
        let house = Householder {
            tau: Complex::new(1.14995, 0.0),
            beta: Complex::new(-9.79796, 0.0),
        };
        house.apply_left_local(&mut m, 1, 2..4, 2..4);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(-4.46897e-08, 0.0),
            Complex::new(-2.94653e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(1.01439e-06, 0.0),
            Complex::new(3.28438e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_4x4_real_2() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.36731, 0.0),
            Complex::new(-4.46897e-08, 0.0),
            Complex::new(-2.94653e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.42179, 0.0),
            Complex::new(1.01439e-06, 0.0),
            Complex::new(3.28438e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data);
        let house = Householder {
            tau: Complex::new(1.14995, 0.0),
            beta: Complex::new(-9.79796, 0.0),
        };
        house.apply_right_local(&mut m, 1, 2..4, 0..4);
        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_4x4_real_3() {
        let data = vec![
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
        let mut m = Matrix::new(4, data);
        let house = Householder::make_householder_local(&mut m, 3, 2);
        assert!((house.beta.real + 2.8054e-07).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(0.0, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_4x4_real_3() {
        let data = vec![
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
        let mut m = Matrix::new(4, data);
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new( 2.8054e-07, 0.0),
        };
        house.apply_left_local(&mut m, 2, 3..4, 3..4);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_4x4_real_3() {
        let data = vec![
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
        let mut m = Matrix::new(4, data);
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new( 2.8054e-07, 0.0),
        };
        house.apply_right_local(&mut m, 2, 3..4, 0..4);
        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_1() {
        let mut m = Matrix::new(5, (0..25).map(|i| Complex::new(i as f32, 0.0)).collect());
        let house = Householder::make_householder_local(&mut m, 1, 0);
        assert!((house.beta.real + 5.47723).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(1.18257, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(15.0, 0.0),
            Complex::new(16.0, 0.0),
            Complex::new(17.0, 0.0),
            Complex::new(18.0, 0.0),
            Complex::new(19.0, 0.0),
            Complex::new(20.0, 0.0),
            Complex::new(21.0, 0.0),
            Complex::new(22.0, 0.0),
            Complex::new(23.0, 0.0),
            Complex::new(24.0, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(15.0, 0.0),
            Complex::new(16.0, 0.0),
            Complex::new(17.0, 0.0),
            Complex::new(18.0, 0.0),
            Complex::new(19.0, 0.0),
            Complex::new(20.0, 0.0),
            Complex::new(21.0, 0.0),
            Complex::new(22.0, 0.0),
            Complex::new(23.0, 0.0),
            Complex::new(24.0, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(1.18257, 0.0),
            beta: Complex::new(-5.47723, 0.0),
        };
        house.apply_left_local(&mut m, 0, 1..5, 1..5);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(-14.6059, 0.0),
            Complex::new(0.63742, 0.0),
            Complex::new(-1.54387, 0.0),
            Complex::new(-3.72516, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(-23.7346, 0.0),
            Complex::new(1.27484, 0.0),
            Complex::new(-3.08774, 0.0),
            Complex::new(-7.45032, 0.0),
            Complex::new(15.0, 0.0),
            Complex::new(-32.8633, 0.0),
            Complex::new(1.91226, 0.0),
            Complex::new(-4.63161, 0.0),
            Complex::new(-11.1755, 0.0),
            Complex::new(20.0, 0.0),
            Complex::new(-41.9921, 0.0),
            Complex::new(2.54968, 0.0),
            Complex::new(-6.17548, 0.0),
            Complex::new(-14.9006, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_1() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(-14.6059, 0.0),
            Complex::new(0.63742, 0.0),
            Complex::new(-1.54387, 0.0),
            Complex::new(-3.72516, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(-23.7346, 0.0),
            Complex::new(1.27484, 0.0),
            Complex::new(-3.08774, 0.0),
            Complex::new(-7.45032, 0.0),
            Complex::new(15.0, 0.0),
            Complex::new(-32.8633, 0.0),
            Complex::new(1.91226, 0.0),
            Complex::new(-4.63161, 0.0),
            Complex::new(-11.1755, 0.0),
            Complex::new(20.0, 0.0),
            Complex::new(-41.9921, 0.0),
            Complex::new(2.54968, 0.0),
            Complex::new(-6.17548, 0.0),
            Complex::new(-14.9006, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(1.18257, 0.0),
            beta: Complex::new(-5.47723, 0.0),
        };
        house.apply_right_local(&mut m, 0, 1..5, 0..5);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(-27.3861, 0.0),
            Complex::new(60.0, 0.0),
            Complex::new(-3.49129, 0.0),
            Complex::new(8.45612, 0.0),
            Complex::new(20.4035, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.698259, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.69122, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.08071, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_2() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(-27.3861, 0.0),
            Complex::new(60.0, 0.0),
            Complex::new(-3.49129, 0.0),
            Complex::new(8.45612, 0.0),
            Complex::new(20.4035, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.698259, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.69122, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.08071, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder::make_householder_local(&mut m, 2, 1);
        assert!((house.beta.real -22.3607).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(1.15614, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(-27.3861, 0.0),
            Complex::new(60.0, 0.0),
            Complex::new(-3.49129, 0.0),
            Complex::new(-0.327098, 0.0),
            Complex::new(-0.789245, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.698259, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.69122, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.08071, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_2() {
        let data = vec![
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
            Complex::new(-0.698259, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.69122, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.08071, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(1.15614, 0.0),
            beta: Complex::new(22.3607, 0.0),
        };
        house.apply_left_local(&mut m, 1, 2..5, 2..5);

        let known = vec![
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
            Complex::new(-0.698259, 0.0),
            Complex::new(-4.35102e-07, 0.0),
            Complex::new(1.42321e-07, 0.0),
            Complex::new(-1.33435e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.69122, 0.0),
            Complex::new(-1.23085e-06, 0.0),
            Complex::new(-5.51065e-07, 0.0),
            Complex::new(1.77714e-08, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.08071, 0.0),
            Complex::new(-8.70203e-07, 0.0),
            Complex::new(2.84642e-07, 0.0),
            Complex::new(-2.6687e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            dbg!(c,k);
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_2() {
        let data = vec![
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
            Complex::new(-0.698259, 0.0),
            Complex::new(-4.35102e-07, 0.0),
            Complex::new(1.42321e-07, 0.0),
            Complex::new(-1.33435e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.69122, 0.0),
            Complex::new(-1.23085e-06, 0.0),
            Complex::new(-5.51065e-07, 0.0),
            Complex::new(1.77714e-08, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.08071, 0.0),
            Complex::new(-8.70203e-07, 0.0),
            Complex::new(2.84642e-07, 0.0),
            Complex::new(-2.6687e-07, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(1.15614, 0.0),
            beta: Complex::new(22.3607, 0.0),
        };
        house.apply_right_local(&mut m, 1, 2..5, 0..5);

        let known = vec![
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
            Complex::new(2.91112e-08, 0.0),
            Complex::new(-2.15958e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.66893e-06, 0.0),
            Complex::new(-9.83413e-07, 0.0),
            Complex::new(-5.14034e-07, 0.0),
            Complex::new(4.47644e-08, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-2.7316e-07, 0.0),
            Complex::new(3.73992e-07, 0.0),
            Complex::new(-2.0174e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_3() {
        let data = vec![
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
            Complex::new(2.91112e-08, 0.0),
            Complex::new(-2.15958e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.66893e-06, 0.0),
            Complex::new(-9.83413e-07, 0.0),
            Complex::new(-5.14034e-07, 0.0),
            Complex::new(4.47644e-08, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-2.7316e-07, 0.0),
            Complex::new(3.73992e-07, 0.0),
            Complex::new(-2.0174e-07, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder::make_householder_local(&mut m, 3, 2);
        assert!((house.beta.real +2.17911e-07).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(1.13359, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
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
            Complex::new(2.91112e-08, 0.0),
            Complex::new(-0.874244, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.66893e-06, 0.0),
            Complex::new(-9.83413e-07, 0.0),
            Complex::new(-5.14034e-07, 0.0),
            Complex::new(4.47644e-08, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-2.7316e-07, 0.0),
            Complex::new(3.73992e-07, 0.0),
            Complex::new(-2.0174e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_3() {
        let data = vec![
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
            Complex::new(-1.66893e-06, 0.0),
            Complex::new(-9.83413e-07, 0.0),
            Complex::new(-5.14034e-07, 0.0),
            Complex::new(4.47644e-08, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-2.7316e-07, 0.0),
            Complex::new(3.73992e-07, 0.0),
            Complex::new(-2.0174e-07, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(1.13359, 0.0),
            beta: Complex::new(-2.17911e-07, 0.0),
        };
        house.apply_left_local(&mut m, 2, 3..5, 3..5);

        let known = vec![
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
            Complex::new(-1.66893e-06, 0.0),
            Complex::new(-9.83413e-07, 0.0),
            Complex::new(1.13034e-07, 0.0),
            Complex::new(-5.03446e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-2.7316e-07, 0.0),
            Complex::new(-2.49894e-07, 0.0),
            Complex::new(3.43689e-07, 0.0),
        ];
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_3() {
        let data = vec![
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
            Complex::new(-1.66893e-06, 0.0),
            Complex::new(-9.83413e-07, 0.0),
            Complex::new(1.13034e-07, 0.0),
            Complex::new(-5.03446e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-9.53674e-07, 0.0),
            Complex::new(-2.7316e-07, 0.0),
            Complex::new(-2.49894e-07, 0.0),
            Complex::new(3.43689e-07, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(1.13359, 0.0),
            beta: Complex::new(-2.17911e-07, 0.0),
        };
        house.apply_right_local(&mut m, 2, 3..5, 0..5);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_4() {
        let data = vec![
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
        let mut m = Matrix::new(5, data);
        let house = Householder::make_householder_local(&mut m, 4, 3);
        assert!((house.beta.real -4.07865e-07).square_norm() < f32::EPSILON);
        assert!((house.tau - Complex::new(0.0, 0.0)).square_norm() < f32::EPSILON);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_4() {
        let data = vec![
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
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new(4.07865e-07, 0.0),
        };
        house.apply_left_local(&mut m, 3, 4..5, 4..5);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_4() {
        let data = vec![
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
        let mut m = Matrix::new(5, data);
        let house = Householder {
            tau: Complex::new(0.0, 0.0),
            beta: Complex::new(4.07865e-07, 0.0),
        };
        house.apply_right_local(&mut m, 3, 4..5, 0..5);

        let known = vec![
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
        m.data().zip(known.iter()).for_each(|(c, k)| {
            assert!((*c - *k).square_norm() < f32::EPSILON);
        })
    }

}

/*        for i in m.data() {
            print!("Complex::new({:?},0.0),", i.real)
        }

*/

use core::{
    borrow::BorrowMut,
    fmt::Debug,
    ops::{Add, Mul},
};
use alloc::vec::Vec;
use alloc::vec;


use super::{float::Float,vector::Vector};

pub struct Matrix<F: Float> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<F>,
}

impl<F: Float> Matrix<F> {
    pub fn new(rows: usize, data: Vec<F>) -> Self {
        debug_assert!(rows > 0);
        let cols = data.len() / rows;
        Matrix {
            rows,
            cols,
            data,
        }
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        debug_assert!(rows > 0);
        let data = vec![F::ZERO; rows * cols];
        Matrix {
            rows,
            cols,
            data,
        }
    }

    pub fn single(rows: usize, cols: usize, c: F) -> Self {
        debug_assert!(rows > 0);
        let data = vec![c; rows * cols];
        Matrix {
            rows,
            cols,
            data,
        }
    }
    pub fn identity(rows: usize, cols: usize) -> Self {
        debug_assert!(rows > 0);
        let data = vec![F::ZERO; rows * cols];
        let mut id = Matrix {
            rows,
            cols,
            data,
        };
        <Vec<&'_ F> as Vector<F>>::add(id.data_diag_ref(), F::IDENTITY);
        id
    }

    pub fn explicit_copy(&self) -> Self {
        let new_data = self.data.clone();

        Self::new(self.rows, new_data)
    }
    #[inline(always)]
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    #[inline(always)]
    pub fn coeff(&self, row: usize, col: usize) -> F {
        self.data[self.index(row, col)]
    }

    #[inline(always)]
    pub fn coeff_ref(&mut self, row: usize, col: usize) -> &mut F {
        let idx = self.index(row, col);
        self.data[idx].borrow_mut()
    }

    #[inline(always)]
    pub fn data(&self) -> impl Iterator<Item = &F> {
        self.data.iter()
    }

    #[inline(always)]
    pub fn data_diag(&self) -> impl Iterator<Item = &F> {
        self.data.iter().step_by(self.cols + 1)
    }

    #[inline(always)]
    pub fn data_col(&self, row: usize) -> impl Iterator<Item = &F> {
        self.data[row..].into_iter().step_by(self.cols)
    }

    #[inline(always)]
    pub fn data_rows(&self) -> impl Iterator<Item = &[F]> {
        self.data.chunks_exact(self.cols)
    }

    #[inline(always)]
    pub fn data_row(&self, col: usize) -> impl Iterator<Item = &F> {
        self.data[col * self.cols..(col + 1) * self.cols].into_iter()
    }

    pub(crate) fn data_north(&self, north: usize) -> impl Iterator<Item = &F> {
        self.data().take(north * self.cols)
    }

    pub(crate) fn data_south(&self, south: usize) -> impl Iterator<Item = &F> {
        self.data().skip((self.rows - south) * self.cols)
    }
    pub(crate) fn data_west(&self, west: usize) -> impl Iterator<Item = &F> {
        self.data()
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    pub(crate) fn data_east(&self, east: usize) -> impl Iterator<Item = &F> {
        self.data()
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols-east)
            .map(|(_, c)| c)
    }

    pub(crate) fn data_north_west(
        &self,
        north: usize,
        west: usize,
    ) -> impl Iterator<Item = &F> {
        self.data_north(north)
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    pub(crate) fn data_north_east(
        &self,
        north: usize,
        east: usize,
    ) -> impl Iterator<Item = &F> {
        self.data_north(north)
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols-east)
            .map(|(_, c)| c)
    }

    pub(crate) fn data_south_west(
        &self,
        south: usize,
        west: usize,
    ) -> impl Iterator<Item = &F> {
        self.data_south(south)
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    pub(crate) fn data_south_east(
        &self,
        south: usize,
        east: usize,
    ) -> impl Iterator<Item = &F> {
        self.data_south(south)
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols-east)
            .map(|(_, c)| c)
    }

    #[inline(always)]
    pub fn data_ref(&mut self) -> impl Iterator<Item = &mut F> {
        self.data.iter_mut()
    }

    #[inline(always)]
    pub fn data_diag_ref(&mut self) -> impl Iterator<Item = &mut F> {
        self.data.iter_mut().step_by(self.cols + 1)
    }

    #[inline(always)]
    pub fn data_col_ref(&mut self, row: usize) -> impl Iterator<Item = &mut F> {
        self.data[row..].iter_mut().step_by(self.cols)
    }

    #[inline(always)]
    pub fn data_rows_ref(&mut self) -> impl Iterator<Item = &mut [F]> {
        self.data.chunks_exact_mut(self.cols)
    }

    #[inline(always)]
    pub fn data_row_ref(&mut self, col: usize) -> impl Iterator<Item = &mut F> {
        self.data[col * self.cols..(col + 1) * self.cols].iter_mut()
    }

    pub(crate) fn data_north_ref(&mut self, north: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_ref().take(north * cols)
    }

    pub(crate) fn data_south_ref(&mut self, south: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        let rows = self.rows;
        self.data_ref().skip((rows - south) * cols)
    }
    pub(crate) fn data_west_ref(&mut self, west: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| i % cols < west)
            .map(|(_, c)| c)
    }
    pub(crate) fn data_east_ref(&mut self, east: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| i % cols >= cols-east)
            .map(|(_, c)| c)
    }

    pub(crate) fn data_north_west_ref(
        &mut self,
        north: usize,
        west: usize,
    ) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_north_ref(north)
            .enumerate()
            .filter(move |(i, _)| i % cols < west)
            .map(|(_, c)| c)
    }
    pub(crate) fn data_north_east_ref(
        &mut self,
        north: usize,
        east: usize,
    ) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_north_ref(north)
            .enumerate()
            .filter(move |(i, _)| i % cols >= cols-east)
            .map(|(_, c)| c)
    }

    pub(crate) fn data_south_west_ref(
        &mut self,
        south: usize,
        west: usize,
    ) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_south_ref(south)
            .enumerate()
            .filter(move |(i, _)| i % cols < west)
            .map(|(_, c)| c)
    }
    pub(crate) fn data_south_east_ref(
        &mut self,
        south: usize,
        east: usize,
    ) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_south_ref(south)
            .enumerate()
            .filter(move |(i, _)| i % cols >= cols-east)
            .map(|(_, c)| c)
    }

    pub fn transposed(&self) -> Self {
        let mut transposed_data = vec![F::ZERO; self.data.len()];

        (0..self.cols).for_each(|i| {
            (0..self.rows).for_each(|j| {
                transposed_data[i * self.rows + j] = self.data[j * self.cols + i];
            })
        });
        Matrix::new(self.cols, transposed_data)
    }

    pub fn col_swap(&mut self, a: usize, b: usize) {
        self.data.chunks_exact_mut(self.cols).for_each(|row| {
            row.swap(a, b);
        });
    }

    pub fn row_swap(&mut self, a: usize, b: usize) {
        let mut rows = self.data_rows_ref();

        let (x, y) = match b >= a {
            true => (a, b),
            false => (b, a),
        };
        let row_x = rows.nth(x).unwrap();
        let row_y = rows.nth(y - x - 1).unwrap();

        row_x.iter_mut().zip(row_y.iter_mut()).for_each(|(ap, bp)| {
            core::mem::swap(&mut (*ap), &mut (*bp));
        });
    }

    /* 
    pub fn covariance(&self) -> Self {
        let mut data = Vec::with_capacity(self.rows * self.rows);

        (0..self.rows).for_each(|i| {
            (0..self.rows).for_each(|j| {
                let x = self.data_row(i);
                let y = self.data_row(j);
                let mx = <Vec<&'_ F> as Vector<T>>::mean(x);
                let my = <Vec<&'_ F> as Vector<T>>::mean(y);
                let x = self.data_row(i);
                let y = self.data_row(j);
                let covariance = x.zip(y).fold(F::ZERO, |acc, (x, y)| {
                    acc + (*x - ComplexFloat::new(mx, T::ZERO)) * (*y - -ComplexFloat::new(my, T::ZERO))
                });
                data.push(covariance / T::usize(self.rows));
            });
        });

        Matrix::new(self.rows, data)
    }*/
}

impl<F: Float> Add<F> for Matrix<F> {
    type Output = Self;
    fn add(self, adder: F) -> Self {
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ F> as Vector<F>>::add(out_matrix.data_ref(), adder);
        out_matrix
    }
}

impl<F: Float> Mul<F> for Matrix<F> {
    type Output = Self;
    fn mul(self, scaler: F) -> Self {
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ F> as Vector<F>>::mul(out_matrix.data_ref(), scaler);
        out_matrix
    }
}

impl<T: Float> Debug for Matrix<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols);
        for row in self.data_rows() {
            write!(f, "{:?}", row).unwrap();
            writeln!(f).unwrap();
        }
        Ok(())
    }
}

#[cfg(test)]
mod matrix_tests {
    use crate::shared::complex::{Complex, ComplexFloat};

    use super::*;

    #[test]
    fn coeff_static() {
        //square case
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2).real - 8.0).l2_norm() < f32::EPSILON);

        //long case
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 3).real - 11.0).l2_norm() < f32::EPSILON);

        //wide case
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 2).real - 11.0).l2_norm() < f32::EPSILON);
    }

    #[test]
    fn coeff_ref_static() {
        //square case
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 8.0).l2_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 4.0).l2_norm() < f32::EPSILON);

        //long case
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 11.0).l2_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 3).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 4.0).l2_norm() < f32::EPSILON);

        //wide case
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 11.0).l2_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(3, 0).real = 5.0;
        m.coeff_ref(3, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 4.0).l2_norm() < f32::EPSILON);
    }

    #[test]
    fn data_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_col_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_col_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let v = m.data_col_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let v = m.data_col_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let v = m.data_col_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
                r[0].real += 1.0;
                r[1].real += 2.0;
                r[2].real += 3.0;
            });
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0] - 1.0).l2_norm() < f32::EPSILON);
                assert!((r[1].real - k[1] - 2.0).l2_norm() < f32::EPSILON);
                assert!((r[2].real - k[2] - 3.0).l2_norm() < f32::EPSILON);
            });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).l2_norm() < f32::EPSILON);
            r[0].real += 1.0;
            r[1].real += 2.0;
            r[2].real += 3.0;
            r[3].real += 4.0;
        });
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0] - 1.0).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).l2_norm() < f32::EPSILON);
            assert!((r[3].real - k[3] - 4.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            r[0].real += 1.0;
            r[1].real += 2.0;
            r[2].real += 3.0;
        });
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0] - 1.0).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).l2_norm() < f32::EPSILON);
        });
    }
    #[test]
    fn transposed_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]
                .into_iter()
                .map(|r| ComplexFloat::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
                .into_iter()
                .map(|r| ComplexFloat::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexFloat::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]
                .into_iter()
                .map(|r| ComplexFloat::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn row_swap_static() {
        let mut m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        m.row_swap(0, 1);
        assert!((m.coeff(0, 0) - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1) - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 2) - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0) - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1) - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 2) - 2.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0) - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 1) - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2) - 8.0).l2_norm() < f32::EPSILON);

        let mut m = Matrix::<f32>::new(3, (0..12).map(|i| i as f32).collect());

        m.row_swap(2, 1);
        assert!((m.coeff(0, 0) - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1) - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 2) - 2.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 3) - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0) - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1) - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 2) - 10.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 3) - 11.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0) - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 1) - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2) - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 3) - 7.0).l2_norm() < f32::EPSILON);

        let mut m = Matrix::<f32>::new(4, (0..12).map(|i| i as f32).collect());

        m.row_swap(3, 1);
        assert!((m.coeff(0, 0) - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1) - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 2) - 2.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0) - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1) - 10.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 2) - 11.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0) - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 1) - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2) - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 0) - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 1) - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 2) - 5.0).l2_norm() < f32::EPSILON);
    }

    #[test]
    fn test_north() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_north(1);
        let kn = (0..3).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north(2);
        let kn = (0..6).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_south() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_south(1);
        let kn = (6..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south(2);
        let kn = (3..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_west() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_west(1);
        let kn = [0, 3, 6].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_west(2);
        let kn = [0, 1, 3, 4, 6, 7].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_west(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }
    #[test]
    fn test_east() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_east(1);
        let kn = [2, 5, 8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_east(2);
        let kn = [1, 2, 4, 5, 7, 8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_east(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_north_east() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_north_east(1,1);
        let kn = [2].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_east(2,1);
        let kn = [2,5].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_east(1,2);
        let kn = [1,2].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_east(2,2);
        let kn = [1,2,4,5].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_north_west() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_north_west(1,1);
        let kn = [0].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_west(2,1);
        let kn = [0,3].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_west(1,2);
        let kn = [0,1].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_west(2,2);
        let kn = [0,1,3,4].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_south_east() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_south_east(1,1);
        let kn = [8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_east(2,1);
        let kn = [5,8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_east(1,2);
        let kn = [7,8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_east(2,2);
        let kn = [4,5,7,8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }
    #[test]
    fn test_south_west() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_south_west(1,1);
        let kn = [6].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_west(2,1);
        let kn = [3,6].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_west(1,2);
        let kn = [6,7].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_west(2,2);
        let kn = [3,4,6,7].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }
}

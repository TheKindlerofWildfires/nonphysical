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
        self.data[row..].iter().step_by(self.cols)
    }

    #[inline(always)]
    pub fn data_rows(&self) -> impl Iterator<Item = &[F]> {
        self.data.chunks_exact(self.cols)
    }

    #[inline(always)]
    pub fn data_row(&self, col: usize) -> impl Iterator<Item = &F> {
        self.data[col * self.cols..(col + 1) * self.cols].iter()
    }

    pub fn data_north(&self, north: usize) -> impl Iterator<Item = &F> {
        self.data().take(north * self.cols)
    }

    pub fn data_south(&self, south: usize) -> impl Iterator<Item = &F> {
        self.data().skip((self.rows - south) * self.cols)
    }
    pub fn data_west(&self, west: usize) -> impl Iterator<Item = &F> {
        self.data()
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    pub fn data_east(&self, east: usize) -> impl Iterator<Item = &F> {
        self.data()
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols-east)
            .map(|(_, c)| c)
    }

    pub fn data_north_west(
        &self,
        north: usize,
        west: usize,
    ) -> impl Iterator<Item = &F> {
        self.data_north(north)
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    pub fn data_north_east(
        &self,
        north: usize,
        east: usize,
    ) -> impl Iterator<Item = &F> {
        self.data_north(north)
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols-east)
            .map(|(_, c)| c)
    }

    pub fn data_south_west(
        &self,
        south: usize,
        west: usize,
    ) -> impl Iterator<Item = &F> {
        self.data_south(south)
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    pub fn data_south_east(
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

    pub fn data_north_ref(&mut self, north: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_ref().take(north * cols)
    }

    pub fn data_south_ref(&mut self, south: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        let rows = self.rows;
        self.data_ref().skip((rows - south) * cols)
    }
    pub fn data_west_ref(&mut self, west: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| i % cols < west)
            .map(|(_, c)| c)
    }
    pub fn data_east_ref(&mut self, east: usize) -> impl Iterator<Item = &mut F> {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| i % cols >= cols-east)
            .map(|(_, c)| c)
    }

    pub fn data_north_west_ref(
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
    pub fn data_north_east_ref(
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

    pub fn data_south_west_ref(
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
    pub fn data_south_east_ref(
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
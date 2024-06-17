use std::{
    borrow::BorrowMut,
    ops::{Add, Mul, MulAssign},
};

use super::{complex::Complex, float::Float, vector::Vector};

pub struct Matrix<T: Float> {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<Complex<T>>,
}

impl<T: Float> Matrix<T> {
    pub fn new(rows: usize, data: Vec<Complex<T>>) -> Self {
        let columns = data.len() / rows;
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn zero(rows: usize, columns: usize) -> Self {
        let data = vec![Complex::<T>::zero(); rows * columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn identity(rows: usize, columns: usize) -> Self {
        let data = vec![Complex::<T>::zero(); rows * columns];
        let mut id = Matrix {
            rows,
            columns,
            data,
        };
        <Vec<&'_ Complex<T>> as Vector<T>>::add(id.data_diag_ref(), Complex::<T>::one());
        id
    }
    #[inline(always)]
    fn index(&self, row: usize, column: usize) -> usize {
        column * self.rows + row
    }

    #[inline(always)]
    pub fn coeff(&self, row: usize, column: usize) -> Complex<T> {
        self.data[self.index(row, column)]
    }

    #[inline(always)]
    pub fn coeff_ref(&mut self, row: usize, column: usize) -> &mut Complex<T> {
        let idx = self.index(row, column);
        self.data[idx].borrow_mut()
    }

    #[inline(always)]
    pub fn row(&self, index: usize) -> usize {
        index % self.rows
    }

    #[inline(always)]
    pub fn column(&self, index: usize) -> usize {
        index / self.rows
    }

    #[inline(always)]
    pub fn data(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter()
    }

    #[inline(always)]
    pub fn data_diag(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter().step_by(self.rows + 1)
    }

    #[inline(always)]
    pub fn data_row(&self, row: usize) -> impl Iterator<Item = &Complex<T>> {
        self.data[row..].iter().step_by(self.columns)
    }

    #[inline(always)]
    pub fn data_rows(&self) -> impl Iterator<Item = &[Complex<T>]> {
        self.data.chunks_exact(self.columns)
    }

    #[inline(always)]
    pub fn data_column(&self, column: usize) -> impl Iterator<Item = &Complex<T>> {
        self.data[column * self.columns..(column + 1) * self.columns].iter()
    }

    #[inline(always)]
    pub fn data_ref(&mut self) -> impl Iterator<Item = &mut Complex<T>> {
        self.data.iter_mut()
    }

    #[inline(always)]
    pub fn data_diag_ref(&mut self) -> impl Iterator<Item = &mut Complex<T>> {
        self.data.iter_mut().step_by(self.columns + 1)
    }

    #[inline(always)]
    pub fn data_row_ref(&mut self, row: usize) -> impl Iterator<Item = &mut Complex<T>> {
        self.data[row..].iter_mut().step_by(self.columns)
    }

    #[inline(always)]
    pub fn data_rows_ref(&mut self) -> impl Iterator<Item = &mut [Complex<T>]> {
        self.data.chunks_exact_mut(self.columns)
    }

    #[inline(always)]
    pub fn data_column_ref(&mut self, column: usize) -> impl Iterator<Item = &mut Complex<T>> {
        self.data[column * self.columns..(column + 1) * self.columns].iter_mut()
    }

    pub fn transpose(&mut self) {
        (0..self.columns).for_each(|i| {
            (i..self.rows).for_each(|j| {
                let tmp = self.data[i * self.columns + j];
                self.data[i * self.rows + j] = self.data[j * self.rows + i];
                self.data[j * self.columns + i] = tmp;
            });
        });
        let temp = self.columns;
        self.columns = self.rows;
        self.rows = temp;
    }

    pub fn transposed(&self) -> Self {
        let mut transposed_data = vec![Complex::<T>::zero(); self.data.len()];

        (0..self.columns).for_each(|i| {
            (i..self.rows).for_each(|j| {
                transposed_data[i * self.rows + j] = self.data[j * self.columns + i];
                transposed_data[j * self.columns + i] = self.data[i * self.rows + j];
            })
        });
        Matrix::<T>::new(self.columns, transposed_data)
    }

    pub fn row_swap(&mut self, a: usize, b: usize) {
        self.data.chunks_exact_mut(self.columns).for_each(|row| {
            let tmp = row[a];
            row[a] = row[b];
            row[b] = tmp;
        });
    }

    pub fn col_swap(&mut self, a: usize, b: usize) {
        let mut rows = self.data_rows_ref();
        let (row_a, row_b) = match b > a {
            true => {
                let ra = rows.nth(a).unwrap();
                let rb = rows.nth(b - a - 1).unwrap();
                (ra, rb)
            }
            false => {
                let rb = rows.nth(b).unwrap();
                let ra = rows.nth(a - b - 1).unwrap();

                (ra, rb)
            }
        };

        row_a.iter_mut().zip(row_b.iter_mut()).for_each(|(ap, bp)| {
            let tmp = *ap;
            *ap = *bp;
            *bp = tmp;
        });
    }
}
impl<T: Float> Add for Matrix<T> {
    type Output = Self;
    fn add(self, rhs: Matrix<T>) -> Self {
        let out_data = self.data().map(|c| c.clone()).collect();
        let mut out_matrix = Matrix::new(self.rows, out_data);
        <Vec<&'_ Complex<T>> as Vector<T>>::acc(out_matrix.data_ref(), rhs.data());
        out_matrix
    }
}

/* This is a bad multiplication impl */
impl<T: Float> Mul for Matrix<T> {
    type Output = Self;
    fn mul(self, rhs: Matrix<T>) -> Self {
        debug_assert!(self.columns == rhs.rows);
        
        let mut output = Matrix::new(
            self.rows,
            vec![Complex::<T>::zero(); self.rows * rhs.columns],
        );

        for c in 0..rhs.columns {
            for r in 0..self.rows {
                let mut tmp = Complex::<T>::zero();
                for a in 0..self.columns {
                    tmp = tmp + self.coeff(r, a) * rhs.coeff(a, c);
                }
                *output.coeff_ref(r, c) = tmp;
            }
        }
        output
    }
}


impl<T: Float> Mul<T> for Matrix<T> {
    type Output = Self;
    fn mul(self, scaler: T) -> Self {
        let out_data = self.data().map(|c| c.clone()).collect();
        let mut out_matrix = Matrix::new(self.rows, out_data);
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(out_matrix.data_ref(), scaler);
        out_matrix
    }
}

impl<T: Float> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(self.data_ref(), rhs);
    }
}

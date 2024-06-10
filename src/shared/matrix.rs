use std::{
    borrow::BorrowMut,
    ops::{Mul, MulAssign},
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
        let data = vec![Complex::<T>::zero();rows*columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn identity(rows: usize, columns: usize) -> Self {
        let data = vec![Complex::<T>::zero();rows*columns];
        let mut id = Matrix {
            rows,
            columns,
            data,
        };
        <Vec<&'_ Complex<T>> as Vector<T>>::add(id.data_diag_ref(),Complex::<T>::new(T::usize(1), T::zero()));
        id
    }
    #[inline(always)]
    fn index(&self, row: usize, column: usize) -> usize {
        column *self.rows +row
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
    pub fn row(&self, index: usize) -> usize{
        index % self.rows
    }

    #[inline(always)]
    pub fn column(&self, index: usize) -> usize{
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
    pub fn data_column_ref(&mut self, column: usize) ->impl Iterator<Item = &mut Complex<T>> {
        self.data[column * self.columns..(column + 1) * self.columns].iter_mut()
    }

    pub fn transpose(&mut self){
        let mut transposed_matrix = vec![Complex::<T>::zero();self.data.len()];
        (0..self.columns).for_each(|i|{
            (0..self.rows).for_each(|j|{
                transposed_matrix[i*self.columns+j] = self.data[j*self.rows+i];
            });
        });
        let temp = self.columns;
        self.columns = self.rows;
        self.rows = temp;
        self.data = transposed_matrix;

    }

    pub fn row_swap(&mut self, a: usize, b: usize){
        let xb:Vec<_> = self.data_row(a).map(|c| c.clone()).collect();
        let yb:Vec<_> = self.data_row(b).map(|c| c.clone()).collect();

        let x = self.data_row_ref(a);
        x.zip(yb).for_each(|(xp,yp)|{
            *xp = yp;
        });
        let y= self.data_row_ref(b);
        y.zip(xb).for_each(|(yp,xp)|{
           *yp = xp; 
        })
    }

    pub fn col_swap(&mut self, a: usize, b: usize){
        let xb:Vec<_> = self.data_column(a).map(|c| c.clone()).collect();
        let yb:Vec<_> = self.data_column(b).map(|c| c.clone()).collect();

        let x = self.data_column_ref(a);
        x.zip(yb).for_each(|(xp,yp)|{
            *xp = yp;
        });
        let y= self.data_column_ref(b);
        y.zip(xb).for_each(|(yp,xp)|{
           *yp = xp; 
        })
    }
}

impl<T: Float> Mul<T> for Matrix<T> {
    type Output = Self;
    fn mul(self, scaler: T) -> Self {
        Self {
            rows: self.rows,
            columns: self.columns,
            data: self.data.iter().map(|c| *c * scaler).collect(),
        }
    }
}

impl<T: Float> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|c| *c = *c * rhs);
    }
}

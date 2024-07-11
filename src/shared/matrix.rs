<<<<<<< HEAD
use core::{
    borrow::BorrowMut,
    fmt::Debug,
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
        debug_assert!(rows > 0);
        let columns = data.len() / rows;
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn zero(rows: usize, columns: usize) -> Self {
        debug_assert!(rows > 0);
        let data = vec![Complex::ZERO; rows * columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn single(rows: usize, columns: usize, c: Complex<T>) -> Self {
        debug_assert!(rows > 0);
        let data = vec![c; rows * columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn identity(rows: usize, columns: usize) -> Self {
        debug_assert!(rows > 0);
        let data = vec![Complex::ZERO; rows * columns];
        let mut id = Matrix {
            rows,
            columns,
            data,
        };
        <Vec<&'_ Complex<T>> as Vector<T>>::add(id.data_diag_ref(), Complex::ONE);
        id
    }

    pub fn explicit_copy(&self) -> Self {
        let new_data = self.data.clone();

        Self::new(self.rows, new_data)
    }
    #[inline(always)]
    pub fn index(&self, row: usize, column: usize) -> usize {
        row * self.columns + column
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
    pub fn data(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter()
    }

    #[inline(always)]
    pub fn data_diag(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter().step_by(self.columns + 1)
    }

    #[inline(always)]
    pub fn data_column(&self, row: usize) -> impl Iterator<Item = &Complex<T>> {
        self.data[row..].iter().step_by(self.columns)
    }

    #[inline(always)]
    pub fn data_rows(&self) -> impl Iterator<Item = &[Complex<T>]> {
        self.data.chunks_exact(self.columns)
    }

    #[inline(always)]
    pub fn data_row(&self, column: usize) -> impl Iterator<Item = &Complex<T>> {
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
    pub fn data_column_ref(&mut self, row: usize) -> impl Iterator<Item = &mut Complex<T>> {
        self.data[row..].iter_mut().step_by(self.columns)
    }

    #[inline(always)]
    pub fn data_rows_ref(&mut self) -> impl Iterator<Item = &mut [Complex<T>]> {
        self.data.chunks_exact_mut(self.columns)
    }

    #[inline(always)]
    pub fn data_row_ref(&mut self, column: usize) -> impl Iterator<Item = &mut Complex<T>> {
        self.data[column * self.columns..(column + 1) * self.columns].iter_mut()
    }

    pub fn transposed(&self) -> Self {
        let mut transposed_data = vec![Complex::ZERO; self.data.len()];

        (0..self.columns).for_each(|i| {
            (0..self.rows).for_each(|j| {
                transposed_data[i * self.rows + j] = self.data[j * self.columns + i];
            })
        });
        Matrix::new(self.columns, transposed_data)
    }

    pub fn col_swap(&mut self, a: usize, b: usize) {
        self.data.chunks_exact_mut(self.columns).for_each(|row| {
            let tmp = row[a];
            row[a] = row[b];
            row[b] = tmp;
        });
    }

    pub fn row_swap(&mut self, a: usize, b: usize) {
        let mut rows = self.data_rows_ref();
        
        let (x,y) = match b>=a{
            true =>{
                (a,b)
            }
            false => {
                (b,a)
            }
        };
        let row_x = rows.nth(x).unwrap();
        let row_y = rows.nth(y-x-1).unwrap();

        row_x.iter_mut().zip(row_y.iter_mut()).for_each(|(ap, bp)| {
            let tmp = *ap;
            *ap = *bp;
            *bp = tmp;
        });
    }


    pub fn acc(&self, other: &Matrix<T>) -> Matrix<T> {
        debug_assert!(self.columns == other.columns);
        debug_assert!(self.rows == other.rows);
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ Complex<T>> as Vector<T>>::acc(out_matrix.data_ref(), other.data());
        out_matrix
    }

    
}

impl<T: Float> Add<Complex<T>> for Matrix<T> {
    type Output = Self;
    fn add(self, adder: Complex<T>) -> Self {
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ Complex<T>> as Vector<T>>::add(out_matrix.data_ref(), adder);
        out_matrix
    }
}

impl<T: Float> Mul<T> for Matrix<T> {
    type Output = Self;
    fn mul(self, scaler: T) -> Self {
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(out_matrix.data_ref(), scaler);
        out_matrix
    }
}

impl<T: Float> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(self.data_ref(), rhs);
    }
}

impl<T: Float> Debug for Matrix<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("columns", &self.columns);
        for row in self.data_rows() {
            write!(f, "{:?}", row).unwrap();
            writeln!(f).unwrap();
        }
        Ok(())
    }
}

#[cfg(test)]
mod matrix_tests {

    use super::*;

    #[test]
    fn coeff_static() {
        //square case
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2).real - 8.0).square_norm() < f32::EPSILON);

        //long case
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 3).real - 11.0).square_norm() < f32::EPSILON);

        //wide case
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(3, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(3, 2).real - 11.0).square_norm() < f32::EPSILON);
    }

    #[test]
    fn coeff_ref_static() {
        //square case
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 8.0).square_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 4.0).square_norm() < f32::EPSILON);

        //long case
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 11.0).square_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 3).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 4.0).square_norm() < f32::EPSILON);

        //wide case
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 11.0).square_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(3, 0).real = 5.0;
        m.coeff_ref(3, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 4.0).square_norm() < f32::EPSILON);
    }

    #[test]
    fn data_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_column_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_column_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let v = m.data_column_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let v = m.data_column_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let v = m.data_column_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
            });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
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
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
                r[0].real += 1.0;
                r[1].real += 2.0;
                r[2].real += 3.0;
            });
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0] - 1.0).square_norm() < f32::EPSILON);
                assert!((r[1].real - k[1] - 2.0).square_norm() < f32::EPSILON);
                assert!((r[2].real - k[2] - 3.0).square_norm() < f32::EPSILON);
            });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).square_norm() < f32::EPSILON);
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
            assert!((r[0].real - k[0] - 1.0).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).square_norm() < f32::EPSILON);
            assert!((r[3].real - k[3] - 4.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
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
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
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
            assert!((r[0].real - k[0] - 1.0).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).square_norm() < f32::EPSILON);
        });
    }
    #[test]
    fn transposed_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).square_norm() < f32::EPSILON);
        });
    }


    #[test]
    fn acc_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| Complex::<f32>::new(c as f32, 0.0)).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let r1 = m33.acc(&m33);
        let r2 = m34.acc(&m34);
        let r3 = m43.acc(&m43);

        let k1 = Matrix::<f32>::new(
            3,
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });

        let k2 = Matrix::<f32>::new(
            3,
            [
                0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k3 = Matrix::<f32>::new(
            4,
            [
                0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn row_swap_static(){
        let mut m = Matrix::<f32>::new(3, (0..9).map(|i| Complex::new(i as f32,0.0)).collect());

        m.row_swap(0, 1);
        assert!((m.coeff(0, 0).real-3.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 1).real-4.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 2).real-5.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 0).real-0.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 1).real-1.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 2).real-2.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 0).real-6.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 1).real-7.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 2).real-8.0).square_norm() <f32::EPSILON);


        let mut m = Matrix::<f32>::new(3, (0..12).map(|i| Complex::new(i as f32,0.0)).collect());

        m.row_swap(2, 1);
        assert!((m.coeff(0, 0).real-0.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 1).real-1.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 2).real-2.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 3).real-3.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 0).real-8.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 1).real-9.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 2).real-10.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 3).real-11.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 0).real-4.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 1).real-5.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 2).real-6.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 3).real-7.0).square_norm() <f32::EPSILON);

        let mut m = Matrix::<f32>::new(4, (0..12).map(|i| Complex::new(i as f32,0.0)).collect());

        m.row_swap(3, 1);
        assert!((m.coeff(0, 0).real-0.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 1).real-1.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 2).real-2.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 0).real-9.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 1).real-10.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 2).real-11.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 0).real-6.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 1).real-7.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 2).real-8.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(3, 0).real-3.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(3, 1).real-4.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(3, 2).real-5.0).square_norm() <f32::EPSILON);

        
    }
}
=======
use core::{
    borrow::BorrowMut,
    fmt::Debug,
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
        debug_assert!(rows > 0);
        let columns = data.len() / rows;
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn zero(rows: usize, columns: usize) -> Self {
        debug_assert!(rows > 0);
        let data = vec![Complex::ZERO; rows * columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn single(rows: usize, columns: usize, c: Complex<T>) -> Self {
        debug_assert!(rows > 0);
        let data = vec![c; rows * columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn identity(rows: usize, columns: usize) -> Self {
        debug_assert!(rows > 0);
        let data = vec![Complex::ZERO; rows * columns];
        let mut id = Matrix {
            rows,
            columns,
            data,
        };
        <Vec<&'_ Complex<T>> as Vector<T>>::add(id.data_diag_ref(), Complex::ONE);
        id
    }

    pub fn explicit_copy(&self) -> Self {
        let new_data = self.data.clone();

        Self::new(self.rows, new_data)
    }
    #[inline(always)]
    pub fn index(&self, row: usize, column: usize) -> usize {
        row * self.columns + column
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
    pub fn data(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter()
    }

    #[inline(always)]
    pub fn data_diag(&self) -> impl Iterator<Item = &Complex<T>> {
        self.data.iter().step_by(self.columns + 1)
    }

    #[inline(always)]
    pub fn data_column(&self, row: usize) -> impl Iterator<Item = &Complex<T>> {
        self.data[row..].iter().step_by(self.columns)
    }

    #[inline(always)]
    pub fn data_rows(&self) -> impl Iterator<Item = &[Complex<T>]> {
        self.data.chunks_exact(self.columns)
    }

    #[inline(always)]
    pub fn data_row(&self, column: usize) -> impl Iterator<Item = &Complex<T>> {
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
    pub fn data_column_ref(&mut self, row: usize) -> impl Iterator<Item = &mut Complex<T>> {
        self.data[row..].iter_mut().step_by(self.columns)
    }

    #[inline(always)]
    pub fn data_rows_ref(&mut self) -> impl Iterator<Item = &mut [Complex<T>]> {
        self.data.chunks_exact_mut(self.columns)
    }

    #[inline(always)]
    pub fn data_row_ref(&mut self, column: usize) -> impl Iterator<Item = &mut Complex<T>> {
        self.data[column * self.columns..(column + 1) * self.columns].iter_mut()
    }

    pub fn transposed(&self) -> Self {
        let mut transposed_data = vec![Complex::ZERO; self.data.len()];

        (0..self.columns).for_each(|i| {
            (0..self.rows).for_each(|j| {
                transposed_data[i * self.rows + j] = self.data[j * self.columns + i];
            })
        });
        Matrix::new(self.columns, transposed_data)
    }

    pub fn col_swap(&mut self, a: usize, b: usize) {
        self.data.chunks_exact_mut(self.columns).for_each(|row| {
            let tmp = row[a];
            row[a] = row[b];
            row[b] = tmp;
        });
    }

    pub fn row_swap(&mut self, a: usize, b: usize) {
        let mut rows = self.data_rows_ref();
        
        let (x,y) = match b>=a{
            true =>{
                (a,b)
            }
            false => {
                (b,a)
            }
        };
        let row_x = rows.nth(x).unwrap();
        let row_y = rows.nth(y-x-1).unwrap();

        row_x.iter_mut().zip(row_y.iter_mut()).for_each(|(ap, bp)| {
            let tmp = *ap;
            *ap = *bp;
            *bp = tmp;
        });
    }


    pub fn acc(&self, other: &Matrix<T>) -> Matrix<T> {
        debug_assert!(self.columns == other.columns);
        debug_assert!(self.rows == other.rows);
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ Complex<T>> as Vector<T>>::acc(out_matrix.data_ref(), other.data());
        out_matrix
    }

    
}

impl<T: Float> Add<Complex<T>> for Matrix<T> {
    type Output = Self;
    fn add(self, adder: Complex<T>) -> Self {
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ Complex<T>> as Vector<T>>::add(out_matrix.data_ref(), adder);
        out_matrix
    }
}

impl<T: Float> Mul<T> for Matrix<T> {
    type Output = Self;
    fn mul(self, scaler: T) -> Self {
        let mut out_matrix = self.explicit_copy();
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(out_matrix.data_ref(), scaler);
        out_matrix
    }
}

impl<T: Float> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(self.data_ref(), rhs);
    }
}

impl<T: Float> Debug for Matrix<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("columns", &self.columns);
        for row in self.data_rows() {
            write!(f, "{:?}", row).unwrap();
            writeln!(f).unwrap();
        }
        Ok(())
    }
}

#[cfg(test)]
mod matrix_tests {

    use super::*;

    #[test]
    fn coeff_static() {
        //square case
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2).real - 8.0).square_norm() < f32::EPSILON);

        //long case
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(2, 3).real - 11.0).square_norm() < f32::EPSILON);

        //wide case
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(3, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff(3, 2).real - 11.0).square_norm() < f32::EPSILON);
    }

    #[test]
    fn coeff_ref_static() {
        //square case
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 8.0).square_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 4.0).square_norm() < f32::EPSILON);

        //long case
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 11.0).square_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 3).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 4.0).square_norm() < f32::EPSILON);

        //wide case
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 11.0).square_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(3, 0).real = 5.0;
        m.coeff_ref(3, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 5.0).square_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 4.0).square_norm() < f32::EPSILON);
    }

    #[test]
    fn data_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_column_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_column_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let v = m.data_column_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let v = m.data_column_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).square_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).square_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let v = m.data_column_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
            });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).square_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
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
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
                r[0].real += 1.0;
                r[1].real += 2.0;
                r[2].real += 3.0;
            });
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0] - 1.0).square_norm() < f32::EPSILON);
                assert!((r[1].real - k[1] - 2.0).square_norm() < f32::EPSILON);
                assert!((r[2].real - k[2] - 3.0).square_norm() < f32::EPSILON);
            });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).square_norm() < f32::EPSILON);
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
            assert!((r[0].real - k[0] - 1.0).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).square_norm() < f32::EPSILON);
            assert!((r[3].real - k[3] - 4.0).square_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
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
            assert!((r[0].real - k[0]).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).square_norm() < f32::EPSILON);
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
            assert!((r[0].real - k[0] - 1.0).square_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).square_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).square_norm() < f32::EPSILON);
        });
    }
    #[test]
    fn transposed_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).square_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).square_norm() < f32::EPSILON);
        });
    }


    #[test]
    fn acc_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| Complex::<f32>::new(c as f32, 0.0)).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let r1 = m33.acc(&m33);
        let r2 = m34.acc(&m34);
        let r3 = m43.acc(&m43);

        let k1 = Matrix::<f32>::new(
            3,
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });

        let k2 = Matrix::<f32>::new(
            3,
            [
                0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k3 = Matrix::<f32>::new(
            4,
            [
                0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn row_swap_static(){
        let mut m = Matrix::<f32>::new(3, (0..9).map(|i| Complex::new(i as f32,0.0)).collect());

        m.row_swap(0, 1);
        assert!((m.coeff(0, 0).real-3.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 1).real-4.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 2).real-5.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 0).real-0.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 1).real-1.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 2).real-2.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 0).real-6.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 1).real-7.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 2).real-8.0).square_norm() <f32::EPSILON);


        let mut m = Matrix::<f32>::new(3, (0..12).map(|i| Complex::new(i as f32,0.0)).collect());

        m.row_swap(2, 1);
        assert!((m.coeff(0, 0).real-0.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 1).real-1.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 2).real-2.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 3).real-3.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 0).real-8.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 1).real-9.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 2).real-10.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 3).real-11.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 0).real-4.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 1).real-5.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 2).real-6.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 3).real-7.0).square_norm() <f32::EPSILON);

        let mut m = Matrix::<f32>::new(4, (0..12).map(|i| Complex::new(i as f32,0.0)).collect());

        m.row_swap(3, 1);
        assert!((m.coeff(0, 0).real-0.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 1).real-1.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(0, 2).real-2.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 0).real-9.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 1).real-10.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(1, 2).real-11.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 0).real-6.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 1).real-7.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(2, 2).real-8.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(3, 0).real-3.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(3, 1).real-4.0).square_norm() <f32::EPSILON);
        assert!((m.coeff(3, 2).real-5.0).square_norm() <f32::EPSILON);

        
    }
}
>>>>>>> master

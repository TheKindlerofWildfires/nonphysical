use std::{
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
        let columns = data.len() / rows;
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn zero(rows: usize, columns: usize) -> Self {
        let data = vec![Complex::zero(); rows * columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn single(rows: usize, columns: usize, c: Complex<T>) -> Self {
        let data = vec![c; rows * columns];
        Matrix {
            rows,
            columns,
            data,
        }
    }

    pub fn identity(rows: usize, columns: usize) -> Self {
        let data = vec![Complex::zero(); rows * columns];
        let mut id = Matrix {
            rows,
            columns,
            data,
        };
        <Vec<&'_ Complex<T>> as Vector<T>>::add(id.data_diag_ref(), Complex::one());
        id
    }

    pub fn explicit_copy(&self) -> Self {
        let new_data = self.data.clone();

        Self::new(self.rows, new_data)
    }
    #[inline(always)]
    fn index(&self, row: usize, column: usize) -> usize {
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

    pub fn transpose(&mut self) {
        (0..self.columns).for_each(|i| {
            (0..self.rows).for_each(|j| {
                let tmp = self.data[i * self.rows + j];
                self.data[i * self.rows + j] = self.data[j * self.columns + i];
                self.data[j * self.columns + i] = tmp;
            });
        });
        let temp = self.columns;
        self.columns = self.rows;
        self.rows = temp;
    }

    pub fn transposed(&self) -> Self {
        let mut transposed_data = vec![Complex::zero(); self.data.len()];

        (0..self.columns).for_each(|i| {
            (0..self.rows).for_each(|j| {
                transposed_data[i * self.rows + j] = self.data[j * self.columns + i];
            })
        });
        Matrix::new(self.columns, transposed_data)
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
    pub fn dot(&self, other: &Matrix<T>) -> Matrix<T> {
        debug_assert!(self.columns == other.rows);

        let mut output = Matrix::new(self.rows, vec![Complex::zero(); self.rows * other.columns]);
        for c in 0..other.columns {
            for r in 0..self.rows {
                let mut tmp = Complex::zero();
                for a in 0..self.columns {
                    tmp = tmp + self.coeff(r, a) * other.coeff(a, c);
                }
                *output.coeff_ref(r, c) = tmp;
            }
        }
        output
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    fn test_coeff() {
        //square case
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::epsilon());
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::epsilon());
        assert!((m.coeff(1, 0).real - 3.0).square_norm() < f32::epsilon());
        assert!((m.coeff(1, 1).real - 4.0).square_norm() < f32::epsilon());
        assert!((m.coeff(2, 0).real - 6.0).square_norm() < f32::epsilon());
        assert!((m.coeff(2, 2).real - 8.0).square_norm() < f32::epsilon());

        //long case
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::epsilon());
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::epsilon());
        assert!((m.coeff(1, 0).real - 4.0).square_norm() < f32::epsilon());
        assert!((m.coeff(1, 1).real - 5.0).square_norm() < f32::epsilon());
        assert!((m.coeff(2, 0).real - 8.0).square_norm() < f32::epsilon());
        assert!((m.coeff(2, 3).real - 11.0).square_norm() < f32::epsilon());

        //wide case
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).square_norm() < f32::epsilon());
        assert!((m.coeff(0, 1).real - 1.0).square_norm() < f32::epsilon());
        assert!((m.coeff(1, 0).real - 3.0).square_norm() < f32::epsilon());
        assert!((m.coeff(1, 1).real - 4.0).square_norm() < f32::epsilon());
        assert!((m.coeff(3, 0).real - 9.0).square_norm() < f32::epsilon());
        assert!((m.coeff(3, 2).real - 11.0).square_norm() < f32::epsilon());
    }

    #[test]
    fn test_coeff_ref() {
        //square case
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 0).real - 3.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 1).real - 4.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 0).real - 6.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 2).real - 8.0).square_norm() < f32::epsilon());

        m.coeff_ref(0,0).real=9.0;
        m.coeff_ref(0,1).real=8.0;
        m.coeff_ref(1,0).real=7.0;
        m.coeff_ref(1,1).real=6.0;
        m.coeff_ref(2,0).real=5.0;
        m.coeff_ref(2,2).real=4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 0).real - 5.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 2).real - 4.0).square_norm() < f32::epsilon());

        //long case
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 0).real - 4.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 1).real - 5.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 0).real - 8.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 3).real - 11.0).square_norm() < f32::epsilon());

        m.coeff_ref(0,0).real=9.0;
        m.coeff_ref(0,1).real=8.0;
        m.coeff_ref(1,0).real=7.0;
        m.coeff_ref(1,1).real=6.0;
        m.coeff_ref(2,0).real=5.0;
        m.coeff_ref(2,3).real=4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 0).real - 5.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(2, 3).real - 4.0).square_norm() < f32::epsilon());

        //wide case
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(0, 1).real - 1.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 0).real - 3.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 1).real - 4.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(3, 0).real - 9.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(3, 2).real - 11.0).square_norm() < f32::epsilon());


        m.coeff_ref(0,0).real=9.0;
        m.coeff_ref(0,1).real=8.0;
        m.coeff_ref(1,0).real=7.0;
        m.coeff_ref(1,1).real=6.0;
        m.coeff_ref(3,0).real=5.0;
        m.coeff_ref(3,2).real=4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(0, 1).real - 8.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 0).real - 7.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(1, 1).real - 6.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(3, 0).real - 5.0).square_norm() < f32::epsilon());
        assert!((m.coeff_ref(3, 2).real - 4.0).square_norm() < f32::epsilon());
    }

    #[test]
    fn test_data() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data();
        d.zip(0..9).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
        });
    }

    #[test]
    fn test_data_ref() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_ref();
        d.zip(0..9).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
            c.real *=2.0;
        });
        let d = m.data_ref();
        d.zip(0..9).for_each(|(c,i)| {
            assert!((c.real-(i as f32)*2.0).square_norm()< f32::epsilon());
        });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
            c.real*=3.0;
        });
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c,i)| {
            assert!((c.real-(i as f32)*3.0).square_norm()< f32::epsilon());
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
            c.real*=4.0;
        });

        let d = m.data_ref();
        d.zip(0..12).for_each(|(c,i)| {
            assert!((c.real-(i as f32)*4.0).square_norm()< f32::epsilon());
        });

    }

    #[test]
    fn test_data_diag() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag();
        d.zip((0..9).step_by(4)).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(5)).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(4)).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
        });
    }

    #[test]
    fn test_data_diag_ref() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
            c.real *=2.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c,i)| {
            assert!((c.real-(i as f32)*2.0).square_norm()< f32::epsilon());
        });

        
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
            c.real*=3.0;
        });
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c,i)| {
            assert!((c.real-(i as f32)*3.0).square_norm()< f32::epsilon());
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(4)).for_each(|(c,i)| {
            assert!((c.real-i as f32).square_norm()< f32::epsilon());
            c.real*=4.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c,i)| {
            assert!((c.real-(i as f32)*4.0).square_norm()< f32::epsilon());
        });

    }

    #[test]
    fn test_data_row() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(0);
        r.zip(vec![0.0,1.0,2.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(1);
        r.zip(vec![4.0,5.0,6.0,7.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(2);
        r.zip(vec![6.0,7.0,8.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_data_row_ref() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(0);
        r.zip(vec![0.0,1.0,2.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
            c.real *= 2.0;
        });
        let r = m.data_row_ref(0);
        r.zip(vec![0.0,1.0,2.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)*2.0).square_norm() < f32::epsilon());
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(1);
        r.zip(vec![4.0,5.0,6.0,7.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
            c.real*=3.0;
        });
        let r = m.data_row_ref(1);
        r.zip(vec![4.0,5.0,6.0,7.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)*3.0).square_norm() < f32::epsilon());
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(2);
        r.zip(vec![6.0,7.0,8.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
            c.real*=4.0;
        });

        let r = m.data_row_ref(2);
        r.zip(vec![6.0,7.0,8.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)*4.0).square_norm() < f32::epsilon());
        });

    }


    #[test]
    fn test_data_column() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(0);
        v.zip(vec![0.0,3.0,6.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(1);
        v.zip(vec![1.0,5.0,9.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column(2);
        v.zip(vec![2.0,5.0,8.0,11.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_data_column_ref() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(0);
        v.zip(vec![0.0,3.0,6.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
            c.real *= 2.0;
        });
        let v = m.data_column_ref(0);
        v.zip(vec![0.0,3.0,6.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)*2.0).square_norm() < f32::epsilon());
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(1);
        v.zip(vec![1.0,5.0,9.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
            c.real*=3.0;
        });
        let v = m.data_column_ref(1);
        v.zip(vec![1.0,5.0,9.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)*3.0).square_norm() < f32::epsilon());
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_column_ref(2);
        v.zip(vec![2.0,5.0,8.0,11.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)).square_norm() < f32::epsilon());
            c.real*=4.0;
        });

        let v = m.data_column_ref(2);
        v.zip(vec![2.0,5.0,8.0,11.0]).for_each(|(c,i)|{
            assert!((c.real-(i as f32)*4.0).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_data_rows() {}

    #[test]
    fn test_data_rows_ref() {}

    #[test]
    fn test_transpose() {}

    #[test]
    fn test_transposed() {}

    #[test]
    fn test_dot() {}

    #[test]
    fn test_acc() {}
}

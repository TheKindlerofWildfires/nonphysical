use crate::shared::{float::Float, vector::{float_vector::FloatVector, Vector}};
use core::{
    borrow::BorrowMut,
    fmt::Debug,
    ops::{Add, Mul},
};

use super::Matrix;

pub struct MatrixStack<F: Float, const N: usize> {
    pub rows: usize,
    pub cols: usize,
    pub data: [F; N],
}

impl<F: Float, const N: usize> Matrix<F> for MatrixStack<F, N> {
    type MatrixInit = (usize, [F; N]);

    #[inline(always)]
    fn n_cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn n_rows(&self) -> usize {
        self.rows
    }
    #[inline(always)]
    fn direct(&self, idx: usize) -> F {
        self.data[idx]
    }

    #[inline(always)]
    fn direct_ref(&mut self, idx: usize) -> &mut F {
        self.data[idx].borrow_mut()
    }

    fn new(init: Self::MatrixInit) -> Self {
        let (rows, data) = init;
        debug_assert!(rows > 0);
        let cols = data.len() / rows;
        Self { rows, cols, data }
    }

    fn zero(rows: usize, cols: usize) -> Self {
        debug_assert!(rows * cols == N);
        let data = [F::ZERO; N];
        Self { rows, cols, data }
    }

    fn single(rows: usize, cols: usize, c: F) -> Self {
        debug_assert!(rows * cols == N);
        let data = [c; N];
        Self { rows, cols, data }
    }
    fn identity(rows: usize, cols: usize) -> Self {
        debug_assert!(rows * cols == N);
        let data = [F::ZERO; N];
        let mut id = Self { rows, cols, data };
        FloatVector::add_ref(id.data_diag_ref(), F::IDENTITY);
        id
    }

    fn explicit_copy(&self) -> Self {
        let new_data = self.data;

        Self::new((self.rows, new_data))
    }
    #[inline(always)]
    fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    #[inline(always)]
    fn coeff(&self, row: usize, col: usize) -> F {
        self.data[self.index(row, col)]
    }

    #[inline(always)]
    fn coeff_ref(&mut self, row: usize, col: usize) -> &mut F {
        let idx = self.index(row, col);
        self.data[idx].borrow_mut()
    }

    #[inline(always)]
    fn data<'a>(&'a self) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data.iter()
    }

    #[inline(always)]
    fn data_diag<'a>(&'a self) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data.iter().step_by(self.cols + 1)
    }

    #[inline(always)]
    fn data_upper_triangular<'a>(&'a self) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data()
            .enumerate()
            .filter(|(i, _)| {
                let row = i / self.cols;
                let col = i % self.cols;
                col > row
            })
            .map(|(_, f)| f)
    }

    #[inline(always)]
    fn data_lower_triangular<'a>(&'a self) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data()
            .enumerate()
            .filter(|(i, _)| {
                let row = i / self.cols;
                let col = i % self.cols;
                row > col
            })
            .map(|(_, f)| f)
    }

    #[inline(always)]
    fn data_row<'a>(&'a self, row: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data[row..].iter().step_by(self.cols)
    }

    #[inline(always)]
    fn data_rows<'a>(&'a self) -> impl Iterator<Item = &[F]>
    where
        F: 'a,
    {
        self.data.chunks_exact(self.cols)
    }

    #[inline(always)]
    fn data_col<'a>(&'a self, col: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data[col * self.cols..(col + 1) * self.cols].iter()
    }

    fn data_north<'a>(&'a self, north: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data().take(north * self.cols)
    }

    fn data_south<'a>(&'a self, south: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data().skip((self.rows - south) * self.cols)
    }
    fn data_west<'a>(&'a self, west: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data()
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    fn data_east<'a>(&'a self, east: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data()
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols - east)
            .map(|(_, c)| c)
    }

    fn data_north_west<'a>(&'a self, north: usize, west: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data_north(north)
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    fn data_north_east<'a>(&'a self, north: usize, east: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data_north(north)
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols - east)
            .map(|(_, c)| c)
    }

    fn data_south_west<'a>(&'a self, south: usize, west: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data_south(south)
            .enumerate()
            .filter(move |(i, _)| i % self.cols < west)
            .map(|(_, c)| c)
    }
    fn data_south_east<'a>(&'a self, south: usize, east: usize) -> impl Iterator<Item = &'a F>
    where
        F: 'a,
    {
        self.data_south(south)
            .enumerate()
            .filter(move |(i, _)| i % self.cols >= self.cols - east)
            .map(|(_, c)| c)
    }

    #[inline(always)]
    fn data_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        self.data.iter_mut()
    }

    #[inline(always)]
    fn data_diag_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        self.data.iter_mut().step_by(self.cols + 1)
    }

    #[inline(always)]
    fn data_upper_triangular_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| {
                let row = i / cols;
                let col = i % cols;
                col > row
            })
            .map(|(_, f)| f)
    }

    #[inline(always)]
    fn data_lower_triangular_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| {
                let row = i / cols;
                let col = i % cols;
                row > col
            })
            .map(|(_, f)| f)
    }

    #[inline(always)]
    fn data_row_ref<'a>(&'a mut self, row: usize) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        self.data[row..].iter_mut().step_by(self.cols)
    }

    #[inline(always)]
    fn data_rows_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut [F]>
    where
        F: 'a,
    {
        self.data.chunks_exact_mut(self.cols)
    }

    #[inline(always)]
    fn data_col_ref<'a>(&'a mut self, col: usize) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        self.data[col * self.cols..(col + 1) * self.cols].iter_mut()
    }

    fn data_north_ref<'a>(&'a mut self, north: usize) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_ref().take(north * cols)
    }

    fn data_south_ref<'a>(&'a mut self, south: usize) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        let rows = self.rows;
        self.data_ref().skip((rows - south) * cols)
    }
    fn data_west_ref<'a>(&'a mut self, west: usize) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| i % cols < west)
            .map(|(_, c)| c)
    }
    fn data_east_ref<'a>(&'a mut self, east: usize) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_ref()
            .enumerate()
            .filter(move |(i, _)| i % cols >= cols - east)
            .map(|(_, c)| c)
    }

    fn data_north_west_ref<'a>(
        &'a mut self,
        north: usize,
        west: usize,
    ) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_north_ref(north)
            .enumerate()
            .filter(move |(i, _)| i % cols < west)
            .map(|(_, c)| c)
    }
    fn data_north_east_ref<'a>(
        &'a mut self,
        north: usize,
        east: usize,
    ) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_north_ref(north)
            .enumerate()
            .filter(move |(i, _)| i % cols >= cols - east)
            .map(|(_, c)| c)
    }

    fn data_south_west_ref<'a>(
        &'a mut self,
        south: usize,
        west: usize,
    ) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_south_ref(south)
            .enumerate()
            .filter(move |(i, _)| i % cols < west)
            .map(|(_, c)| c)
    }
    fn data_south_east_ref<'a>(
        &'a mut self,
        south: usize,
        east: usize,
    ) -> impl Iterator<Item = &'a mut F>
    where
        F: 'a,
    {
        let cols = self.cols;
        self.data_south_ref(south)
            .enumerate()
            .filter(move |(i, _)| i % cols >= cols - east)
            .map(|(_, c)| c)
    }

    fn transposed(&self) -> Self {
        let mut transposed_data = [F::ZERO; N];

        (0..self.cols).for_each(|i| {
            (0..self.rows).for_each(|j| {
                transposed_data[i * self.rows + j] = self.data[j * self.cols + i];
            })
        });
        MatrixStack::new((self.cols, transposed_data))
    }

    fn col_swap(&mut self, a: usize, b: usize) {
        self.data.chunks_exact_mut(self.cols).for_each(|row| {
            row.swap(a, b);
        });
    }

    fn row_swap(&mut self, a: usize, b: usize) {
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
}

impl<F: Float, const N: usize> Add<F> for MatrixStack<F, N> {
    type Output = Self;
    fn add(self, adder: F) -> Self {
        let mut out = self.explicit_copy();
        FloatVector::add_ref(out.data_ref(), adder);
        out
    }
}

impl<F: Float, const N: usize> Mul<F> for MatrixStack<F, N> {
    type Output = Self;
    fn mul(self, scaler: F) -> Self {
        let mut out = self.explicit_copy();
        FloatVector::mul_ref(out.data_ref(), scaler);
        out
    }
}

impl<F: Float, const N: usize> Debug for MatrixStack<F, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MatrixStack")
            .field("rows", &self.rows)
            .field("cols", &self.cols);
        for row in self.data_rows() {
            write!(f, "{:?}", row).unwrap();
            writeln!(f).unwrap();
        }
        Ok(())
    }
}

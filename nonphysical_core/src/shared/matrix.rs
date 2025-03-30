use core::ops::{Add, Mul};

use super::float::Float;
pub mod matrix_heap;
pub mod matrix_stack;
pub trait Matrix<F: Float>: Add<F> + Mul<F> + Sized +Clone{
    type MatrixInit;
    fn new(init: Self::MatrixInit) -> Self;
    fn n_rows(&self) -> usize;
    fn n_cols(&self)->usize;
    fn zero(rows: usize, cols: usize) -> Self;
    fn single(rows: usize, cols: usize, c: F) -> Self;
    fn identity(rows: usize, cols: usize) -> Self;
    fn index(&self, row: usize, col: usize) -> usize;
    fn coeff(&self, row: usize, col: usize) -> F;
    fn coeff_ref(&mut self, row: usize, col: usize) -> &mut F;
    fn direct(&self, idx: usize) -> F;
    fn direct_ref(&mut self,idx: usize) -> &mut F;
    fn data<'a>(&'a self) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_diag<'a>(&'a self) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_upper_triangular<'a>(&'a self) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_lower_triangular<'a>(&'a self) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_col<'a>(&'a self, col: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_rows<'a>(&'a self) -> impl Iterator<Item = &'a [F]> where F: 'a;
    fn data_row<'a>(&'a self, row: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_north<'a>(&'a self, north: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_south<'a>(&'a self, south: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_west<'a>(&'a self, west: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_east<'a>(&'a self, east: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_north_west<'a>(&'a self, north: usize, west: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_north_east<'a>(&'a self, north: usize, east: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_south_west<'a>(&'a self, south: usize, west: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_south_east<'a>(&'a self, south: usize, east: usize) -> impl Iterator<Item = &'a F> where F: 'a;
    fn data_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_diag_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_upper_triangular_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_lower_triangular_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_col_ref<'a>(&'a mut self, col: usize) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_rows_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut[F]> where F: 'a;
    fn data_row_ref<'a>(&'a mut self, row: usize) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_north_ref<'a>(&'a mut self, north: usize) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_south_ref<'a>(&'a mut self, south: usize) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_west_ref<'a>(&'a mut self, west: usize) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_east_ref<'a>(&'a mut self, east: usize) -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_north_west_ref<'a>(&'a mut self, north: usize, west: usize)
        -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_north_east_ref<'a>(&'a mut self, north: usize, east: usize)
        -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_south_west_ref<'a>(&'a mut self, south: usize, west: usize)
        -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn data_south_east_ref<'a>(&'a mut self, south: usize, east: usize)
        -> impl Iterator<Item = &'a mut F> where F: 'a;
    fn transposed(&self) -> Self;
    fn col_swap(&mut self, a: usize, b: usize);
    fn row_swap(&mut self, a: usize, b: usize);
    fn dot(&self, other: &Self)->Self;
}

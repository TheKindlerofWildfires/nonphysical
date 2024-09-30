use crate::shared::{float::Float, matrix::Matrix};

pub trait Layer<F: Float> {
    type Matrix: Matrix<F>;
    type Parameters;
    fn new(params: Self::Parameters, size: usize, previous_size: usize) -> Self
    where
        Self: Sized;
    fn forward(&self, x: &Self::Matrix) -> Self::Matrix;
    fn backward(&self, gradient: &Self::Matrix, memory: &Self::Matrix, lambda: F, epsilon: F) -> Self::Matrix;
    fn forward_ref(&self, x: &mut Self::Matrix);
    fn backward_ref(&self, gradient: &mut Self::Matrix, memory: &Self::Matrix, lambda: F, epsilon: F);
}
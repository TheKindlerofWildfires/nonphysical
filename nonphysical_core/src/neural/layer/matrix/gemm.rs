use alloc::{format, string::String, vec, vec::Vec};

use crate::{
    linear::gemm::Gemm,
    neural::layer::{EulerLayer, Layer},
    shared::{float::Float, matrix::matrix_heap::MatrixHeap},
};

pub struct GemmLayer<F: Float> {
    name: String,
    weights: MatrixHeap<F>,
}

impl<F: Float> Layer<F> for GemmLayer<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = Self::Matrix;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, weights: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name, weights }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size[1] != self.weights.rows {
            Err(format!(
                "Cannot multiple input of {}x{} by weights of {}x{}",
                size[0], size[1], self.weights.rows, self.weights.cols
            ))
        } else {
            Ok(vec![size[0], self.weights.cols])
        }
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        <MatrixHeap<F> as Gemm<F>>::gemm(x, &self.weights)
    }

    fn backward(&self, _gradient: &Self::Matrix, _memory: &Self::Matrix) -> Self::Matrix {
        todo!()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        *x = <MatrixHeap<F> as Gemm<F>>::gemm(x, &self.weights)
    }

    fn backward_ref(&self, _: &mut Self::Matrix, _: &Self::Matrix) {
        todo!()
    }
}
impl<F: Float> EulerLayer<F> for GemmLayer<F> {
    fn update(&mut self, _: &Self::Matrix, _: F) {
        todo!()
    }
}

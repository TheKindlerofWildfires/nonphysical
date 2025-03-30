use alloc::string::String;
use alloc::vec::Vec;
use crate::shared::{float::Float, matrix::Matrix};

pub mod vector;
pub mod matrix;


pub trait Layer<F: Float> {
    type Matrix: Matrix<F>;
    type Parameters;
    fn name(&self)->String;
    fn new(name: String, params: Self::Parameters) -> Self
    where
        Self: Sized;
    fn compile(&self,shape: Vec<usize>)->Result<Vec<usize>,String>;
    fn forward(&self, x: &Self::Matrix) -> Self::Matrix;
    fn backward(&self, gradient: &Self::Matrix, memory: &Self::Matrix) -> Self::Matrix;
    fn forward_ref(&self, x: &mut Self::Matrix);
    fn backward_ref(&self, gradient: &mut Self::Matrix, memory: &Self::Matrix);
}

pub trait EulerLayer<F:Float> : Layer<F>{
    fn update(&mut self, gradient: &Self::Matrix, lambda: F);
}
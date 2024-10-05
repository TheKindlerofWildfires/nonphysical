use core::marker::PhantomData;


use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::matrix_heap::MatrixHeap,
    },
};

use super::Activation;

pub struct Gelu<F: Float> {
    parameters: GeluParameters<F>,
}

pub struct GeluParameters<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> Activation<F> for Gelu<F> {}

impl<F: Float> Layer<F> for Gelu<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = GeluParameters<F>;
    fn new(parameters: Self::Parameters, _size: usize, _previous_size: usize) -> Self
    where
        Self: Sized,
    {
        Self { parameters }
    }
    /*
       f(x) =  log(exp(x)+1)
    */
    fn forward(&self, _x: &Self::Matrix) -> Self::Matrix {
        todo!()
    }
    /*
       f'(x) =  1/(exp(-x)+1)

    */
    fn backward(
        &self,
        _gradient: &Self::Matrix, //incoming grad
        _memory: &Self::Matrix,   //historical input
        _lambda: F,
        _epsilon: F,
    ) -> Self::Matrix {
        todo!()
    }

    fn forward_ref(&self, _x: &mut Self::Matrix) {
        todo!()
    }

    fn backward_ref(
        &self,
        _gradient: &mut Self::Matrix,
        _memory: &Self::Matrix,
        _lambda: F,
        _epsilon: F,
    ) {
        todo!()
    }
}

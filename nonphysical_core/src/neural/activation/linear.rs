use core::marker::PhantomData;


use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
    },
};

use super::Activation;

pub struct Linear<F: Float> {
    parameters: LinearParameters<F>,
}

pub struct LinearParameters<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> Activation<F> for Linear<F> {}

impl<F: Float> Layer<F> for Linear<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = LinearParameters<F>;
    fn new(parameters: Self::Parameters, _size: usize, _previous_size: usize) -> Self
    where
        Self: Sized,
    {
        Self { parameters }
    }
    /*
       f(x) =  log(exp(x)+1)
    */
    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        x.explicit_copy()
    }
    /*
       f'(x) =  1/(exp(-x)+1)

    */
    fn backward(
        &self,
        gradient: &Self::Matrix, //incoming grad
        _memory: &Self::Matrix,   //historical input
        _lambda: F,
        _epsilon: F,
    ) -> Self::Matrix {
        gradient.explicit_copy()
    }

    fn forward_ref(&self, _x: &mut Self::Matrix) {
        
    }

    fn backward_ref(
        &self,
        _gradient: &mut Self::Matrix,
        _memory: &Self::Matrix,
        _lambda: F,
        _epsilon: F,
    ) {
        
    }
}

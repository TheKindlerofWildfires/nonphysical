use core::marker::PhantomData;

use alloc::vec::Vec;

use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

use super::Activation;

pub struct Relu6<F: Float> {
    parameters: Relu6Parameters<F>,
}

pub struct Relu6Parameters<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> Activation<F> for Relu6<F> {}

impl<F: Float> Layer<F> for Relu6<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = Relu6Parameters<F>;
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
        todo!()
    }
    /*
       f'(x) =  1/(exp(-x)+1)

    */
    fn backward(
        &self,
        gradient: &Self::Matrix, //incoming grad
        memory: &Self::Matrix,   //historical input
        _lambda: F,
        _epsilon: F,
    ) -> Self::Matrix {
        todo!()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        todo!()
    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,
        _lambda: F,
        _epsilon: F,
    ) {
        todo!()
    }
}

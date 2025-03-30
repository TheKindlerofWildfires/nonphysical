use core::marker::PhantomData;


use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

use super::Activation;

pub struct Exponential<F: Float> {
    parameters: ExponentialParameters<F>,
}

pub struct ExponentialParameters<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> Activation<F> for Exponential<F> {}

impl<F: Float> Layer<F> for Exponential<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = ExponentialParameters<F>;
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
        let mut out = x.clone();
        FloatVector::exp_ref(out.data_ref());
        out
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
        let factor = FloatVector::exp(memory.data());
        let mut out = gradient.clone();
        FloatVector::mul_vec_ref_direct(out.data_ref(), factor);
        out
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::exp_ref(x.data_ref());

    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,
        _lambda: F,
        _epsilon: F,
    ) {
        let factor = FloatVector::exp(memory.data());
        FloatVector::mul_vec_ref_direct(gradient.data_ref(), factor);
    }
}

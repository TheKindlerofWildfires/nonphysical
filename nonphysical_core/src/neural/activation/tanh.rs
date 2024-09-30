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

pub struct Tanh<F: Float> {
    parameters: SoftplusParameters<F>,
}

pub struct SoftplusParameters<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> Activation<F> for Tanh<F> {}

impl<F: Float> Layer<F> for Tanh<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = SoftplusParameters<F>;
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
        let data = FloatVector::tanh(x.data());
        Matrix::new((x.rows, data.collect::<Vec<_>>()))
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
        let factor = FloatVector::recip_direct(FloatVector::l2_norm_direct(FloatVector::cosh(memory.data())));
        let mut out = gradient.explicit_copy();
        FloatVector::scale_vec_ref_direct(out.data_ref(), factor);
        out
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::tanh_ref(x.data_ref());
    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,
        _lambda: F,
        _epsilon: F,
    ) {
        let factor = FloatVector::recip_direct(FloatVector::l2_norm_direct(FloatVector::cosh(memory.data())));
        FloatVector::scale_vec_ref_direct(gradient.data_ref(), factor);
    }
}

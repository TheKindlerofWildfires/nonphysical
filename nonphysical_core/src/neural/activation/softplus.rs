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

pub struct Softplus<F: Float> {
    parameters: SoftplusParameters<F>,
}

pub struct SoftplusParameters<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> Activation<F> for Softplus<F> {}

impl<F: Float> Layer<F> for Softplus<F> {
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
        let data = FloatVector::ln_direct(FloatVector::add_direct(FloatVector::exp(x.data()), F::IDENTITY));
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
        let factor = FloatVector::recip_direct(FloatVector::add_direct(FloatVector::exp_direct(FloatVector::neg(memory.data())),F::IDENTITY));
        let mut out = gradient.explicit_copy();
        FloatVector::mul_vec_ref_direct(out.data_ref(), factor);
        out
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::exp_ref(x.data_ref());
        FloatVector::add_ref(x.data_ref(),F::IDENTITY);
        FloatVector::ln_ref(x.data_ref());
    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,
        _lambda: F,
        _epsilon: F,
    ) {
        let factor = FloatVector::recip_direct(FloatVector::add_direct(FloatVector::exp_direct(FloatVector::neg(memory.data())),F::IDENTITY));
        FloatVector::mul_vec_ref_direct(gradient.data_ref(), factor);
    }
}

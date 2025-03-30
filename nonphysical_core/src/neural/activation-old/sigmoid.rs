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

pub struct Sigmoid<F: Float> {
    parameters: SigmoidParameters<F>,
}

pub struct SigmoidParameters<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> Activation<F> for Sigmoid<F> {}

impl<F: Float> Layer<F> for Sigmoid<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = SigmoidParameters<F>;
    fn new(parameters: Self::Parameters, _size: usize, _previous_size: usize) -> Self
    where
        Self: Sized,
    {
        Self { parameters }
    }
    /*
       f(x) =  1/(I+exp(-x))
    */
    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        let data = FloatVector::recip_direct(FloatVector::add_direct(
            FloatVector::exp_direct(FloatVector::neg(x.data())),
            F::IDENTITY,
        ));
        Matrix::new((x.rows, data.collect::<Vec<_>>()))
    }
    /*
       f'(x) =  exp(-x)/(I+exp(-x))^2

    */
    fn backward(
        &self,
        gradient: &Self::Matrix, //incoming grad
        memory: &Self::Matrix,   //historical input
        _lambda: F,
        _epsilon: F,
    ) -> Self::Matrix {
        let exp = FloatVector::exp_direct(FloatVector::neg(memory.data()));
        let factor = exp.map(|exp| exp/(exp+F::IDENTITY).l2_norm());
        let mut out = gradient.clone();
        FloatVector::mul_vec_ref_direct(out.data_ref(), factor);
        out
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::neg_ref(x.data_ref());
        FloatVector::exp_ref(x.data_ref());
        FloatVector::add_ref(x.data_ref(),F::IDENTITY);
        FloatVector::recip_ref(x.data_ref());
    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,
        _lambda: F,
        _epsilon: F,
    ) {
        let exp = FloatVector::exp_direct(FloatVector::neg(memory.data()));
        let factor = exp.map(|exp| exp/(exp+F::IDENTITY).l2_norm());
        FloatVector::mul_vec_ref_direct(gradient.data_ref(), factor);
    }
}

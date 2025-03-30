use core::marker::PhantomData;


use crate::{
    neural::layer::Layer,
    shared::{
        float::Float, matrix::{matrix_heap::MatrixHeap, Matrix}, primitive::Primitive, vector::{float_vector::FloatVector, Vector}
    },
};

use super::Activation;

pub struct Softsign<P: Primitive> {
    parameters: SoftsignParameters<P>,
}

pub struct SoftsignParameters<P: Primitive> {
    phantom_data: PhantomData<P>,
}

impl<P: Primitive> Activation<P> for Softsign<P> {}

impl<P: Primitive> Layer<P> for Softsign<P> {
    type Matrix = MatrixHeap<P>;
    type Parameters = SoftsignParameters<P>;
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
        FloatVector::descale_vec_ref_direct(out.data_ref(), FloatVector::add_direct(FloatVector::l1_norm(x.data()), P::Primitive::IDENTITY));
        out
    }
    /*
       f'(x) =  - sgn(x)^3 / (sig(x)+x)^3

    */
    fn backward(
        &self,
        _gradient: &Self::Matrix, //incoming grad
        _memory: &Self::Matrix,   //historical input
        _lambda: P,
        _epsilon: P,
    ) -> Self::Matrix {
        todo!()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        let val = x.clone();
        FloatVector::descale_vec_ref_direct(x.data_ref(), FloatVector::add_direct(FloatVector::l1_norm(val.data()), P::Primitive::IDENTITY));
    }

    fn backward_ref(
        &self,
        _gradient: &mut Self::Matrix,
        _memory: &Self::Matrix,
        _lambda: P,
        _epsilon: P,
    ) {
        todo!()
    }
}

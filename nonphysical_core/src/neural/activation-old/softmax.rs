use core::marker::PhantomData;


use crate::{
    linear::gemm::Gemm, neural::layer::Layer, shared::{float::Float, matrix::{matrix_heap::MatrixHeap, Matrix}, vector::{float_vector::FloatVector, Vector}}
};

use super::Activation;

pub struct Softmax<F: Float> {
    parameters: SoftmaxParameters<F>,
}

pub struct SoftmaxParameters<F: Float> {
    phantom_data: PhantomData<F>
}

impl<F: Float> Activation<F> for Softmax<F> {}

impl<F: Float> Layer<F> for Softmax<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = SoftmaxParameters<F>;
    fn new(parameters: Self::Parameters, _size: usize, _previous_size: usize) -> Self
    where
        Self: Sized,
    {
        Self { parameters }
    }
    /*
        f(x) =  exp(x)/sum(exp(x))
     */ 
    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        let exp = FloatVector::exp(x.data());
        let mut out =Self::Matrix::new((x.rows, exp.collect()));
        let sum = FloatVector::sum(out.data());
        FloatVector::div_ref(out.data_ref(), sum);
        out
    }
    /*
        f'(x) = exp(x)*(kronecker-)
    
     */
    fn backward(
        &self,
        gradient: &Self::Matrix, //incoming grad
        memory: &Self::Matrix, //historical input
        _lambda: F,
        _epsilon: F,
    ) -> Self::Matrix {
        let forward = {
            let exp = FloatVector::exp(memory.data());
            let mut out =Self::Matrix::new((memory.rows, exp.collect()));
            let sum = FloatVector::sum(out.data());
            FloatVector::div_ref(out.data_ref(), sum);
            out
        };
        let tmp = Self::Matrix::new((memory.rows, forward.data.repeat(memory.rows)));
        let mut trans_tmp = tmp.transposed();
        FloatVector::sub_ref(trans_tmp.data_ref(), F::IDENTITY);
        FloatVector::neg_ref(trans_tmp.data_ref());
        let tmp2 = <Self::Matrix as Gemm<F>>::gemm(tmp, trans_tmp);
        tmp2.dot(gradient)
    }
    
    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::exp_ref(x.data_ref());
        let sum = FloatVector::sum(x.data());
        FloatVector::div_ref(x.data_ref(), sum);
    }
    
    fn backward_ref(&self, gradient: &mut Self::Matrix, memory: &Self::Matrix, _lambda: F, _epsilon: F) {
        let forward = {
            let exp = FloatVector::exp(memory.data());
            let mut out =Self::Matrix::new((memory.rows, exp.collect()));
            let sum = FloatVector::sum(out.data());
            FloatVector::div_ref(out.data_ref(), sum);
            out
        };
        let tmp = Self::Matrix::new((memory.rows, forward.data.repeat(memory.rows)));
        let mut trans_tmp = tmp.transposed();
        FloatVector::sub_ref(trans_tmp.data_ref(), F::IDENTITY);
        FloatVector::neg_ref(trans_tmp.data_ref());
        let tmp2 = <Self::Matrix as Gemm<F>>::gemm(tmp, trans_tmp);
        *gradient = tmp2.dot(gradient);
    }
}

use alloc::{string::String, vec::Vec};

use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

pub struct TanhLayer {
    name: String,
}

impl<F: Float> Layer<F> for TanhLayer {
    type Matrix = MatrixHeap<F>;

    type Parameters = ();

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, _: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        Ok(size)
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        let mut out = x.clone();
        FloatVector::tanh_ref(out.data_ref());
        out
    }

    fn backward(
        &self,
        gradient: &Self::Matrix,
        memory: &Self::Matrix,

    ) -> Self::Matrix {
        //grad * cos(mem)
        let delta = FloatVector::cosh(memory.data());
        let delta = FloatVector::recip_direct(delta);
        let delta = FloatVector::powf_direct(delta, F::IDENTITY+F::IDENTITY);
        let mut out = gradient.clone();
        FloatVector::mul_vec_ref_direct(out.data_ref(), delta);
        out
        
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::tanh_ref(x.data_ref());

    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,

    ) {
        let delta = FloatVector::cosh(memory.data());
        let delta = FloatVector::recip_direct(delta);
        let delta = FloatVector::powf_direct(delta, F::IDENTITY+F::IDENTITY);
        FloatVector::mul_vec_ref_direct(gradient.data_ref(), delta);
    }
}

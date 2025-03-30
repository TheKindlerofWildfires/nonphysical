use alloc::{string::String, vec::Vec};

use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

pub struct SinLayer {
    name: String,
}

impl<F: Float> Layer<F> for SinLayer {
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
        FloatVector::sin_ref(out.data_ref());
        out
    }

    fn backward(
        &self,
        gradient: &Self::Matrix,
        memory: &Self::Matrix,

    ) -> Self::Matrix {
        //grad * cos(mem)
        let delta = FloatVector::cos(memory.data());
        let mut out = gradient.clone();
        FloatVector::mul_vec_ref_direct(out.data_ref(), delta);
        out
        
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::sin_ref(x.data_ref());

    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,

    ) {
        let delta = FloatVector::cos(memory.data());
        FloatVector::mul_vec_ref_direct(gradient.data_ref(), delta);
    }
}

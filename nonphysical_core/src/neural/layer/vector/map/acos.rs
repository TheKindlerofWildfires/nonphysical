use alloc::{string::String, vec::Vec};

use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix}, vector::{float_vector::FloatVector, Vector},
    },
};

pub struct AcosLayer {
    name: String,
}

impl<F: Float> Layer<F> for AcosLayer {
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
        FloatVector::acos_ref(out.data_ref());
        out
    }

    fn backward(
        &self,
        _: &Self::Matrix,
        _: &Self::Matrix,
    ) -> Self::Matrix {
        todo!()
        
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::acos_ref(x.data_ref());
    }

    fn backward_ref(
        &self,
        _: &mut Self::Matrix,
        _: &Self::Matrix,
    ) {
        todo!()
    }
}

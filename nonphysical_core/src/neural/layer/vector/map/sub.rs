use alloc::{string::String, vec::Vec};

use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::matrix_heap::MatrixHeap,
    },
};

pub struct SubLayer {
    name: String,
}

impl<F: Float> Layer<F> for SubLayer {
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

    fn forward(&self, _: &Self::Matrix) -> Self::Matrix {
        todo!()
    }

    fn backward(
        &self,
        _: &Self::Matrix,
        _: &Self::Matrix,

    ) -> Self::Matrix {
        todo!()
        
    }

    fn forward_ref(&self, _: &mut Self::Matrix) {
        todo!()
    }

    fn backward_ref(
        &self,
        _: &mut Self::Matrix,
        _: &Self::Matrix,

    ) {
        todo!()
    }
}

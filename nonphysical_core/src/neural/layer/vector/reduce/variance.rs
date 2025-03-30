use alloc::{string::String, vec::Vec};

use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::matrix_heap::MatrixHeap,
    },
};

pub struct VarianceLayerTotal {
    name: String,
    include_mean: bool,
}

impl<F: Float> Layer<F> for VarianceLayerTotal {
    type Matrix = MatrixHeap<F>;

    type Parameters = bool;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, include_mean: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name,include_mean }
    }

    fn compile(&self, _: Vec<usize>) -> Result<Vec<usize>, String> {
        todo!()
    }

    fn forward(&self, _x: &Self::Matrix) -> Self::Matrix {
        todo!()
    }

    fn backward(
        &self,
        _gradient: &Self::Matrix,
        _memory: &Self::Matrix,
    ) -> Self::Matrix {
        todo!()
        
    }

    fn forward_ref(&self, _x: &mut Self::Matrix) {
        todo!()
    }

    fn backward_ref(
        &self,
        _gradient: &mut Self::Matrix,
        _memory: &Self::Matrix,
    ) {
        todo!()
    }
}


pub struct VarianceLayerAxis {
    name: String,
    axis: usize,
}

impl<F: Float> Layer<F> for VarianceLayerAxis {
    type Matrix = MatrixHeap<F>;

    type Parameters = usize;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, axis: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        assert!(axis<2, "Tensors not supported");
        Self { name,axis }
    }

    fn compile(&self, _size: Vec<usize>) -> Result<Vec<usize>, String> {
        todo!()
    }


    fn forward(&self, _x: &Self::Matrix) -> Self::Matrix {
        todo!()
    }

    fn backward(
        &self,
        _gradient: &Self::Matrix,
        _memory: &Self::Matrix,
    ) -> Self::Matrix {
        todo!()
        
    }

    fn forward_ref(&self, _x: &mut Self::Matrix) {
        todo!()
    }

    fn backward_ref(
        &self,
        _gradient: &mut Self::Matrix,
        _memory: &Self::Matrix,
    ) {
        todo!()
    }
}
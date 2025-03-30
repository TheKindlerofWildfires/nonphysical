use alloc::{format, string::{String,ToString}, vec::Vec};

use crate::{
    neural::layer::{EulerLayer, Layer},
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

pub struct FmaLayerTotal<F: Float> {
    name: String,
    add: F,
    mul: F,
}

impl<F: Float> Layer<F> for FmaLayerTotal<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = (F, F);

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, param: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        let (add, mul) = param;
        Self { name, add, mul }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        Ok(size)
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        let mut out = x.clone();
        FloatVector::fma_ref(out.data_ref(), self.mul, self.add);
        out
    }

    fn backward(&self, _: &Self::Matrix, _: &Self::Matrix) -> Self::Matrix {
        todo!()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::fma_ref(x.data_ref(), self.mul, self.add);
    }

    fn backward_ref(&self, _: &mut Self::Matrix, _: &Self::Matrix) {
        todo!()
    }
}

impl<F: Float> EulerLayer<F> for FmaLayerTotal<F> {
    fn update(&mut self, _: &Self::Matrix, _: F) {
        todo!()
    }
}

pub struct FmaLayerAxis<F: Float> {
    name: String,
    axis: usize,
    muls: Vec<F>,
    adds: Vec<F>,
}

impl<F: Float> Layer<F> for FmaLayerAxis<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = (usize, Vec<F>, Vec<F>);

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, params: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        let (axis, muls,adds) = params;
        assert!(axis < 2, "Tensors not supported");
        Self { name, axis, muls,adds }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size.len() > self.axis {
            let out_size = size.clone();
            let target = out_size[self.axis];
            if target == self.muls.len() {
                Ok(out_size)
            } else {
                Err(format!(
                    "Mismatch input: {}, constants: {} ",
                    target,
                    self.muls.len()
                ))
            }
        } else {
            Err(format!(
                "Couldn't compile map on axis {} for input of size {:?}",
                self.axis, size
            ))
        }
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        //tensor issue avoidance
        let mut out = x.clone();
        if self.axis == 0 {
            out.data_rows_ref()
                .zip(self.muls.iter())
                .zip(self.adds.iter())
                .for_each(|((row, mul),add)| {
                    FloatVector::fma_ref(row.iter_mut(), *mul, *add);
                });
        } else {
            (0..out.cols).zip(self.muls.iter()).zip(self.adds.iter()).for_each(|((i, mul),add)| {
                FloatVector::fma_ref(out.data_col_ref(i), *mul,*add);
            });
        }
        out
    }

    fn backward(&self, _: &Self::Matrix, _: &Self::Matrix) -> Self::Matrix {
        todo!()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        if self.axis == 0 {
            x.data_rows_ref()
                .zip(self.muls.iter())
                .zip(self.adds.iter())
                .for_each(|((row, mul),add)| {
                    FloatVector::fma_ref(row.iter_mut(), *mul, *add);
                });
        } else {
            (0..x.cols).zip(self.muls.iter()).zip(self.adds.iter()).for_each(|((i, mul),add)| {
                FloatVector::fma_ref(x.data_col_ref(i), *mul,*add);
            });
        }
    }

    fn backward_ref(&self, _: &mut Self::Matrix, _: &Self::Matrix) {
        todo!()
    }
}

impl<F: Float> EulerLayer<F> for FmaLayerAxis<F> {
    fn update(&mut self, _: &Self::Matrix, _: F) {
        todo!()
    }
}

pub struct FmaLayerElement<F: Float> {
    name: String,
    muls: MatrixHeap<F>,
}

impl<F: Float> Layer<F> for FmaLayerElement<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = Self::Matrix;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, muls: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name, muls }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size.len() != 2 {
            if size[0] != self.muls.rows {
                Err(format!(
                    "Input rows {} does not match mul rows {}",
                    size[0], self.muls.rows
                ))
            } else if size[1] != self.muls.cols {
                Err(format!(
                    "Input cols {} does not match mul cols {}",
                    size[1], self.muls.cols
                ))
            } else {
                Ok(size)
            }
        } else {
            Err("Tensors not supported".to_string())
        }
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        //tensor issue avoidance
        let mut out = x.clone();
        FloatVector::mul_vec_ref(out.data_ref(), self.muls.data());
        out
    }

    fn backward(&self, gradient: &Self::Matrix, _: &Self::Matrix) -> Self::Matrix {
        gradient.clone()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::mul_vec_ref(x.data_ref(), self.muls.data());
    }

    fn backward_ref(&self, _: &mut Self::Matrix, _: &Self::Matrix) {}
}

impl<F: Float> EulerLayer<F> for FmaLayerElement<F> {
    fn update(&mut self, gradient: &Self::Matrix, lambda: F) {
        let update = FloatVector::mul(gradient.data(), lambda);
        FloatVector::mul_vec_ref_direct(self.muls.data_ref(), update);
    }
}

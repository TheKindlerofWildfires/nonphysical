use alloc::{format, string::{String,ToString}, vec::Vec};

use crate::{
    neural::layer::{EulerLayer, Layer},
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

pub struct PowfLayerTotal<F: Float> {
    name: String,
    exp: F,
}

impl<F: Float> Layer<F> for PowfLayerTotal<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = F;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, exp: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name, exp }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        Ok(size)
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        let mut out = x.clone();
        FloatVector::powf_ref(out.data_ref(), self.exp);
        out
    }

    fn backward(&self, _: &Self::Matrix, _: &Self::Matrix) -> Self::Matrix {
        todo!()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::powf_ref(x.data_ref(), self.exp);
    }

    fn backward_ref(&self, _: &mut Self::Matrix, _: &Self::Matrix) {}
}

impl<F: Float> EulerLayer<F> for PowfLayerTotal<F> {
    fn update(&mut self, _: &Self::Matrix, _: F) {
        todo!()
    }
}

pub struct PowfLayerAxis<F: Float> {
    name: String,
    axis: usize,
    exps: Vec<F>,
}

impl<F: Float> Layer<F> for PowfLayerAxis<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = (usize, Vec<F>);

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, params: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        let (axis, exps) = params;
        assert!(axis < 2, "Tensors not supported");
        Self { name, axis, exps }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size.len() > self.axis {
            let out_size = size.clone();
            let target = out_size[self.axis];
            if target == self.exps.len() {
                Ok(out_size)
            } else {
                Err(format!(
                    "Mismatch input: {}, constants: {} ",
                    target,
                    self.exps.len()
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
                .zip(self.exps.iter())
                .for_each(|(row, exp)| {
                    FloatVector::powf_ref(row.iter_mut(), *exp);
                });
        } else {
            (0..out.cols).zip(self.exps.iter()).for_each(|(i, exp)| {
                FloatVector::powf_ref(out.data_col_ref(i), *exp);
            });
        }
        out
    }

    fn backward(&self, gradient: &Self::Matrix, _: &Self::Matrix) -> Self::Matrix {
        gradient.clone()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        if self.axis == 0 {
            x.data_rows_ref()
                .zip(self.exps.iter())
                .for_each(|(row, exp)| {
                    FloatVector::powf_ref(row.iter_mut(), *exp);
                });
        } else {
            (0..x.cols).zip(self.exps.iter()).for_each(|(i, exp)| {
                FloatVector::powf_ref(x.data_col_ref(i), *exp);
            });
        }
    }

    fn backward_ref(&self, _: &mut Self::Matrix, _: &Self::Matrix) {}
}

impl<F: Float> EulerLayer<F> for PowfLayerAxis<F> {
    fn update(&mut self, _: &Self::Matrix, _: F) {
        todo!()
    }
}

pub struct PowfLayerElement<F: Float> {
    name: String,
    exps: MatrixHeap<F>,
}

impl<F: Float> Layer<F> for PowfLayerElement<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = Self::Matrix;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, exps: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name, exps }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size.len() != 2 {
            if size[0] != self.exps.rows {
                Err(format!(
                    "Input rows {} does not match add rows {}",
                    size[0], self.exps.rows
                ))
            } else if size[1] != self.exps.cols {
                Err(format!(
                    "Input cols {} does not match add cols {}",
                    size[1], self.exps.cols
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
        FloatVector::powf_vec_ref(out.data_ref(), self.exps.data());
        out
    }

    fn backward(&self, gradient: &Self::Matrix, _: &Self::Matrix) -> Self::Matrix {
        gradient.clone()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::powf_vec_ref(x.data_ref(), self.exps.data());
    }

    fn backward_ref(&self, _: &mut Self::Matrix, _: &Self::Matrix) {}
}

impl<F: Float> EulerLayer<F> for PowfLayerElement<F> {
    fn update(&mut self, _: &Self::Matrix, _: F) {
        todo!()
    }
}

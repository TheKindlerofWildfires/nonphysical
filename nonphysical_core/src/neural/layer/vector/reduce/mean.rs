use alloc::{format,string::String, vec, vec::Vec};

use crate::{
    neural::layer::Layer,
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

pub struct MeanLayerTotal {
    name: String,
}

impl<F: Float> Layer<F> for MeanLayerTotal {
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

    fn compile(&self, _: Vec<usize>) -> Result<Vec<usize>, String> {
        Ok(vec![1])
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        Self::Matrix::new((1, vec![FloatVector::mean(x.data())]))
    }

    fn backward(
        &self,
        gradient: &Self::Matrix,
        memory: &Self::Matrix,
    ) -> Self::Matrix {
        //grad/out * xi *lambda
        let distribution = FloatVector::mean(gradient.data());
        let output =  FloatVector::mean(memory.data());
        let mut out = memory.clone();
        FloatVector::mul_ref(out.data_ref(), distribution/output);
        out
        
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        *x = Self::Matrix::new((1, vec![FloatVector::mean(x.data())]))
    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,
    ) {
        let distribution = FloatVector::mean(gradient.data());
        let output =  FloatVector::mean(memory.data());
        let mut out = memory.clone();
        FloatVector::mul_ref(out.data_ref(), distribution/output);
        *gradient = out
    }
}


pub struct MeanLayerAxis {
    name: String,
    axis: usize,
}

impl<F: Float> Layer<F> for MeanLayerAxis {
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

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size.len()>self.axis{
            let mut out_size = size.clone();
            out_size[self.axis]=1;
            Ok(out_size)
        }else{
            Err(format!("Couldn't compile reduce on axis {} for input of size {:?}",self.axis, size))
        }
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        //tensor issue avoidance
        if self.axis == 0{
            let data = x.data_rows().map(|row|{
                FloatVector::mean(row.iter())
            });
            Self::Matrix::new((1, data.collect()))
        }else{
            let data = (0..x.cols).map(|i|{
                FloatVector::mean(x.data_col(i))
            });
            Self::Matrix::new((x.rows, data.collect()))
        }
    }

    fn backward(
        &self,
        gradient: &Self::Matrix,
        memory: &Self::Matrix,
    ) -> Self::Matrix {
        if self.axis == 0{
            let mut out = memory.clone();
            memory.data_rows().zip(out.data_rows_ref()).zip(gradient.data()).for_each(|((memory_row, out_row),distribution)|{
                let output = FloatVector::mean(memory_row.iter());
                FloatVector::mul_ref(out_row.iter_mut(),output/ *distribution);
            });
            out
        }else{
            let mut out = memory.clone();
            (0..memory.cols).zip(gradient.data()).for_each(|(i,distribution)|{
                let output = FloatVector::mean(memory.data_col(i));
                FloatVector::mul_ref(out.data_col_ref(i),output/ *distribution);
            });
            out
        }
        
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        if self.axis == 0{
            let data = x.data_rows().map(|row|{
                FloatVector::mean(row.iter())
            });
            *x = Self::Matrix::new((1, data.collect()))
        }else{
            let data = (0..x.cols).map(|i|{
                FloatVector::mean(x.data_col(i))
            });
            *x = Self::Matrix::new((x.rows, data.collect()))
        }
    }

    fn backward_ref(
        &self,
        gradient: &mut Self::Matrix,
        memory: &Self::Matrix,
    ) {
        if self.axis == 0{
            let mut out = memory.clone();
            memory.data_rows().zip(out.data_rows_ref()).zip(gradient.data()).for_each(|((memory_row, out_row),distribution)|{
                let output = FloatVector::mean(memory_row.iter());
                FloatVector::mul_ref(out_row.iter_mut(),output/ *distribution);
            });
            *gradient = out
        }else{
            let mut out = memory.clone();
            (0..memory.cols).zip(gradient.data()).for_each(|(i,distribution)|{
                let output = FloatVector::mean(memory.data_col(i));
                FloatVector::mul_ref(out.data_col_ref(i),output/ *distribution);
            });
            *gradient = out
        }
    }
}
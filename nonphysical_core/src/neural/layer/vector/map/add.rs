use alloc::{format,string::{String,ToString}, vec::Vec};

use crate::{
    neural::layer::{EulerLayer, Layer},
    shared::{
        float::Float,
        matrix::{matrix_heap::MatrixHeap, Matrix},
        vector::{float_vector::FloatVector, Vector},
    },
};

pub struct AddLayerTotal<F:Float> {
    name: String,
    add: F
}

impl<F: Float> Layer<F> for AddLayerTotal<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = F;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, add: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name,add }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        Ok(size)
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        let mut out = x.clone();
        FloatVector::add_ref(out.data_ref(),self.add);
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
        FloatVector::add_ref(x.data_ref(),self.add);


    }

    fn backward_ref(
        &self,
        _: &mut Self::Matrix,
        _: &Self::Matrix,

    ) {

    }
}

impl<F:Float> EulerLayer<F> for AddLayerTotal<F>{
    fn update(&mut self, gradient: &Self::Matrix, lambda: F) {
        self.add += FloatVector::sum(gradient.data())*lambda;
    }
}

pub struct AddLayerAxis<F:Float> {
    name: String,
    axis: usize,
    adds: Vec<F>
}

impl<F: Float> Layer<F> for AddLayerAxis<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = (usize, Vec<F>);

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, params: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        let (axis,adds) = params;
        assert!(axis<2, "Tensors not supported");
        Self { name,axis,adds }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size.len()>self.axis{
            let out_size = size.clone();
            let target = out_size[self.axis];
            if target == self.adds.len(){
                Ok(out_size)
            }else{
                Err(format!("Mismatch input: {}, constants: {} ",target, self.adds.len()))

            }
            
        }else{
            Err(format!("Couldn't compile map on axis {} for input of size {:?}",self.axis, size))
        }
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        //tensor issue avoidance
        let mut out = x.clone();
        if self.axis == 0{
            out.data_rows_ref().zip(self.adds.iter()).for_each(|(row,add)|{
                FloatVector::add_ref(row.iter_mut(), *add);
            });
        }else{
            (0..out.cols).zip(self.adds.iter()).for_each(|(i,add)|{
                FloatVector::add_ref(out.data_col_ref(i), *add);
            });
        }
        out
    }

    fn backward(
        &self,
        gradient: &Self::Matrix,
        _: &Self::Matrix,

    ) -> Self::Matrix {
        gradient.clone()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        if self.axis == 0{
            x.data_rows_ref().zip(self.adds.iter()).for_each(|(row,add)|{
                FloatVector::add_ref(row.iter_mut(), *add);
            });
        }else{
            (0..x.cols).zip(self.adds.iter()).for_each(|(i,add)|{
                FloatVector::add_ref(x.data_col_ref(i), *add);
            });
        }
    }

    fn backward_ref(
        &self,
        _: &mut Self::Matrix,
        _: &Self::Matrix,

    ) {
    }
}

impl<F:Float> EulerLayer<F> for AddLayerAxis<F>{
    fn update(&mut self, gradient: &Self::Matrix, lambda: F) {
        self.adds.iter_mut().zip(gradient.data_rows()).for_each(|(add,row)|{
            *add += FloatVector::sum(row.iter())*lambda;
        });
    }
}


pub struct AddLayerElement<F:Float> {
    name: String,
    adds: MatrixHeap<F>
}

impl<F: Float> Layer<F> for AddLayerElement<F> {
    type Matrix = MatrixHeap<F>;

    type Parameters = Self::Matrix;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn new(name: String, adds: Self::Parameters) -> Self
    where
        Self: Sized,
    {
        Self { name,adds }
    }

    fn compile(&self, size: Vec<usize>) -> Result<Vec<usize>, String> {
        if size.len()!=2{
            if size[0] != self.adds.rows{
                Err(format!("Input rows {} does not match add rows {}", size[0], self.adds.rows))
            }else if size[1] != self.adds.cols{
                Err(format!("Input cols {} does not match add cols {}", size[1], self.adds.cols))
            }else{
                Ok(size)
            }
        }else{
            Err("Tensors not supported".to_string())
        }
    }

    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        //tensor issue avoidance
        let mut out = x.clone();
        FloatVector::mul_vec_ref(out.data_ref(), self.adds.data());
        out
    }

    fn backward(
        &self,
        gradient: &Self::Matrix,
        _: &Self::Matrix,

    ) -> Self::Matrix {
        gradient.clone()
    }

    fn forward_ref(&self, x: &mut Self::Matrix) {
        FloatVector::mul_vec_ref(x.data_ref(), self.adds.data());
    }

    fn backward_ref(
        &self,
        _: &mut Self::Matrix,
        _: &Self::Matrix,

    ) {
    }
}

impl<F:Float> EulerLayer<F> for AddLayerElement<F>{
    fn update(&mut self, gradient: &Self::Matrix, lambda: F) {
        let update = FloatVector::mul(gradient.data(), lambda);
        FloatVector::add_vec_ref_direct(self.adds.data_ref(), update);
    }
}
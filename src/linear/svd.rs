use crate::shared::{complex::{Complex, Float}, matrix::Matrix};

pub struct SingularValueDecomposition<T:Float>{
    columns: usize,
    ph: T
}

impl<T:Float> SingularValueDecomposition<T>{
    pub fn new(columns: usize){
        
    }
    pub fn svd(&self, matrix: Matrix<T>){
        let rows = matrix.data.len()/self.columns;
        let dim = self.columns.min(rows);
        let max_mag = matrix.norm_max();


    }
}
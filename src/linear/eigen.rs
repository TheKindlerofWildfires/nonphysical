use crate::shared::{complex::Complex, float::Float, matrix::Matrix};


pub trait Eigen<T: Float> {
    fn symmetric_eigen(&mut self) -> (Vec<Complex<T>>, Self);
}

impl<T:Float> Eigen<T> for Matrix<T>{
    fn symmetric_eigen(&mut self) -> (Vec<Complex<T>>, Self) {
        //compute the shur decomposition

        //recover eigen values

        //recalcualte eigenvectors

        //sort the matrixices by eigen vector/eigen values

        todo!()
    }
}
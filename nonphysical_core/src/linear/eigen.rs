use crate::{linear::schur::Schur, shared::{complex::Complex, float::Float, matrix::Matrix}};


pub trait Eigen<T: Float> {
    fn eigen(&mut self) -> (Vec<Complex<T>>, Self);
}

impl<T:Float> Eigen<T> for Matrix<T>{
    fn eigen(&mut self) -> (Vec<Complex<T>>, Self) {

        //compute the shur decomposition
        let q = <Matrix<T> as Schur<T>>::schur(self);
        let t = self;
        //get the eigen values
        let eigen_values = t.data_diag().cloned().collect::<Vec<_>>();

        //recover eigen values

        //recalcualte eigenvectors

        //sort the matrixices by eigen vector/eigen values

        todo!()
    }
}

#[cfg(test)]
mod eigen_tests {
    use super::*;
    #[test]
    fn basic_eigen() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Eigen<f32>>::eigen(&mut m);
    }
}
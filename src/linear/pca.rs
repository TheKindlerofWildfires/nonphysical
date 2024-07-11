use crate::shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector};

use super::{eigen::Eigen, gemm::Gemm};

pub trait PrincipleComponentAnalysis<T: Float> {
    fn pca(&mut self, components: usize) -> Self
    where
        Self: Sized;
    fn normalize(&mut self);
    fn transform(&mut self, other: &Self) -> Self;
}
impl<T: Float> PrincipleComponentAnalysis<T> for Matrix<T> {
    fn pca(&mut self, components: usize) -> Self
    where
        Self: Sized,
    {
        self.normalize();
        let mut covariance = self.covariance();
        let (eigen_values,eigen_vectors) = <Matrix<T> as Eigen<T>>::symmetric_eigen(&mut covariance);
        let partial_principles = Matrix::new(
            components,
            eigen_vectors
                .data_rows()
                .take(components)
                .flat_map(|row| row.iter().map(|c| c.clone()))
                .collect::<Vec<_>>(),
        );
        self.transform(&partial_principles)
    }

    fn normalize(&mut self) {
        self.data_rows_ref().for_each(|row| {
            let (mean, variance) = <Vec<&'_ Complex<T>> as Vector<T>>::variance(row.iter());
            row.iter_mut()
                .for_each(|c| *c = (*c - Complex::new(mean, T::ZERO)) / variance);
        });
    }
    fn transform(&mut self, other: &Self) -> Self {
        <Matrix<T> as Gemm<T>>::gemm(self,other)
    }
}

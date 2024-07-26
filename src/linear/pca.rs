use crate::{linear::svd::SingularValueDecomposition, shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector}};

use super::{eigen::Eigen, gemm::Gemm, von_mises::VonMises};

pub trait PrincipleComponentAnalysis<T: Float> {
    fn pca(&mut self, components: usize) -> Self
    where
        Self: Sized;
    fn primary(&mut self) -> Self where Self: Sized;
    fn normalize(&mut self);
    fn transform(&mut self, other: &Self) -> Self;
}
impl<T: Float> PrincipleComponentAnalysis<T> for Matrix<T> {
    fn pca(&mut self, components: usize) -> Self
    where
        Self: Sized,
    {
        //Get the mean along axis 0
        //remove the means with something like normalize
        //take the SVD value
        //the pure answer is in U
        dbg!(&self);
        self.normalize();
        dbg!(&self);
        //*self = self.transposed();
        let (u, s, v) =
            <Matrix<T> as SingularValueDecomposition<T>>::jacobi_svd_full(self);

        //Need to do svd flip
        //then multiply by s
        //let data = <Matrix<T> as Gemm<T>>::gemm(&u,&s);
        //dbg!(data);
        todo!()

        /* 
        self.normalize();
        let mut covariance = self.covariance();
        let (_,eigen_vectors) = <Matrix<T> as Eigen<T>>::symmetric_eigen(&mut covariance);
        let partial_principles = Matrix::new(
            components,
            eigen_vectors
                .data_rows()
                .take(components)
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>(),
        );
        self.transform(&partial_principles)*/
    }

    //Returns only the first component using the von_mises short cut
    //Also has a numerical stability problem around normalizing after eigen solving but its fast
    fn primary(&mut self) -> Self{
        //self.normalize();
        let covariance = self.covariance();
        let (vector,_) = <Matrix<T> as VonMises<T>>::auto_vector(&covariance);
        let mut out = self.transform(&vector);
        let mean = <Vec<&Complex<T>> as Vector<T>>::mean(out.data());
        <Vec<&Complex<T>> as Vector<T>>::add(out.data_ref(), Complex::new(-mean,T::ZERO));
        out

    }

    fn normalize(&mut self) {
        /*(0..self.columns).for_each(|i| {
            let mean = <Vec<&'_ Complex<T>> as Vector<T>>::mean(self.data_column(i));
            <Vec<&'_ Complex<T>> as Vector<T>>::add(self.data_column_ref(i),Complex::new(mean, T::ZERO))
        }
        );*/
        self.data_rows_ref().for_each(|row| {
            let (mean, variance) = <Vec<&'_ Complex<T>> as Vector<T>>::variance(row.iter());
            dbg!(mean);
            let deviation = variance.sqrt();
            row.iter_mut()
                .for_each(|c| *c = (*c - Complex::new(mean, T::ZERO)));
        });
    }
    fn transform(&mut self, other: &Self) -> Self {
        <Matrix<T> as Gemm<T>>::gemm(self,other)
    }
}
//The best version of this probably uses SVD
#[cfg(test)]
mod pca_tests {
    use std::time::SystemTime;

    use super::*;
    #[test]
    fn primary_3x3() {
        let mut m = Matrix::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let transformed = <Matrix<f32> as PrincipleComponentAnalysis<f32>>::primary(&mut m);
        let known_values = vec![Complex::new(-5.19615242,0.0),Complex::new(0.0,0.0),Complex::new(5.19615242,0.0)];
        transformed.data().zip(known_values.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });

    }

    #[test]
    fn pca_3x3() {
        let mut m = Matrix::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let transformed = <Matrix<f32> as PrincipleComponentAnalysis<f32>>::pca(&mut m,1);
        let known_values = vec![Complex::new(-5.19615242,0.0),Complex::new(0.0,0.0),Complex::new(5.19615242,0.0)];
        transformed.data().zip(known_values.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });

    }
    #[test]
    fn primary_4x4() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let transformed = <Matrix<f32> as PrincipleComponentAnalysis<f32>>::primary(&mut m);
        let known_values = vec![Complex::new(-12.0,0.0),Complex::new(-4.0,0.0),Complex::new(4.0,0.0),Complex::new(12.0,0.0),];
        transformed.data().zip(known_values.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });

    }
    #[test]
    fn primary_5x5() {
        let mut m = Matrix::new(5, (0..25).map(|i| Complex::new(i as f32, 0.0)).collect());
        let transformed = <Matrix<f32> as PrincipleComponentAnalysis<f32>>::primary(&mut m);
        let known_values = vec![Complex::new(-22.36067977,0.0),Complex::new(-11.18033989,0.0),Complex::new(0.0,0.0),Complex::new(11.18033989,0.0),Complex::new(22.36067977,0.0)];
        transformed.data().zip(known_values.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });

    }

    #[test]
    fn primary_speed() {
        let now = SystemTime::now();
        let mut m = Matrix::new(512, (0..512*512).map(|i| Complex::new(i as f32, 0.0)).collect());
        let transformed = <Matrix<f32> as PrincipleComponentAnalysis<f32>>::primary(&mut m);
        dbg!(now.elapsed());
        assert!(1==2);

    }
}
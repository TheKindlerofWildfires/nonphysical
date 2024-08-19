#[cfg(test)]
mod eigen_tests {
    use super::*;
    #[test]
    fn basic_eigen() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Eigen<f32>>::eigen(&mut m);
    }
}


#[cfg(test)]
mod bidiagonal_tests {
    use nonphysical_core::shared::matrix::Matrix;

    use super::*;

    #[test]
    fn unblocked_r_4x4(){
        let data = (0..4*4).map(|i| i as f32/15.0).collect();
        let mut m = Matrix::new(4,data);
        let b = RealBidiagonal::new(&mut m);
    }
}
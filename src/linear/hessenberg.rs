use crate::{
    linear::householder::Householder,
    shared::{complex::Complex, float::Float, matrix::Matrix},
};

pub trait Hessenberg<T: Float> {
    fn hessenberg(&mut self) -> Vec<Complex<T>>;

    fn sequence(&self, coefficients: &[Complex<T>]) -> Self;
}

impl<T: Float> Hessenberg<T> for Matrix<T> {
    fn hessenberg(&mut self) -> Vec<Complex<T>> {
        debug_assert!(self.rows == self.columns);
        let n: usize = self.rows - 1;
        let mut h_coefficients = vec![Complex::ZERO; n];
        for (i,h_coefficient) in h_coefficients.iter_mut().enumerate() {
            let householder = Householder::make_householder_local(self, i + 1, i);
            *self.coeff_ref(i, i + 1) = householder.beta;
            *h_coefficient = householder.tau;
            householder.apply_left_local(self, i, [i + 1, self.rows], [i + 1, self.rows]);
            householder.apply_right_local(self, i, [i + 1, self.rows], [0, self.rows]);
        }
        h_coefficients
    }
    fn sequence(&self, coefficients: &[Complex<T>]) -> Self {
        let mut output = Matrix::identity(self.rows, self.rows);
        for i in (0..self.rows - 1).rev() {
            let householder = Householder {
                tau: coefficients[i],
                beta: Complex::ZERO,
            };
            let mut vec = self.data_row(i).map(|c| c.conj()).collect::<Vec<_>>();
            vec[i+1] = Complex::new(T::ONE, T::ZERO);
            householder.apply_left_local_vec(&mut output, &vec, [i + 1, self.rows], [i + 1, self.rows]);
        }
        output
    }
}

#[cfg(test)]
mod hessenberg_tests {
    use super::*;
    #[test]
    fn basic_hessenberg() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Hessenberg<f32>>::hessenberg(&mut m);

        let known_coefficients = vec![Complex::new(1.2672611,0.0), Complex::new(1.1499536,0.0), Complex::new(-4.5084846e-7,0.0)];
        coefficients.iter().zip(known_coefficients.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        let known_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.7416575, 0.0),
            Complex::new(0.42179346, 0.0),
            Complex::new(0.6326902, 0.0),
            Complex::new(-14.966629, 0.0),
            Complex::new(29.999996, 0.0),
            Complex::new( -9.797959, 0.0),
            Complex::new(0.85976774, 0.0),
            Complex::new(1.6074603e-7, 0.0),
            Complex::new(-2.4494874, 0.0),
            Complex::new(-6.5759485e-7, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.51821e-8, 0.0),
            Complex::new(5.328711e-7, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new( 5.5941918e-8, 0.0),
        ];
        m.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
    #[test]
    fn basic_sequence() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data).transposed();
        let coefficients: Vec<Complex<f32>> = vec![
            Complex::new(1.26726, 0.0),
            Complex::new(1.14995, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let m2 = <Matrix<f32> as Hessenberg<f32>>::sequence(&mut m,&coefficients);
        let known_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.26725996, 0.0),
            Complex::new(-0.5345214, 0.0),
            Complex::new(-0.8017827, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.8728662, 0.0),
            Complex::new( 0.21821885, 0.0),
            Complex::new( -0.43643647, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.40824404, 0.0),
            Complex::new(-0.8164957, 0.0),
            Complex::new( 0.40824777, 0.0),
        ];

        m2.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
}

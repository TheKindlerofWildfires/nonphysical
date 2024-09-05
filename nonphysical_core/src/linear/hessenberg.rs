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
            householder.apply_left_local(self, i, (i + 1)..self.rows, (i + 1)..self.rows);
            householder.apply_right_local(self, i, (i + 1)..self.rows, 0..self.rows);
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
            householder.apply_left_local_vec(&mut output, &vec, (i + 1)..self.rows,(i + 1)..self.rows);
        }
        output
    }
}

#[cfg(test)]
mod hessenberg_tests {
    use super::*;
    #[test]
    fn hessenberg_3x3() {
        let mut m = Matrix::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Hessenberg<f32>>::hessenberg(&mut m);

        let known_coefficients = vec![Complex::new(1.44721,0.0), Complex::new(0.0,0.0)];
        coefficients.iter().zip(known_coefficients.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        let known_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-2.23607, 0.0),
            Complex::new(0.61803395, 0.0),
            Complex::new(-6.7082, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
        ];
        m.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
    #[test]
    fn hessenberg_sequence_3x3() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-2.23607, 0.0),
            Complex::new(0.61803395, 0.0),
            Complex::new(-6.7082, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(-4.76837e-07, 0.0),
        ];
        let mut m = Matrix::new(3, data);
        let coefficients = vec![Complex::new(1.44721,0.0), Complex::new(0.0,0.0)];
        let m2 = <Matrix<f32> as Hessenberg<f32>>::sequence(&mut m,&coefficients);
        let known_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.447214, 0.0),
            Complex::new(-0.894427, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.894427, 0.0),
            Complex::new(0.447214, 0.0),
        ];

        m2.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
    #[test]
    fn hessenberg_4x4() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Hessenberg<f32>>::hessenberg(&mut m);

        let known_coefficients = vec![Complex::new(1.26726,0.0), Complex::new(1.14995,0.0), Complex::new(0.0,0.0)];
        coefficients.iter().zip(known_coefficients.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        let known_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        m.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
    #[test]
    fn hessenberg_sequence_4x4() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-3.74166, 0.0),
            Complex::new(0.421793, 0.0),
            Complex::new(0.63269, 0.0),
            Complex::new(-14.9666, 0.0),
            Complex::new(30.0, 0.0),
            Complex::new(-9.79796, 0.0),
            Complex::new(0.859768, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-2.44949, 0.0),
            Complex::new(-9.96223e-07, 0.0),
            Complex::new(-2.8054e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.76837e-07, 0.0),
            Complex::new(1.96297e-07, 0.0),
            Complex::new(3.40572e-07, 0.0),
        ];
        let mut m = Matrix::new(4, data);
        let coefficients = vec![Complex::new(1.26726,0.0), Complex::new(1.14995,0.0), Complex::new(0.0,0.0)];
        let m2 = <Matrix<f32> as Hessenberg<f32>>::sequence(&mut m,&coefficients);
        let known_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.267261, 0.0),
            Complex::new(-0.534522, 0.0),
            Complex::new(-0.801784, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.872871, 0.0),
            Complex::new(0.218218, 0.0),
            Complex::new(-0.436436, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.408248, 0.0),
            Complex::new(-0.816497, 0.0),
            Complex::new(0.408248, 0.0),
        ];

        m2.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
    #[test]
    fn hessenberg_5x5() {
        let mut m = Matrix::new(5, (0..25).map(|i| Complex::new(i as f32, 0.0)).collect());
        let coefficients = <Matrix<f32> as Hessenberg<f32>>::hessenberg(&mut m);

        let known_coefficients = vec![Complex::new(1.18257,0.0), Complex::new(1.15614,0.0), Complex::new(1.13359,0.0), Complex::new(0.0,0.0)];
        coefficients.iter().zip(known_coefficients.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        let known_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(-27.3861, 0.0),
            Complex::new(60.0, 0.0),
            Complex::new(22.3607, 0.0),
            Complex::new(-0.327098, 0.0),
            Complex::new(-0.789245, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.47213, 0.0),
            Complex::new(-1.19158e-06, 0.0),
            Complex::new(-2.17911e-07, 0.0),
            Complex::new(-0.874244, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-7.2217e-07, 0.0),
            Complex::new(-1.39336e-07, 0.0),
            Complex::new(-2.62754e-07, 0.0),
            Complex::new(4.07865e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.78137e-06, 0.0),
            Complex::new(-1.01109e-06, 0.0),
            Complex::new(7.8637e-08, 0.0),
            Complex::new(-4.53019e-07, 0.0),
        ];
        m.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }

    #[test]
    fn hessenberg_sequence_5x5() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(-5.47723, 0.0),
            Complex::new(0.308774, 0.0),
            Complex::new(0.463161, 0.0),
            Complex::new(0.617548, 0.0),
            Complex::new(-27.3861, 0.0),
            Complex::new(60.0, 0.0),
            Complex::new(22.3607, 0.0),
            Complex::new(-0.327098, 0.0),
            Complex::new(-0.789245, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.47213, 0.0),
            Complex::new(-1.19158e-06, 0.0),
            Complex::new(-2.17911e-07, 0.0),
            Complex::new(-0.874244, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-7.2217e-07, 0.0),
            Complex::new(-1.39336e-07, 0.0),
            Complex::new(-2.62754e-07, 0.0),
            Complex::new(4.07865e-07, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.78137e-06, 0.0),
            Complex::new(-1.01109e-06, 0.0),
            Complex::new(7.8637e-08, 0.0),
            Complex::new(-4.53019e-07, 0.0),
        ];
        let mut m = Matrix::new(5, data);
        let coefficients = vec![Complex::new(1.18257,0.0), Complex::new(1.15614,0.0), Complex::new(1.13359,0.0), Complex::new(0.0,0.0)];
        let m2 = <Matrix<f32> as Hessenberg<f32>>::sequence(&mut m,&coefficients);
        let known_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.182574, 0.0),
            Complex::new(-0.365148, 0.0),
            Complex::new(-0.547723, 0.0),
            Complex::new(-0.730297, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.816496, 0.0),
            Complex::new(-0.408248, 0.0),
            Complex::new(-5.96046e-08, 0.0),
            Complex::new(0.408248, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.317271, 0.0),
            Complex::new(0.75581, 0.0),
            Complex::new(-0.559808, 0.0),
            Complex::new(0.121268, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-0.446474, 0.0),
            Complex::new(0.358819, 0.0),
            Complex::new(0.621784, 0.0),
            Complex::new(-0.534129, 0.0),
        ];

        m2.data().zip(known_data.iter()).for_each(|(a, b)| {
            assert!((*a - *b).square_norm() < f32::EPSILON)
        });
    }
}

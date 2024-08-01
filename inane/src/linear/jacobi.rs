#[cfg(test)]
mod jacobi_tests {
    use baseless::{linear::jacobi::{ComplexJacobi, Jacobian, RealJacobi}, shared::{complex::{Complex, ComplexScaler}, float::Float, matrix::Matrix}};


    #[test]
    fn test_left_1_r() {
        let data: Vec<f32> = vec![
            0.0,
            -3.74166,
            0.0,
            0.0,
            -14.9666,
            30.0,
            -9.79796,
            0.0,
            0.0,
            -2.44949,
            -9.96223e-7,
            -2.8054e-7,
            0.0,
            4.76837e-7,
            1.96297e-7,
            3.40572e-7,
        ];
        let mut m = Matrix::new(4, data);

        let jacobi = RealJacobi {
            c: -7.964988e-8,
            s: 1.0,
        };
        jacobi.apply_left(&mut m, 1, 0,0..4);
        let known_data = vec![
            3.74166,
            2.9802277e-7,
            0.0,
            0.0,
            -29.999999,
            -14.966603,
            -9.79796,
            0.0,
            2.44949,
            1.9510159e-7,
            -9.96223e-7,
            -2.8054e-7,
            -4.76837e-7,
            -3.7980009829560004e-14,
            1.96297e-7,
            3.40572e-7,
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k-c).l2_norm() < f32::EPSILON));
    }

    #[test]
    fn test_right_1_r() {
        let data: Vec<f32> = vec![
            3.74166,
            2.9802277e-7,
            0.0,
            0.0,
            -29.999998,
            -14.966603,
            -9.79796,
            0.0,
            2.44949,
            1.9510159e-7,
            -9.96223e-7,
            -2.8054e-7,
            -4.76837e-7,
            -3.798001e-14,
            1.96297e-7,
            3.40572e-7,
        ];
        let mut m = Matrix::new(4, data);

        let jacobi = RealJacobi {
            c: -7.964988e-8,
            s: 1.0,
        };
        jacobi.apply_right(&mut m, 1, 0,0..4);
        let known_data = vec![
            29.999998,
            14.966603,
            9.79796,
            0.0,
            3.7416625,
            1.4901109e-6,
            7.8040637e-7,
            0.0,
            2.44949,
            1.9510159e-7,
            -9.96223e-7,
            -2.8054e-7,
            -4.76837e-7,
            -3.798001e-14,
            1.96297e-7,
            3.40572e-7,
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k-c).l2_norm() < f32::EPSILON));
    }

    #[test]
    fn test_left_1_c() {
        let data: Vec<ComplexScaler<f32>> = vec![
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(-3.74166, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(-14.9666, 0.0),
            ComplexScaler::new(30.0, 0.0),
            ComplexScaler::new(-9.79796, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(-2.44949, 0.0),
            ComplexScaler::new(-9.96223e-7, 0.0),
            ComplexScaler::new(-2.8054e-7, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(4.76837e-7, 0.0),
            ComplexScaler::new(1.96297e-7, 0.0),
            ComplexScaler::new(3.40572e-7, 0.0),
        ];
        let mut m = Matrix::new(4, data);

        let jacobi = ComplexJacobi {
            c: ComplexScaler::new(-7.964988e-8, 0.0),
            s: ComplexScaler::new(1.0, 0.0),
        };
        jacobi.apply_left(&mut m, 1, 0,0..4);
        let known_data = vec![
            ComplexScaler::new(3.74166, 0.0),
            ComplexScaler::new(2.9802277e-7, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(-29.999999, 0.0),
            ComplexScaler::new(-14.966603, 0.0),
            ComplexScaler::new(-9.79796, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(2.44949, 0.0),
            ComplexScaler::new(1.9510159e-7, 0.0),
            ComplexScaler::new(-9.96223e-7, 0.0),
            ComplexScaler::new(-2.8054e-7, 0.0),
            ComplexScaler::new(-4.76837e-7, 0.0),
            ComplexScaler::new(-3.7980009829560004e-14, 0.0),
            ComplexScaler::new(1.96297e-7, 0.0),
            ComplexScaler::new(3.40572e-7, 0.0),
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k-*c).l2_norm() < f32::EPSILON));
    }

    #[test]
    fn test_right_1_c() {
        let data: Vec<ComplexScaler<f32>> = vec![
            ComplexScaler::new(3.74166, 0.0),
            ComplexScaler::new(2.9802277e-7, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(-29.999998, 0.0),
            ComplexScaler::new(-14.966603, 0.0),
            ComplexScaler::new(-9.79796, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(2.44949, 0.0),
            ComplexScaler::new(1.9510159e-7, 0.0),
            ComplexScaler::new(-9.96223e-7, 0.0),
            ComplexScaler::new(-2.8054e-7, 0.0),
            ComplexScaler::new(-4.76837e-7, 0.0),
            ComplexScaler::new(-3.798001e-14, 0.0),
            ComplexScaler::new(1.96297e-7, 0.0),
            ComplexScaler::new(3.40572e-7, 0.0),
        ];
        let mut m = Matrix::new(4, data);

        let jacobi = ComplexJacobi {
            c: ComplexScaler::new(-7.964988e-8, 0.0),
            s: ComplexScaler::new(1.0, 0.0),
        };
        jacobi.apply_right(&mut m, 1, 0,0..4);
        let known_data = vec![
            ComplexScaler::new(29.999998, 0.0),
            ComplexScaler::new(14.966603, 0.0),
            ComplexScaler::new(9.79796, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(3.7416625, 0.0),
            ComplexScaler::new(1.4901109e-6, 0.0),
            ComplexScaler::new(7.8040637e-7, 0.0),
            ComplexScaler::new(0.0, 0.0),
            ComplexScaler::new(2.44949, 0.0),
            ComplexScaler::new(1.9510159e-7, 0.0),
            ComplexScaler::new(-9.96223e-7, 0.0),
            ComplexScaler::new(-2.8054e-7, 0.0),
            ComplexScaler::new(-4.76837e-7, 0.0),
            ComplexScaler::new(-3.798001e-14, 0.0),
            ComplexScaler::new(1.96297e-7, 0.0),
            ComplexScaler::new(3.40572e-7, 0.0),
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k-*c).l2_norm() < f32::EPSILON));
    }
}
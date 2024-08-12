#[cfg(test)]
mod jacobi_tests {
    use nonphysical_core::{
        linear::jacobi::{ComplexJacobi, Jacobian, RealJacobi},
        shared::{
            complex::{Complex, ComplexScaler},
            float::Float,
            matrix::{heap::MatrixHeap, Matrix},
            primitive::Primitive,
        },
    };
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn test_left_1_r() {
        let data: Vec<F32> = vec![
            F32(0.0),
            F32(-3.74166),
            F32(0.0),
            F32(0.0),
            F32(-14.9666),
            F32(30.0),
            F32(-9.79796),
            F32(0.0),
            F32(0.0),
            F32(-2.44949),
            F32(-9.96223e-7),
            F32(-2.8054e-7),
            F32(0.0),
            F32(4.76837e-7),
            F32(1.96297e-7),
            F32(3.40572e-7),
        ];
        let mut m = MatrixHeap::new((4, data));

        let jacobi = RealJacobi {
            c: F32(-7.964988e-8),
            s: F32(1.0),
        };
        jacobi.apply_left(&mut m, 1, 0, 0..4);
        let known_data = vec![
            F32(3.74166),
            F32(2.9802277e-7),
            F32(0.0),
            F32(0.0),
            F32(-29.999999),
            F32(-14.966603),
            F32(-9.79796),
            F32(0.0),
            F32(2.44949),
            F32(1.9510159e-7),
            F32(-9.96223e-7),
            F32(-2.8054e-7),
            F32(-4.76837e-7),
            F32(-3.7980009829560004e-14),
            F32(1.96297e-7),
            F32(3.40572e-7),
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k - *c).l2_norm() < F32::EPSILON));
    }

    #[test]
    fn test_right_1_r() {
        let data: Vec<F32> = vec![
            F32(3.74166),
            F32(2.9802277e-7),
            F32(0.0),
            F32(0.0),
            F32(-29.999998),
            F32(-14.966603),
            F32(-9.79796),
            F32(0.0),
            F32(2.44949),
            F32(1.9510159e-7),
            F32(-9.96223e-7),
            F32(-2.8054e-7),
            F32(-4.76837e-7),
            F32(-3.798001e-14),
            F32(1.96297e-7),
            F32(3.40572e-7),
        ];
        let mut m = MatrixHeap::new((4, data));

        let jacobi = RealJacobi {
            c: F32(-7.964988e-8),
            s: F32(1.0),
        };
        jacobi.apply_right(&mut m, 1, 0, 0..4);
        let known_data = vec![
            F32(29.999998),
            F32(14.966603),
            F32(9.79796),
            F32(0.0),
            F32(3.7416625),
            F32(1.4901109e-6),
            F32(7.8040637e-7),
            F32(0.0),
            F32(2.44949),
            F32(1.9510159e-7),
            F32(-9.96223e-7),
            F32(-2.8054e-7),
            F32(-4.76837e-7),
            F32(-3.798001e-14),
            F32(1.96297e-7),
            F32(3.40572e-7),
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k - *c).l2_norm() < F32::EPSILON));
    }

    #[test]
    fn test_left_1_c() {
        let data: Vec<ComplexScaler<F32>> = vec![
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(-3.74166), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(-14.9666), F32(0.0)),
            ComplexScaler::new(F32(30.0), F32(0.0)),
            ComplexScaler::new(F32(-9.79796), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(-2.44949), F32(0.0)),
            ComplexScaler::new(F32(-9.96223e-7), F32(0.0)),
            ComplexScaler::new(F32(-2.8054e-7), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(4.76837e-7), F32(0.0)),
            ComplexScaler::new(F32(1.96297e-7), F32(0.0)),
            ComplexScaler::new(F32(3.40572e-7), F32(0.0)),
        ];
        let mut m = MatrixHeap::new((4, data));

        let jacobi = ComplexJacobi {
            c: ComplexScaler::new(F32(-7.964988e-8), F32(0.0)),
            s: ComplexScaler::new(F32(1.0), F32(0.0)),
        };
        jacobi.apply_left(&mut m, 1, 0, 0..4);
        let known_data = vec![
            ComplexScaler::new(F32(3.74166), F32(0.0)),
            ComplexScaler::new(F32(2.9802277e-7), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(-29.999999), F32(0.0)),
            ComplexScaler::new(F32(-14.966603), F32(0.0)),
            ComplexScaler::new(F32(-9.79796), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(2.44949), F32(0.0)),
            ComplexScaler::new(F32(1.9510159e-7), F32(0.0)),
            ComplexScaler::new(F32(-9.96223e-7), F32(0.0)),
            ComplexScaler::new(F32(-2.8054e-7), F32(0.0)),
            ComplexScaler::new(F32(-4.76837e-7), F32(0.0)),
            ComplexScaler::new(F32(-3.7980009829560004e-14), F32(0.0)),
            ComplexScaler::new(F32(1.96297e-7), F32(0.0)),
            ComplexScaler::new(F32(3.40572e-7), F32(0.0)),
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k - *c).l2_norm() < F32::EPSILON));
    }

    #[test]
    fn test_right_1_c() {
        let data: Vec<ComplexScaler<F32>> = vec![
            ComplexScaler::new(F32(3.74166), F32(0.0)),
            ComplexScaler::new(F32(2.9802277e-7), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(-29.999998), F32(0.0)),
            ComplexScaler::new(F32(-14.966603), F32(0.0)),
            ComplexScaler::new(F32(-9.79796), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(2.44949), F32(0.0)),
            ComplexScaler::new(F32(1.9510159e-7), F32(0.0)),
            ComplexScaler::new(F32(-9.96223e-7), F32(0.0)),
            ComplexScaler::new(F32(-2.8054e-7), F32(0.0)),
            ComplexScaler::new(F32(-4.76837e-7), F32(0.0)),
            ComplexScaler::new(F32(-3.798001e-14), F32(0.0)),
            ComplexScaler::new(F32(1.96297e-7), F32(0.0)),
            ComplexScaler::new(F32(3.40572e-7), F32(0.0)),
        ];
        let mut m = MatrixHeap::new((4, data));

        let jacobi = ComplexJacobi {
            c: ComplexScaler::new(F32(-7.964988e-8), F32(0.0)),
            s: ComplexScaler::new(F32(1.0), F32(0.0)),
        };
        jacobi.apply_right(&mut m, 1, 0, 0..4);
        let known_data = vec![
            ComplexScaler::new(F32(29.999998), F32(0.0)),
            ComplexScaler::new(F32(14.966603), F32(0.0)),
            ComplexScaler::new(F32(9.79796), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(3.7416625), F32(0.0)),
            ComplexScaler::new(F32(1.4901109e-6), F32(0.0)),
            ComplexScaler::new(F32(7.8040637e-7), F32(0.0)),
            ComplexScaler::new(F32(0.0), F32(0.0)),
            ComplexScaler::new(F32(2.44949), F32(0.0)),
            ComplexScaler::new(F32(1.9510159e-7), F32(0.0)),
            ComplexScaler::new(F32(-9.96223e-7), F32(0.0)),
            ComplexScaler::new(F32(-2.8054e-7), F32(0.0)),
            ComplexScaler::new(F32(-4.76837e-7), F32(0.0)),
            ComplexScaler::new(F32(-3.798001e-14), F32(0.0)),
            ComplexScaler::new(F32(1.96297e-7), F32(0.0)),
            ComplexScaler::new(F32(3.40572e-7), F32(0.0)),
        ];
        known_data
            .into_iter()
            .zip(m.data())
            .for_each(|(k, c)| assert!((k - *c).l2_norm() < F32::EPSILON));
    }
}


#[cfg(test)]
mod fourier_tests {
    use baseless::{shared::{complex::{Complex, ComplexScaler}, float::Float}, signal::fourier::FastFourierTransform};

    #[test]
    fn forward_2_static() {
        let mut x = (0..2)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexScaler::<f32>::new(1.0, 3.0),
            ComplexScaler::<f32>::new(-1.0, -1.0),
        ];
        let truth_2 = [
            ComplexScaler::<f32>::new(0.0, 2.0),
            ComplexScaler::<f32>::new(2.0, 4.0),
        ];
        let truth_3 = [
            ComplexScaler::<f32>::new(2.0, 6.0),
            ComplexScaler::<f32>::new(-2.0, -2.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn forward_4_static() {
        let mut x = (0..4)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexScaler::<f32>::new(6.0, 10.0),
            ComplexScaler::<f32>::new(-4.0, 0.0),
            ComplexScaler::<f32>::new(-2.0, -2.0),
            ComplexScaler::<f32>::new(0.0, -4.0),
        ];
        let truth_2 = [
            ComplexScaler::<f32>::new(0.0, 4.0),
            ComplexScaler::<f32>::new(12.0, 16.0),
            ComplexScaler::<f32>::new(8.0, 12.0),
            ComplexScaler::<f32>::new(4.0, 8.0),
        ];
        let truth_3 = [
            ComplexScaler::<f32>::new(24.0, 40.0),
            ComplexScaler::<f32>::new(0.0, -16.0),
            ComplexScaler::<f32>::new(-8.0, -8.0),
            ComplexScaler::<f32>::new(-16.0, 0.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn forward_8_static() {
        let mut x = (0..8)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexScaler::<f32>::new(28.0, 36.0),
            ComplexScaler::<f32>::new(-13.65685425, 5.65685425),
            ComplexScaler::<f32>::new(-8.0, 0.0),
            ComplexScaler::<f32>::new(-5.65685425, -2.34314575),
            ComplexScaler::<f32>::new(-4.0, -4.0),
            ComplexScaler::<f32>::new(-2.34314575, -5.65685425),
            ComplexScaler::<f32>::new(0.0, -8.0),
            ComplexScaler::<f32>::new(5.65685425, -13.65685425),
        ];
        let truth_2 = [
            ComplexScaler::<f32>::new(0.0, 8.0),
            ComplexScaler::<f32>::new(56.0, 64.0),
            ComplexScaler::<f32>::new(48.0, 56.0),
            ComplexScaler::<f32>::new(40.0, 48.0),
            ComplexScaler::<f32>::new(32.0, 40.0),
            ComplexScaler::<f32>::new(24.0, 32.0),
            ComplexScaler::<f32>::new(16.0, 24.0),
            ComplexScaler::<f32>::new(8.0, 16.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn forward_16_static() {
        let mut x = (0..16)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexScaler::<f32>::new(120.0, 136.0),
            ComplexScaler::<f32>::new(-48.21871594, 32.21871594),
            ComplexScaler::<f32>::new(-27.3137085, 11.3137085),
            ComplexScaler::<f32>::new(-19.9728461, 3.9728461),
            ComplexScaler::<f32>::new(-16.0, 0.0),
            ComplexScaler::<f32>::new(-13.3454291, -2.6545709),
            ComplexScaler::<f32>::new(-11.3137085, -4.6862915),
            ComplexScaler::<f32>::new(-9.59129894, -6.40870106),
            ComplexScaler::<f32>::new(-8.0, -8.0),
            ComplexScaler::<f32>::new(-6.40870106, -9.59129894),
            ComplexScaler::<f32>::new(-4.6862915, -11.3137085),
            ComplexScaler::<f32>::new(-2.6545709, -13.3454291),
            ComplexScaler::<f32>::new(0.0, -16.0),
            ComplexScaler::<f32>::new(3.9728461, -19.9728461),
            ComplexScaler::<f32>::new(11.3137085, -27.3137085),
            ComplexScaler::<f32>::new(32.21871594, -48.21871594),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn inverse_2_static() {
        let mut x = [
            ComplexScaler::<f32>::new(2.0, 6.0),
            ComplexScaler::<f32>::new(-2.0, -2.0),
        ];

        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexScaler::<f32>::new(0.0, 2.0),
            ComplexScaler::<f32>::new(2.0, 4.0),
        ];
        let truth_2 = [
            ComplexScaler::<f32>::new(1.0, 3.0),
            ComplexScaler::<f32>::new(-1.0, -1.0),
        ];
        let truth_3 = (0..2)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn inverse_4_static() {
        let mut x = [
            ComplexScaler::<f32>::new(24.0, 40.0),
            ComplexScaler::<f32>::new(0.0, -16.0),
            ComplexScaler::<f32>::new(-8.0, -8.0),
            ComplexScaler::<f32>::new(-16.0, 0.0),
        ];

        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexScaler::<f32>::new(0.0, 4.0),
            ComplexScaler::<f32>::new(12.0, 16.0),
            ComplexScaler::<f32>::new(8.0, 12.0),
            ComplexScaler::<f32>::new(4.0, 8.0),
        ];
        let truth_2 = [
            ComplexScaler::<f32>::new(6.0, 10.0),
            ComplexScaler::<f32>::new(-4.0, 0.0),
            ComplexScaler::<f32>::new(-2.0, -2.0),
            ComplexScaler::<f32>::new(0.0, -4.0),
        ];
        let truth_3 = (0..4)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn inverse_8_static() {
        let mut x = [
            ComplexScaler::<f32>::new(0.0, 8.0),
            ComplexScaler::<f32>::new(56.0, 64.0),
            ComplexScaler::<f32>::new(48.0, 56.0),
            ComplexScaler::<f32>::new(40.0, 48.0),
            ComplexScaler::<f32>::new(32.0, 40.0),
            ComplexScaler::<f32>::new(24.0, 32.0),
            ComplexScaler::<f32>::new(16.0, 24.0),
            ComplexScaler::<f32>::new(8.0, 16.0),
        ];
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexScaler::<f32>::new(28.0, 36.0),
            ComplexScaler::<f32>::new(-13.65685425, 5.65685425),
            ComplexScaler::<f32>::new(-8.0, 0.0),
            ComplexScaler::<f32>::new(-5.65685425, -2.34314575),
            ComplexScaler::<f32>::new(-4.0, -4.0),
            ComplexScaler::<f32>::new(-2.34314575, -5.65685425),
            ComplexScaler::<f32>::new(0.0, -8.0),
            ComplexScaler::<f32>::new(5.65685425, -13.65685425),
        ];
        let truth_2 = (0..8)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn inverse_16_static() {
        let mut x = [
            ComplexScaler::<f32>::new(120.0, 136.0),
            ComplexScaler::<f32>::new(-48.21871594, 32.21871594),
            ComplexScaler::<f32>::new(-27.3137085, 11.3137085),
            ComplexScaler::<f32>::new(-19.9728461, 3.9728461),
            ComplexScaler::<f32>::new(-16.0, 0.0),
            ComplexScaler::<f32>::new(-13.3454291, -2.6545709),
            ComplexScaler::<f32>::new(-11.3137085, -4.6862915),
            ComplexScaler::<f32>::new(-9.59129894, -6.40870106),
            ComplexScaler::<f32>::new(-8.0, -8.0),
            ComplexScaler::<f32>::new(-6.40870106, -9.59129894),
            ComplexScaler::<f32>::new(-4.6862915, -11.3137085),
            ComplexScaler::<f32>::new(-2.6545709, -13.3454291),
            ComplexScaler::<f32>::new(0.0, -16.0),
            ComplexScaler::<f32>::new(3.9728461, -19.9728461),
            ComplexScaler::<f32>::new(11.3137085, -27.3137085),
            ComplexScaler::<f32>::new(32.21871594, -48.21871594),
        ];
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = (0..16)
            .map(|i| ComplexScaler::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }
}

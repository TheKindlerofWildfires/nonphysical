
#[cfg(test)]
mod wavelet_tests {
    use std::time::SystemTime;

    use baseless::{random::pcg::PermutedCongruentialGenerator, shared::{complex::{Complex, ComplexScaler}, float::Float}, signal::wavelet::{DaubechiesFirstComplexWavelet, DaubechiesFirstRealWavelet, DiscreteWavelet}};

    #[test]
    fn daubechies_c_first_2_static() {
        let signal = vec![
            ComplexScaler::<f32>::new(0.5, 0.0),
            ComplexScaler::<f32>::new(1.5, 0.0),
        ];
        let knowns = vec![
            vec![ComplexScaler::new(2.0.sqrt(), 0.0)],
            vec![ComplexScaler::new(-(2.0).sqrt() / 2.0, 0.0)],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_4_static() {
        let signal = vec![
            ComplexScaler::<f32>::new(0.5, 0.0),
            ComplexScaler::<f32>::new(1.5, 0.0),
            ComplexScaler::<f32>::new(-1.0, 0.0),
            ComplexScaler::<f32>::new(-2.5, 0.0),
        ];
        let knowns = vec![
            vec![
                ComplexScaler::new(2.0.sqrt(), 0.0),
                ComplexScaler::new(-7.0 * 2.0.sqrt() / 4.0, 0.0),
            ],
            vec![
                ComplexScaler::new(-(2.0).sqrt() / 2.0, 0.0),
                ComplexScaler::new(3.0 * 2.0.sqrt() / 4.0, 0.0),
            ],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_8_static() {
        let signal = vec![
            ComplexScaler::<f32>::new(0.5, 0.0),
            ComplexScaler::<f32>::new(1.5, 0.0),
            ComplexScaler::<f32>::new(-1.0, 0.0),
            ComplexScaler::<f32>::new(-2.5, 0.0),
            ComplexScaler::<f32>::new(-1.5, 0.0),
            ComplexScaler::<f32>::new(1.5, 0.0),
            ComplexScaler::<f32>::new(-2.0, 0.0),
            ComplexScaler::<f32>::new(2.0, 0.0),
        ];
        let knowns = vec![
            vec![
                ComplexScaler::new(2.0.sqrt(), 0.0),
                ComplexScaler::new(-7.0 * 2.0.sqrt() / 4.0, 0.0),
                ComplexScaler::ZERO,
                ComplexScaler::ZERO,
            ],
            vec![
                ComplexScaler::new(-(2.0).sqrt() / 2.0, 0.0),
                ComplexScaler::new(3.0 * 2.0.sqrt() / 4.0, 0.0),
                ComplexScaler::new(-3.0 * 2.0.sqrt() / 2.0, 0.0),
                ComplexScaler::new(-2.0 * 2.0.sqrt(), 0.0),
            ],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_2_static() {
        let signal = vec![
            0.5,
            1.5
        ];
        let knowns = vec![
            vec![2.0.sqrt()],
            vec![-(2.0).sqrt() / 2.0],
        ];

        let dfw: DaubechiesFirstRealWavelet<f32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_4_static() {
        let signal = vec![
            0.5,
            1.5,
            -1.0,
            -2.5,
        ];
        let knowns = vec![
            vec![
                2.0.sqrt(),
                -7.0 * 2.0.sqrt() / 4.0,
            ],
            vec![
                -(2.0).sqrt() / 2.0,
                3.0 * 2.0.sqrt() / 4.0,
            ],
        ];

        let dfw: DaubechiesFirstRealWavelet<f32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_8_static() {
        let signal = vec![
            0.5,
            1.5,
            -1.0,
            -2.5,
            -1.5,
            1.5,
            -2.0,
            2.0,
        ];
        let knowns = vec![
            vec![
                2.0.sqrt(),
                -7.0 * 2.0.sqrt() / 4.0,
                0.0,
                0.0,
            ],
            vec![
                -(2.0).sqrt() / 2.0,
                3.0 * 2.0.sqrt() / 4.0,
                -3.0 * 2.0.sqrt() / 2.0,
                -2.0 * 2.0.sqrt(),
            ],
        ];

        let dfw: DaubechiesFirstRealWavelet<f32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let signal =
            (0..2048*4096).map(|_| ComplexScaler::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();
        let dfw = DaubechiesFirstComplexWavelet::new();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}",now.elapsed());

        let signal =
            (0..1024*4096).map(|_| ComplexScaler::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}",now.elapsed());

        let signal =
        (0..512*4096).map(|_| ComplexScaler::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}",now.elapsed());
    }
}

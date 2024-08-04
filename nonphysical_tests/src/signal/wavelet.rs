#[cfg(test)]
mod wavelet_tests {
    use std::time::SystemTime;

    use nonphysical_core::{
        random::pcg::PermutedCongruentialGenerator,
        shared::{
            complex::{Complex, ComplexScaler},
            float::Float,
            primitive::Primitive,
        },
        signal::wavelet::{
            DaubechiesFirstComplexWavelet, DaubechiesFirstRealWavelet, DiscreteWavelet,
        },
    };
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn daubechies_c_first_2_static() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
        ];
        let knowns = vec![
            vec![ComplexScaler::new(F32(2.0).sqrt(), F32(0.0))],
            vec![ComplexScaler::new(-F32(2.0).sqrt() / F32(2.0), F32(0.0))],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < F32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_4_static() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
        ];
        let knowns = vec![
            vec![
                ComplexScaler::new(F32(2.0).sqrt(), F32(0.0)),
                ComplexScaler::new(F32(-7.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
            ],
            vec![
                ComplexScaler::new(-F32(2.0).sqrt() / F32(2.0), F32(0.0)),
                ComplexScaler::new(F32(3.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
            ],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < F32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_8_static() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(2.0), F32(0.0)),
        ];
        let knowns = vec![
            vec![
                ComplexScaler::new(F32(2.0).sqrt(), F32(0.0)),
                ComplexScaler::new(F32(-7.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
                ComplexScaler::ZERO,
                ComplexScaler::ZERO,
            ],
            vec![
                ComplexScaler::new(-F32(2.0).sqrt() / F32(2.0), F32(0.0)),
                ComplexScaler::new(F32(3.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
                ComplexScaler::new(-F32(3.0) * F32(2.0).sqrt() / F32(2.0), F32(0.0)),
                ComplexScaler::new(F32(-2.0) * F32(2.0).sqrt(), F32(0.0)),
            ],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < F32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_2_static() {
        let signal = vec![F32(0.5), F32(1.5)];
        let knowns = vec![vec![F32(2.0).sqrt()], vec![-F32(2.0).sqrt() / F32(2.0)]];

        let dfw: DaubechiesFirstRealWavelet<F32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < F32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_4_static() {
        let signal = vec![F32(0.5), F32(1.5), F32(-1.0), F32(-2.5)];
        let knowns = vec![
            vec![F32(2.0).sqrt(), F32(-7.0) * F32(2.0).sqrt() / F32(4.0)],
            vec![
                -F32(2.0).sqrt() / F32(2.0),
                F32(3.0) * F32(2.0).sqrt() / F32(4.0),
            ],
        ];

        let dfw: DaubechiesFirstRealWavelet<F32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < F32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_8_static() {
        let signal = vec![F32(0.5), F32(1.5), F32(-1.0), F32(-2.5), F32(-1.5), F32(1.5), F32(-2.0), F32(2.0)];
        let knowns = vec![
            vec![
                F32(2.0).sqrt(),
                F32(-7.0) * F32(2.0).sqrt() / F32(4.0),
                F32(0.0),
                F32(0.0),
            ],
            vec![
                -F32(2.0).sqrt() / F32(2.0),
                F32(3.0) * F32(2.0).sqrt() / F32(4.0),
                -F32(3.0) * F32(2.0).sqrt() / F32(2.0),
                F32(-2.0) * F32(2.0).sqrt(),
            ],
        ];

        let dfw: DaubechiesFirstRealWavelet<F32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < F32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let signal = (0..2048 * 4096)
            .map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)))
            .collect::<Vec<_>>();
        let dfw = DaubechiesFirstComplexWavelet::new();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}", now.elapsed());

        let signal = (0..1024 * 4096)
            .map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)))
            .collect::<Vec<_>>();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}", now.elapsed());

        let signal = (0..512 * 4096)
            .map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)))
            .collect::<Vec<_>>();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}", now.elapsed());
    }
}

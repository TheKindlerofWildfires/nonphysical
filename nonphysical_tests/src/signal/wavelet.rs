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
        signal::wavelet::{wavelet_heap::DaubechiesFirstWaveletHeap, DiscreteWavelet},
    };
    use nonphysical_std::shared::primitive::F32;
    use nonphysical_core::shared::matrix::Matrix;
    #[test]
    fn daubechies_c_first_2_static() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
        ];
        let knowns = vec![
            ComplexScaler::new(F32(2.0).sqrt(), F32(0.0)),
            ComplexScaler::new(-F32(2.0).sqrt() / F32(2.0), F32(0.0)),
        ];

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < F32::EPSILON);
        });

        let reconstruction = dfw.backward(&deconstruction);
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
            ComplexScaler::new(F32(2.0).sqrt(), F32(0.0)),
            ComplexScaler::new(F32(-7.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
            ComplexScaler::new(-F32(2.0).sqrt() / F32(2.0), F32(0.0)),
            ComplexScaler::new(F32(3.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
        ];

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < F32::EPSILON);
        });

        let reconstruction = dfw.backward(&deconstruction);
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
            ComplexScaler::new(F32(2.0).sqrt(), F32(0.0)),
            ComplexScaler::new(F32(-7.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
            ComplexScaler::ZERO,
            ComplexScaler::ZERO,
            ComplexScaler::new(-F32(2.0).sqrt() / F32(2.0), F32(0.0)),
            ComplexScaler::new(F32(3.0) * F32(2.0).sqrt() / F32(4.0), F32(0.0)),
            ComplexScaler::new(-F32(3.0) * F32(2.0).sqrt() / F32(2.0), F32(0.0)),
            ComplexScaler::new(F32(-2.0) * F32(2.0).sqrt(), F32(0.0)),
        ];

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < F32::EPSILON);
        });

        let reconstruction = dfw.backward(&deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }
    #[test]
    fn daubechies_c_first_4_decompose() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
        ];

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.decompose(&signal);
        let valid = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.4142137), F32(0.0)),
            ComplexScaler::<F32>::new(F32( -2.474874), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.7071068), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.0606602), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.75000024), F32(0.0)),
            ComplexScaler::<F32>::new(F32( 2.7500005), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.25000006 ), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.2500002), F32(0.0)),
        ];
        deconstruction.data().zip(valid.iter()).for_each(|(a,b)|{
            assert!((*a-*b).l2_norm()<F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_8_decompose() {
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

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.decompose(&signal);
        let valid = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(2.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.4142137), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.474874), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.7071068), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.0606602), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.1213205), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.8284273 ), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.75000024), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(2.7500005), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32( 0.25000006), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-3.5000005), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.2500002), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.5303303), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.5303303), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.9445441), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.9445441), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.2980976), F32(0.0)),
            ComplexScaler::<F32>::new(F32(2.651651), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.5303303), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.2374371), F32(0.0)),
        ];
        deconstruction.data().zip(valid.iter()).for_each(|(a,b)|{
            assert!((*a-*b).l2_norm()<F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_4_cis_detail() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
        ];

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.cis_detail(&signal);
        let valid = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.50000006), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.50000006), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.7500001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.7500001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.62500024), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.62500024), F32(0.0)),
            ComplexScaler::<F32>::new(F32( 0.62500024), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.62500024), F32(0.0)),
        ];
        deconstruction.data().zip(valid.iter()).for_each(|(a,b)|{
            assert!((*a-*b).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn daubechies_c_first_4_cis_approx() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
        ];

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.cis_approx(&signal);
        let valid = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.0000001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.0000001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.7500004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.7500004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.37500018), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.37500018), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.37500018), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.37500018), F32(0.0)),
        ];
        deconstruction.data().zip(valid.iter()).for_each(|(a,b)|{
            assert!((*a-*b).l2_norm()<F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_4_trans_detail() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
        ];
        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.trans_detail(&signal);
        let valid = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.50000006), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.50000006), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.7500001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-0.7500001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.3750004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.3750004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.3750004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.3750004), F32(0.0)),
        ];
        deconstruction.data().zip(valid.iter()).for_each(|(a,b)|{
            assert!((*a-*b).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn daubechies_c_first_4_trans_approx() {
        let signal = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
        ];

        let dfw = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.trans_approx(&signal);
        let valid = vec![
            ComplexScaler::<F32>::new(F32(0.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.0), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-2.5), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.0000001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(1.0000001), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.7500004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(-1.7500004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.12500004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.12500004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.12500004), F32(0.0)),
            ComplexScaler::<F32>::new(F32(0.12500004), F32(0.0)),
        ];
        deconstruction.data().zip(valid.iter()).for_each(|(a,b)|{
            assert!((*a-*b).l2_norm()<F32::EPSILON);
        });
    }
    #[test]
    fn daubechies_r_first_2_static() {
        let signal = vec![F32(0.5), F32(1.5)];
        let knowns = vec![F32(2.0).sqrt(), -F32(2.0).sqrt() / F32(2.0)];

        let dfw: DaubechiesFirstWaveletHeap<F32> = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < F32::EPSILON);
        });

        let reconstruction = dfw.backward(&deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_4_static() {
        let signal = vec![F32(0.5), F32(1.5), F32(-1.0), F32(-2.5)];
        let knowns = vec![
            F32(2.0).sqrt(),
            F32(-7.0) * F32(2.0).sqrt() / F32(4.0),
            -F32(2.0).sqrt() / F32(2.0),
            F32(3.0) * F32(2.0).sqrt() / F32(4.0),
        ];

        let dfw: DaubechiesFirstWaveletHeap<F32> = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < F32::EPSILON);
        });

        let reconstruction = dfw.backward(&deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_8_static() {
        let signal = vec![
            F32(0.5),
            F32(1.5),
            F32(-1.0),
            F32(-2.5),
            F32(-1.5),
            F32(1.5),
            F32(-2.0),
            F32(2.0),
        ];
        let knowns = vec![
            F32(2.0).sqrt(),
            F32(-7.0) * F32(2.0).sqrt() / F32(4.0),
            F32(0.0),
            F32(0.0),
            -F32(2.0).sqrt() / F32(2.0),
            F32(3.0) * F32(2.0).sqrt() / F32(4.0),
            -F32(3.0) * F32(2.0).sqrt() / F32(2.0),
            F32(-2.0) * F32(2.0).sqrt(),
        ];

        let dfw: DaubechiesFirstWaveletHeap<F32> = DaubechiesFirstWaveletHeap::new(());
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < F32::EPSILON);
        });

        let reconstruction = dfw.backward(&deconstruction);
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
        let dfw = DaubechiesFirstWaveletHeap::new(());
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

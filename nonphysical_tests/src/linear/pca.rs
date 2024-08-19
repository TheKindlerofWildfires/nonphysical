
#[cfg(test)]
mod pca_tests {
    use std::time::SystemTime;

    use nonphysical_core::{linear::pca::{PrincipleComponentAnalysis, RealPrincipleComponentAnalysis}, shared::{float::Float, matrix::{matrix_heap::MatrixHeap, Matrix}, primitive::Primitive}};
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn pca_3x3() {
        let mut m = MatrixHeap::new((3, (0..9).map(|i| F32(i as f32)).collect()));
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        dbg!(&transformed);
        let known_values = vec![
            -1.7320502, -2.1976344e-8, 1.7320505,
            -2.8048786e-10, -3.5588445e-18, -2.8048786e-10,
        ];
        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c - F32(k)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn pca_3x3_t() {
        let mut m = MatrixHeap::new((3, (0..9).map(|i| F32(i as f32)).collect())).transposed();
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        dbg!(&transformed);
        let known_values = vec![
            -5.19615242,
            0.0,
            5.19615242,
            -8.414635e-10,
            -1.06765335e-17,
            -8.414635e-10,
        ];
        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c - F32(k)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn pca_4x4() {
        let mut m = MatrixHeap::new((4, (0..16).map(|i| F32(i as f32)).collect())).transposed();
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        let known_values = vec![
            -1.1999999e+01,
            -4.0000005e+00,
            4.0000005e+00,
            1.1999999e+01,
            2.7701734e-07,
            3.7073377e-08,
            -3.7073377e-08,
            3.0173291e-07,
        ];
        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c - F32(k)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn pca_5x5() {
        let mut m = MatrixHeap::new((5, (0..25).map(|i| F32(i as f32)).collect()));
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        let known_values = vec![
            -2.2360683e+01,
            -1.1180341e+01,
            0.0000000e+00,
            1.1180341e+01,
            2.2360682e+01,
            8.0921933e-07,
            -2.6973981e-07,
            0.0000000e+00,
            2.6973976e-07,
            5.3947952e-07,
        ];
        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c -F32(k)).l2_norm() < F32::EPSILON);
            });
    }
    #[test]
    fn pca_4x3() {
        //wrong in svd
        let mut m = MatrixHeap::new((3, (0..12).map(|i| F32(i as f32)).collect()));
        dbg!(&m);
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        dbg!(&transformed);
        let known_values = vec![
            -2.5980763,-8.6602551,-8.6602551,2.5980763,
            6.5293634e-08,8.7382812e-09,-8.7382812e-09,7.1119146e-08
        ];

        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c -F32(k)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn pca_4x3_t() {
        //wrong in svd
        let mut m = MatrixHeap::new((4, (0..12).map(|i| F32(i as f32)).collect())).transposed();
        dbg!(&m);
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        dbg!(&transformed);
        let known_values = vec![
            -7.7942300,-2.5980766,2.5980766,7.7942295,
            4.3529084e-08,-9.1508454e-08,9.1508454e-08,-1.7476561e-08
        ];

        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c -F32(k)).l2_norm() < F32::EPSILON);
            });
    }
    #[test]
    fn pca_3x4() {
        //bugged in svd
        let mut m = MatrixHeap::new((3, (0..12).map(|i| F32(i as f32)).collect())).transposed();
        dbg!(&m);
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        dbg!(&transformed);
        let known_values = vec![
            -2.0,0.0,2.0,1.0e-7, 0.0, 1.0e-7
        ];

        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c -F32(k)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn pca_3x4_t() {
        //bugged in svd
        let mut m = MatrixHeap::new((3, (0..12).map(|i| F32(i as f32)).collect())).transposed();
        dbg!(&m);
        let transformed = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        dbg!(&transformed);
        let known_values = vec![
            -7.9999995, -0.0, 7.9999995,4.1295317e-07, 0.0, 4.1295311e-07
        ];

        transformed
            .data()
            .zip(known_values.into_iter())
            .for_each(|(c, k)| {
                assert!((*c -F32(k)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn pca_time() {
        let mut m = MatrixHeap::new((1024, (0..1024*1024).map(|i| F32(i as f32)).collect())).transposed();
        let now = SystemTime::now();
        let _ = RealPrincipleComponentAnalysis::pca(&mut m, 2);

        let _ = println!("{:?}", now.elapsed());
    }
    /* 
    #[test]
    fn longer_pca_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let data = (0..2048 * 2048)
            .map(|_| pcg.next_u32() as f32 / u32::MAX as f32)
            .collect::<Vec<_>>();
        let mut m = MatrixHeap::new(2048, data);
        let now = SystemTime::now();
        let _ = RealPrincipleComponentAnalysis::pca(&mut m, 2);
        let _ = println!("{:?}", now.elapsed());
        
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let data = (0..2048 * 2048)
            .map(|_| pcg.next_u32() as f32 / u32::MAX as f32)
            .collect::<Vec<_>>();
        let mut m = MatrixHeap::new(2048, data);
        let now = SystemTime::now();
        let _ = RealPrincipleComponentAnalysis::pca(&mut m, 3);
        let _ = println!("{:?}", now.elapsed());
    }*/
}

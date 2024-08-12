#[cfg(test)]
pub mod svd_tests {
    use std::time::SystemTime;

    use nonphysical_core::{
        linear::svd::{RealSingularValueDecomposition, SingularValueDecomposition},
        shared::{
            float::Float,
            matrix::{heap::MatrixHeap, Matrix},
            primitive::Primitive,
        },
    };
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn jacobi_svd_square_r() {
        let mut in_mat = MatrixHeap::new((3, (0..9).map(|i| F32(i as f32)).collect()));
        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([14.2267, 1.26523, 7.16572e-8])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });

        let mut in_mat = MatrixHeap::new((5, (0..25).map(|i| F32(i as f32)).collect()));

        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([69.9086, 3.5761, 1.4977e-6, 1.0282e-6, 2.22847e-7])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn jacobi_svd_col_r() {
        let mut in_mat =
            MatrixHeap::new((4, (0..12).map(|i| F32(i as f32)).collect())).transposed();
        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([22.4468, 1.46406, 8.76493e-07])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });

        let mut in_mat =
            MatrixHeap::new((5, (0..20).map(|i| F32(i as f32)).collect())).transposed();

        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([49.6337, 2.54849, 9.15324e-07, 7.48904e-07])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn jacobi_svd_full_col_r() {
        let mut in_mat =
            MatrixHeap::new((4, (0..12).map(|i| F32(i as f32)).collect())).transposed();
        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);

        let known_u = vec![
            -0.497573, -0.573971, -0.650368, 0.765346, 0.0623779, -0.640589, -0.408248, 0.816497,
            -0.408248,
        ];
        let known_v = vec![
            -0.08351782, -0.31365094, -0.54378414, -0.751945, -0.832481, -0.44902518, -0.06556851, 0.366851,
            0.43411967, -0.32989407, -0.6425695, 0.5383443, -0.33397692, 0.7688756, -0.5358213, 0.100922346,
        ];

        s.into_iter()
            .zip([22.4468, 1.46406, 8.76493e-07])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });
        u.data().zip(known_u.iter()).for_each(|(ui,ki)|{
            assert!((*ui - F32(*ki)).l2_norm() < F32::EPSILON);
        });

        v.data().zip(known_v.iter()).for_each(|(vi,ki)|{
            assert!((*vi - F32(*ki)).l2_norm() < F32::EPSILON);
        });
        let mut in_mat =
            MatrixHeap::new((5, (0..20).map(|i| F32(i as f32)).collect())).transposed();
            let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
            let known_u = vec![
                -0.43989652, -0.47870785, -0.5175191, -0.5563305, 0.7116815, 0.26615584, -0.17937064, -0.62489724,
                0.47809163, -0.8366599, 0.2390457, 0.11952255,0.26726142, 0.0, -0.8017837, 0.53452224
            ];
            let known_v = vec![
                -0.06412447, -0.22469716, -0.3852698, -0.5458424, -0.6869464,
                -0.7719378, -0.4995108, -0.22708385, 0.045342598, 0.35791713,
                0.5084658, -0.51264495, -0.0058990587, -0.48412958, 0.49420792,
                0.36759225, -0.24614105, -0.66537243, 0.5987976, -0.05487675,
                -0.079614125, 0.61368537, -0.59769964, -0.32720026, 0.39082867,
            ];
    
            s.into_iter()
            .zip([49.6337, 2.54849, 9.15324e-07, 7.48904e-07])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });
            u.data().zip(known_u.iter()).for_each(|(ui,ki)|{
                assert!((*ui - F32(*ki)).l2_norm() < F32::EPSILON);
            });
    
            v.data().zip(known_v.iter()).for_each(|(vi,ki)|{
                assert!((*vi - F32(*ki)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn jacobi_svd_col_speed() {
        let now = SystemTime::now();
        let mut in_mat =
            MatrixHeap::new((4096, (0..4096 * 4).map(|i| F32(i as f32)).collect())).transposed();
        let _ = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);
        let _ = dbg!(now.elapsed());
    }

    #[test]
    fn jacobi_svd_col_full_speed() {
        let now = SystemTime::now();
        let mut in_mat =
            MatrixHeap::new((4096, (0..4096 * 4).map(|i| F32(i as f32)).collect())).transposed();
        let (u,s,v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        dbg!(u.rows, u.cols, v.rows, v.cols);
        let _ = dbg!(now.elapsed());
    }

    #[test]
    fn jacobi_svd_full_square_r() {
        let mut in_mat = MatrixHeap::new((3, (0..9).map(|i| F32(i as f32)).collect()));
        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter()
            .zip([14.2267, 1.26523, 9.010921e-8])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });

        let kv = vec![
            0.46632808,
            0.5709908,
            0.6756534,
            -0.7847747,
            -0.085456595,
            0.6138613,
            -0.40824822,
            0.8164965,
            -0.40824825,
        ];
        u.data()
            .zip(kv.into_iter())
            .for_each(|(up, k)| assert!((*up - F32(k)).l2_norm() < F32::EPSILON));

        let ku = vec![
            0.13511896,
            0.49633512,
            0.85755134,
            0.9028158,
            0.29493162,
            -0.31295204,
            -0.40824804,
            0.81649673,
            -0.40824836,
        ];

        v.data()
            .zip(ku.into_iter())
            .for_each(|(up, k)| assert!((*up - F32(k)).l2_norm() < F32::EPSILON));
    }
    #[test]
    fn jacobi_svd_full_square_r_pca() {
        let mut in_mat = MatrixHeap::zero(3, 3);
        in_mat.data_row_ref(0).for_each(|rp| *rp = F32(-3.0));
        in_mat.data_row_ref(2).for_each(|rp| *rp = F32(3.0));

        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter()
            .zip([7.348469, 0.0, 0.0])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });

        let ku = vec![
            -0.57735026,
            -0.57735026,
            -0.57735026,
            0.70710677,
            -0.70710677,
            0.0,
            0.40824828,
            0.40824828,
            -0.8164966,
        ];
        let kv = vec![
            0.70710677,
            0.0,
            -0.70710677,
            0.0,
            1.0,
            0.0,
            0.70710677,
            0.0,
            0.70710677,
        ];

        u.data()
            .zip(ku.into_iter())
            .for_each(|(up, k)| assert!((*up - F32(k)).l2_norm() < F32::EPSILON));

        v.data()
            .zip(kv.into_iter())
            .for_each(|(up, k)| assert!((*up - F32(k)).l2_norm() < F32::EPSILON));
    }

    #[test]
    fn jacobi_svd_full_square_r_pca2() {
        let mut in_mat = MatrixHeap::zero(3, 3);
        in_mat.data_row_ref(0).for_each(|rp| *rp = F32(-3.0));
        in_mat.data_row_ref(2).for_each(|rp| *rp = F32(3.0));
        in_mat = in_mat.transposed();

        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter()
            .zip([7.348469, 0.0, 0.0])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });

        let ku = vec![
            -0.70710677,
            0.0,
            0.70710677,
            -0.70710677,
            0.0,
            -0.70710677,
            0.0,
            1.0,
            0.0,
        ];
        let kv = vec![
            0.57735026,
            0.57735026,
            0.57735026,
            -0.40824828,
            -0.40824828,
            0.8164966,
            0.70710677,
            -0.70710677,
            0.0,
        ];
        u.data()
            .zip(ku.into_iter())
            .for_each(|(up, k)| assert!((*up - F32(k)).l2_norm() < F32::EPSILON));

        v.data()
            .zip(kv.into_iter())
            .for_each(|(up, k)| assert!((*up - F32(k)).l2_norm() < F32::EPSILON));
    }
    #[test]
    fn jacobi_svd_speed_square() {
        let mut in_mat = MatrixHeap::new((512, (0..512 * 512).map(|i| F32(i as f32)).collect()));
        let now = SystemTime::now();
        let (_, _, _) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        let _ = println!("{:?}", now.elapsed());
    }
}

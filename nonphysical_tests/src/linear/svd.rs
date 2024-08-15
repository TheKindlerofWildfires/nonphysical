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
#[cfg(test)]
mod svd_col_pivot_tests {
    use nonphysical_core::{
        linear::{qr_decomposition::{QRDecomposition, RealQRDecomposition}, svd::{RealSingularValueDecomposition, SingularValueDecomposition}},
        shared::{
            float::Float,
            matrix::{heap::MatrixHeap, Matrix},
            primitive::Primitive,
            vector::Vector,
        },
    };
    use nonphysical_std::shared::primitive::F32;
    /* 
    #[test]
    fn svd_4x3_t() {
        let data = vec![0.0, 3.0, 6.0, 9.0, 
        1.0, 4.0, 7.0, 10.0, 
        2.0, 5.0, 8.0, 11.0];
        let mut m = MatrixHeap::new((3, data.iter().map(|d| F32(*d)).collect())).transposed();
        let s_solo = RealSingularValueDecomposition::jacobi_svd(&mut m);
        let (u,s,v) = RealSingularValueDecomposition::jacobi_svd_full(&mut m);

        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            -1.3298854, -1.0067357, -1.1683105,0.30068427, 0.16675025, 0.08337511,
            0.48109484, -0.3106717, 7.6731034e-8,0.6615054, -0.7737445, 0.048664965
        ];
        let known_s = vec![    1.1367172,
        1.1798036,
        1.9952745,];
        let known_perm = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let known_seq = vec![
            -0.1367172, -0.8254141, 0.4314262, -0.33744848,
            -0.34179297, -0.42799264, -0.3237158, 0.7714973,
            -0.5468688, -0.030570894, -0.646847, -0.53064954,
            -0.75194454, 0.36685067, 0.53913665, 0.09660053,
        ];
        m.data().zip(known_m.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.tau.iter().zip(known_tau.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.col_permutations()
            .data()
            .zip(known_perm.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
        qr.householder_sequence(&m)
            .data()
            .zip(known_seq.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
    }
    #[test]
    fn col_pivot_svd_4x3() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut m = MatrixHeap::new((3, data.iter().map(|d| F32(*d)).collect())).transposed();
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            -1.7391933, -0.29461744, -1.0169054,
            0.33172232, 0.17000894, 0.08500443,
            0.36858037, -0.41411272, -3.3527613e-8,
            0.4054384, -0.8668052, 0.0
        ];
        let known_tau = vec![    1.4181666,
        1.0401279,
        0.0,];
        let known_perm = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let known_seq = vec![
            -0.41816664, -0.7246632, -0.41743037, -0.35461512,-0.47043753, -0.28051484, 0.29225922, 0.7839545,
            -0.52270836, 0.16363356, 0.66777253, -0.50406337,-0.5749792, 0.607782, -0.54260147, 0.074724
        ];
        m.data().zip(known_m.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.tau.iter().zip(known_tau.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.col_permutations()
            .data()
            .zip(known_perm.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
        qr.householder_sequence(&m)
            .data()
            .zip(known_seq.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn col_pivot_svd_3x4() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut m = MatrixHeap::new((4, data.iter().map(|d| F32(*d)).collect()));
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            -1.32989,
            -1.00674,
            -1.16831,
            0.300684,
            0.16675,
            0.0833751,
            0.48109484,
            -0.3106717,
            7.6731034e-8,
            0.6615054,
            -0.7737445,
            0.048664965,
        ];
        let known_tau = vec![1.1367172, 1.1798036, 1.9952745];
        let known_perm = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let known_seq = vec![
            -0.1367172,
            -0.8254141,
            0.4314262,
            -0.33744848,
            -0.34179297,
            -0.42799264,
            -0.3237158,
            0.7714973,
            -0.5468688,
            -0.030570894,
            -0.646847,
            -0.53064954,
            -0.75194454,
            0.36685067,
            0.53913665,
            0.09660053,
        ];
        m.data().zip(known_m.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.tau.iter().zip(known_tau.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.col_permutations()
            .data()
            .zip(known_perm.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
        qr.householder_sequence(&m)
            .data()
            .zip(known_seq.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn col_pivot_svd_3x4_t() {
        let data = vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0];
        let mut m = MatrixHeap::new((4, data.iter().map(|d| F32(*d)).collect()));
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            -1.73919,
            -0.294617,
            -1.01691,
            0.331722,
            0.170009,
            0.0850044,
            0.36858,
            -0.414113,
            -3.3527613e-8,
            0.405438,
            -0.866805,
            0.0,
        ];
        let known_tau = vec![1.4181666, 1.0401279, 0.0];
        let known_seq = vec![
            -0.41816664,
            -0.7246632,
            -0.41743037,
            -0.35461512,
            -0.47043753,
            -0.28051484,
            0.29225922,
            0.7839545,
            -0.52270836,
            0.16363356,
            0.66777253,
            -0.50406337,
            -0.5749792,
            0.607782,
            -0.54260147,
            0.074724,
        ];
        let known_perm = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        m.data().zip(known_m.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.tau.iter().zip(known_tau.iter()).for_each(|(mp, kp)| {
            assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
        });
        qr.col_permutations()
            .data()
            .zip(known_perm.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
        qr.householder_sequence(&m)
            .data()
            .zip(known_seq.iter())
            .for_each(|(mp, kp)| {
                assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON);
            });
    }*/
}

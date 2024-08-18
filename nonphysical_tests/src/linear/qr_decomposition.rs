#[cfg(test)]
mod qr_col_pivot_pca_tests {
    use nonphysical_core::{
        linear::qr_decomposition::{QRDecomposition, RealQRDecomposition},
        shared::{
            float::Float,
            matrix::{matrix_heap::MatrixHeap, Matrix},
            primitive::Primitive,
            vector::Vector,
        },
    };
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn col_pivot_pca_4x3_t() {
        let data = vec![
            -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut m = MatrixHeap::new((3, data.iter().map(|d| F32(*d)).collect())).transposed();
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            2.0, 0.0, -2.0, 0.33333334, 0.0, 0.0, 0.33333334, 0.0, 0.0, 0.33333334, 0.0, 0.0,
        ];
        let known_tau = vec![1.5, 0.0, 0.0];
        let known_perm = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let known_seq = vec![
            -0.5, -0.5, -0.5, -0.5, -0.5, 0.833333, -0.166667, -0.166667, -0.5, -0.166667,
            0.833333, -0.166667, -0.5, -0.166667, -0.166667, 0.833333,
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
    fn col_pivot_pca_4x3() {
        let data = vec![
            -4.0, -4.0, -4.0, -4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0,
        ];
        let mut m = MatrixHeap::new((3, data.iter().map(|d| F32(*d)).collect())).transposed();
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            2.0, 0.0, -2.0, 0.33333334, 0.0, 0.0, 0.33333334, 0.0, 0.0, 0.33333334, 0.0, 0.0,
            0.33333334, 0.0, 0.0,
        ];
        let known_tau = vec![1.5, 0.0, 0.0];
        let known_perm = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let known_seq = vec![
            -0.5, -0.5, -0.5, -0.5, -0.5, 0.833333, -0.166667, -0.166667, -0.5, -0.166667,
            0.833333, -0.166667, -0.5, -0.166667, -0.166667, 0.833333,
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
    fn col_pivot_pca_3x4() {
        let data = vec![
            -1.5, -1.5, -1.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5,
        ];
        let mut m = MatrixHeap::new((4, data.iter().map(|d| F32(*d)).collect()));
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            1.49071,
            1.49071,
            1.49071,
            0.133831,
            1.26441e-07,
            1.26441e-07,
            -0.133831,
            -0.190744,
            1.46482e-14,
            -0.401492,
            -0.762974,
            0.780776,
        ];
        let known_tau = vec![1.6708204, 1.2357023, 1.2425356];
        let known_seq = vec![
            -0.67082,
            0.737865,
            0.051131,
            -0.0542326,
            -0.223607,
            -0.136953,
            -0.964982,
            -0.00725807,
            0.223607,
            0.136953,
            -0.064009,
            -0.962885,
            0.67082,
            0.646563,
            -0.249193,
            0.26431,
        ];
        let known_perm = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
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
    fn col_pivot_pca_3x4_t() {
        let data = vec![
            -4.5, -4.5, -4.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5, 4.5, 4.5, 4.5,
        ];
        let mut m = MatrixHeap::new((4, data.iter().map(|d| F32(*d)).collect()));
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            1.49071,
            1.49071,
            1.49071,
            0.133831,
            1.26441e-07,
            1.26441e-07,
            -0.133831,
            -0.190744,
            1.46482e-14,
            -0.401492,
            -0.762974,
            0.780776,
        ];
        let known_tau = vec![1.6708204, 1.2357023, 1.2425356];
        let known_seq = vec![
            -0.67082,
            0.737865,
            0.051131,
            -0.0542326,
            -0.223607,
            -0.136953,
            -0.964982,
            -0.00725807,
            0.223607,
            0.136953,
            -0.064009,
            -0.962885,
            0.67082,
            0.646563,
            -0.249193,
            0.26431,
        ];
        let known_perm = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
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
}
#[cfg(test)]
mod qr_col_pivot_svd_tests {
    use nonphysical_core::{
        linear::qr_decomposition::{QRDecomposition, RealQRDecomposition},
        shared::{
            float::Float,
            matrix::{matrix_heap::MatrixHeap, Matrix},
            primitive::Primitive,
            vector::Vector,
        },
    };
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn col_pivot_svd_4x3_t() {
        let data = vec![0.0, 3.0, 6.0, 9.0, 
        1.0, 4.0, 7.0, 10.0, 
        2.0, 5.0, 8.0, 11.0];
        let mut m = MatrixHeap::new((3, data.iter().map(|d| F32(*d)).collect())).transposed();
        let scale = Vector::l1_max(m.data());
        Vector::div(m.data_ref(), scale);
        let qr = RealQRDecomposition::col_pivot(&mut m);

        let known_m = vec![
            -1.3298854, -1.0067357, -1.1683105,0.30068427, 0.16675025, 0.08337511,
            0.48109484, -0.3106717, 7.6731034e-8,0.6615054, -0.7737445, 0.048664965
        ];
        let known_tau = vec![    1.1367172,
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
    }
}

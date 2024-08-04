
#[cfg(test)]
pub mod svd_tests {
    use std::time::SystemTime;

    use nonphysical_core::{linear::svd::{RealSingularValueDecomposition, SingularValueDecomposition}, shared::{float::Float, matrix::Matrix, primitive::Primitive}};
    use nonphysical_std::shared::primitive::F32;


    #[test]
    fn jacobi_svd_square_r() {
        let mut in_mat = Matrix::new(3, (0..9).map(|i| F32(i as f32)).collect());
        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([14.2267, 1.26523, 7.16572e-8])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });

        let mut in_mat = Matrix::new(5, (0..25).map(|i| F32(i as f32)).collect());

        let s = RealSingularValueDecomposition::jacobi_svd(&mut in_mat);

        s.into_iter()
            .zip([69.9086, 3.5761, 1.4977e-6, 1.0282e-6, 2.22847e-7])
            .for_each(|(si, ki)| {
                assert!((si - F32(ki)).l2_norm() < F32::EPSILON);
            });
    }

    #[test]
    fn jacobi_svd_full_square_r() {
        let mut in_mat = Matrix::new(3, (0..9).map(|i| F32(i as f32)).collect());
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
        let mut in_mat: Matrix<F32> = Matrix::zero(3, 3);
        in_mat.data_row_ref(0).for_each(|rp| *rp = F32(-3.0));
        in_mat.data_row_ref(2).for_each(|rp| *rp = F32(3.0));

        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter().zip([7.348469, 0.0, 0.0]).for_each(|(si, ki)| {
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
        let mut in_mat: Matrix<F32> = Matrix::zero(3, 3);
        in_mat.data_row_ref(0).for_each(|rp| *rp = F32(-3.0));
        in_mat.data_row_ref(2).for_each(|rp| *rp = F32(3.0));
        in_mat = in_mat.transposed();

        let (u, s, v) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        s.into_iter().zip([7.348469, 0.0, 0.0]).for_each(|(si, ki)| {
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
        let mut in_mat = Matrix::new(512, (0..512 * 512).map(|i| F32(i as f32)).collect());
        let now = SystemTime::now();
        let (_, _, _) = RealSingularValueDecomposition::jacobi_svd_full(&mut in_mat);
        let _ = println!("{:?}", now.elapsed());
    }
}

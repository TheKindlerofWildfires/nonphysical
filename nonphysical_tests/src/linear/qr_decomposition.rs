#[cfg(test)]
mod qr_row_decomposition_tests {
    use nonphysical_core::{
        linear::qr_decomposition::{QRDecomposition, RealQRDecomposition},
        shared::{
            float::Float,
            matrix::{heap::MatrixHeap, Matrix},
            primitive::Primitive,
        },
    };
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn row_pivot_4x3() {
        let mut m =
            MatrixHeap::new((4, (0..12).map(|i| F32(i as f32 / 11.0)).collect())).transposed();
        let qr = RealQRDecomposition::row_pivot(&mut m);

        let known_tau = vec![1.1367172, 1.1798036, 1.9615307];
        qr.tau
            .iter()
            .zip(known_tau.iter())
            .for_each(|(mp, kp)| assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON));
        let known_perm = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        qr.row_permutations()
            .data()
            .zip(known_perm.iter())
            .for_each(|(mp, kp)| assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON));

        let known_sequence = vec![
            -0.1367172, -0.34179297, -0.5468688, -0.75194454,-0.8254141, -0.42799264, -0.03057095, 0.3668507,0.48512945, -0.45732266, -0.54074234, 0.5129358,-0.25426286, 0.70061105, -0.63843375, 0.19208547
        ];
        qr.householder_sequence(&m)
            .data()
            .zip(known_sequence.iter())
            .for_each(|(mp, kp)| assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON));
        let known = vec![
            -1.3298854, 0.30068427, 0.48109484, 0.6615054,
            -1.0067357, 0.16675025, -0.31067163, -0.7737445,
            -1.1683105, 0.08337509, 5.3618187e-8, 0.14004244,
        ];
        m.data()
            .zip(known.iter())
            .for_each(|(mp, kp)| assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON));
    }

    #[test]
    fn row_pivot_5x4() {
        let mut m =
            MatrixHeap::new((5, (0..20).map(|i| F32(i as f32 / 19.0)).collect())).transposed();
        RealQRDecomposition::row_pivot(&mut m);

        let known = vec![
            -1.4557176,
            0.22832067,
            0.35878962,
            0.48925862,
            0.61972755,
            -1.1417392,
            0.1614663,
            -0.05390935,
            -0.34482676,
            -0.63574475,
            -1.3510581,
            0.053822137,
            -5.5213885e-8,
            -0.16811231,
            0.27135494,
            -1.2463986,
            0.10764424,
            -3.5769535e-8,
            -6.8260874e-8,
            0.08068039,
        ];
        m.data()
            .zip(known.iter())
            .for_each(|(mp, kp)| assert!((*mp - F32(*kp)).l2_norm() < F32::EPSILON));
    }
}

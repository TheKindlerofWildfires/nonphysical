
#[cfg(test)]
mod gemm_test {
    use std::time::SystemTime;

    use nonphysical_core::{linear::gemm::Gemm, random::pcg::PermutedCongruentialGenerator, shared::{complex::{Complex, ComplexScaler}, float::Float, matrix::Matrix, primitive::Primitive}};
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn naive_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| ComplexScaler::new(F32(c as f32), F32(0.0))).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::new(F32(c as f32), F32(0.0)))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::new(F32(c as f32), F32(0.0)))
                .collect(),
        );

        let r1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m33, &m33);
        let r2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m33, &m34);
        let r3 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m43, &m33);
        let r4 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m34, &m43);
        let r5 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m43, &m34);

        let k1 = Matrix::new(
            3,
            [15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0]
                .into_iter()
                .map(|r| ComplexScaler::real(F32(r)))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });

        let k2 = Matrix::new(
            3,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0,
            ]
            .into_iter()
            .map(|r| ComplexScaler::real(F32(r)))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
        let k3 = Matrix::new(
            4,
            [
                15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0, 96.0, 126.0, 156.0,
            ]
            .into_iter()
            .map(|r| ComplexScaler::real(F32(r)))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
        let k4 = Matrix::new(
            3,
            [42.0, 48.0, 54.0, 114.0, 136.0, 158.0, 186.0, 224.0, 262.0]
                .into_iter()
                .map(|r| ComplexScaler::real(F32(r)))
                .collect(),
        );
        r4.data().zip(k4.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
        let k5 = Matrix::new(
            4,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0,
                158.0, 188.0, 218.0,
            ]
            .into_iter()
            .map(|r| ComplexScaler::real(F32(r)))
            .collect(),
        );
        r5.data().zip(k5.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn gemm_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| ComplexScaler::new(F32(c as f32), F32(0.0))).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::new(F32(c as f32), F32(0.0)))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::new(F32(c as f32), F32(0.0)))
                .collect(),
        );
        let r1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m33, &m33);
        let r2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m33, &m34);
        let r3 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m43, &m33);
        let r4 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m34, &m43);
        let r5 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m43, &m34);

        let k1 = Matrix::new(
            3,
            [15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0]
                .into_iter()
                .map(|r| ComplexScaler::real(F32(r)))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });

        let k2 = Matrix::new(
            3,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0,
            ]
            .into_iter()
            .map(|r| ComplexScaler::real(F32(r)))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
        let k3 = Matrix::new(
            4,
            [
                15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0, 96.0, 126.0, 156.0,
            ]
            .into_iter()
            .map(|r| ComplexScaler::real(F32(r)))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
        let k4 = Matrix::new(
            3,
            [42.0, 48.0, 54.0, 114.0, 136.0, 158.0, 186.0, 224.0, 262.0]
                .into_iter()
                .map(|r| ComplexScaler::real(F32(r)))
                .collect(),
        );
        r4.data().zip(k4.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
        let k5 = Matrix::new(
            4,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0,
                158.0, 188.0, 218.0,
            ]
            .into_iter()
            .map(|r| ComplexScaler::real(F32(r)))
            .collect(),
        );
        r5.data().zip(k5.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn gemm_lesser() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);

        let n1 = (pcg.next_u32() as usize % (Matrix::<F32>::KC - 1) + 4) as usize;
        let n2 = (pcg.next_u32() as usize % (Matrix::<F32>::KC  - 1) + 4) as usize;
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m1, &m2);
        let g2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m2, &m1);

        let k1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m1, &m2);
        let k2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).l2_norm() < F32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).l2_norm() < F32::EPSILON);
        });
    }
    #[test]
    fn gemm_middle() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);

        let n1 = (pcg.next_u32() as usize % Matrix::<F32>::KC  + Matrix::<F32>::KC ) as usize;
        let n2 = (pcg.next_u32() as usize % Matrix::<F32>::KC  + Matrix::<F32>::KC ) as usize;
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m1, &m2);
        let g2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m2, &m1);

        let k1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m1, &m2);
        let k2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).l2_norm() < F32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn gemm_greater() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let n1 = (pcg.next_u32() as usize % Matrix::<F32>::MC  + Matrix::<F32>::MC ) as usize;
        let n2 = (pcg.next_u32() as usize % Matrix::<F32>::MC  + Matrix::<F32>::MC ) as usize;
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m1, &m2);
        let g2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m2, &m1);

        let k1 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m1, &m2);
        let k2 = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).l2_norm() < F32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).l2_norm() < F32::EPSILON);
        });
    }

    #[test]
    fn gemm_speed() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let n1 = 1000;
        let n2 = 1000;
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| ComplexScaler::new(F32(pcg.next_u32() as f32 / u32::MAX as f32), F32(0.0)));
        let m2 = Matrix::new(n2, data.collect());

        let now = SystemTime::now();
        let _ = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m1, &m2);
        let _ = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::gemm(&m2, &m1);
        let _ = println!("{:?}",now.elapsed());
        let now = SystemTime::now();
        let _ = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m1, &m2);
        let _ = <Matrix<ComplexScaler<F32>> as Gemm<ComplexScaler<F32>>>::naive(&m2, &m1);
        let _ = println!("{:?}",now.elapsed());
    }
}

#[cfg(test)]
mod bidiagonal_tests {
    use nonphysical_core::{
        linear::bidiagonal::{Bidiagonal, RealBidiagonal},
        shared::{float::Float, matrix::{heap::MatrixHeap, Matrix}, primitive::Primitive},
    };
    use nonphysical_std::shared::primitive::F32;

    

    #[test]
    fn unblocked_r_3x3() {
        let mut m = MatrixHeap::new((3, (0..9).map(|i| F32(i as f32 / 8.0)).collect()));
        let mut bidiagonal = MatrixHeap::zero(3, 3);
        RealBidiagonal::unblocked(&mut m, &mut bidiagonal, 0, 3, 3);
        let known = vec![
            1.0,
            0.447214,
            0.894427,
            1.51995,
            1.0431,
            0.95779,
            0.561992,
            0.0,
            0.0,
        ]
        .iter()
        .map(|i| F32(*i as f32))
        .collect::<Vec<_>>();
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - k).l2_norm() < F32::EPSILON);
        })
    }
    #[test]
    fn bidiagonal_r_3x3() {
        let mut m = MatrixHeap::new((3, (0..9).map(|i| F32(i as f32 / 8.0)).collect()));
        RealBidiagonal::new(&mut m);
        let known = vec![
            1.0,
            0.447214,
            0.894427,
            1.51995,
            1.0431,
            0.95779,
            0.561992,
            0.0,
            0.0,
        ]
        .iter()
        .map(|i| F32(*i as f32))
        .collect::<Vec<_>>();
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - k).l2_norm() < F32::EPSILON);
        })
    }

    /*
    #[test]
    fn right_local_3x3_real_2() {
        let data = vec![1.0, 0.447214, 0.894427, 1.51995, 1.0431 ,0.95779, 0.561992, 0.0,-2.23517e-08]
            .iter()
            .map(|i| F32(*i as f32))
            .collect();
        let mut m = MatrixHeap::new((3, data));
        let house: RealHouseholder<F32> = RealHouseholder {
            tau: F32(1.51995),
            beta: F32(1.5052),
        };
        let essential = m.data_col(0).skip(2).cloned().collect::<Vec<_>>();

        house.apply_right(&mut m, &essential, [2,3],[2,3]);

        let known = vec![1.0, 0.447214, 0.894427, 1.51995, 1.0431 ,0.95779, 0.561992, 0.0,-2.23517e-08]
            .iter()
            .map(|i| F32(*i as f32))
            .collect::<Vec<_>>();
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - k).l2_norm() < F32::EPSILON);
        })
    }
    */
    /*
    #[test]
    fn unblocked_r_4x4(){
        let data = (0..4*4).map(|i| i as f32/15.0).collect();
        let mut m = Matrix::new(4,data);
        let b = RealBidiagonal::new(&mut m);
    }
    */
}

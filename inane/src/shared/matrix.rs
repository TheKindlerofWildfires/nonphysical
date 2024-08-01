
#[cfg(test)]
mod matrix_tests {
    use baseless::shared::{complex::{Complex, ComplexScaler}, float::Float, matrix::Matrix};


    #[test]
    fn coeff_static() {
        //square case
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2).real - 8.0).l2_norm() < f32::EPSILON);

        //long case
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 3).real - 11.0).l2_norm() < f32::EPSILON);

        //wide case
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 2).real - 11.0).l2_norm() < f32::EPSILON);
    }

    #[test]
    fn coeff_ref_static() {
        //square case
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 8.0).l2_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 2).real - 4.0).l2_norm() < f32::EPSILON);

        //long case
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 11.0).l2_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(2, 0).real = 5.0;
        m.coeff_ref(2, 3).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 0).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(2, 3).real - 4.0).l2_norm() < f32::EPSILON);

        //wide case
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        assert!((m.coeff_ref(0, 0).real - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 11.0).l2_norm() < f32::EPSILON);

        m.coeff_ref(0, 0).real = 9.0;
        m.coeff_ref(0, 1).real = 8.0;
        m.coeff_ref(1, 0).real = 7.0;
        m.coeff_ref(1, 1).real = 6.0;
        m.coeff_ref(3, 0).real = 5.0;
        m.coeff_ref(3, 2).real = 4.0;

        assert!((m.coeff_ref(0, 0).real - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(0, 1).real - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 0).real - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(1, 1).real - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 0).real - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff_ref(3, 2).real - 4.0).l2_norm() < f32::EPSILON);
    }

    #[test]
    fn data_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let d = m.data_ref();
        d.zip(0..9).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_ref();
        d.zip(0..12).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_diag_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(5)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let d = m.data_diag_ref();
        d.zip((0..12).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - i as f32).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let d = m.data_diag_ref();
        d.zip((0..9).step_by(4)).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_row_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let r = m.data_row_ref(0);
        r.zip(vec![0.0, 1.0, 2.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let r = m.data_row_ref(1);
        r.zip(vec![4.0, 5.0, 6.0, 7.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let r = m.data_row_ref(2);
        r.zip(vec![6.0, 7.0, 8.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_col_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_col_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 2.0;
        });
        let v = m.data_col_ref(0);
        v.zip(vec![0.0, 3.0, 6.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 2.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 3.0;
        });
        let v = m.data_col_ref(1);
        v.zip(vec![1.0, 5.0, 9.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 3.0).l2_norm() < f32::EPSILON);
        });

        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let v = m.data_col_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32)).l2_norm() < f32::EPSILON);
            c.real *= 4.0;
        });

        let v = m.data_col_ref(2);
        v.zip(vec![2.0, 5.0, 8.0, 11.0]).for_each(|(c, i)| {
            assert!((c.real - (i as f32) * 4.0).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            });
        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).l2_norm() < f32::EPSILON);
        });
        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows();
        rs.zip([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn data_rows_ref_static() {
        let mut m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
                assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
                assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
                r[0].real += 1.0;
                r[1].real += 2.0;
                r[2].real += 3.0;
            });
        let rs = m.data_rows_ref();
        rs.zip([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            .for_each(|(r, k)| {
                assert!((r[0].real - k[0] - 1.0).l2_norm() < f32::EPSILON);
                assert!((r[1].real - k[1] - 2.0).l2_norm() < f32::EPSILON);
                assert!((r[2].real - k[2] - 3.0).l2_norm() < f32::EPSILON);
            });
        let mut m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            assert!((r[3].real - k[3]).l2_norm() < f32::EPSILON);
            r[0].real += 1.0;
            r[1].real += 2.0;
            r[2].real += 3.0;
            r[3].real += 4.0;
        });
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0] - 1.0).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).l2_norm() < f32::EPSILON);
            assert!((r[3].real - k[3] - 4.0).l2_norm() < f32::EPSILON);
        });
        let mut m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0]).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1]).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2]).l2_norm() < f32::EPSILON);
            r[0].real += 1.0;
            r[1].real += 2.0;
            r[2].real += 3.0;
        });
        let rs = m.data_rows_ref();
        rs.zip([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ])
        .for_each(|(r, k)| {
            assert!((r[0].real - k[0] - 1.0).l2_norm() < f32::EPSILON);
            assert!((r[1].real - k[1] - 2.0).l2_norm() < f32::EPSILON);
            assert!((r[2].real - k[2] - 3.0).l2_norm() < f32::EPSILON);
        });
    }
    #[test]
    fn transposed_static() {
        let m = Matrix::new(
            3,
            (0..9)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]
                .into_iter()
                .map(|r| ComplexScaler::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            3,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
                .into_iter()
                .map(|r| ComplexScaler::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).l2_norm() < f32::EPSILON);
        });

        let m = Matrix::new(
            4,
            (0..12)
                .map(|c| ComplexScaler::<f32>::new(c as f32, c as f32))
                .collect(),
        );
        let mp = Matrix::new(
            3,
            [0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]
                .into_iter()
                .map(|r| ComplexScaler::<f32>::new(r, r))
                .collect(),
        );
        m.transposed().data().zip(mp.data()).for_each(|(c1, c2)| {
            assert!((c1.real - c2.real).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn row_swap_static() {
        let mut m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        m.row_swap(0, 1);
        assert!((m.coeff(0, 0) - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1) - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 2) - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0) - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1) - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 2) - 2.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0) - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 1) - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2) - 8.0).l2_norm() < f32::EPSILON);

        let mut m = Matrix::<f32>::new(3, (0..12).map(|i| i as f32).collect());

        m.row_swap(2, 1);
        assert!((m.coeff(0, 0) - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1) - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 2) - 2.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 3) - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0) - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1) - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 2) - 10.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 3) - 11.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0) - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 1) - 5.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2) - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 3) - 7.0).l2_norm() < f32::EPSILON);

        let mut m = Matrix::<f32>::new(4, (0..12).map(|i| i as f32).collect());

        m.row_swap(3, 1);
        assert!((m.coeff(0, 0) - 0.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 1) - 1.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(0, 2) - 2.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 0) - 9.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 1) - 10.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(1, 2) - 11.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 0) - 6.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 1) - 7.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(2, 2) - 8.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 0) - 3.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 1) - 4.0).l2_norm() < f32::EPSILON);
        assert!((m.coeff(3, 2) - 5.0).l2_norm() < f32::EPSILON);
    }

    #[test]
    fn test_north() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_north(1);
        let kn = (0..3).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north(2);
        let kn = (0..6).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_south() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_south(1);
        let kn = (6..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south(2);
        let kn = (3..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_west() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_west(1);
        let kn = [0, 3, 6].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_west(2);
        let kn = [0, 1, 3, 4, 6, 7].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_west(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }
    #[test]
    fn test_east() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_east(1);
        let kn = [2, 5, 8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_east(2);
        let kn = [1, 2, 4, 5, 7, 8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_east(3);
        let kn = (0..9).map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_north_east() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_north_east(1,1);
        let kn = [2].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_east(2,1);
        let kn = [2,5].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_east(1,2);
        let kn = [1,2].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_east(2,2);
        let kn = [1,2,4,5].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_north_west() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_north_west(1,1);
        let kn = [0].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_west(2,1);
        let kn = [0,3].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_west(1,2);
        let kn = [0,1].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_north_west(2,2);
        let kn = [0,1,3,4].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }

    #[test]
    fn test_south_east() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_south_east(1,1);
        let kn = [8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_east(2,1);
        let kn = [5,8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_east(1,2);
        let kn = [7,8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_east(2,2);
        let kn = [4,5,7,8].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }
    #[test]
    fn test_south_west() {
        let m = Matrix::<f32>::new(3, (0..9).map(|i| i as f32).collect());

        let n = m.data_south_west(1,1);
        let kn = [6].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_west(2,1);
        let kn = [3,6].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_west(1,2);
        let kn = [6,7].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });

        let n = m.data_south_west(2,2);
        let kn = [3,4,6,7].map(|i| i as f32);
        n.zip(kn).for_each(|(n, k)| {
            assert!(*n == k);
        });
    }
}

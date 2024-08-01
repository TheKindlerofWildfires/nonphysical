
#[cfg(test)]
mod householder_tests {
    use baseless::{linear::householder::{Householder, RealHouseholder}, shared::{float::Float, matrix::Matrix}};

    #[test]
    fn make_local_3x3_real_1() {
        let mut m = Matrix::new(3, (0..9).map(|i| i as f32).collect());

        //This is the result of a fight with the borrow checker
        let prep = RealHouseholder::make_householder_prep(&mut m.data_row(0).skip(1));
        let house: RealHouseholder<f32> =
            RealHouseholder::make_householder_local(&mut m.data_row_ref(0).skip(1),prep);
        dbg!(house.tau,house.beta);
        assert!((house.beta + 2.23607).l2_norm() < f32::EPSILON);
        assert!((house.tau - 1.44721).l2_norm() < f32::EPSILON);

        let known = vec![0.0, 1.0, 0.61803395, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((c - k).l2_norm() < f32::EPSILON);
        })
    }
    /*
    #[test]
    fn left_local_3x3_real_1() {
        let data = vec![0.0, -2.23607, 0.61803395, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut m = Matrix::new(3, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.44721,
            beta: -2.23607,
        };
        house.apply_left_local(&mut m, 0, 1..3, 1..3);

        let known = vec![
            0.0, -2.23607, 0.61803395, 3.0, -6.26099, -1.34164, 6.0, -10.2859, -2.68328,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_3x3_real_1() {
        let data = vec![
            0.0, -2.23607, 0.61803395, 3.0, -6.26099, -1.34164, 6.0, -10.2859, -2.68328,
        ];
        let mut m = Matrix::new(3, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.44721,
            beta: -2.23607,
        };
        house.apply_right_local(&mut m, 0, 1..3, 0..3);

        let known = vec![
            0.0,
            -2.23607,
            0.61803395,
            -6.7082,
            12.0,
            3.0,
            -4.76837e-07,
            1.0,
            -4.76837e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_3x3_real_2() {
        let data = vec![
            0.0,
            -2.23607,
            0.61803395,
            -6.7082,
            12.0,
            3.0,
            -4.76837e-07,
            1.0,
            -4.76837e-07,
        ];
        let mut m = Matrix::new(3, data);
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 2, 1);
        assert!((house.beta - 3.0).l2_norm() < f32::EPSILON);
        assert!((house.tau - 0.0).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0,
            -2.23607,
            0.61803395,
            -6.7082,
            12.0,
            3.0,
            -4.76837e-07,
            1.0,
            -4.76837e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_3x3_real_2() {
        let data = vec![
            0.0,
            -2.23607,
            0.61803395,
            -6.7082,
            12.0,
            3.0,
            -4.76837e-07,
            1.0,
            -4.76837e-07,
        ];
        let mut m = Matrix::new(3, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 0.0,
            beta: 3.0,
        };
        house.apply_left_local(&mut m, 0, 2..3, 2..3);

        let known = vec![
            0.0,
            -2.23607,
            0.61803395,
            -6.7082,
            12.0,
            3.0,
            -4.76837e-07,
            1.0,
            -4.76837e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_3x3_real_2() {
        let data = vec![
            0.0,
            -2.23607,
            0.61803395,
            -6.7082,
            12.0,
            3.0,
            -4.76837e-07,
            1.0,
            -4.76837e-07,
        ];
        let mut m = Matrix::new(3, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 0.0,
            beta: 3.0,
        };
        house.apply_right_local(&mut m, 0, 2..3, 0..3);

        let known = vec![
            0.0,
            -2.23607,
            0.61803395,
            -6.7082,
            12.0,
            3.0,
            -4.76837e-07,
            1.0,
            -4.76837e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_4x4_real_1() {
        let mut m = Matrix::new(4, (0..16).map(|i| i as f32).collect());
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 1, 0);
        assert!((house.beta + 3.74166).l2_norm() < f32::EPSILON);
        assert!((house.tau - 1.26726).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0, 1.0, 0.421793, 0.63269, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_4x4_real_1() {
        let data = vec![
            0.0, -3.74166, 0.421793, 0.63269, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            14.0, 15.0,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.26726,
            beta: -3.74166,
        };
        house.apply_left_local(&mut m, 0, 1..4, 1..4);

        let known = vec![
            0.0, -3.74166, 0.421793, 0.63269, 4.0, -10.1559, -0.392671, -2.58901, 8.0, -16.5702,
            -0.785341, -5.17801, 12.0, -22.9845, -1.17801, -7.76702,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_4x4_real_1() {
        let data = vec![
            0.0, -3.74166, 0.421793, 0.63269, 4.0, -10.1559, -0.392671, -2.58901, 8.0, -16.5702,
            -0.785341, -5.17801, 12.0, -22.9845, -1.17801, -7.76702,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.26726,
            beta: -3.74166,
        };
        house.apply_right_local(&mut m, 0, 1..4, 0..4);

        let known = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            1.46924,
            9.68717,
            0.0,
            0.36731,
            2.98023e-07,
            0.0,
            0.0,
            2.42179,
            -4.76837e-07,
            -9.53674e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_4x4_real_2() {
        let data = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            1.46924,
            9.68717,
            0.0,
            0.36731,
            2.98023e-07,
            0.0,
            0.0,
            2.42179,
            -4.76837e-07,
            -9.53674e-07,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 2, 1);
        assert!((house.beta + 9.79796).l2_norm() < f32::EPSILON);
        assert!((house.tau - 1.14995).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            1.46924,
            0.859768,
            0.0,
            0.36731,
            2.98023e-07,
            0.0,
            0.0,
            2.42179,
            -4.76837e-07,
            -9.53674e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_4x4_real_2() {
        let data = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            0.36731,
            2.98023e-07,
            0.0,
            0.0,
            2.42179,
            -4.76837e-07,
            -9.53674e-07,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.14995,
            beta: -9.79796,
        };
        house.apply_left_local(&mut m, 1, 2..4, 2..4);

        let known = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            0.36731,
            -4.46897e-08,
            -2.94653e-07,
            0.0,
            2.42179,
            1.01439e-06,
            3.28438e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_4x4_real_2() {
        let data = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            0.36731,
            -4.46897e-08,
            -2.94653e-07,
            0.0,
            2.42179,
            1.01439e-06,
            3.28438e-07,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.14995,
            beta: -9.79796,
        };
        house.apply_right_local(&mut m, 1, 2..4, 0..4);
        let known = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            -2.44949,
            -9.96223e-07,
            -2.8054e-07,
            0.0,
            4.76837e-07,
            1.96297e-07,
            3.40572e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_4x4_real_3() {
        let data = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            -2.44949,
            -9.96223e-07,
            -2.8054e-07,
            0.0,
            4.76837e-07,
            1.96297e-07,
            3.40572e-07,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 3, 2);
        assert!((house.beta + 2.8054e-07).l2_norm() < f32::EPSILON);
        assert!((house.tau - 0.0).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            -2.44949,
            -9.96223e-07,
            -2.8054e-07,
            0.0,
            4.76837e-07,
            1.96297e-07,
            3.40572e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_4x4_real_3() {
        let data = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            -2.44949,
            -9.96223e-07,
            -2.8054e-07,
            0.0,
            4.76837e-07,
            1.96297e-07,
            3.40572e-07,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 0.0,
            beta: 2.8054e-07,
        };
        house.apply_left_local(&mut m, 2, 3..4, 3..4);

        let known = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            -2.44949,
            -9.96223e-07,
            -2.8054e-07,
            0.0,
            4.76837e-07,
            1.96297e-07,
            3.40572e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_4x4_real_3() {
        let data = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            -2.44949,
            -9.96223e-07,
            -2.8054e-07,
            0.0,
            4.76837e-07,
            1.96297e-07,
            3.40572e-07,
        ];
        let mut m = Matrix::new(4, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 0.0,
            beta: 2.8054e-07,
        };
        house.apply_right_local(&mut m, 2, 3..4, 0..4);
        let known = vec![
            0.0,
            -3.74166,
            0.421793,
            0.63269,
            -14.9666,
            30.0,
            -9.79796,
            0.859768,
            0.0,
            -2.44949,
            -9.96223e-07,
            -2.8054e-07,
            0.0,
            4.76837e-07,
            1.96297e-07,
            3.40572e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_1() {
        let mut m = Matrix::new(5, (0..25).map(|i| i as f32).collect());
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 1, 0);
        assert!((house.beta + 5.47723).l2_norm() < f32::EPSILON);
        assert!((house.tau - 1.18257).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0, 1.0, 0.308774, 0.463161, 0.617548, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_1() {
        let data = vec![
            0.0, -5.47723, 0.308774, 0.463161, 0.617548, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.18257,
            beta: -5.47723,
        };
        house.apply_left_local(&mut m, 0, 1..5, 1..5);

        let known = vec![
            0.0, -5.47723, 0.308774, 0.463161, 0.617548, 5.0, -14.6059, 0.63742, -1.54387,
            -3.72516, 10.0, -23.7346, 1.27484, -3.08774, -7.45032, 15.0, -32.8633, 1.91226,
            -4.63161, -11.1755, 20.0, -41.9921, 2.54968, -6.17548, -14.9006,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_1() {
        let data = vec![
            0.0, -5.47723, 0.308774, 0.463161, 0.617548, 5.0, -14.6059, 0.63742, -1.54387,
            -3.72516, 10.0, -23.7346, 1.27484, -3.08774, -7.45032, 15.0, -32.8633, 1.91226,
            -4.63161, -11.1755, 20.0, -41.9921, 2.54968, -6.17548, -14.9006,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.18257,
            beta: -5.47723,
        };
        house.apply_right_local(&mut m, 0, 1..5, 0..5);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            -3.49129,
            8.45612,
            20.4035,
            0.0,
            -0.698259,
            0.0,
            0.0,
            -4.76837e-07,
            0.0,
            1.69122,
            0.0,
            -9.53674e-07,
            -9.53674e-07,
            0.0,
            4.08071,
            0.0,
            0.0,
            -9.53674e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_2() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            -3.49129,
            8.45612,
            20.4035,
            0.0,
            -0.698259,
            0.0,
            0.0,
            -4.76837e-07,
            0.0,
            1.69122,
            0.0,
            -9.53674e-07,
            -9.53674e-07,
            0.0,
            4.08071,
            0.0,
            0.0,
            -9.53674e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 2, 1);
        assert!((house.beta - 22.3607).l2_norm() < f32::EPSILON);
        assert!((house.tau - 1.15614).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            -3.49129,
            -0.327098,
            -0.789245,
            0.0,
            -0.698259,
            0.0,
            0.0,
            -4.76837e-07,
            0.0,
            1.69122,
            0.0,
            -9.53674e-07,
            -9.53674e-07,
            0.0,
            4.08071,
            0.0,
            0.0,
            -9.53674e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_2() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            -0.698259,
            0.0,
            0.0,
            -4.76837e-07,
            0.0,
            1.69122,
            0.0,
            -9.53674e-07,
            -9.53674e-07,
            0.0,
            4.08071,
            0.0,
            0.0,
            -9.53674e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.15614,
            beta: 22.3607,
        };
        house.apply_left_local(&mut m, 1, 2..5, 2..5);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            -0.698259,
            -4.35102e-07,
            1.42321e-07,
            -1.33435e-07,
            0.0,
            1.69122,
            -1.23085e-06,
            -5.51065e-07,
            1.77714e-08,
            0.0,
            4.08071,
            -8.70203e-07,
            2.84642e-07,
            -2.6687e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_2() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            -0.698259,
            -4.35102e-07,
            1.42321e-07,
            -1.33435e-07,
            0.0,
            1.69122,
            -1.23085e-06,
            -5.51065e-07,
            1.77714e-08,
            0.0,
            4.08071,
            -8.70203e-07,
            2.84642e-07,
            -2.6687e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.15614,
            beta: 22.3607,
        };
        house.apply_right_local(&mut m, 1, 2..5, 0..5);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            2.91112e-08,
            -2.15958e-07,
            0.0,
            -1.66893e-06,
            -9.83413e-07,
            -5.14034e-07,
            4.47644e-08,
            0.0,
            -9.53674e-07,
            -2.7316e-07,
            3.73992e-07,
            -2.0174e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_3() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            2.91112e-08,
            -2.15958e-07,
            0.0,
            -1.66893e-06,
            -9.83413e-07,
            -5.14034e-07,
            4.47644e-08,
            0.0,
            -9.53674e-07,
            -2.7316e-07,
            3.73992e-07,
            -2.0174e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 3, 2);
        assert!((house.beta + 2.17911e-07).l2_norm() < f32::EPSILON);
        assert!((house.tau - 1.13359).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            2.91112e-08,
            -0.874244,
            0.0,
            -1.66893e-06,
            -9.83413e-07,
            -5.14034e-07,
            4.47644e-08,
            0.0,
            -9.53674e-07,
            -2.7316e-07,
            3.73992e-07,
            -2.0174e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_3() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -1.66893e-06,
            -9.83413e-07,
            -5.14034e-07,
            4.47644e-08,
            0.0,
            -9.53674e-07,
            -2.7316e-07,
            3.73992e-07,
            -2.0174e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.13359,
            beta: -2.17911e-07,
        };
        house.apply_left_local(&mut m, 2, 3..5, 3..5);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -1.66893e-06,
            -9.83413e-07,
            1.13034e-07,
            -5.03446e-07,
            0.0,
            -9.53674e-07,
            -2.7316e-07,
            -2.49894e-07,
            3.43689e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_3() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -1.66893e-06,
            -9.83413e-07,
            1.13034e-07,
            -5.03446e-07,
            0.0,
            -9.53674e-07,
            -2.7316e-07,
            -2.49894e-07,
            3.43689e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 1.13359,
            beta: -2.17911e-07,
        };
        house.apply_right_local(&mut m, 2, 3..5, 0..5);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -7.2217e-07,
            -1.39336e-07,
            -2.62754e-07,
            4.07865e-07,
            0.0,
            -1.78137e-06,
            -1.01109e-06,
            7.8637e-08,
            -4.53019e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn make_local_5x5_real_4() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -7.2217e-07,
            -1.39336e-07,
            -2.62754e-07,
            4.07865e-07,
            0.0,
            -1.78137e-06,
            -1.01109e-06,
            7.8637e-08,
            -4.53019e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder::make_householder_local(&mut m, 4, 3);
        assert!((house.beta - 4.07865e-07).l2_norm() < f32::EPSILON);
        assert!((house.tau - 0.0).l2_norm() < f32::EPSILON);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -7.2217e-07,
            -1.39336e-07,
            -2.62754e-07,
            4.07865e-07,
            0.0,
            -1.78137e-06,
            -1.01109e-06,
            7.8637e-08,
            -4.53019e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn left_local_5x5_real_4() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -7.2217e-07,
            -1.39336e-07,
            -2.62754e-07,
            4.07865e-07,
            0.0,
            -1.78137e-06,
            -1.01109e-06,
            7.8637e-08,
            -4.53019e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 0.0,
            beta: 4.07865e-07,
        };
        house.apply_left_local(&mut m, 3, 4..5, 4..5);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -7.2217e-07,
            -1.39336e-07,
            -2.62754e-07,
            4.07865e-07,
            0.0,
            -1.78137e-06,
            -1.01109e-06,
            7.8637e-08,
            -4.53019e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    #[test]
    fn right_local_5x5_real_4() {
        let data = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -7.2217e-07,
            -1.39336e-07,
            -2.62754e-07,
            4.07865e-07,
            0.0,
            -1.78137e-06,
            -1.01109e-06,
            7.8637e-08,
            -4.53019e-07,
        ];
        let mut m = Matrix::new(5, data);
        let house: RealHouseholder<f32> = RealHouseholder {
            tau: 0.0,
            beta: 4.07865e-07,
        };
        house.apply_right_local(&mut m, 3, 4..5, 0..5);

        let known = vec![
            0.0,
            -5.47723,
            0.308774,
            0.463161,
            0.617548,
            -27.3861,
            60.0,
            22.3607,
            -0.327098,
            -0.789245,
            0.0,
            4.47213,
            -1.19158e-06,
            -2.17911e-07,
            -0.874244,
            0.0,
            -7.2217e-07,
            -1.39336e-07,
            -2.62754e-07,
            4.07865e-07,
            0.0,
            -1.78137e-06,
            -1.01109e-06,
            7.8637e-08,
            -4.53019e-07,
        ];
        m.data().zip(known.into_iter()).for_each(|(c, k)| {
            assert!((*c - *k).l2_norm() < f32::EPSILON);
        })
    }
    */
}

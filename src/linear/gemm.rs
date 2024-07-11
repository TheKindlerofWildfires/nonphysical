<<<<<<< HEAD
use core::cmp::min;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix};

pub trait Gemm<'a, T: Float + 'a> {
    const MC: usize = 256;
    const KC: usize = 128;

    fn naive(x: &Matrix<T>, y: &Matrix<T>) -> Matrix<T> {
        debug_assert!(x.columns == y.rows);
        let mut z = Matrix::new(x.rows, vec![Complex::ZERO; x.rows * y.columns]);
        (0..y.columns).for_each(|c| {
            (0..x.rows).for_each(|r| {
                let mut tmp = Complex::<T>::ZERO;
                (0..x.columns).for_each(|a| tmp += x.coeff(r, a) * y.coeff(a, c));
                *z.coeff_ref(r, c) = tmp;
            });
        });
        z
    }
    fn gemm(x: &Matrix<T>, y: &Matrix<T>) -> Matrix<T> {
        debug_assert!(x.columns == y.rows);
        let x_rows = x.rows;
        let y_cols = y.columns;

        //pad out to size 4
        let x = Self::pad(&x);
        let y = Self::pad(&y);

        let mut z = Matrix::zero(x.rows, y.columns); //this is now oversize

        let n_chunk_size = y.columns;
        //it's probable a chunk operation would work better here
        for p_index in (0..x.columns).step_by(Self::KC) {
            let p_chunk_size = min(x.columns - p_index, Self::KC);
            for i_index in (0..x.rows).step_by(Self::MC) {
                let i_chunk_size = min(x.rows - i_index, Self::MC);
                Self::kernel(
                    i_chunk_size,
                    n_chunk_size,
                    p_chunk_size,
                    p_index,
                    i_index,
                    &x,
                    &y,
                    &mut z,
                );
            }
        }
        let z = Self::cut(x_rows, y_cols, &z);
        z
    }

    #[inline(always)]
    fn pad(x: &Matrix<T>) -> Matrix<T> {
        let x_rows = ((x.rows >> 2) << 2) + 4;
        let x_cols = ((x.columns >> 2) << 2) + 4;
        let mut output = Matrix::<T>::zero(x_rows, x_cols);
        x.data_rows()
            .zip(output.data_rows_ref())
            .for_each(|(x_row, o_row)| {
                x_row.iter().zip(o_row.iter_mut()).for_each(|(x, o)| {
                    *o = *x;
                })
            });
        output
    }

    #[inline(always)]
    fn cut(r: usize, c: usize, z: &Matrix<T>) -> Matrix<T> {
        let mut output = Matrix::<T>::zero(r, c);
        output
            .data_rows_ref()
            .zip(z.data_rows())
            .for_each(|(o_row, z_row)| {
                o_row.iter_mut().zip(z_row.iter()).for_each(|(o, z)| {
                    *o = *z;
                });
            });

        output
    }

    fn kernel(
        i_chunk_size: usize,
        n_chunk_size: usize,
        p_chunk_size: usize,
        p_index: usize,
        i_index: usize,
        x: &Matrix<T>,
        y: &Matrix<T>,
        z: &mut Matrix<T>,
    ) {
        let mut packed_x = Matrix::<T>::zero(i_chunk_size, p_chunk_size);

        for row in (0..i_chunk_size).step_by(4) {
            Self::pack_x(row, p_chunk_size, p_index, x, i_index, &mut packed_x);
        }
        for col in (0..n_chunk_size).step_by(4) {
            for row in (0..i_chunk_size).step_by(4) {
                Self::gemm4x4(row, col, p_chunk_size, p_index, i_index, &packed_x, y, z);
            }
        }
    }

    fn pack_x(
        row: usize,
        p_chunk_size: usize,
        p_index: usize,
        x: &Matrix<T>,
        i_index: usize,
        packed: &mut Matrix<T>,
    ) {
        let mut pack_index = row * p_chunk_size;
        for j in 0..p_chunk_size {
            packed.data[pack_index] = x.coeff(row + i_index, j + p_index); //buggy
            packed.data[pack_index + 1] = x.coeff(row + i_index + 1, j + p_index);
            packed.data[pack_index + 2] = x.coeff(row + i_index + 2, j + p_index);
            packed.data[pack_index + 3] = x.coeff(row + i_index + 3, j + p_index);
            pack_index += 4;
        }
    }
    fn gemm4x4(
        row: usize,
        col: usize,
        k: usize,
        p_index: usize,
        i_index: usize,
        packed: &Matrix<T>,
        y: &Matrix<T>,
        z: &mut Matrix<T>,
    ) {
        let mut local_z = [Complex::<T>::ZERO; 16];
        let mut local_x = [Complex::<T>::ZERO; 4];
        let mut local_y = [Complex::<T>::ZERO; 4];

        let mut y_index = y.index(p_index, col);
        let mut x_index = row * k;
        for _ in 0..k {
            local_x[0] = packed.data[x_index];
            local_x[1] = packed.data[x_index + 1];
            local_x[2] = packed.data[x_index + 2];
            local_x[3] = packed.data[x_index + 3];

            local_y[0] = y.data[y_index];
            local_y[1] = y.data[y_index + 1];
            local_y[2] = y.data[y_index + 2];
            local_y[3] = y.data[y_index + 3];

            local_z[0] += local_x[0] * local_y[0];
            local_z[4] += local_x[1] * local_y[0];
            local_z[1] += local_x[0] * local_y[1];
            local_z[5] += local_x[1] * local_y[1];

            local_z[2] += local_x[0] * local_y[2];
            local_z[6] += local_x[1] * local_y[2];
            local_z[3] += local_x[0] * local_y[3];
            local_z[7] += local_x[1] * local_y[3];

            local_z[8] += local_x[2] * local_y[0];
            local_z[12] += local_x[3] * local_y[0];
            local_z[9] += local_x[2] * local_y[1];
            local_z[13] += local_x[3] * local_y[1];

            local_z[10] += local_x[2] * local_y[2];
            local_z[14] += local_x[3] * local_y[2];
            local_z[11] += local_x[2] * local_y[3];
            local_z[15] += local_x[3] * local_y[3];

            y_index += y.columns;
            x_index += 4;
        }
        let mut z_index = z.index(row + i_index, col);

        (0..16).step_by(4).for_each(|i| {
            z.data[z_index] += local_z[i];
            z.data[z_index + 1] += local_z[i + 1];
            z.data[z_index + 2] += local_z[i + 2];
            z.data[z_index + 3] += local_z[i + 3];
            z_index += z.columns;
        });
    }
}

impl<'a, T: Float + 'a> Gemm<'a, T> for Matrix<T> {}

#[cfg(test)]
mod gemm_test {
    use crate::{random::pcg::PermutedCongruentialGenerator, shared::matrix};

    use super::*;

    #[test]
    fn naive_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| Complex::<f32>::new(c as f32, 0.0)).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );

        let r1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m33, &m33);
        let r2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m33, &m34);
        let r3 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m43, &m33);
        let r4 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m34, &m43);
        let r5 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m43, &m34);

        let k1 = Matrix::<f32>::new(
            3,
            [15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });

        let k2 = Matrix::<f32>::new(
            3,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k3 = Matrix::<f32>::new(
            4,
            [
                15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0, 96.0, 126.0, 156.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k4 = Matrix::<f32>::new(
            3,
            [42.0, 48.0, 54.0, 114.0, 136.0, 158.0, 186.0, 224.0, 262.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r4.data().zip(k4.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k5 = Matrix::<f32>::new(
            4,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0,
                158.0, 188.0, 218.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r5.data().zip(k5.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn gemm_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| Complex::<f32>::new(c as f32, 0.0)).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let r1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m33, &m33);
        let r2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m33, &m34);
        let r3 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m43, &m33);
        let r4 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m34, &m43);
        let r5 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m43, &m34);

        let k1 = Matrix::<f32>::new(
            3,
            [15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });

        let k2 = Matrix::<f32>::new(
            3,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k3 = Matrix::<f32>::new(
            4,
            [
                15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0, 96.0, 126.0, 156.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k4 = Matrix::<f32>::new(
            3,
            [42.0, 48.0, 54.0, 114.0, 136.0, 158.0, 186.0, 224.0, 262.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r4.data().zip(k4.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k5 = Matrix::<f32>::new(
            4,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0,
                158.0, 188.0, 218.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r5.data().zip(k5.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn gemm_lesser() {
        let mut pcg = PermutedCongruentialGenerator::<f32>::new(3, 0);

        let n1 = (pcg.next_u32() as usize % (Matrix::<f32>::KC - 1) + 4) as usize;
        let n2 = (pcg.next_u32() as usize % (Matrix::<f32>::KC - 1) + 4) as usize;
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m1, &m2);
        let g2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m2, &m1);

        let k1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m1, &m2);
        let k2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });
    }
    #[test]
    fn gemm_middle() {
        let mut pcg = PermutedCongruentialGenerator::<f32>::new(3, 0);

        let n1 = (pcg.next_u32() as usize % Matrix::<f32>::KC + Matrix::<f32>::KC) as usize;
        let n2 = (pcg.next_u32() as usize % Matrix::<f32>::KC + Matrix::<f32>::KC) as usize;
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m1, &m2);
        let g2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m2, &m1);

        let k1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m1, &m2);
        let k2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn gemm_greater() {
        let mut pcg = PermutedCongruentialGenerator::<f32>::new(3, 0);
        let n1 = (pcg.next_u32() as usize % Matrix::<f32>::MC + Matrix::<f32>::MC) as usize;
        let n2 = (pcg.next_u32() as usize % Matrix::<f32>::MC + Matrix::<f32>::MC) as usize;
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m1, &m2);
        let g2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m2, &m1);

        let k1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m1, &m2);
        let k2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });
    }
}
=======
use core::cmp::min;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix};

pub trait Gemm<'a, T: Float + 'a> {
    const MC: usize = 256;
    const KC: usize = 128;

    fn naive(x: &Matrix<T>, y: &Matrix<T>) -> Matrix<T> {
        debug_assert!(x.columns == y.rows);
        let mut z = Matrix::new(x.rows, vec![Complex::ZERO; x.rows * y.columns]);
        (0..y.columns).for_each(|c| {
            (0..x.rows).for_each(|r| {
                let mut tmp = Complex::<T>::ZERO;
                (0..x.columns).for_each(|a| tmp += x.coeff(r, a) * y.coeff(a, c));
                *z.coeff_ref(r, c) = tmp;
            });
        });
        z
    }
    fn gemm(x: &Matrix<T>, y: &Matrix<T>) -> Matrix<T> {
        debug_assert!(x.columns == y.rows);
        let x_rows = x.rows;
        let y_cols = y.columns;

        //pad out to size 4
        let x = Self::pad(&x);
        let y = Self::pad(&y);

        let mut z = Matrix::zero(x.rows, y.columns); //this is now oversize

        let n_chunk_size = y.columns;
        //it's probable a chunk operation would work better here
        for p_index in (0..x.columns).step_by(Self::KC) {
            let p_chunk_size = min(x.columns - p_index, Self::KC);
            for i_index in (0..x.rows).step_by(Self::MC) {
                let i_chunk_size = min(x.rows - i_index, Self::MC);
                Self::kernel(
                    i_chunk_size,
                    n_chunk_size,
                    p_chunk_size,
                    p_index,
                    i_index,
                    &x,
                    &y,
                    &mut z,
                );
            }
        }
        let z = Self::cut(x_rows, y_cols, &z);
        z
    }

    #[inline(always)]
    fn pad(x: &Matrix<T>) -> Matrix<T> {
        let x_rows = ((x.rows >> 2) << 2) + 4;
        let x_cols = ((x.columns >> 2) << 2) + 4;
        let mut output = Matrix::<T>::zero(x_rows, x_cols);
        x.data_rows()
            .zip(output.data_rows_ref())
            .for_each(|(x_row, o_row)| {
                x_row.iter().zip(o_row.iter_mut()).for_each(|(x, o)| {
                    *o = *x;
                })
            });
        output
    }

    #[inline(always)]
    fn cut(r: usize, c: usize, z: &Matrix<T>) -> Matrix<T> {
        let mut output = Matrix::<T>::zero(r, c);
        output
            .data_rows_ref()
            .zip(z.data_rows())
            .for_each(|(o_row, z_row)| {
                o_row.iter_mut().zip(z_row.iter()).for_each(|(o, z)| {
                    *o = *z;
                });
            });

        output
    }

    fn kernel(
        i_chunk_size: usize,
        n_chunk_size: usize,
        p_chunk_size: usize,
        p_index: usize,
        i_index: usize,
        x: &Matrix<T>,
        y: &Matrix<T>,
        z: &mut Matrix<T>,
    ) {
        let mut packed_x = Matrix::<T>::zero(i_chunk_size, p_chunk_size);

        for row in (0..i_chunk_size).step_by(4) {
            Self::pack_x(row, p_chunk_size, p_index, x, i_index, &mut packed_x);
        }
        for col in (0..n_chunk_size).step_by(4) {
            for row in (0..i_chunk_size).step_by(4) {
                Self::gemm4x4(row, col, p_chunk_size, p_index, i_index, &packed_x, y, z);
            }
        }
    }

    fn pack_x(
        row: usize,
        p_chunk_size: usize,
        p_index: usize,
        x: &Matrix<T>,
        i_index: usize,
        packed: &mut Matrix<T>,
    ) {
        let mut pack_index = row * p_chunk_size;
        for j in 0..p_chunk_size {
            packed.data[pack_index] = x.coeff(row + i_index, j + p_index); //buggy
            packed.data[pack_index + 1] = x.coeff(row + i_index + 1, j + p_index);
            packed.data[pack_index + 2] = x.coeff(row + i_index + 2, j + p_index);
            packed.data[pack_index + 3] = x.coeff(row + i_index + 3, j + p_index);
            pack_index += 4;
        }
    }
    fn gemm4x4(
        row: usize,
        col: usize,
        k: usize,
        p_index: usize,
        i_index: usize,
        packed: &Matrix<T>,
        y: &Matrix<T>,
        z: &mut Matrix<T>,
    ) {
        let mut local_z = [Complex::<T>::ZERO; 16];
        let mut local_x = [Complex::<T>::ZERO; 4];
        let mut local_y = [Complex::<T>::ZERO; 4];

        let mut y_index = y.index(p_index, col);
        let mut x_index = row * k;
        for _ in 0..k {
            local_x[0] = packed.data[x_index];
            local_x[1] = packed.data[x_index + 1];
            local_x[2] = packed.data[x_index + 2];
            local_x[3] = packed.data[x_index + 3];

            local_y[0] = y.data[y_index];
            local_y[1] = y.data[y_index + 1];
            local_y[2] = y.data[y_index + 2];
            local_y[3] = y.data[y_index + 3];

            local_z[0] += local_x[0] * local_y[0];
            local_z[4] += local_x[1] * local_y[0];
            local_z[1] += local_x[0] * local_y[1];
            local_z[5] += local_x[1] * local_y[1];

            local_z[2] += local_x[0] * local_y[2];
            local_z[6] += local_x[1] * local_y[2];
            local_z[3] += local_x[0] * local_y[3];
            local_z[7] += local_x[1] * local_y[3];

            local_z[8] += local_x[2] * local_y[0];
            local_z[12] += local_x[3] * local_y[0];
            local_z[9] += local_x[2] * local_y[1];
            local_z[13] += local_x[3] * local_y[1];

            local_z[10] += local_x[2] * local_y[2];
            local_z[14] += local_x[3] * local_y[2];
            local_z[11] += local_x[2] * local_y[3];
            local_z[15] += local_x[3] * local_y[3];

            y_index += y.columns;
            x_index += 4;
        }
        let mut z_index = z.index(row + i_index, col);

        (0..16).step_by(4).for_each(|i| {
            z.data[z_index] += local_z[i];
            z.data[z_index + 1] += local_z[i + 1];
            z.data[z_index + 2] += local_z[i + 2];
            z.data[z_index + 3] += local_z[i + 3];
            z_index += z.columns;
        });
    }
}

impl<'a, T: Float + 'a> Gemm<'a, T> for Matrix<T> {}

#[cfg(test)]
mod gemm_test {
    use crate::{random::pcg::PermutedCongruentialGenerator, shared::matrix};

    use super::*;

    #[test]
    fn naive_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| Complex::<f32>::new(c as f32, 0.0)).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );

        let r1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m33, &m33);
        let r2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m33, &m34);
        let r3 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m43, &m33);
        let r4 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m34, &m43);
        let r5 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m43, &m34);

        let k1 = Matrix::<f32>::new(
            3,
            [15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });

        let k2 = Matrix::<f32>::new(
            3,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k3 = Matrix::<f32>::new(
            4,
            [
                15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0, 96.0, 126.0, 156.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k4 = Matrix::<f32>::new(
            3,
            [42.0, 48.0, 54.0, 114.0, 136.0, 158.0, 186.0, 224.0, 262.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r4.data().zip(k4.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k5 = Matrix::<f32>::new(
            4,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0,
                158.0, 188.0, 218.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r5.data().zip(k5.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn gemm_static() {
        let m33 = Matrix::new(
            3,
            (0..9).map(|c| Complex::<f32>::new(c as f32, 0.0)).collect(),
        );
        let m34 = Matrix::new(
            3,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let m43 = Matrix::new(
            4,
            (0..12)
                .map(|c| Complex::<f32>::new(c as f32, 0.0))
                .collect(),
        );
        let r1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m33, &m33);
        let r2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m33, &m34);
        let r3 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m43, &m33);
        let r4 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m34, &m43);
        let r5 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m43, &m34);

        let k1 = Matrix::<f32>::new(
            3,
            [15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r1.data().zip(k1.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });

        let k2 = Matrix::<f32>::new(
            3,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r2.data().zip(k2.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k3 = Matrix::<f32>::new(
            4,
            [
                15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0, 96.0, 126.0, 156.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r3.data().zip(k3.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k4 = Matrix::<f32>::new(
            3,
            [42.0, 48.0, 54.0, 114.0, 136.0, 158.0, 186.0, 224.0, 262.0]
                .iter()
                .map(|&r| Complex::<f32>::new(r, 0.0))
                .collect(),
        );
        r4.data().zip(k4.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
        let k5 = Matrix::<f32>::new(
            4,
            [
                20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0,
                158.0, 188.0, 218.0,
            ]
            .iter()
            .map(|&r| Complex::<f32>::new(r, 0.0))
            .collect(),
        );
        r5.data().zip(k5.data()).for_each(|(r, k)| {
            assert!((r.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn gemm_lesser() {
        let mut pcg = PermutedCongruentialGenerator::<f32>::new(3, 0);

        let n1 = (pcg.next_u32() as usize % (Matrix::<f32>::KC - 1) + 4) as usize;
        let n2 = (pcg.next_u32() as usize % (Matrix::<f32>::KC - 1) + 4) as usize;
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m1, &m2);
        let g2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m2, &m1);

        let k1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m1, &m2);
        let k2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });
    }
    #[test]
    fn gemm_middle() {
        let mut pcg = PermutedCongruentialGenerator::<f32>::new(3, 0);

        let n1 = (pcg.next_u32() as usize % Matrix::<f32>::KC + Matrix::<f32>::KC) as usize;
        let n2 = (pcg.next_u32() as usize % Matrix::<f32>::KC + Matrix::<f32>::KC) as usize;
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m1, &m2);
        let g2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m2, &m1);

        let k1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m1, &m2);
        let k2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn gemm_greater() {
        let mut pcg = PermutedCongruentialGenerator::<f32>::new(3, 0);
        let n1 = (pcg.next_u32() as usize % Matrix::<f32>::MC + Matrix::<f32>::MC) as usize;
        let n2 = (pcg.next_u32() as usize % Matrix::<f32>::MC + Matrix::<f32>::MC) as usize;
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m1 = Matrix::new(n1, data.collect());
        let data =
            (0..n1 * n2).map(|_| Complex::<f32>::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0));
        let m2 = Matrix::new(n2, data.collect());

        let g1 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m1, &m2);
        let g2 = <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m2, &m1);

        let k1 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m1, &m2);
        let k2 = <matrix::Matrix<f32> as Gemm<f32>>::naive(&m2, &m1);

        g1.data().zip(k1.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });

        g2.data().zip(k2.data()).for_each(|(g, k)| {
            assert!((g.real - k.real).square_norm() < f32::EPSILON);
        });
    }
}
>>>>>>> master

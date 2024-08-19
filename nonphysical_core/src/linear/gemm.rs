use core::cmp::min;

use crate::shared::{float::Float, matrix::{heap::MatrixHeap, Matrix}};

pub trait Gemm<'a, F: Float + 'a> {
    type GemmMatrix: Matrix<F>;

    const MC: usize = 256;
    const KC: usize = 128;

    fn naive(x: &Self::GemmMatrix, y: &Self::GemmMatrix) -> Self::GemmMatrix {
        debug_assert!(x.n_cols() == y.n_rows());
        let mut z =  Self::GemmMatrix::zero(x.n_rows(), y.n_cols());
        (0..y.n_cols()).for_each(|c| {
            (0..x.n_rows()).for_each(|r| {
                let mut tmp =F::ZERO;
                (0..x.n_cols()).for_each(|a| {
                    //tmp = x.coeff(r, a).fma(y.coeff(a, c),tmp)
                    tmp += x.coeff(r, a) * y.coeff(a, c);
                });
                *z.coeff_ref(r, c) = tmp;
            });
        });
        z
    }
    fn gemm(x: Self::GemmMatrix, y: Self::GemmMatrix) -> Self::GemmMatrix {
        debug_assert!(x.n_cols() == y.n_rows());
        let x_rows = x.n_rows();
        let y_cols = y.n_cols();

        //pad out to size 4
        let x = Self::pad(x);
        let y = Self::pad(y);

        let mut z =Self::GemmMatrix::zero(x.n_rows(), y.n_cols()); //this is now oversize

        let n_chunk_size = y.n_cols();
        //it's probable a chunk operation would work better here
        for p_index in (0..x.n_cols()).step_by(Self::KC) {
            let p_chunk_size = min(x.n_cols() - p_index, Self::KC);
            for i_index in (0..x.n_rows()).step_by(Self::MC) {
                let i_chunk_size = min(x.n_rows() - i_index, Self::MC);
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
        Self::cut(x_rows, y_cols, z)
    }

    #[inline(always)]
    fn pad(x: Self::GemmMatrix) -> Self::GemmMatrix {
        let x_rows = ((x.n_rows() >> 2) << 2) + 4;
        let x_cols = ((x.n_cols() >> 2) << 2) + 4;
        let mut output =Self::GemmMatrix::zero(x_rows, x_cols);
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
    fn cut(r: usize, c: usize, z: Self::GemmMatrix) -> Self::GemmMatrix{
        let mut output =Self::GemmMatrix::zero(r, c);
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
        x: &Self::GemmMatrix,
        y: &Self::GemmMatrix,
        z: &mut Self::GemmMatrix,
    ){
        let mut packed_x =Self::GemmMatrix::zero(i_chunk_size, p_chunk_size);

        for row in (0..i_chunk_size).step_by(4) {
            Self::pack_x(row, p_chunk_size, p_index, x, i_index, &mut packed_x );
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
        x: &Self::GemmMatrix,
        i_index: usize,
        packed: &mut Self::GemmMatrix,
    ) {
        let mut pack_index = row * p_chunk_size;
        for j in 0..p_chunk_size {
            *packed.direct_ref(pack_index) = x.coeff(row + i_index, j + p_index); //buggy
            *packed.direct_ref(pack_index + 1) = x.coeff(row + i_index + 1, j + p_index);
            *packed.direct_ref(pack_index + 2) = x.coeff(row + i_index + 2, j + p_index);
            *packed.direct_ref(pack_index + 3) = x.coeff(row + i_index + 3, j + p_index);
            pack_index += 4;
        }
    }
    fn gemm4x4(
        row: usize,
        col: usize,
        k: usize,
        p_index: usize,
        i_index: usize,
        packed: &Self::GemmMatrix,
        y: &Self::GemmMatrix,
        z: &mut Self::GemmMatrix,
    ) {
        let mut local_z = [F::ZERO; 16];
        let mut local_x = [F::ZERO; 4];
        let mut local_y = [F::ZERO; 4];

        let mut y_index = y.index(p_index, col);
        let mut x_index = row * k;
        for _ in 0..k {
            local_x[0] = packed.direct(x_index);
            local_x[1] = packed.direct(x_index + 1);
            local_x[2] = packed.direct(x_index + 2);
            local_x[3] = packed.direct(x_index + 3);

            local_y[0] = y.direct(y_index);
            local_y[1] = y.direct(y_index + 1);
            local_y[2] = y.direct(y_index + 2);
            local_y[3] = y.direct(y_index + 3);

            local_z[0] = local_x[0].fma(local_y[0], local_z[0]);
            local_z[4] = local_x[1].fma(local_y[0], local_z[4]);
            local_z[1] = local_x[0].fma(local_y[1], local_z[1]);
            local_z[5] = local_x[1].fma(local_y[1], local_z[5]);

            local_z[2] = local_x[0].fma(local_y[2], local_z[2]);
            local_z[6] = local_x[1].fma(local_y[2], local_z[6]);
            local_z[3] = local_x[0].fma(local_y[3], local_z[3]);
            local_z[7] = local_x[1].fma(local_y[3], local_z[7]);

            local_z[8] = local_x[2].fma(local_y[0], local_z[8]);
            local_z[12] = local_x[3].fma(local_y[0], local_z[12]);
            local_z[9] = local_x[2].fma(local_y[1], local_z[9]);
            local_z[13] = local_x[3].fma(local_y[1], local_z[13]);

            local_z[10] = local_x[2].fma(local_y[2], local_z[10]);
            local_z[14] = local_x[3].fma(local_y[2], local_z[14]);
            local_z[11] = local_x[2].fma(local_y[3], local_z[11]);
            local_z[15] = local_x[3].fma(local_y[3], local_z[15]);

            y_index += y.n_cols();
            x_index += 4;
        }
        let mut z_index = z.index(row + i_index, col);

        (0..16).step_by(4).for_each(|i| {
            *z.direct_ref(z_index) += local_z[i];
            *z.direct_ref(z_index + 1) += local_z[i + 1];
            *z.direct_ref(z_index + 2) += local_z[i + 2];
            *z.direct_ref(z_index + 3) += local_z[i + 3];
            z_index += z.n_cols();
        });
    }
}

impl<'a, F: Float + 'a> Gemm<'a, F> for MatrixHeap<F> {
    type GemmMatrix = MatrixHeap<F>;
}
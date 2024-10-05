use super::{DiscreteWavelet, WaveletFamily};
use crate::shared::{
    float::Float,
    matrix::{matrix_heap::MatrixHeap, Matrix},
};
use alloc::vec;
use alloc::vec::Vec;
pub struct DaubechiesFirstWaveletHeap<F: Float> {
    coefficients: [F; 2],
}

impl<F: Float> DiscreteWavelet<F> for DaubechiesFirstWaveletHeap<F> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type Matrix = MatrixHeap<F>;
    type DiscreteWaveletInit = ();

    fn new(_: Self::DiscreteWaveletInit) -> Self {
        let first = (F::IDENTITY + F::IDENTITY).sqrt().recip();
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&self, input: &[F]) -> Vec<F> {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;
        let mut out = vec![F::ZERO; n];
        let (detail, approx) = out.split_at_mut(half_n);

        input
            .chunks_exact(2)
            .zip(detail.iter_mut())
            .zip(approx.iter_mut())
            .for_each(|((chunk, d), a)| {
                let cache_a = chunk[0] * self.coefficients[0];
                let cache_b = chunk[1] * self.coefficients[1];
                *a = cache_a - cache_b;
                *d = cache_a + cache_b;
            });
        out
    }

    fn backward(&self, input: &[F]) -> Vec<F> {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;
        let mut out = vec![F::ZERO; n];
        let (detail, approx) = input.split_at(half_n);
        out.chunks_exact_mut(2)
            .zip(detail.iter())
            .zip(approx.iter())
            .for_each(|((chunk, d), a)| {
                let cache_a = *d * self.coefficients[0];
                let cache_b = *a * self.coefficients[1];
                chunk[0] = cache_a + cache_b;
                chunk[1] = cache_a - cache_b;
            });
        out
    }
    //bugs known in these functions, need to compare to right answers

    fn decompose(&self, input: &[F]) -> Self::Matrix {
        let n = input.len();
        let depth = n.ilog2() as usize + 1;
        assert!(n & (n - 1) == 0); //power of two
        let mut out = Self::Matrix::zero(depth, input.len());
        //base case copy in the row
        out.data_row_ref(0)
            .zip(input.iter())
            .for_each(|(i, j)| *i = *j);
        let mut prev = input;

        //use the previous row to fill in the next one with decreasing chunk size
        out.data_rows_ref()
            .enumerate()
            .skip(1)
            .for_each(|(i, row)| {
                let chunk_size = 1 << (depth - i);
                row.chunks_exact_mut(chunk_size)
                    .zip(prev.chunks_exact(chunk_size))
                    .for_each(|(r_chunk, p_chunk)| {
                        r_chunk.copy_from_slice(&self.forward(p_chunk));
                    });
                prev = row;
            });
        out
    }

    fn cis_detail(&self, input: &[F]) -> Self::Matrix {
        //Take the detail, at every level create a recon from it
        let n = input.len();
        let depth = n.ilog2() as usize + 1;
        assert!(n & (n - 1) == 0); //power of two
        let mut out = Self::Matrix::zero(depth, input.len());
        //base case copy in the row
        out.data_row_ref(0)
            .zip(input.iter())
            .for_each(|(i, j)| *i = *j);

        let mut carry = input.to_vec();
        out.data_rows_ref()
            .enumerate()
            .skip(1)
            .for_each(|(i, row)| {
                let chunk_size = 1 << (depth - i - 1);
                //set up next chunk from detail
                let next = self.forward(&carry);
                //use detail to reconstruct
                carry = next[chunk_size..].to_vec();

                while carry.len() < row.len() {
                    let mut recon = vec![F::ZERO; carry.len()];
                    recon.extend_from_slice(&carry);
                    carry = self.backward(&recon);
                }
                //put carry back into row
                row.copy_from_slice(&carry);
                carry = next[chunk_size..].to_vec();
            });
        out
    }

    fn cis_approx(&self, input: &[F]) -> Self::Matrix {
        //Take the approx, at every level create a recon from it
        let n = input.len();
        let depth = n.ilog2() as usize + 1;
        assert!(n & (n - 1) == 0); //power of two
        let mut out = Self::Matrix::zero(depth, input.len());
        //base case copy in the row
        out.data_row_ref(0)
            .zip(input.iter())
            .for_each(|(i, j)| *i = *j);

        let mut carry = input.to_vec();
        out.data_rows_ref()
            .enumerate()
            .skip(1)
            .for_each(|(i, row)| {
                let chunk_size = 1 << (depth - i - 1);
                //set up next chunk from approx
                let next = self.forward(&carry);
                //use approx to reconstruct
                carry = next[..chunk_size].to_vec();

                while carry.len() < row.len() {
                    let mut recon = carry.clone();
                    recon.extend_from_slice(&vec![F::ZERO; carry.len()]);
                    carry = self.backward(&recon);
                }
                //put carry back into row
                row.copy_from_slice(&carry);
                carry = next[..chunk_size].to_vec();
            });
        out
    }

    fn trans_detail(&self, input: &[F]) -> Self::Matrix {
        //Take the approx, at every level create a recon from the detail
        let n = input.len();
        let depth = n.ilog2() as usize + 1;
        assert!(n & (n - 1) == 0); //power of two
        let mut out = Self::Matrix::zero(depth, input.len());
        //base case copy in the row
        out.data_row_ref(0)
            .zip(input.iter())
            .for_each(|(i, j)| *i = *j);

        let mut carry = input.to_vec();
        out.data_rows_ref()
            .enumerate()
            .skip(1)
            .for_each(|(i, row)| {
                let chunk_size = 1 << (depth - i - 1);
                //set up next chunk from approx
                let next = self.forward(&carry);
                //use detail to reconstruct
                carry = next[chunk_size..].to_vec();

                while carry.len() < row.len() {
                    let mut recon = vec![F::ZERO; carry.len()];
                    recon.extend_from_slice(&carry);
                    carry = self.backward(&recon);
                }
                //put carry back into row
                row.copy_from_slice(&carry);
                carry = next[..chunk_size].to_vec();
            });
        out
    }

    fn trans_approx(&self, input: &[F]) -> Self::Matrix {
        //Take the detail, at every level create a recon from the approx
        let n = input.len();
        let depth = n.ilog2() as usize + 1;
        assert!(n & (n - 1) == 0); //power of two
        let mut out = Self::Matrix::zero(depth, input.len());
        //base case copy in the row
        out.data_row_ref(0)
            .zip(input.iter())
            .for_each(|(i, j)| *i = *j);

        let mut carry = input.to_vec();
        out.data_rows_ref()
            .enumerate()
            .skip(1)
            .for_each(|(i, row)| {
                let chunk_size = 1 << (depth - i - 1);
                //set up next chunk from approx
                let next = self.forward(&carry);
                //use approx to reconstruct
                carry = next[..chunk_size].to_vec();

                while carry.len() < row.len() {
                    let mut recon = carry.clone();
                    recon.extend_from_slice(&vec![F::ZERO; carry.len()]);
                    carry = self.backward(&recon);
                }
                //put carry back into row
                row.copy_from_slice(&carry);
                carry = next[chunk_size..].to_vec();
            });
        out
    }
}

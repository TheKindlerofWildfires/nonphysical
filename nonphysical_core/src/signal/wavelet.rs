use alloc::vec::Vec;

use crate::shared::{float::Float, matrix::Matrix};


pub mod wavelet_heap;
pub enum WaveletFamily {
    ReverseBiorthogonal,
    Daubechies,
    Symlet,
    Coiflets,
    Biorthogonal,
    DiscreteMeyer,
}

pub trait DiscreteWavelet<F: Float> {
    const SYMMETRY: usize;
    const ORTHOGONAL: usize;
    const BIORTHOGONAL: usize;
    const FAMILY: WaveletFamily;
    type Matrix: Matrix<F>;
    type DiscreteWaveletInit;

    fn new(init: Self::DiscreteWaveletInit) -> Self;

    fn forward(&self, input: &[F]) -> Vec<F>;

    fn backward(&self, input: &[F])-> Vec<F>;

    fn decompose(&self, input: &[F])->Self::Matrix;

    fn cis_detail(&self, input: &[F])->Self::Matrix;

    fn cis_approx(&self, input: &[F])->Self::Matrix;

    fn trans_detail(&self, input: &[F])->Self::Matrix;

    fn trans_approx(&self, input: &[F])->Self::Matrix;
}


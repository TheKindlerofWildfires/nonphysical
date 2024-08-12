use crate::shared::float::Float;


pub mod heap;
pub mod stack;
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
    type DiscreteWaveletInit;

    fn new(init: Self::DiscreteWaveletInit) -> Self;

    fn forward(&self, input: &mut [F]);

    fn backward(&self, input: &mut [F]);
}


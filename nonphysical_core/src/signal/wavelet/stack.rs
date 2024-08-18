use crate::shared::{complex::Complex, float::Float, primitive::Primitive};

use super::{DiscreteWavelet, WaveletFamily};
pub struct DaubechiesFirstRealWaveletStack<P: Primitive, const N: usize> {
    coefficients: [P; 2],
}

impl<P: Primitive, const N: usize> DiscreteWavelet<P> for DaubechiesFirstRealWaveletStack<P, N> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type DiscreteWaveletInit = ();

    fn new(_: Self::DiscreteWaveletInit) -> Self {
        let first = P::usize(2).sqrt().recip();
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&mut self, input: &mut [P]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let mut deconstruction = [P::ZERO;N];

        input.chunks_exact(2).enumerate().for_each(|(i,chunk)| {
            let cache_a = chunk[0] * self.coefficients[0];
            let cache_b = chunk[1] * self.coefficients[1];
            deconstruction[i] = cache_a+cache_b;
            deconstruction[i+N/2] = cache_a-cache_b;
        });

        input.copy_from_slice(&deconstruction);
    }

    fn backward(&mut self, input: &mut [P]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut reconstruction = [P::ZERO;N];

        input[0..half_n]
            .iter()
            .zip(input[half_n..].iter()).enumerate()
            .for_each(|(i,(lp, hp))| {
                let cache_a = *lp * self.coefficients[0];
                let cache_b = *hp * self.coefficients[1];
                reconstruction[i*2]=cache_a+cache_b;
                reconstruction[i*2+1] = cache_a-cache_b;
            });

        input.copy_from_slice(&reconstruction);
    }
}
pub struct DaubechiesFirstComplexWaveletStack<C: Complex, const N: usize> {
    coefficients: [C; 2],
}

impl<C: Complex, const N: usize> DiscreteWavelet<C> for DaubechiesFirstComplexWaveletStack<C, N> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type DiscreteWaveletInit = ();

    fn new(_: Self::DiscreteWaveletInit) -> Self {
        let first = C::new(C::Primitive::usize(2).sqrt().recip(), C::Primitive::ZERO);
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&mut self, input: &mut [C]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let mut deconstruction = [C::ZERO;N];

        input.chunks_exact_mut(2).enumerate().for_each(|(i,chunk)| {
            let cache_a = chunk[0] * self.coefficients[0];
            let cache_b = chunk[1] * self.coefficients[1];
            deconstruction[i*2]=cache_a+cache_b;
            deconstruction[i*2+1] = cache_a-cache_b;
        });

        input.copy_from_slice(&deconstruction);
    }

    fn backward(&mut self, input: &mut [C]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut reconstruction = [C::ZERO;N];

        input[0..half_n]
            .iter()
            .zip(input[half_n..].iter()).enumerate()
            .for_each(|(i,(lp, hp))| {
                let cache_a = *lp * self.coefficients[0];
                let cache_b = *hp * self.coefficients[1];
                reconstruction[i*2]=cache_a+cache_b;
                reconstruction[i*2+1] = cache_a-cache_b;
            });

        input.copy_from_slice(&reconstruction);
    }
}

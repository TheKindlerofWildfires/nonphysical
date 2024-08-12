use crate::shared::{complex::Complex, float::Float, primitive::Primitive};
use alloc::vec::Vec;

use super::{DiscreteWavelet, WaveletFamily};
pub struct DaubechiesFirstRealWaveletHeap<P: Primitive> {
    coefficients: [P; 2],
}

impl<P: Primitive> DiscreteWavelet<P> for DaubechiesFirstRealWaveletHeap<P> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type DiscreteWaveletInit = ();

    fn new(_: Self::DiscreteWaveletInit) -> Self{
        let first = P::usize(2).sqrt().recip();
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&self, input: &mut [P]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut low_pass = Vec::with_capacity(half_n);
        let mut high_pass = Vec::with_capacity(half_n);

        input.chunks_exact(2).for_each(|chunk| {
            let cache_a = chunk[0] * self.coefficients[0];
            let cache_b = chunk[1] * self.coefficients[1];
            low_pass.push(cache_a + cache_b);
            high_pass.push(cache_a - cache_b);
        });

        input[0..half_n].copy_from_slice(&low_pass);
        input[half_n..].copy_from_slice(&high_pass);

    }

    fn backward(&self, input: &mut [P]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut reconstruction = Vec::with_capacity(n);

        input[0..half_n].iter().zip(input[half_n..].iter()).for_each(|(lp, hp)| {
            let cache_a = *lp * self.coefficients[0];
            let cache_b = *hp * self.coefficients[1];
            reconstruction.push(cache_a + cache_b);
            reconstruction.push(cache_a - cache_b);
        });

        input.copy_from_slice(&reconstruction);

    }
}
pub struct DaubechiesFirstComplexWaveletHeap<C: Complex> {
    coefficients: [C; 2],
}

impl<C: Complex> DiscreteWavelet<C> for DaubechiesFirstComplexWaveletHeap<C> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type DiscreteWaveletInit = ();

    fn new(_: Self::DiscreteWaveletInit) -> Self{
        let first = C::new(C::Primitive::usize(2).sqrt().recip(), C::Primitive::ZERO);
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&self, input: &mut [C]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut low_pass = Vec::with_capacity(half_n);
        let mut high_pass = Vec::with_capacity(half_n);

        input.chunks_exact(2).for_each(|chunk| {
            let cache_a = chunk[0] * self.coefficients[0];
            let cache_b = chunk[1] * self.coefficients[1];
            low_pass.push(cache_a + cache_b);
            high_pass.push(cache_a - cache_b);
        });

        input[0..half_n].copy_from_slice(&low_pass);
        input[half_n..].copy_from_slice(&high_pass);

    }

    fn backward(&self, input: &mut [C]) {
        let n = input.len();
        assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut reconstruction = Vec::with_capacity(n);

        input[0..half_n].iter().zip(input[half_n..].iter()).for_each(|(lp, hp)| {
            let cache_a = *lp * self.coefficients[0];
            let cache_b = *hp * self.coefficients[1];
            reconstruction.push(cache_a + cache_b);
            reconstruction.push(cache_a - cache_b);
        });

        input.copy_from_slice(&reconstruction);

    }

}
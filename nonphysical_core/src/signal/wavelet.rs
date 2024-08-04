use crate::shared::{complex::Complex, float::Float, primitive::Primitive};
use alloc::vec::Vec;
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

    fn new() -> Self;

    fn forward(&self, input: &[F]) -> [Vec<F>; 2];

    fn backward(&self, input: [Vec<F>; 2]) -> Vec<F>;

    //fn forward_multi(&self, input: &mut M, level: usize) -> Vec<M>;
    //fn forward_multi(&self, input: &mut M, level: usize) -> Vec<M>;
}
pub struct DaubechiesFirstRealWavelet<P: Primitive> {
    coefficients: [P; 2],
}

impl<P: Primitive> DiscreteWavelet<P> for DaubechiesFirstRealWavelet<P> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;

    fn new() -> Self {
        let first = P::usize(2).sqrt().recip();
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&self, input: &[P]) -> [Vec<P>; 2] {
        let n = input.len();
        debug_assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut low_pass = Vec::with_capacity(half_n);
        let mut high_pass = Vec::with_capacity(half_n);

        input.chunks_exact(2).for_each(|chunk| {
            let cache_a = chunk[0] * self.coefficients[0];
            let cache_b = chunk[1] * self.coefficients[1];
            low_pass.push(cache_a + cache_b);
            high_pass.push(cache_a - cache_b);
        });

        [low_pass, high_pass]
    }

    fn backward(&self, input: [Vec<P>; 2]) -> Vec<P> {
        debug_assert!(input.len() == 2);
        let low_pass = &input[0];
        let high_pass = &input[1];
        let n = low_pass.len() + high_pass.len();

        let mut output = Vec::with_capacity(n);

        low_pass.iter().zip(high_pass.iter()).for_each(|(lp, hp)| {
            let cache_a = *lp * self.coefficients[0];
            let cache_b = *hp * self.coefficients[1];
            output.push(cache_a + cache_b);
            output.push(cache_a - cache_b);
        });
        output
    }
}
pub struct DaubechiesFirstComplexWavelet<C: Complex> {
    coefficients: [C; 2],
}

impl<C: Complex> DiscreteWavelet<C> for DaubechiesFirstComplexWavelet<C> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;

    fn new() -> Self {
        let first = C::new(C::Primitive::usize(2).sqrt().recip(), C::Primitive::ZERO);
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&self, input: &[C]) -> [Vec<C>; 2] {
        let n = input.len();
        debug_assert!(n % 2 == 0);
        let half_n = n / 2;

        let mut low_pass = Vec::with_capacity(half_n);
        let mut high_pass = Vec::with_capacity(half_n);

        input.chunks_exact(2).for_each(|chunk| {
            let cache_a = chunk[0] * self.coefficients[0];
            let cache_b = chunk[1] * self.coefficients[1];
            low_pass.push(cache_a + cache_b);
            high_pass.push(cache_a - cache_b);
        });

        [low_pass, high_pass]
    }
    /*
    fn forward_multi(input: &mut Vec<F>, levels: usize) -> Vec<Vec<F>> {
        let mut dwt_result = Vec::with_capacity(levels + 1);
        let mut current_signal = signal.clone();

        for _ in 0..levels {
            let (low_pass, high_pass) = dwt_1d(&current_signal);
            dwt_result.push(high_pass);
            current_signal = low_pass;
        }
        dwt_result.push(current_signal); // Approximation at the final level

        dwt_result.reverse(); // To have high-pass details first and approximation last

        dwt_result
    }*/

    fn backward(&self, input: [Vec<C>; 2]) -> Vec<C> {
        debug_assert!(input.len() == 2);
        let low_pass = &input[0];
        let high_pass = &input[1];
        let n = low_pass.len() + high_pass.len();

        let mut output = Vec::with_capacity(n);

        low_pass.iter().zip(high_pass.iter()).for_each(|(lp, hp)| {
            let cache_a = *lp * self.coefficients[0];
            let cache_b = *hp * self.coefficients[1];
            output.push(cache_a + cache_b);
            output.push(cache_a - cache_b);
        });
        output
    }

    /*
        fn backwards_multi(input: &mut Vec<Vec<F>>) -> Vec<F> {
        let mut current_signal = dwt_result.last().unwrap().clone(); // Get the approximation signal at the final level

        for detail_coefficients in dwt_result.iter().rev().skip(1) {
            current_signal = idwt_1d(&current_signal, &detail_coefficients);
        }

        current_signal
    }
    */
}
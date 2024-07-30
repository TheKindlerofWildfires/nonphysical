use crate::shared::{complex::Complex, float::Float, real::Real};
use alloc::vec::Vec;
pub enum WaveletFamily {
    ReverseBiorthogonal,
    Daubechies,
    Symlet,
    Coiflets,
    Biorthogonal,
    DiscreteMeyer,
}

trait DiscreteWavelet<F: Float> {
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
pub struct DaubechiesFirstRealWavelet<R: Real> {
    coefficients: [R; 2],
}

impl<R: Real> DiscreteWavelet<R> for DaubechiesFirstRealWavelet<R> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;

    fn new() -> Self {
        let first = R::usize(2).sqrt().recip();
        let coefficients = [first, first];
        Self { coefficients }
    }

    fn forward(&self, input: &[R]) -> [Vec<R>; 2] {
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

    fn backward(&self, input: [Vec<R>; 2]) -> Vec<R> {
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

#[cfg(test)]
mod wavelet_tests {
    use std::time::SystemTime;

    use alloc::vec;

    use crate::{random::pcg::PermutedCongruentialGenerator, shared::complex::ComplexFloat};

    use super::*;

    #[test]
    fn daubechies_c_first_2_static() {
        let signal = vec![
            ComplexFloat::<f32>::new(0.5, 0.0),
            ComplexFloat::<f32>::new(1.5, 0.0),
        ];
        let knowns = vec![
            vec![ComplexFloat::new(2.0.sqrt(), 0.0)],
            vec![ComplexFloat::new(-(2.0).sqrt() / 2.0, 0.0)],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_4_static() {
        let signal = vec![
            ComplexFloat::<f32>::new(0.5, 0.0),
            ComplexFloat::<f32>::new(1.5, 0.0),
            ComplexFloat::<f32>::new(-1.0, 0.0),
            ComplexFloat::<f32>::new(-2.5, 0.0),
        ];
        let knowns = vec![
            vec![
                ComplexFloat::new(2.0.sqrt(), 0.0),
                ComplexFloat::new(-7.0 * 2.0.sqrt() / 4.0, 0.0),
            ],
            vec![
                ComplexFloat::new(-(2.0).sqrt() / 2.0, 0.0),
                ComplexFloat::new(3.0 * 2.0.sqrt() / 4.0, 0.0),
            ],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_first_8_static() {
        let signal = vec![
            ComplexFloat::<f32>::new(0.5, 0.0),
            ComplexFloat::<f32>::new(1.5, 0.0),
            ComplexFloat::<f32>::new(-1.0, 0.0),
            ComplexFloat::<f32>::new(-2.5, 0.0),
            ComplexFloat::<f32>::new(-1.5, 0.0),
            ComplexFloat::<f32>::new(1.5, 0.0),
            ComplexFloat::<f32>::new(-2.0, 0.0),
            ComplexFloat::<f32>::new(2.0, 0.0),
        ];
        let knowns = vec![
            vec![
                ComplexFloat::new(2.0.sqrt(), 0.0),
                ComplexFloat::new(-7.0 * 2.0.sqrt() / 4.0, 0.0),
                ComplexFloat::ZERO,
                ComplexFloat::ZERO,
            ],
            vec![
                ComplexFloat::new(-(2.0).sqrt() / 2.0, 0.0),
                ComplexFloat::new(3.0 * 2.0.sqrt() / 4.0, 0.0),
                ComplexFloat::new(-3.0 * 2.0.sqrt() / 2.0, 0.0),
                ComplexFloat::new(-2.0 * 2.0.sqrt(), 0.0),
            ],
        ];

        let dfw = DaubechiesFirstComplexWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_2_static() {
        let signal = vec![
            0.5,
            1.5
        ];
        let knowns = vec![
            vec![2.0.sqrt()],
            vec![-(2.0).sqrt() / 2.0],
        ];

        let dfw: DaubechiesFirstRealWavelet<f32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_4_static() {
        let signal = vec![
            0.5,
            1.5,
            -1.0,
            -2.5,
        ];
        let knowns = vec![
            vec![
                2.0.sqrt(),
                -7.0 * 2.0.sqrt() / 4.0,
            ],
            vec![
                -(2.0).sqrt() / 2.0,
                3.0 * 2.0.sqrt() / 4.0,
            ],
        ];

        let dfw: DaubechiesFirstRealWavelet<f32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_r_first_8_static() {
        let signal = vec![
            0.5,
            1.5,
            -1.0,
            -2.5,
            -1.5,
            1.5,
            -2.0,
            2.0,
        ];
        let knowns = vec![
            vec![
                2.0.sqrt(),
                -7.0 * 2.0.sqrt() / 4.0,
                0.0,
                0.0,
            ],
            vec![
                -(2.0).sqrt() / 2.0,
                3.0 * 2.0.sqrt() / 4.0,
                -3.0 * 2.0.sqrt() / 2.0,
                -2.0 * 2.0.sqrt(),
            ],
        ];

        let dfw: DaubechiesFirstRealWavelet<f32> = DaubechiesFirstRealWavelet::new();
        let deconstruction = dfw.forward(&signal);
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).l2_norm() < f32::EPSILON);
            });
        });

        let reconstruction = dfw.backward(deconstruction);
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_c_time() {
        let mut pcg = PermutedCongruentialGenerator::new(3, 0);
        let signal =
            (0..2048*4096).map(|_| ComplexFloat::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();
        let dfw = DaubechiesFirstComplexWavelet::new();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}",now.elapsed());

        let signal =
            (0..1024*4096).map(|_| ComplexFloat::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}",now.elapsed());

        let signal =
        (0..512*4096).map(|_| ComplexFloat::new(pcg.next_u32() as f32 / u32::MAX as f32, 0.0)).collect::<Vec<_>>();
        let now = SystemTime::now();
        let _ = dfw.forward(&signal);
        let _ = println!("{:?}",now.elapsed());
    }
}

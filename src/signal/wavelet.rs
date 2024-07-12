use crate::shared::{complex::Complex, float::Float};

pub enum WaveletFamily {
    ReverseBiorthogonal,
    DaubechiesFirst,
    Symlet,
    Coiflets,
    Biorthogonal,
    DiscreteMeyer,
}

trait DiscreteWavelet<T: Float> {
    const SYMMETRY: usize;
    const ORTHOGONAL: usize;
    const BIORTHOGONAL: usize;
    const FAMILY: WaveletFamily;

    fn coefficients() -> Vec<Complex<T>>;

    fn forward(input: &Self, coefficients: &[Complex<T>]) -> Vec<Self>
    where
        Self: Sized;

    fn backward(input: Vec<Self>, coefficients:&[Complex<T>]) -> Self
    where
        Self: Sized;

    //fn forward_multi(&self, input: &mut M, level: usize) -> Vec<M>;
    //fn forward_multi(&self, input: &mut M, level: usize) -> Vec<M>;
}

trait DaubechiesFirstWavelet<T: Float>: DiscreteWavelet<T> {
    fn daubechies_first_coefficients() -> Vec<Complex<T>> {
        vec![
            Complex::new(T::usize(2).sqrt().recip(), T::ZERO),
            Complex::new(T::usize(2).sqrt().recip(), T::ZERO),
        ]
    }

    fn daubechies_first_forward(input: &Self, coefficients: &[Complex<T>]) -> Vec<Self>
    where
        Self: Sized;

    fn daubechies_first_backward(input: Vec<Self>, coefficients: &[Complex<T>]) -> Self
    where
        Self: Sized;
}

impl<T: Float, F: DaubechiesFirstWavelet<T>> DiscreteWavelet<T> for F {
    const SYMMETRY: usize = 0;

    const ORTHOGONAL: usize = 1;

    const BIORTHOGONAL: usize = 1;

    const FAMILY: WaveletFamily = WaveletFamily::DaubechiesFirst;

    fn coefficients() -> Vec<Complex<T>> {
        Self::daubechies_first_coefficients()
    }

    fn forward(input: &Self, coefficients: &[Complex<T>]) -> Vec<Self>
    where
        Self: Sized,
    {
        Self::daubechies_first_forward(input, coefficients)
    }

    fn backward(input: Vec<Self>, coefficients: &[Complex<T>]) -> Self
    where
        Self: Sized,
    {
        Self::daubechies_first_backward(input, coefficients)
    }
}

impl<T: Float> DaubechiesFirstWavelet<T> for Vec<Complex<T>> {
    fn daubechies_first_forward(input: &Self, coefficients: &[Complex<T>]) -> Vec<Vec<Complex<T>>> {
        let n = input.len();
        debug_assert!(n % 2 == 0);
        let half_n = n/2;
        
        let mut low_pass = Vec::with_capacity(half_n);
        let mut high_pass = Vec::with_capacity(half_n);

        input.chunks_exact(2).for_each(|chunk|{
            let cache_a = chunk[0] * coefficients[0];
            let cache_b = chunk[1] * coefficients[1];
            low_pass.push(cache_a+cache_b);
            high_pass.push(cache_a-cache_b);
        });

        vec![low_pass, high_pass]
    }
    /*
    fn forward_multi(input: &mut Vec<Complex<T>>, levels: usize) -> Vec<Vec<Complex<T>>> {
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

    fn daubechies_first_backward(input: Vec<Self>, coefficients:&[Complex<T>]) -> Vec<Complex<T>> {
        debug_assert!(input.len() == 2);
        let low_pass = &input[0];
        let high_pass = &input[1];
        let n = low_pass.len() + high_pass.len();

        let mut output = Vec::with_capacity(n);

        low_pass.iter().zip(high_pass.iter()).for_each(|(lp,hp)|{
            let cache_a = *lp * coefficients[0];
            let cache_b = *hp * coefficients[1];
            output.push(cache_a + cache_b);
            output.push(cache_a - cache_b);
        });
        output
    }

    /*
        fn backwards_multi(input: &mut Vec<Vec<Complex<T>>>) -> Vec<Complex<T>> {
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

    use super::*;

    #[test]
    fn daubechies_first_2_static() {
        let signal = vec![Complex::<f32>::new(0.5, 0.0), Complex::<f32>::new(1.5, 0.0)];
        let knowns = vec![
            vec![Complex::new(2.0.sqrt(), 0.0)],
            vec![Complex::new(-2.0.sqrt() / 2.0, 0.0)],
        ];

        let coefficients = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_coefficients();

        let deconstruction = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_forward(
            &signal,
            &coefficients,
        );
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).square_norm() < f32::EPSILON);
            });
        });

        let reconstruction = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_backward(
            deconstruction,
            &coefficients,
        );
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_first_4_static() {
        let signal = vec![
            Complex::<f32>::new(0.5, 0.0),
            Complex::<f32>::new(1.5, 0.0),
            Complex::<f32>::new(-1.0, 0.0),
            Complex::<f32>::new(-2.5, 0.0),
        ];
        let knowns = vec![
            vec![
                Complex::new(2.0.sqrt(), 0.0),
                Complex::new(-7.0 * 2.0.sqrt() / 4.0, 0.0),
            ],
            vec![
                Complex::new(-2.0.sqrt() / 2.0, 0.0),
                Complex::new(3.0 * 2.0.sqrt() / 4.0, 0.0),
            ],
        ];

        let coefficients = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_coefficients();

        let deconstruction = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_forward(
            &signal,
            &coefficients,
        );
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).square_norm() < f32::EPSILON);
            });
        });

        let reconstruction = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_backward(
            deconstruction,
            &coefficients,
        );
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).square_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn daubechies_first_8_static() {
        let signal = vec![
            Complex::<f32>::new(0.5, 0.0),
            Complex::<f32>::new(1.5, 0.0),
            Complex::<f32>::new(-1.0, 0.0),
            Complex::<f32>::new(-2.5, 0.0),
            Complex::<f32>::new(-1.5, 0.0),
            Complex::<f32>::new(1.5, 0.0),
            Complex::<f32>::new(-2.0, 0.0),
            Complex::<f32>::new(2.0, 0.0),
        ];
        let knowns = vec![
            vec![
                Complex::new(2.0.sqrt(), 0.0),
                Complex::new(-7.0 * 2.0.sqrt() / 4.0, 0.0),
                Complex::ZERO,
                Complex::ZERO,
            ],
            vec![
                Complex::new(-2.0.sqrt() / 2.0, 0.0),
                Complex::new(3.0 * 2.0.sqrt() / 4.0, 0.0),
                Complex::new(-3.0 * 2.0.sqrt() / 2.0, 0.0),
                Complex::new(-2.0 * 2.0.sqrt(), 0.0),
            ],
        ];

        let coefficients = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_coefficients();

        let deconstruction = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_forward(
            &signal,
            &coefficients,
        );
        deconstruction.iter().zip(knowns.iter()).for_each(|(c, k)| {
            c.iter().zip(k.iter()).for_each(|(cc, kk)| {
                assert!((*cc - *kk).square_norm() < f32::EPSILON);
            });
        });

        let reconstruction = <Vec<Complex<f32>> as DaubechiesFirstWavelet<f32>>::daubechies_first_backward(
            deconstruction,
            &coefficients,
        );
        reconstruction.iter().zip(signal.iter()).for_each(|(r, k)| {
            assert!((*r - *k).square_norm() < f32::EPSILON);
        });
    }

}

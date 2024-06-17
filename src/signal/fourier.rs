use crate::shared::{complex::Complex, float::Float};

pub struct FastFourierTransform<T: Float> {
    twiddles: Vec<Complex<T>>,
}
impl<T: Float> FastFourierTransform<T> {
    pub fn new(len: usize) -> Self {
        let twiddles = Self::generate_twiddles(len >> 1);
        FastFourierTransform { twiddles }
    }

    pub fn fft(&self, x: &mut [Complex<T>]) {
        let twiddles = &mut self.twiddles.clone();

        let n: usize = x.len().ilog2() as usize;

        (0..n).rev().for_each(|t| {
            let dist = 1 << t;
            let chunk_size = dist << 1;

            if chunk_size > 4 {
                if t < n - 1 {
                    Self::collapse_twiddles(twiddles);
                }
                Self::fft_chunk_n(x, twiddles, dist);
            } else if chunk_size == 2 {
                Self::fft_chunk_2(x);
            } else if chunk_size == 4 {
                Self::fft_chunk_4(x);
            }
        });
        Self::reverse(x, n);
    }

    fn ifft(&self, x: &mut [Complex<T>]) {
        x.iter_mut().for_each(|c| *c = c.conj());
        self.fft(x);
        let sf = T::one() / T::usize(x.len());
        x.iter_mut().for_each(|c| *c = c.conj()*sf);
    }

    #[inline]
    fn generate_twiddles(dist: usize) -> Vec<Complex<T>> {
        let angle = -T::pi() / T::usize(dist);
        let mut twiddles = Vec::with_capacity(dist);
        (0..dist).for_each(|i| {
            let phase = angle * T::usize(i);
            let (sin, cos) = phase.sin_cos();
            twiddles.push(Complex::<T>::new(cos, sin));
        });
        twiddles
    }

    #[inline]
    fn collapse_twiddles(twiddles: &mut Vec<Complex<T>>) {
        let len = twiddles.len() >> 1;
        (0..len).for_each(|i| {
            twiddles[i] = twiddles[i << 1];
        });
        twiddles.resize(len, Complex::<T>::new(T::zero(), T::zero()));
    }

    #[inline]
    fn fft_chunk_n(x: &mut [Complex<T>], twiddles: &[Complex<T>], dist: usize) {
        let chunk_size = dist << 1;
        x.chunks_exact_mut(chunk_size).for_each(|chunk| {
            let (complex_s0, complex_s1) = chunk.split_at_mut(dist);
            complex_s0
                .iter_mut()
                .zip(complex_s1.iter_mut())
                .zip(twiddles)
                .for_each(|((c_s0, c_s1), w)| {
                    let temp = *c_s0 - *c_s1;
                    *c_s0 = *c_s0 + *c_s1;
                    *c_s1 = temp * *w;
                });
        });
    }

    #[inline]
    fn fft_chunk_4(x: &mut [Complex<T>]) {
        x.chunks_exact_mut(4).for_each(|chunk| {
            let (complex_s0, complex_s1) = chunk.split_at_mut(2);

            let temp = complex_s0[0];
            complex_s0[0] = complex_s0[0] + complex_s1[0];
            complex_s1[0] = temp - complex_s1[0];

            let temp = complex_s0[1];
            complex_s0[1] = complex_s0[1] + complex_s1[1];
            complex_s1[1] = (temp - complex_s1[1]).mul_ni();
        });
    }

    #[inline]
    fn fft_chunk_2(x: &mut [Complex<T>]) {
        x.chunks_exact_mut(2).for_each(|chunk| {
            let temp = chunk[0];
            chunk[0] = chunk[0] + chunk[1];
            chunk[1] = temp - chunk[1];
        });
    }
    #[inline]
    fn reverse(buf: &mut [Complex<T>], log_n: usize) {
        let big_n = 1 << log_n;
        let half_n = big_n >> 1;
        let quart_n = big_n >> 2;
        let min = big_n - 1;

        let mut forward = half_n;
        let mut rev = 1;
        (0..quart_n).rev().for_each(|i| {
            let zeros = (i as usize).trailing_ones();

            forward ^= 2 << zeros;
            rev ^= quart_n >> zeros;

            if forward < rev {
                buf.swap(forward, rev);
                buf.swap(min ^ forward, min ^ rev);
            }

            buf.swap(forward ^ 1, rev ^ half_n);
        });
    }
}

#[cfg(test)]
mod fourier_tests {
    use super::*;

    #[test]
    fn test_forward_2() {
        let mut x = (0..2)
            .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            Complex::<f32>::new(1.0, 3.0),
            Complex::<f32>::new(-1.0, -1.0),
        ];
        let truth_2 = [Complex::<f32>::new(0.0, 2.0), Complex::<f32>::new(2.0, 4.0)];
        let truth_3 = [
            Complex::<f32>::new(2.0, 6.0),
            Complex::<f32>::new(-2.0, -2.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_forward_4() {
        let mut x = (0..4)
            .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            Complex::<f32>::new(6.0, 10.0),
            Complex::<f32>::new(-4.0, 0.0),
            Complex::<f32>::new(-2.0, -2.0),
            Complex::<f32>::new(0.0, -4.0),
        ];
        let truth_2 = [
            Complex::<f32>::new(0.0, 4.0),
            Complex::<f32>::new(12.0, 16.0),
            Complex::<f32>::new(8.0, 12.0),
            Complex::<f32>::new(4.0, 8.0),
        ];
        let truth_3 = [
            Complex::<f32>::new(24.0, 40.0),
            Complex::<f32>::new(0.0, -16.0),
            Complex::<f32>::new(-8.0, -8.0),
            Complex::<f32>::new(-16.0, 0.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_forward_8() {
        let mut x = (0..8)
            .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            Complex::<f32>::new(28.0, 36.0),
            Complex::<f32>::new(-13.65685425, 5.65685425),
            Complex::<f32>::new(-8.0, 0.0),
            Complex::<f32>::new(-5.65685425, -2.34314575),
            Complex::<f32>::new(-4.0, -4.0),
            Complex::<f32>::new(-2.34314575, -5.65685425),
            Complex::<f32>::new(0.0, -8.0),
            Complex::<f32>::new(5.65685425, -13.65685425),
        ];
        let truth_2 = [
            Complex::<f32>::new(0.0, 8.0),
            Complex::<f32>::new(56.0, 64.0),
            Complex::<f32>::new(48.0, 56.0),
            Complex::<f32>::new(40.0, 48.0),
            Complex::<f32>::new(32.0, 40.0),
            Complex::<f32>::new(24.0, 32.0),
            Complex::<f32>::new(16.0, 24.0),
            Complex::<f32>::new(8.0, 16.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn test_forward_16() {
        let mut x = (0..16)
            .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            Complex::<f32>::new(120.0, 136.0),
            Complex::<f32>::new(-48.21871594, 32.21871594),
            Complex::<f32>::new(-27.3137085, 11.3137085),
            Complex::<f32>::new(-19.9728461, 3.9728461),
            Complex::<f32>::new(-16.0, 0.0),
            Complex::<f32>::new(-13.3454291, -2.6545709),
            Complex::<f32>::new(-11.3137085, -4.6862915),
            Complex::<f32>::new(-9.59129894, -6.40870106),
            Complex::<f32>::new(-8.0, -8.0),
            Complex::<f32>::new(-6.40870106, -9.59129894),
            Complex::<f32>::new(-4.6862915, -11.3137085),
            Complex::<f32>::new(-2.6545709, -13.3454291),
            Complex::<f32>::new(0.0, -16.0),
            Complex::<f32>::new(3.9728461, -19.9728461),
            Complex::<f32>::new(11.3137085, -27.3137085),
            Complex::<f32>::new(32.21871594, -48.21871594),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_inverse_2() {
        let mut x = [
            Complex::<f32>::new(2.0, 6.0),
            Complex::<f32>::new(-2.0, -2.0),
        ];

        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [Complex::<f32>::new(0.0, 2.0), Complex::<f32>::new(2.0, 4.0)];
        let truth_2 = [
            Complex::<f32>::new(1.0, 3.0),
            Complex::<f32>::new(-1.0, -1.0),
        ];
        let truth_3 = (0..2)
            .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_inverse_4() {
        let mut x = [
            Complex::<f32>::new(24.0, 40.0),
            Complex::<f32>::new(0.0, -16.0),
            Complex::<f32>::new(-8.0, -8.0),
            Complex::<f32>::new(-16.0, 0.0),
        ];

        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            Complex::<f32>::new(0.0, 4.0),
            Complex::<f32>::new(12.0, 16.0),
            Complex::<f32>::new(8.0, 12.0),
            Complex::<f32>::new(4.0, 8.0),
        ];
        let truth_2 = [
            Complex::<f32>::new(6.0, 10.0),
            Complex::<f32>::new(-4.0, 0.0),
            Complex::<f32>::new(-2.0, -2.0),
            Complex::<f32>::new(0.0, -4.0),
        ];
        let truth_3 = (0..4)
            .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });
    }

    #[test]
    fn test_inverse_8() {
        let mut x = [
            Complex::<f32>::new(0.0, 8.0),
            Complex::<f32>::new(56.0, 64.0),
            Complex::<f32>::new(48.0, 56.0),
            Complex::<f32>::new(40.0, 48.0),
            Complex::<f32>::new(32.0, 40.0),
            Complex::<f32>::new(24.0, 32.0),
            Complex::<f32>::new(16.0, 24.0),
            Complex::<f32>::new(8.0, 16.0),
        ];
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            Complex::<f32>::new(28.0, 36.0),
            Complex::<f32>::new(-13.65685425, 5.65685425),
            Complex::<f32>::new(-8.0, 0.0),
            Complex::<f32>::new(-5.65685425, -2.34314575),
            Complex::<f32>::new(-4.0, -4.0),
            Complex::<f32>::new(-2.34314575, -5.65685425),
            Complex::<f32>::new(0.0, -8.0),
            Complex::<f32>::new(5.65685425, -13.65685425),
        ];
        let truth_2 = (0..8)
        .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
        .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn test_inverse_16() {
        let mut x = [
            Complex::<f32>::new(120.0, 136.0),
            Complex::<f32>::new(-48.21871594, 32.21871594),
            Complex::<f32>::new(-27.3137085, 11.3137085),
            Complex::<f32>::new(-19.9728461, 3.9728461),
            Complex::<f32>::new(-16.0, 0.0),
            Complex::<f32>::new(-13.3454291, -2.6545709),
            Complex::<f32>::new(-11.3137085, -4.6862915),
            Complex::<f32>::new(-9.59129894, -6.40870106),
            Complex::<f32>::new(-8.0, -8.0),
            Complex::<f32>::new(-6.40870106, -9.59129894),
            Complex::<f32>::new(-4.6862915, -11.3137085),
            Complex::<f32>::new(-2.6545709, -13.3454291),
            Complex::<f32>::new(0.0, -16.0),
            Complex::<f32>::new(3.9728461, -19.9728461),
            Complex::<f32>::new(11.3137085, -27.3137085),
            Complex::<f32>::new(32.21871594, -48.21871594),
        ];
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = (0..16)
        .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
        .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).square_norm() < f32::epsilon());
        });
    }

}

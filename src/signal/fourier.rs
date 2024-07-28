use crate::shared::{complex::Complex, float::Float, real::Real};

pub struct FastFourierTransform<C: Complex> {
    twiddles: Vec<C>,
}
impl<C: Complex> FastFourierTransform<C> {
    pub fn new(len: usize) -> Self {
        let twiddles = Self::generate_twiddles(len >> 1);
        FastFourierTransform { twiddles }
    }

    pub fn fft(&self, x: &mut [C]) {
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

    fn ifft(&self, x: &mut [C]) {
        x.iter_mut().for_each(|c| *c = c.conjugate());
        self.fft(x);
        let sf = C::Primitive::ONE / C::Primitive::usize(x.len());
        x.iter_mut().for_each(|c| *c = c.conjugate() * sf);
    }

    #[inline]
    fn generate_twiddles(dist: usize) -> Vec<C> {
        let angle = -C::Primitive::PI / C::Primitive::usize(dist);
        let mut twiddles = Vec::with_capacity(dist);
        (0..dist).for_each(|i| {
            let phase = angle * C::Primitive::usize(i);
            let (sin, cos) = phase.sin_cos();
            twiddles.push(C::new(cos, sin));
        });
        twiddles
    }

    #[inline]
    fn collapse_twiddles(twiddles: &mut Vec<C>) {
        let len = twiddles.len() >> 1;
        (0..len).for_each(|i| {
            twiddles[i] = twiddles[i << 1];
        });
        twiddles.resize(len, C::new(C::Primitive::ZERO, C::Primitive::ZERO));
    }

    #[inline]
    fn fft_chunk_n(x: &mut [C], twiddles: &[C], dist: usize) {
        let chunk_size = dist << 1;
        x.chunks_exact_mut(chunk_size).for_each(|chunk| {
            let (c_s0, c_s1) = chunk.split_at_mut(dist);
            c_s0
                .iter_mut()
                .zip(c_s1.iter_mut())
                .zip(twiddles)
                .for_each(|((c_s0, c_s1), w)| {
                    let temp = *c_s0 - *c_s1;
                    *c_s0 += *c_s1;
                    *c_s1 = temp * *w;
                });
        });
    }

    #[inline]
    fn fft_chunk_4(x: &mut [C]) {
        x.chunks_exact_mut(4).for_each(|chunk| {
            let (c_s0, c_s1) = chunk.split_at_mut(2);

            let temp = c_s0[0];
            c_s0[0] += c_s1[0];
            c_s1[0] = temp - c_s1[0];

            let temp = c_s0[1];
            c_s0[1] += c_s1[1];
            c_s1[1] = (temp - c_s1[1]).mul_ni();
        });
    }

    #[inline]
    fn fft_chunk_2(x: &mut [C]) {
        x.chunks_exact_mut(2).for_each(|chunk| {
            let temp = chunk[0];
            chunk[0] += chunk[1];
            chunk[1] = temp - chunk[1];
        });
    }
    #[inline]
    fn reverse(buf: &mut [C], log_n: usize) {
        let big_n = 1 << log_n;
        let half_n = big_n >> 1;
        let quart_n = big_n >> 2;
        let min = big_n - 1;

        let mut forward = half_n;
        let mut rev = 1;
        (0..quart_n).rev().for_each(|i: usize| {
            let zeros = i.trailing_ones();

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
    use crate::shared::{complex::ComplexFloat, float::Float};

    use super::*;

    #[test]
    fn forward_2_static() {
        let mut x = (0..2)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexFloat::<f32>::new(1.0, 3.0),
            ComplexFloat::<f32>::new(-1.0, -1.0),
        ];
        let truth_2 = [
            ComplexFloat::<f32>::new(0.0, 2.0),
            ComplexFloat::<f32>::new(2.0, 4.0),
        ];
        let truth_3 = [
            ComplexFloat::<f32>::new(2.0, 6.0),
            ComplexFloat::<f32>::new(-2.0, -2.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn forward_4_static() {
        let mut x = (0..4)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexFloat::<f32>::new(6.0, 10.0),
            ComplexFloat::<f32>::new(-4.0, 0.0),
            ComplexFloat::<f32>::new(-2.0, -2.0),
            ComplexFloat::<f32>::new(0.0, -4.0),
        ];
        let truth_2 = [
            ComplexFloat::<f32>::new(0.0, 4.0),
            ComplexFloat::<f32>::new(12.0, 16.0),
            ComplexFloat::<f32>::new(8.0, 12.0),
            ComplexFloat::<f32>::new(4.0, 8.0),
        ];
        let truth_3 = [
            ComplexFloat::<f32>::new(24.0, 40.0),
            ComplexFloat::<f32>::new(0.0, -16.0),
            ComplexFloat::<f32>::new(-8.0, -8.0),
            ComplexFloat::<f32>::new(-16.0, 0.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn forward_8_static() {
        let mut x = (0..8)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexFloat::<f32>::new(28.0, 36.0),
            ComplexFloat::<f32>::new(-13.65685425, 5.65685425),
            ComplexFloat::<f32>::new(-8.0, 0.0),
            ComplexFloat::<f32>::new(-5.65685425, -2.34314575),
            ComplexFloat::<f32>::new(-4.0, -4.0),
            ComplexFloat::<f32>::new(-2.34314575, -5.65685425),
            ComplexFloat::<f32>::new(0.0, -8.0),
            ComplexFloat::<f32>::new(5.65685425, -13.65685425),
        ];
        let truth_2 = [
            ComplexFloat::<f32>::new(0.0, 8.0),
            ComplexFloat::<f32>::new(56.0, 64.0),
            ComplexFloat::<f32>::new(48.0, 56.0),
            ComplexFloat::<f32>::new(40.0, 48.0),
            ComplexFloat::<f32>::new(32.0, 40.0),
            ComplexFloat::<f32>::new(24.0, 32.0),
            ComplexFloat::<f32>::new(16.0, 24.0),
            ComplexFloat::<f32>::new(8.0, 16.0),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.fft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn forward_16_static() {
        let mut x = (0..16)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexFloat::<f32>::new(120.0, 136.0),
            ComplexFloat::<f32>::new(-48.21871594, 32.21871594),
            ComplexFloat::<f32>::new(-27.3137085, 11.3137085),
            ComplexFloat::<f32>::new(-19.9728461, 3.9728461),
            ComplexFloat::<f32>::new(-16.0, 0.0),
            ComplexFloat::<f32>::new(-13.3454291, -2.6545709),
            ComplexFloat::<f32>::new(-11.3137085, -4.6862915),
            ComplexFloat::<f32>::new(-9.59129894, -6.40870106),
            ComplexFloat::<f32>::new(-8.0, -8.0),
            ComplexFloat::<f32>::new(-6.40870106, -9.59129894),
            ComplexFloat::<f32>::new(-4.6862915, -11.3137085),
            ComplexFloat::<f32>::new(-2.6545709, -13.3454291),
            ComplexFloat::<f32>::new(0.0, -16.0),
            ComplexFloat::<f32>::new(3.9728461, -19.9728461),
            ComplexFloat::<f32>::new(11.3137085, -27.3137085),
            ComplexFloat::<f32>::new(32.21871594, -48.21871594),
        ];

        fft.fft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn inverse_2_static() {
        let mut x = [
            ComplexFloat::<f32>::new(2.0, 6.0),
            ComplexFloat::<f32>::new(-2.0, -2.0),
        ];

        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexFloat::<f32>::new(0.0, 2.0),
            ComplexFloat::<f32>::new(2.0, 4.0),
        ];
        let truth_2 = [
            ComplexFloat::<f32>::new(1.0, 3.0),
            ComplexFloat::<f32>::new(-1.0, -1.0),
        ];
        let truth_3 = (0..2)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn inverse_4_static() {
        let mut x = [
            ComplexFloat::<f32>::new(24.0, 40.0),
            ComplexFloat::<f32>::new(0.0, -16.0),
            ComplexFloat::<f32>::new(-8.0, -8.0),
            ComplexFloat::<f32>::new(-16.0, 0.0),
        ];

        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexFloat::<f32>::new(0.0, 4.0),
            ComplexFloat::<f32>::new(12.0, 16.0),
            ComplexFloat::<f32>::new(8.0, 12.0),
            ComplexFloat::<f32>::new(4.0, 8.0),
        ];
        let truth_2 = [
            ComplexFloat::<f32>::new(6.0, 10.0),
            ComplexFloat::<f32>::new(-4.0, 0.0),
            ComplexFloat::<f32>::new(-2.0, -2.0),
            ComplexFloat::<f32>::new(0.0, -4.0),
        ];
        let truth_3 = (0..4)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_3).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn inverse_8_static() {
        let mut x = [
            ComplexFloat::<f32>::new(0.0, 8.0),
            ComplexFloat::<f32>::new(56.0, 64.0),
            ComplexFloat::<f32>::new(48.0, 56.0),
            ComplexFloat::<f32>::new(40.0, 48.0),
            ComplexFloat::<f32>::new(32.0, 40.0),
            ComplexFloat::<f32>::new(24.0, 32.0),
            ComplexFloat::<f32>::new(16.0, 24.0),
            ComplexFloat::<f32>::new(8.0, 16.0),
        ];
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = [
            ComplexFloat::<f32>::new(28.0, 36.0),
            ComplexFloat::<f32>::new(-13.65685425, 5.65685425),
            ComplexFloat::<f32>::new(-8.0, 0.0),
            ComplexFloat::<f32>::new(-5.65685425, -2.34314575),
            ComplexFloat::<f32>::new(-4.0, -4.0),
            ComplexFloat::<f32>::new(-2.34314575, -5.65685425),
            ComplexFloat::<f32>::new(0.0, -8.0),
            ComplexFloat::<f32>::new(5.65685425, -13.65685425),
        ];
        let truth_2 = (0..8)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });

        fft.ifft(&mut x);

        x.iter().zip(truth_2).for_each(|(xp, yp)| {
            assert_eq!(*xp, yp);
        });
    }

    #[test]
    fn inverse_16_static() {
        let mut x = [
            ComplexFloat::<f32>::new(120.0, 136.0),
            ComplexFloat::<f32>::new(-48.21871594, 32.21871594),
            ComplexFloat::<f32>::new(-27.3137085, 11.3137085),
            ComplexFloat::<f32>::new(-19.9728461, 3.9728461),
            ComplexFloat::<f32>::new(-16.0, 0.0),
            ComplexFloat::<f32>::new(-13.3454291, -2.6545709),
            ComplexFloat::<f32>::new(-11.3137085, -4.6862915),
            ComplexFloat::<f32>::new(-9.59129894, -6.40870106),
            ComplexFloat::<f32>::new(-8.0, -8.0),
            ComplexFloat::<f32>::new(-6.40870106, -9.59129894),
            ComplexFloat::<f32>::new(-4.6862915, -11.3137085),
            ComplexFloat::<f32>::new(-2.6545709, -13.3454291),
            ComplexFloat::<f32>::new(0.0, -16.0),
            ComplexFloat::<f32>::new(3.9728461, -19.9728461),
            ComplexFloat::<f32>::new(11.3137085, -27.3137085),
            ComplexFloat::<f32>::new(32.21871594, -48.21871594),
        ];
        let fft = FastFourierTransform::new(x.len());

        let truth_1 = (0..16)
            .map(|i| ComplexFloat::<f32>::new(i as f32, (i + 1) as f32))
            .collect::<Vec<_>>();

        fft.ifft(&mut x);
        x.iter().zip(truth_1).for_each(|(xp, yp)| {
            assert!((*xp - yp).l2_norm() < f32::EPSILON);
        });
    }
}

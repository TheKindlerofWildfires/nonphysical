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

    //Untested
    fn ifft(&self, x: &mut [Complex<T>]) {
        let twiddles = &mut self.twiddles.clone();

        x.iter_mut().for_each(|c| *c = (*c).conj());
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

        let sf = T::usize(1) / T::usize(x.len());
        x.iter_mut().for_each(|c| *c = (*c) * -sf);
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

                    (*c_s1).real = temp.real * w.real - temp.imag * w.imag;
                    (*c_s1).imag = temp.real * w.imag + temp.imag * w.real;
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
            complex_s1[1] = (temp - complex_s1[1]).swap().conj();
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

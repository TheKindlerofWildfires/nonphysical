use crate::shared::complex::Complex64;

pub struct FastFourierTransform {
    twiddles: Vec<Complex64>,
}
impl FastFourierTransform {
    pub fn new(len: usize) -> Self {
        let twiddles = Self::generate_twiddles(len >> 1);
        FastFourierTransform { twiddles }
    }

    pub fn fft(&self, x: &mut [Complex64]) {
        let twiddles = &mut self.twiddles.clone();

        let n: usize = x.len().ilog2() as usize;

        for t in (0..n).rev() {
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
        }
        Self::reverse(x, n);
    }

    #[inline]
    fn generate_twiddles(dist: usize) -> Vec<Complex64> {
        let angle = -std::f32::consts::PI / dist as f32;
        let mut twiddles = Vec::with_capacity(dist);
        for i in 0..dist {
            let phase = angle * i as f32;
            twiddles.push(Complex64::new(phase.cos(), phase.sin()));
        }
        twiddles
    }
    #[inline]
    fn fft_chunk_n(x: &mut [Complex64], twiddles: &[Complex64], dist: usize) {
        let chunk_size = dist << 1;

        x.chunks_exact_mut(chunk_size).for_each(|chunk| {
            let (complex_s0, complex_s1) = chunk.split_at_mut(dist);

            for ((c_s0, c_s1), w) in complex_s0
                .iter_mut()
                .zip(complex_s1.iter_mut())
                .zip(twiddles)
            {
                let temp = *c_s0 - *c_s1;
                *c_s0 = *c_s0 + *c_s1;

                (*c_s1).real = temp.real * w.real - temp.imaginary * w.imaginary;
                (*c_s1).imaginary = temp.real * w.imaginary + temp.imaginary * w.real;
            }
        });
    }
    #[inline]
    fn collapse_twiddles(twiddles: &mut Vec<Complex64>) {
        let len = twiddles.len() >> 1;
        for i in 0..len {
            twiddles[i] = twiddles[i << 1];
        }
        twiddles.resize(len, Complex64::new(0.0, 0.0));
    }

    #[inline]
    fn fft_chunk_4(x: &mut [Complex64]) {
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
    fn fft_chunk_2(x: &mut [Complex64]) {
        x.chunks_exact_mut(2).for_each(|chunk| {
            let temp = chunk[0];
            chunk[0] = chunk[0] + chunk[1];
            chunk[1] = temp - chunk[1];
        });
    }
    #[inline]
    fn reverse(buf: &mut [Complex64], log_n: usize) {

        let big_n = 1 << log_n;
        let half_n = big_n >> 1;
        let quart_n = big_n >> 2;
        let nmin1 = big_n - 1;

        let mut forward = half_n;
        let mut rev = 1;
        for i in (0..quart_n).rev(){
            let zeros = (i as usize).trailing_ones();

            forward ^= 2 << zeros;
            rev ^= quart_n >> zeros;

            if forward < rev {
                buf.swap(forward, rev);
                buf.swap(nmin1 ^ forward, nmin1 ^ rev);
            }

            buf.swap(forward ^ 1,  rev ^ half_n);
        }
    }
}

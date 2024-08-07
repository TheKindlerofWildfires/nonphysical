use crate::shared::{complex::Complex, primitive::Primitive};
use super::FourierTransform;
pub struct ComplexFourierTransformStack<C: Complex, const N: usize> {
    pub twiddles: [C;N],
}
impl<C: Complex,const N: usize> FourierTransform<C> for ComplexFourierTransformStack <C,N> {
    type FourierInit = ();
    fn new(_: Self::FourierInit) -> Self {
        let twiddles = Self::generate_twiddles();
        Self { twiddles }
    }

    fn fft(&self, x: &mut [C]) {
        let n: usize = x.len().ilog2() as usize;
        let mut step = 1;
        (0..n).rev().for_each(|t| {
            let dist = 1 << t;
            let chunk_size = dist << 1;

            if chunk_size > 4 {
                Self::fft_chunk_n(x, & self.twiddles, dist,step);
                step<<=1;
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
}
impl<C:Complex,const N: usize> ComplexFourierTransformStack<C,N>{
    #[inline]
    pub fn generate_twiddles() -> [C;N] {
        let angle = -C::Primitive::PI / C::Primitive::usize(N);
        let mut twiddles = [C::ZERO;N];
        twiddles.iter_mut().enumerate().for_each(|(i,twiddle)|{
            let phase = angle * C::Primitive::usize(i);
            let (sin, cos) = phase.sin_cos();
            *twiddle = C::new(cos, sin);
        });
        twiddles
    }

    #[inline]
    fn fft_chunk_n(x: &mut [C], twiddles: &[C], dist: usize,step:usize) {
        let chunk_size = dist << 1;
        x.chunks_exact_mut(chunk_size).for_each(|chunk| {
            let (c_s0, c_s1) = chunk.split_at_mut(dist);
            c_s0
                .iter_mut()
                .zip(c_s1.iter_mut())
                .zip(twiddles.iter().step_by(step))
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
    pub fn reverse(buf: &mut [C], log_n: usize) {
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
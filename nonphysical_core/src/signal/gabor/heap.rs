use alloc::vec::Vec;

use crate::{shared::{complex::Complex, matrix::{heap::MatrixHeap, Matrix}}, signal::fourier::{heap::ComplexFourierTransformHeap, FourierTransform}};

use super::GaborTransform;

pub struct GaborTransformHeap<C: Complex> {
    over_sample: usize,
    window: Vec<C>,
    fourier: ComplexFourierTransformHeap<C>,
}

impl<C: Complex> GaborTransform<C> for GaborTransformHeap<C> {
    type GaborInit = (usize, Vec<C>);
    type GaborMatrix = MatrixHeap<C>;
    fn new(init: Self::GaborInit) -> Self {
        let (over_sample, window) = init;
        let fourier = ComplexFourierTransformHeap::new(window.len());
        GaborTransformHeap {
            over_sample,
            window,
            fourier,
        }
    }

    fn gabor(&self, x: &mut [C]) -> Self::GaborMatrix {
        let win_step = self.window.len() / self.over_sample;
        let win_count = x.len() / win_step - self.over_sample + 1;
        let size = win_count * self.window.len();
        let mut gabor_data = Vec::with_capacity(size);
        (0..win_count).for_each(|i| {
            gabor_data.extend_from_slice(&x[i * win_step..i * win_step + self.window.len()])
        });

        gabor_data
            .chunks_exact_mut(self.window.len())
            .for_each(|g_chunk| {
                Self::convolve(g_chunk, &self.window);
                self.fourier.fft(g_chunk);
            });

        MatrixHeap::new((self.window.len(), gabor_data))
    }


}
impl<C:Complex> GaborTransformHeap<C>{
    #[inline(always)]
    pub fn square_len(&self) -> usize {
        self.window.len() / self.over_sample * (self.window.len() - 1 + self.over_sample)
    }

    #[inline(always)]
    fn convolve(x: &mut [C], y: &[C]) {
        x.iter_mut()
            .zip(y)
            .for_each(|(xi, yi)| *xi *= yi.conjugate());
    }
}
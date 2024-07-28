use crate::shared::complex::Complex;
use crate::shared::matrix::Matrix;
use crate::signal::fourier::FastFourierTransform;
pub struct GaborTransform<C: Complex> {
    over_sample: usize,
    window: Vec<C>,
    fourier: FastFourierTransform<C>,
}

impl<C: Complex> GaborTransform<C> {
    pub fn new(over_sample: usize, window: Vec<C>) -> Self {
        let fourier = FastFourierTransform::new(window.len());
        GaborTransform {
            over_sample,
            window,
            fourier,
        }
    }

    pub fn gabor(&self, x: &mut [C]) -> Matrix<C> {
        let win_step = self.window.len() / self.over_sample;
        let win_count = x.len() / win_step - self.over_sample + 1;
        let size = win_count * self.window.len();
        let mut gabor_data = Vec::with_capacity(size);
        (0..win_count).for_each(|i| {
            gabor_data.extend_from_slice(&x[i*win_step..i*win_step+self.window.len()])
        });

        gabor_data
            .chunks_exact_mut(self.window.len())
            .for_each(|g_chunk| {
                Self::convolve(g_chunk, &self.window);
                self.fourier.fft(g_chunk);
            });

        Matrix::new(self.window.len(), gabor_data)
    }

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

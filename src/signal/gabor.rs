use crate::shared::complex::Complex64;
use crate::signal::fourier::FastFourierTransform;
pub struct GaborTransform {
    over_sample: usize,
    window: Vec<Complex64>,
    fourier: FastFourierTransform,
}

impl GaborTransform {
    pub fn new(over_sample: usize, window: Vec<Complex64>) -> GaborTransform {
        let fourier = FastFourierTransform::new(window.len());
        GaborTransform {
            over_sample,
            window,
            fourier,
        }
    }
    pub fn gaussian(nfft: usize, std: f32) -> Vec<Complex64> {
        let sigma2 = 2.0 * std * std;
        let sub = (nfft as f32 - 1.0) / 2.0;
        (0..nfft)
            .map(|i| Complex64::new(((i as f32 - sub) / sigma2).exp(), 0.0))
            .collect()
    }

    pub fn gabor(&self, x: &mut [Complex64]) -> Vec<Complex64> {
        let win_step = self.window.len() / self.over_sample;
        let win_count = x.len() / win_step;
        let size = self.window.len() * win_count;
        let mut gabor: Vec<Complex64> = Vec::with_capacity(size);

        for i in 0..win_count {
            gabor.extend_from_slice(&x[i * win_step..i * win_step + self.window.len()]);
            Self::convolve(
                &mut gabor[i * self.window.len()..(i + 1) * self.window.len()],
                &self.window,
            );

            self.fourier
                .fft(&mut gabor[i * self.window.len()..(i + 1) * self.window.len()]);
        }
        gabor
    }
    fn convolve(x: &mut [Complex64], y: &[Complex64]) {
        x.iter_mut().zip(y).for_each(|(xi, yi)| *xi = *xi * *yi);
    }
}
//baseline

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
        let sub = (nfft >> 1) as f32 - 0.5;
        let mut window = Vec::with_capacity(nfft);
        for i in 0..nfft {
            let value = ((i as f32 - sub) / sigma2).exp();
            window.push(Complex64::new(value, 0.0));
        }
        window
    }

    pub fn gabor(&self, x: &mut [Complex64]) -> Vec<Complex64> {
        let win_step = self.window.len() / self.over_sample;
        let win_count = x.len() / win_step - self.over_sample + 1;
        let size = x.len() * self.over_sample;
        let mut gabor = Vec::with_capacity(size);

        for i in 0..win_count {
            let gs = i * self.window.len();
            let ge = gs + self.window.len();
            gabor.extend_from_slice(&x[i * win_step..i * win_step + self.window.len()]);
            Self::convolve(&mut gabor[gs..ge], &self.window);
            self.fourier.fft(&mut gabor[gs..ge]);
        }
        gabor
    }

    pub fn square_len(&self) -> usize {
        self.window.len() / self.over_sample * (self.window.len() - 1 + self.over_sample)
    }
    #[inline(always)]
    fn convolve(x: &mut [Complex64], y: &[Complex64]) {
        x.iter_mut()
            .zip(y)
            .for_each(|(xi, yi)| *xi = *xi * yi.conj());
    }
}
//baseline

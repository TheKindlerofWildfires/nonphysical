use crate::shared::complex::{Complex, Float};
use crate::shared::matrix::Matrix;
use crate::signal::fourier::FastFourierTransform;
pub struct GaborTransform<T: Float> {
    over_sample: usize,
    window: Vec<Complex<T>>,
    fourier: FastFourierTransform<T>,
}

impl<T: Float> GaborTransform<T> {
    pub fn new(over_sample: usize, window: Vec<Complex<T>>) -> Self {
        let fourier = FastFourierTransform::new(window.len());
        GaborTransform {
            over_sample,
            window,
            fourier,
        }
    }
    pub fn gaussian(nfft: usize, std: T) -> Vec<Complex<T>> {
        let sigma2 = T::usize(2) * std * std;
        let sub = T::usize(nfft >> 1) - T::float(0.5);
        let mut window = Vec::with_capacity(nfft);
        for i in 0..nfft {
            let value = ((T::usize(i) - sub) / sigma2).exp();
            window.push(Complex::<T>::new(value, T::usize(0)));
        }
        window
    }

    pub fn gabor(&self, x: &mut [Complex<T>]) -> Matrix<T> {
        let win_step = self.window.len() / self.over_sample;
        let win_count = x.len() / win_step - self.over_sample + 1;
        let size = x.len() * self.over_sample;
        let mut gabor_data = Vec::with_capacity(size);

        for i in 0..win_count {
            let gs = i * self.window.len();
            let ge = gs + self.window.len();
            gabor_data.extend_from_slice(&x[i * win_step..i * win_step + self.window.len()]);
            self.convolve(&mut gabor_data[gs..ge], &self.window);
            self.fourier.fft(&mut gabor_data[gs..ge]);
        }
        Matrix::new(self.window.len(), gabor_data)
    }

    pub fn square_len(&self) -> usize {
        self.window.len() / self.over_sample * (self.window.len() - 1 + self.over_sample)
    }
    #[inline(always)]
    fn convolve(&self, x: &mut [Complex<T>], y: &[Complex<T>]) {
        x.iter_mut()
            .zip(y)
            .for_each(|(xi, yi)| *xi = *xi * yi.conj());
    }
}

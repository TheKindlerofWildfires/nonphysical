use crate::shared::float::Float;

pub mod fourier_heap;
pub mod fourier_stack;
pub trait FourierTransform<F:Float>{
    type FourierInit;
    fn new(init: Self::FourierInit) -> Self;
    fn fft(&self, x: &mut [F]);
    fn ifft(&self, x: &mut [F]);
    fn shift(x: &mut [F]);
}

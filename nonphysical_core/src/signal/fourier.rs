use crate::shared::float::Float;

pub mod heap;
pub mod stack;
pub trait FourierTransform<F:Float>{
    type FourierInit;
    fn new(init: Self::FourierInit) -> Self;
    fn fft(&self, x: &mut [F]);
    fn ifft(&self, x: &mut [F]);
}
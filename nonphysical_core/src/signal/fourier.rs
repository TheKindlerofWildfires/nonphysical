use crate::shared::float::Float;

pub mod fourier_heap;
pub trait FourierTransform<F:Float>{
    type FourierInit;
    fn new(init: Self::FourierInit) -> Self;
    fn forward(&self, x: &mut [F]);
    fn backward(&self, x: &mut [F]);
    fn forward_shifted(&self, x: &mut [F]);
    fn backward_shifted(&self, x: &mut [F]);
}

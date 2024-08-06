use crate::shared::complex::Complex;
use crate::shared::matrix::Matrix;
use alloc::vec::Vec;

pub mod heap;
pub trait GaborTransform<C:Complex>{
    type GaborInit;
    type GaborMatrix: Matrix<C>;
    fn new(init: Self::GaborInit) -> Self;
    fn gabor(&self, x: &mut [C]) -> Self::GaborMatrix;
}


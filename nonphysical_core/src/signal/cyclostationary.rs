use crate::shared::complex::Complex;
use crate::shared::matrix::Matrix;
pub mod cyclostationary_heap;
pub trait CycloStationaryTransform<C:Complex>{
    type CycloInit;
    type Matrix: Matrix<C>;
    fn new(init: Self::CycloInit) -> Self;
    fn fam(&self, x: &mut [C]) -> Self::Matrix;
}


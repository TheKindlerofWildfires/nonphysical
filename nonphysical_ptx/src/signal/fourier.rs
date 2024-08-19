use crate::cuda::global::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_core::shared::complex::Complex;

#[cfg(target_arch = "nvptx64")]
pub mod fourier_ptx;

#[cfg(not(target_arch = "nvptx64"))]
pub mod fourier_cuda;

pub struct FourierArguments<'a, C: Complex> {
    pub x: CuGlobalSliceRef<'a, C>,
    pub twiddles: CuGlobalSlice<'a, C>,
}

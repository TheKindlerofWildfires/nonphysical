use crate::cuda::global::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_core::shared::complex::Complex;

#[cfg(target_arch = "nvptx64")]
pub mod wavelet_ptx;

#[cfg(not(target_arch = "nvptx64"))]
pub mod wavelet_cuda;

pub struct WaveletArguments<'a, C: Complex> {
    pub x: CuGlobalSliceRef<'a, C>,
    pub ndwt: CuGlobalSlice<'a,usize>,
}

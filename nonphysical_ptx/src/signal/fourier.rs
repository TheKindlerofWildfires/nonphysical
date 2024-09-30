#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::{CuGlobalSlice, CuGlobalSliceRef};


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice,CuGlobalSliceRef};
use nonphysical_core::shared::complex::Complex;

#[cfg(target_arch = "nvptx64")]
pub mod ptx_fourier;

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda_fourier;

pub struct FourierArguments<'a, C: Complex> {
    pub x: CuGlobalSliceRef<'a, C>,
    pub twiddles: CuGlobalSlice<'a, C>,
}
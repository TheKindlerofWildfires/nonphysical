#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::{CuGlobalSlice, CuGlobalSliceRef};


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice,CuGlobalSliceRef};
use nonphysical_core::shared::float::Float;

#[cfg(target_arch = "nvptx64")]
pub mod ptx_wavelet;

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda_wavelet;

pub struct WaveletArguments<'a, F: Float> {
    pub coefficients: CuGlobalSlice<'a,F>,
    pub input: CuGlobalSlice<'a, F>,
    pub output: CuGlobalSliceRef<'a, F>,
}
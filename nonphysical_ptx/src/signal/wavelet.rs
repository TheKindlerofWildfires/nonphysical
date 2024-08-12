use crate::cuda::cu_slice::{CuSlice, CuSliceRef};
use nonphysical_core::shared::complex::Complex;

#[cfg(target_arch = "nvptx64")]
pub mod ptx;

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda;

pub struct WaveletArguments<'a, C: Complex> {
    pub x: CuSliceRef<'a, C>,
    pub ndwt: CuSlice<'a,usize>,
}

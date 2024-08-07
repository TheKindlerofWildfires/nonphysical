/*

    Takes in the data as a mutable slice,

*/
//#[cfg(target_arch = "nvptx64")]
//use core::arch::nvptx;

use crate::cuda::cu_slice::{CuSlice, CuSliceRef};
use nonphysical_core::shared::complex::Complex;
#[cfg(target_arch = "nvptx64")]
pub mod ptx;

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda;

pub struct FourierArguments<'a, C: Complex> {
    pub x: CuSliceRef<'a, C>,
    pub twiddles: CuSlice<'a, C>,
}

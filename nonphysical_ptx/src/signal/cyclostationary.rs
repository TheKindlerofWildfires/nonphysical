#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::{CuGlobalSlice, CuGlobalSliceRef};


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice,CuGlobalSliceRef};
use nonphysical_core::shared::complex::Complex;

#[cfg(target_arch = "nvptx64")]
pub mod ptx_cyclostationary;

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda_cyclostationary;

pub struct CyclostationaryIntermediateArguments<'a, C: Complex> {
    pub x: CuGlobalSliceRef<'a, C>,
    pub twiddles: CuGlobalSlice<'a, C>,
    pub phase: CuGlobalSlice<'a, C>,
}
pub struct CyclostationaryCompleteArguments<'a, C: Complex> {
    pub x: CuGlobalSliceRef<'a, C>,
    pub twiddles: CuGlobalSlice<'a, C>,
    pub result: CuGlobalSliceRef<'a, C>,
}

//Plan is precalc phase, twiddles
//Submit x vector to produce intermediate form a lot like mass fourier
//Precalc a new set of twiddles
//submit x,dest matrix, new twiddles, slice through where each grid is a col outer loop (so can keep col in sm)
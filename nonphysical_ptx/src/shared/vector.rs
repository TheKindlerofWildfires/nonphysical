use nonphysical_core::shared::float::Float;

#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::{CuGlobalSlice,CuGlobalSliceRef};


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice,CuGlobalSliceRef};


#[cfg(target_arch = "nvptx64")]
pub mod vector_ptx;

#[cfg(not(target_arch = "nvptx64"))]
pub mod vector_driver;


//This number has huge perf consequences and is hard to know
//Ideally I'd run a pre-build step that gets a curve of best fit, picks points on it, and then generates the right kernels based on expected range
//Then at runtime picks the right kernel based on size
const CYCLE_COMPARE: usize = 128;


pub struct VectorArgumentsReduce<'a, F: Float> {
    pub data: CuGlobalSlice<'a, F>,
    pub acc: CuGlobalSliceRef<'a, F>,
}

pub struct VectorArgumentsReducePrim<'a, F: Float> {
    pub data: CuGlobalSlice<'a, F>,
    pub acc: CuGlobalSliceRef<'a, F::Primitive>,
}

pub struct VectorArgumentsMap<'a, F: Float> {
    pub data: CuGlobalSliceRef<'a, F>,
    pub map: CuGlobalSlice<'a, F>,
}

pub struct VectorArgumentsMapReduce<'a, F: Float> {
    pub data: CuGlobalSliceRef<'a, F>,
    pub map: CuGlobalSlice<'a, F>,
    pub acc: CuGlobalSliceRef<'a, F>,
}
use nonphysical_core::shared::float::Float;

#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::{CuGlobalSlice,CuGlobalSliceRef};


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice,CuGlobalSliceRef};


#[cfg(target_arch = "nvptx64")]
pub mod ptx_vector;

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda_vector;


pub struct VectorArgumentsReduce<'a, Fa: Float, Fb: Float> {
    pub data: CuGlobalSlice<'a, Fa>,
    pub acc: CuGlobalSliceRef<'a, Fb>,
}

pub struct VectorArgumentsMap<'a, Fa: Float, Fb: Float, Fc:Float> {
    pub data: CuGlobalSlice<'a, Fa>,
    pub output: CuGlobalSliceRef<'a, Fb>,
    pub map: CuGlobalSlice<'a, Fc>,
}

pub struct VectorArgumentsApply<'a, Fa: Float, Fb: Float> {
    pub data: CuGlobalSliceRef<'a, Fa>,
    pub map: CuGlobalSlice<'a, Fb>,
}

pub struct VectorArgumentsMapReduce<'a, Fa: Float, Fb: Float, Fc:Float>  {
    pub data: CuGlobalSlice<'a, Fa>,
    pub acc: CuGlobalSliceRef<'a, Fb>,
    pub map: CuGlobalSlice<'a, Fc>,

}


//solution to the prim problem is to mark each of them with F1/F2/F3 so it doesn't matter and carry it in invoke
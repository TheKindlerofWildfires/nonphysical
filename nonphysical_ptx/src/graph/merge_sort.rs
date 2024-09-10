use nonphysical_core::shared::float::Float;

#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::{CuGlobalSlice, CuGlobalSliceRef};
#[cfg(target_arch = "nvptx64")]
use crate::shared::unsigned::U32;


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice,CuGlobalSliceRef};

#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_std::shared::unsigned::U32;


#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda_merge_sort;

#[cfg(target_arch = "nvptx64")]
pub mod ptx_merge_sort;


pub struct MergeSortArgumentsLocal<'a, Fa:Float>{
    pub src: CuGlobalSliceRef<'a,Fa>,
    pub dst: CuGlobalSliceRef<'a, Fa>,
}

pub struct MergeSortArgumentsGlobal<'a, Fa:Float>{
    pub src: CuGlobalSliceRef<'a,Fa>,
    pub dst: CuGlobalSliceRef<'a, Fa>,
    pub params: CuGlobalSlice<'a,U32>,
}


const INSERT_SIZE: usize = 4;
const THREAD_MAX: usize = 8;

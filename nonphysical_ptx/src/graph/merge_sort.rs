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

/* 
const CTA_SIZE: usize = 4;
const BLOCK_SIZE: usize = 256;
const K_BITS: usize = 8;
const N_BINS: usize = 1<<K_BITS;
const EPB: usize = 2*CTA_SIZE*BLOCK_SIZE;

const CTA_SIZE_32: U32 = U32(4);
const BLOCK_SIZE_32: U32 =  U32(256);
const K_BITS_32: U32 =  U32(8);
const N_BINS_32: U32 =  U32(1<<K_BITS);
const EPB_32: U32 =  U32((2*CTA_SIZE*BLOCK_SIZE) as u32 );
*/
const INSERT_SIZE: usize = 4;
const THREAD_MAX: usize = 8;

/*
    Design is theres about 12288 bytes of SM, and merge sort takes aux space N
    division is as follows
        f32 - 8 per thread, 32 per warp, 16 per block
        f64 - 4 per thread, 32 per warp, 16 per block
        c32 - 4 per thread, 32 per warp, 16 per block
        c64 - 2 per thread, 32 per warp, 16 per block

    for points divide out block size until it becomes 1 (support up to D=16)
    could support up to D=32-128 if crashed down per threads... but meh 
*/
use nonphysical_core::shared::float::Float;
#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda_hash_table;

#[cfg(target_arch = "nvptx64")]
pub mod ptx_hash_table;


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_std::shared::unsigned::U32;
#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::CuGlobalSliceRef;

#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::CuGlobalSliceRef;


#[cfg(target_arch = "nvptx64")]
use crate::shared::unsigned::U32;


pub struct HashTableArguments<'a, Fa:Float,T:Copy>{
    pub keys: CuGlobalSliceRef<'a,Fa>,
    pub values: CuGlobalSliceRef<'a,T>,
    pub table_keys: CuGlobalSliceRef<'a, Fa>,
    pub table_values: CuGlobalSliceRef<'a, T>,
    pub ctr: CuGlobalSliceRef<'a,U32>,
}



const CAPACITY: u32 = 1024*1024*128;

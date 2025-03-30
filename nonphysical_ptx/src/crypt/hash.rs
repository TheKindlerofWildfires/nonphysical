#[cfg(target_arch = "nvptx64")]
use crate::cuda::global::device::{CuGlobalSlice, CuGlobalSliceRef};

#[cfg(target_arch = "nvptx64")]
use crate::shared::unsigned::U8;

#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice, CuGlobalSliceRef};
#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_std::shared::unsigned::U8;

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda;
#[cfg(target_arch = "nvptx64")]
pub mod ptx;
#[cfg(target_arch = "nvptx64")]
pub mod ptx_md4;
#[cfg(target_arch = "nvptx64")]
pub mod ptx_md5;
#[cfg(target_arch = "nvptx64")]
pub mod ptx_sha1;
#[cfg(target_arch = "nvptx64")]
pub mod ptx_sha2;

pub struct HashArguments<'a> {
    pub target: CuGlobalSlice<'a, U8>,
    pub base: CuGlobalSlice<'a, U8>,
    pub hit: CuGlobalSliceRef<'a, U8>,
}
/*
Plan is
    Move target into shared memory at start ->
    create a state for each sub thread to work against & a/b/c/d in shared memory


*/

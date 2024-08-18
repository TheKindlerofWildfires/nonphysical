use crate::cuda::ffi::CUstream;

pub struct CuStream{
    pub stream: CUstream,
}
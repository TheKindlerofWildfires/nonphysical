use crate::cuda::{
    error::CuError,
    ffi::{cuStreamCreate, cuStreamDestroy_v2, CUstream,CUstream_flags_enum,CUstream_flags_enum_CU_STREAM_NON_BLOCKING,CUstream_flags_enum_CU_STREAM_DEFAULT},
};
use std::ptr;

use super::ffi::cuStreamSynchronize;

pub struct CuStream {
    pub stream: CUstream,
}

impl CuStream {
    pub fn new(flags: CUstream_flags_enum) -> Self {
        let mut stream = ptr::null_mut(); 
        CuError::check(unsafe { cuStreamCreate(&mut stream, flags as u32) }).expect("Failed to create a stream");
        Self {stream}
    }

    pub fn blocking()-> Self{
        let mut stream = ptr::null_mut(); 
        CuError::check(unsafe { cuStreamCreate(&mut stream, CUstream_flags_enum_CU_STREAM_DEFAULT as u32) }).expect("Failed to create a stream");
        Self {stream}
    }
    pub fn non_blocking()-> Self{
        let mut stream = ptr::null_mut(); 
        CuError::check(unsafe { cuStreamCreate(&mut stream, CUstream_flags_enum_CU_STREAM_NON_BLOCKING as u32) }).expect("Failed to create a stream");
        Self {stream}
    }
    pub fn sync(&self){
        CuError::check(unsafe { cuStreamSynchronize(self.stream) }).expect("Failed to sync a stream");
    }
}

impl Drop for CuStream {
    fn drop(&mut self) {
        CuError::check(unsafe { cuStreamDestroy_v2(self.stream) }).expect("Failed to destroy a stream");
    }
}

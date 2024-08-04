#[cfg(not(target_arch = "nvptx64"))]
use super::{ffi,runtime::CuError};

#[cfg(not(target_arch = "nvptx64"))]
use alloc::vec;

#[cfg(not(target_arch = "nvptx64"))]
use alloc::vec::Vec;

#[cfg(target_arch = "nvptx64")]
use core::ops::{Deref,DerefMut};

#[cfg(not(target_arch = "nvptx64"))]
use std::time::SystemTime;
#[cfg(not(target_arch = "nvptx64"))]
use std::dbg;

pub struct CuSlice<'a, T> {
    ptr: &'a [T],
}

pub struct CuSliceRef<'a, T> {
    ptr: &'a mut [T],
}
#[cfg(target_arch = "nvptx64")]
impl<'a, T> Deref for CuSlice<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

#[cfg(target_arch = "nvptx64")]
impl<'a, T> Deref for CuSliceRef<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

#[cfg(target_arch = "nvptx64")]
impl<'a, T> DerefMut for CuSliceRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}

impl<'a, T> CuSlice<'a, T> {
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn new(ptr: ffi::DevicePtr, len: usize) -> CuSlice<'a, T> {
        unsafe {
            CuSlice {
                ptr: core::slice::from_raw_parts(ptr as *const T, len),
            }
        }
    }

    // get reference
    #[cfg(target_arch = "nvptx64")]
    pub fn get(&self) -> &[T] {
        self.ptr
    }

    // get host T
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn to_host(&self) -> Result<Vec<T>, CuError> {
        let mut res = vec![];
        unsafe {
            for _ in 0..self.ptr.len() {
                res.push(core::mem::MaybeUninit::zeroed().assume_init());
            }
        }
        let result = unsafe {
            ffi::cuMemcpyDtoH_v2(
                res.as_mut_ptr() as *mut std::os::raw::c_void,
                self.ptr.as_ptr() as ffi::DevicePtr,
                std::mem::size_of::<T>() * self.ptr.len(),
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CuError::new(result));
        }
        Ok(res)
    }
}


impl<'a, T> CuSliceRef<'a, T> {
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn new(ptr: ffi::DevicePtr, len: usize) -> CuSliceRef<'a, T> {
        unsafe {
            CuSliceRef {
                ptr: core::slice::from_raw_parts_mut(ptr as *mut T, len),
            }
        }
    }

    // get reference
    #[cfg(target_arch = "nvptx64")]
    pub fn get(&mut self) -> &mut [T] {
        self.ptr
    }

    // get host T
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn to_host(&self) -> Result<Vec<T>, CuError> {
        let now = SystemTime::now();
        unsafe{ffi::cuCtxSynchronize();}
        dbg!(now.elapsed());
        let now = SystemTime::now();
        let mut res = Vec::with_capacity(self.ptr.len());
        unsafe {
            for _ in 0..self.ptr.len() {
                res.push(core::mem::MaybeUninit::zeroed().assume_init());
            }
        }
        dbg!(now.elapsed());
        let now = SystemTime::now();

        let result = unsafe {
            ffi::cuMemcpyDtoH_v2(
                res.as_mut_ptr() as *mut std::os::raw::c_void,
                self.ptr.as_ptr() as ffi::DevicePtr,
                std::mem::size_of::<T>() * self.ptr.len(),
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CuError::new(result));
        }
        dbg!(now.elapsed());
        Ok(res)
    }
}
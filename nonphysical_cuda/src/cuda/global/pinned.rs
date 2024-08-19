use crate::cuda::{error::CuError, ffi};
use std::{ffi::c_void, ptr, slice::from_raw_parts_mut};

use super::host::{CuGlobalBox, CuGlobalBoxRef, CuGlobalSlice, CuGlobalSliceRef};

pub struct CuPinnedSlice<'a, T: Copy> {
    pub ptr: &'a mut [T],
}

pub struct CuPinnedSliceRef<'a, T: Copy> {
    pub ptr: &'a mut [T],
}

//This pins memory to the CPU for fast write to device access
impl<'a, T: Copy> CuPinnedSlice<'a, T> {
    pub fn alloc(data: &[T]) -> Self {
        let size = size_of::<T>() * data.len();
        let mut ptr = ptr::null_mut();
        CuError::check(unsafe {
            ffi::cuMemHostAlloc(
                &mut ptr as *mut _ as *mut *mut c_void,
                size,
                ffi::CU_MEMHOSTALLOC_WRITECOMBINED| ffi::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        })
        .expect("Failed to allocate a pinned buffer");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }
    pub fn to_global(&mut self) -> CuGlobalSlice<'a, T> {
        let mut d_ptr = 0u64;
        CuError::check(unsafe {
            ffi::cuMemHostGetDevicePointer_v2(&mut d_ptr, self.ptr as *mut _ as *mut c_void, 0)
        })
        .expect("Failed to get device pointer from pinned buffer");

        CuGlobalSlice {
            ptr: unsafe { from_raw_parts_mut(d_ptr as *mut T, self.ptr.len()) },
        }
    }
    pub fn store(&mut self, data: &[T]) {
        assert!(data.len() == self.ptr.len());
        self.ptr.copy_from_slice(data);
    }
}

/*
impl<'a, T: Copy> Drop for CuPinnedSlice<'a, T> {
    fn drop(&mut self) {
        CuError::check(unsafe { ffi::cuMemFreeHost(self.ptr.as_mut_ptr() as *mut c_void) })
            .expect("Failed to free a pinned buffer");
    }
}*/
//This pins memory to the CPU for fast read on host access
impl<'a, T: Copy> CuPinnedSliceRef<'a, T> {
    pub fn alloc(data: &[T]) -> Self {
        let size = size_of::<T>() * data.len();
        let mut ptr = ptr::null_mut();
        CuError::check(unsafe {
            ffi::cuMemHostAlloc(&mut ptr as *mut _ as *mut *mut c_void, size, 0| ffi::CU_MEMHOSTALLOC_DEVICEMAP)
        })
        .expect("Failed to allocate a pinned buffer");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }
    pub fn to_global(&mut self) -> CuGlobalSliceRef<'a, T> {
        let mut d_ptr = 0u64;
        CuError::check(unsafe {
            ffi::cuMemHostGetDevicePointer_v2(&mut d_ptr, self.ptr as *mut _ as *mut c_void, 0)
        })
        .expect("Failed to get device pointer from pinned buffer");

        CuGlobalSliceRef {
            ptr: unsafe { from_raw_parts_mut(d_ptr as *mut T, self.ptr.len()) },
        }
    }

    pub fn store(&mut self, data: &[T]) {
        assert!(data.len() == self.ptr.len());
        self.ptr.copy_from_slice(data);
    }

    pub fn load(&self, data: &mut [T]) {
        assert!(data.len() == self.ptr.len());
        data.copy_from_slice(self.ptr);
    }
}
/*
//Rust 'helps' here by automatically cleaning up the memory
impl<'a, T: Copy> Drop for CuPinnedSliceRef<'a, T> {
    fn drop(&mut self) {
        CuError::check(unsafe { ffi::cuMemFreeHost(self.ptr.as_mut_ptr() as *mut c_void) })
            .expect("Failed to free a pinned buffer");
    }
} */

pub struct CuPinnedBox<'a, T: Copy> {
    pub ptr: &'a mut T,
}

pub struct CuPinnedBoxRef<'a, T: Copy> {
    pub ptr: &'a mut T,
}

//This pins memory to the CPU for fast write to device access
impl<'a, T: Copy> CuPinnedBox<'a, T> {
    pub fn alloc(_: &'a T) -> Self {
        let size = size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();
        CuError::check(unsafe {
            ffi::cuMemHostAlloc(
                &mut ptr,
                size,
                ffi::CU_MEMHOSTALLOC_WRITECOMBINED | ffi::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        })
        .expect("Failed to allocate a pinned buffer");
        Self {
            ptr: unsafe {
                (ptr as *mut T)
                    .as_mut()
                    .expect("Failed to convert pinned box to pointer")
            },
        }
    }

    pub fn to_global(&mut self) -> CuGlobalBox<'a, T> {
        let mut d_ptr = 0u64;
        CuError::check(unsafe {
            ffi::cuMemHostGetDevicePointer_v2(&mut d_ptr, self.ptr as *mut _ as *mut c_void, 0)
        })
        .expect("Failed to get device pointer from pinned buffer");

        CuGlobalBox {
            ptr: unsafe { (d_ptr as *mut T).as_mut() }.expect("Failed to convert global buffer"),
        }
    }

    pub fn store(&mut self, data: T) {
        *self.ptr = data;
    }
}
/*
impl<'a, T: Copy> Drop for CuPinnedBox<'a, T> {
    fn drop(&mut self) {
        CuError::check(unsafe { ffi::cuMemFreeHost(self.ptr as *mut T as *mut c_void) })
            .expect("Failed to free a pinned buffer");
    }
}*/
//This pins memory to the CPU for fast read on host access
impl<'a, T: Copy> CuPinnedBoxRef<'a, T> {
    pub fn alloc(_: &'a T) -> Self {
        let size = size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();
        CuError::check(unsafe {
            ffi::cuMemHostAlloc(&mut ptr, size, 0 | ffi::CU_MEMHOSTALLOC_DEVICEMAP)
        })
        .expect("Failed to allocate a pinned buffer");
        Self {
            ptr: unsafe {
                (ptr as *mut T)
                    .as_mut()
                    .expect("Failed to convert pinned box to pointer")
            },
        }
    }
    pub fn to_global(&mut self) -> CuGlobalBoxRef<'a, T> {
        let mut d_ptr = 0u64;
        CuError::check(unsafe {
            ffi::cuMemHostGetDevicePointer_v2(&mut d_ptr, self.ptr as *mut _ as *mut c_void, 0)
        })
        .expect("Failed to get device pointer from pinned buffer");

        CuGlobalBoxRef {
            ptr: unsafe { (d_ptr as *mut T).as_mut() }.expect("Failed to convert global buffer"),
        }
    }
    pub fn store(&mut self, data: &T) {
        *self.ptr = *data;
    }

    pub fn load(&self, data: &mut T) {
        *data = *self.ptr;
    }
}
/*
//Rust 'helps' here by automatically cleaning up the memory
impl<'a, T: Copy> Drop for CuPinnedBoxRef<'a, T> {
    fn drop(&mut self) {
        CuError::check(unsafe { ffi::cuMemFreeHost(self.ptr as *mut T as *mut c_void) })
            .expect("Failed to free a pinned buffer");
    }
}
    */

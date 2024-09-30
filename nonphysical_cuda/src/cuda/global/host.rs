use crate::cuda::{
    error::CuError,
    ffi::{
        cuMemAllocAsync, cuMemAlloc_v2, cuMemcpyDtoHAsync_v2, cuMemcpyDtoH_v2,
        cuMemcpyHtoDAsync_v2, cuMemcpyHtoD_v2, DevicePtr,
    },
    //global::pinned::{CuPinnedBox, CuPinnedBoxRef, CuPinnedSlice, CuPinnedSliceRef},
    stream::CuStream,
};
use std::{ffi::c_void, slice::from_raw_parts_mut};

pub struct CuGlobalSlice<'a, T> {
    pub ptr: &'a mut [T],
}

pub struct CuGlobalSliceRef<'a, T> {
    pub ptr: &'a mut [T],
}

impl<'a, T> CuGlobalSlice<'a, T> {
    pub fn alloc(data: &[T]) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of_val(data);
        CuError::check(unsafe { cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn alloc_async(data: &[T], stream: &CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of_val(data);
        CuError::check(unsafe { cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn store(&mut self, data: &[T]) {
        assert!(self.ptr.len() == data.len());
        let size = size_of_val(data);
        CuError::check(unsafe {
            cuMemcpyHtoD_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.as_ptr() as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }
    
    pub fn store_async(&mut self, data: &[T], stream: &CuStream) {
        assert!(self.ptr.len() == data.len());
        let size = size_of::<T>() * data.len();
        CuError::check(unsafe {
            cuMemcpyHtoDAsync_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.as_ptr() as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }
}

impl<'a, T> CuGlobalSliceRef<'a, T> {
    pub fn alloc(data: &[T]) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of_val(data);
        CuError::check(unsafe { cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn alloc_async(data: &[T], stream: &CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of_val(data);
        CuError::check(unsafe { cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn store(&mut self, data: &[T]) {
        assert!(self.ptr.len() == data.len());
        let size = size_of_val(data);
        CuError::check(unsafe {
            cuMemcpyHtoD_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.as_ptr() as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }

    pub fn store_async(&mut self, data: &[T], stream: &CuStream) {
        assert!(self.ptr.len() == data.len());
        let size = size_of::<T>() * data.len();
        CuError::check(unsafe {
            cuMemcpyHtoDAsync_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.as_ptr() as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }

    pub fn load(&self, data: &mut [T]) {
        assert!(data.len() == self.ptr.len());
        let size = size_of_val(data);
        CuError::check(unsafe {
            cuMemcpyDtoH_v2(
                data.as_mut_ptr() as *mut c_void,
                self.ptr.as_ptr() as DevicePtr,
                size,
            )
        })
        .expect("Failed to read a global buffer");
    }

    pub fn load_async(&self, data: &mut [T], stream: &CuStream) {
        assert!(data.len() == self.ptr.len());
        let size = size_of::<T>() * data.len();
        CuError::check(unsafe {
            cuMemcpyDtoHAsync_v2(
                data.as_mut_ptr() as *mut c_void,
                self.ptr.as_ptr() as DevicePtr,
                size,
                stream.stream,
            )
        })
        .expect("Failed to read a global buffer async");
    }
}

pub struct CuGlobalBox<'a, T> {
    pub ptr: &'a mut T,
}

pub struct CuGlobalBoxRef<'a, T> {
    pub ptr: &'a mut T,
}

impl<'a, T> CuGlobalBox<'a, T> {
    pub fn alloc(_: &T) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut() }.expect("Failed to convert global buffer"),
        }
    }

    pub fn alloc_async(_: &T, stream: &CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut() }.expect("Failed to convert global buffer"),
        }
    }

    pub fn store(&mut self, data: &T) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            cuMemcpyHtoD_v2(
                self.ptr as *mut T as DevicePtr,
                data as *const T as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }
    pub fn get(&self) -> &T {
        self.ptr
    }

    pub fn store_async(&mut self, data: &T, stream: &CuStream) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            cuMemcpyHtoDAsync_v2(
                self.ptr as *mut T as DevicePtr,
                data as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }
}

impl<'a, T> CuGlobalBoxRef<'a, T> {
    pub fn alloc(_: &T) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut() }.expect("Failed to convert global buffer"),
        }
    }

    pub fn alloc_async(_: &T, stream: &CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut() }.expect("Failed to convert global buffer"),
        }
    }

    pub fn store(&mut self, data: &T) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            cuMemcpyHtoD_v2(
                self.ptr as *mut T as DevicePtr,
                data as *const T as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }

    pub fn store_async(&mut self, data: &T, stream: &CuStream) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            cuMemcpyHtoDAsync_v2(
                self.ptr as *mut T as DevicePtr,
                data as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }

    pub fn load(&self, data: &mut T) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            cuMemcpyDtoH_v2(
                data as *mut T as *mut c_void,
                self.ptr as *const T as DevicePtr,
                size,
            )
        })
        .expect("Failed to read a global buffer");
    }

    pub fn load_async(&self, data: &mut T, stream: &CuStream) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            cuMemcpyDtoHAsync_v2(
                data as *mut T as *mut c_void,
                self.ptr as *const T as DevicePtr,
                size,
                stream.stream,
            )
        })
        .expect("Failed to read a global buffer async");
    }
}

use crate::cuda::error::CuError;
use crate::cuda::ffi::DevicePtr;
use crate::cuda::stream::CuStream;
use crate::cuda::ffi;
use std::ffi::c_void;
use std::slice::from_raw_parts_mut;
use crate::cuda::global::pinned::{CuPinnedSlice,CuPinnedSliceRef, CuPinnedBox, CuPinnedBoxRef};

pub struct CuGlobalSlice<'a, T:Copy> {
    pub ptr: &'a mut [T],
}

pub struct CuGlobalSliceRef<'a, T:Copy> {
    pub ptr: &'a mut [T],
}

impl<'a, T:Copy> CuGlobalSlice<'a, T> {
    pub fn alloc(data: &[T]) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>() * data.len();
        CuError::check(unsafe { ffi::cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn alloc_async(data: &[T], stream: CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>() * data.len();
        CuError::check(unsafe { ffi::cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn store(&mut self, data: &CuPinnedSlice<T>) {
        assert!(self.ptr.len()==data.ptr.len());
        let size = size_of::<T>() * data.ptr.len();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoD_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.ptr.as_ptr() as *const T as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }

    pub fn store_async(&mut self, data: &CuPinnedSlice<T>, stream: CuStream) {
        assert!(self.ptr.len()==data.ptr.len());
        let size = size_of::<T>() * data.ptr.len();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoDAsync_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.ptr.as_ptr() as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }
}
impl<'a, T:Copy> Drop for CuGlobalSlice<'a,T>{
    fn drop(&mut self) { 
        CuError::check(unsafe {
            ffi::cuMemFreeHost(self.ptr.as_mut_ptr() as *mut c_void)
        })
        .expect("Failed to free a global buffer");

     }
}


impl<'a, T:Copy> CuGlobalSliceRef<'a, T> {
    pub fn alloc(data: &[T]) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>() * data.len();
        CuError::check(unsafe { ffi::cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn alloc_async(data: &[T], stream: CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>() * data.len();
        CuError::check(unsafe { ffi::cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { from_raw_parts_mut(ptr as *mut T, data.len()) },
        }
    }

    pub fn store(&mut self, data: &CuPinnedSliceRef<T>) {
        assert!(self.ptr.len()==data.ptr.len());
        let size = size_of::<T>() * data.ptr.len();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoD_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.ptr.as_ptr() as *const T as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }

    pub fn store_async(&mut self, data: &CuPinnedSliceRef<T>, stream: CuStream) {
        assert!(self.ptr.len()==data.ptr.len());
        let size = size_of::<T>() * data.ptr.len();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoDAsync_v2(
                self.ptr.as_ptr() as DevicePtr,
                data.ptr.as_ptr() as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }

    pub fn load(&self, data: &mut CuPinnedSliceRef<T>) {
        assert!(data.ptr.len() == self.ptr.len());
        let size = size_of::<T>() * data.ptr.len();
        CuError::check(unsafe {
            ffi::cuMemcpyDtoH_v2(
                data.ptr.as_mut_ptr() as *mut c_void,
                self.ptr.as_ptr() as DevicePtr,
                size,
            )
        })
        .expect("Failed to read a global buffer");
    }

    pub fn load_async(&self, data: &mut CuPinnedSliceRef<T>, stream: CuStream) {
        assert!(data.ptr.len() == self.ptr.len());
        let size = size_of::<T>() * data.ptr.len();
        CuError::check(unsafe {
            ffi::cuMemcpyDtoHAsync_v2(
                data.ptr.as_mut_ptr() as *mut c_void,
                self.ptr.as_ptr() as DevicePtr,
                size,
                stream.stream,
            )
        })
        .expect("Failed to read a global buffer async");
    }
}

impl<'a, T:Copy> Drop for CuGlobalSliceRef<'a,T>{
    fn drop(&mut self) { 
        CuError::check(unsafe {
            ffi::cuMemFreeHost(self.ptr.as_mut_ptr() as *mut c_void)
        })
        .expect("Failed to free a global buffer");

     }
}

pub struct CuGlobalBox<'a, T:Copy>{
    pub ptr: &'a mut T,
}

pub struct CuGlobalBoxRef<'a, T:Copy>{
    pub ptr: &'a mut T,
}

impl<'a, T:Copy> CuGlobalBox<'a, T> {
    pub fn alloc(_: &T) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { ffi::cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut()}.expect("Failed to convert global buffer"),
        }
    }

    pub fn alloc_async(_: &T, stream: CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { ffi::cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut()}.expect("Failed to convert global buffer"),
        }
    }

    pub fn store(&mut self, data: &CuPinnedBox<T>) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoD_v2(
                self.ptr as *mut T as DevicePtr,
                data.ptr as *const T as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }

    pub fn store_async(&mut self, data: &CuPinnedBox<T>, stream: CuStream) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoDAsync_v2(
                self.ptr as *mut T as DevicePtr,
                data.ptr as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }
}
impl<'a, T:Copy> Drop for CuGlobalBox<'a,T>{
    fn drop(&mut self) { 
        CuError::check(unsafe {
            ffi::cuMemFreeHost(self.ptr as *mut T as *mut c_void)
        })
        .expect("Failed to free a global buffer");

     }
}


impl<'a, T:Copy> CuGlobalBoxRef<'a, T> {
    pub fn alloc(_: &T) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { ffi::cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate a global buffer");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut()}.expect("Failed to convert global buffer"),
        }
    }

    pub fn alloc_async(_: &T, stream: CuStream) -> Self {
        let mut ptr: DevicePtr = 0;
        let size = size_of::<T>();
        CuError::check(unsafe { ffi::cuMemAllocAsync(&mut ptr as *mut u64, size, stream.stream) })
            .expect("Failed to allocate a global buffer async");
        Self {
            ptr: unsafe { (ptr as *mut T).as_mut()}.expect("Failed to convert global buffer"),
        }
    }

    pub fn store(&mut self, data: &CuPinnedBoxRef<T>) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoD_v2(
                self.ptr as *mut T as DevicePtr,
                data.ptr as *const T as *const c_void,
                size,
            )
        })
        .expect("Failed to write a global buffer");
    }

    pub fn store_async(&mut self, data: &CuPinnedBoxRef<T>, stream: CuStream) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            ffi::cuMemcpyHtoDAsync_v2(
                self.ptr as *mut T as DevicePtr,
                data.ptr as *const T as *const c_void,
                size,
                stream.stream,
            )
        })
        .expect("Failed to write a global buffer async");
    }

    pub fn load(&self, data: &mut CuPinnedBoxRef<T>) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            ffi::cuMemcpyDtoH_v2(
                data.ptr as *mut T as *mut c_void,
                self.ptr as *const T as DevicePtr,
                size,
            )
        })
        .expect("Failed to read a global buffer");
    }

    pub fn load_async(&self, data: &mut CuPinnedBoxRef<T>, stream: CuStream) {
        let size = size_of::<T>();
        CuError::check(unsafe {
            ffi::cuMemcpyDtoHAsync_v2(
                data.ptr as *mut T as *mut c_void,
                self.ptr as *const T as DevicePtr,
                size,
                stream.stream,
            )
        })
        .expect("Failed to read a global buffer async");
    }
}

impl<'a, T:Copy> Drop for CuGlobalBoxRef<'a,T>{
    fn drop(&mut self) { 
        CuError::check(unsafe {
            ffi::cuMemFreeHost(self.ptr as *mut T as *mut c_void)
        })
        .expect("Failed to free a global buffer");

     }
}
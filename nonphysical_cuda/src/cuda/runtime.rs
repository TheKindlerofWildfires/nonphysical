use crate::cuda::{
    error::CuError,
    ffi::{
        self, cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSynchronize, cuDeviceGetCount, cuInit,
        cuLaunchKernel, cuModuleGetFunction, cuModuleLoad, CUcontext, CUctx_flags_enum, CUfunction,
        CUmodule, DevicePtr,
    },
};

use std::{
    ffi::{c_void, CString},
    os::raw::c_uint,
    ptr,
    string::String,
};

use super::global::host::CuGlobalBox;

static mut CUDA_INITIALIZED: bool = false;

pub struct Runtime {
    context: CUcontext,
    module: CUmodule,
}
pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
pub struct CUDABox<'a, T> {
    ptr: &'a T,
}

impl<'a, T> CUDABox<'a, T> {
    pub fn new(ptr: &'a T) -> CUDABox<'a, T> {
        CUDABox { ptr }
    }

    // get reference
    pub fn get(&self) -> &T {
        self.ptr
    }
}
impl Runtime {
    pub fn new(device_id: i32, kernel_str: &str) -> Runtime {
        // cuInit
        unsafe {
            if !CUDA_INITIALIZED {
                CuError::check(cuInit(0)).expect("Failed to initialize a cuda context");
                CUDA_INITIALIZED = true;
            }
        }
        // device check
        let mut device_num: i32 = 0;
        CuError::check(unsafe { cuDeviceGetCount(&mut device_num as *mut i32) })
            .expect("Failed to could the available devices");
        assert!(device_id < device_num);

        // context
        let mut context: CUcontext = ptr::null_mut();
        CuError::check(unsafe {
            cuCtxCreate_v2(
                &mut context as *mut CUcontext,
                CUctx_flags_enum::CU_CTX_SCHED_AUTO as u32,
                device_id,
            )
        })
        .expect("Failed to create a cuda context");

        // module
        let mut module: CUmodule = ptr::null_mut();
        let kernel_cstr = CString::new(kernel_str).expect("Could not create kernel string");
        CuError::check(unsafe { cuModuleLoad(&mut module as *mut CUmodule, kernel_cstr.as_ptr()) })
            .expect("Failed to load the module");
        // res
        Runtime { context, module }
    }

    pub fn wrap_args<Args>(args: &Args) -> Vec<*mut c_void> {
        let mut ptr: DevicePtr = 0;
        let size = std::mem::size_of::<Args>();
        CuError::check(unsafe { ffi::cuMemAlloc_v2(&mut ptr as *mut u64, size) })
            .expect("Failed to allocate space for args");
        CuError::check(unsafe {
            ffi::cuMemcpyHtoD_v2(
                ptr,
                args as *const Args as *const std::ffi::c_void,
                std::mem::size_of::<Args>(),
            )
        })
        .expect("Failed to copy args");
        let mut ptr = unsafe { (ptr as *mut Args).as_mut() }.expect("Failed too convert args");
        vec![&mut ptr as *mut _ as *mut c_void]
    }

    /// # Safety
    ///
    /// This function isn't safe, the result of the launch is checked but there are many places a raw pointer can be dereferenced
    pub unsafe fn launch<Args>(
        &self,
        function: CUfunction,
        args: &Args,
        grid_dim: Dim3,
        block_dim: Dim3,
    ) {
        // launch
        let mut args_d = CuGlobalBox::alloc(args);
        args_d.store(args);
        let mut ptr = args_d.get();
        let mut launch_args = vec![&mut ptr as *mut &Args as *mut c_void];
        let shared_mem_bytes = 0;

        CuError::check(unsafe {
            cuLaunchKernel(
                function,
                grid_dim.x as c_uint,
                grid_dim.y as c_uint,
                grid_dim.z as c_uint,
                block_dim.x as c_uint,
                block_dim.y as c_uint,
                block_dim.z as c_uint,
                shared_mem_bytes as c_uint,
                ptr::null_mut(),
                launch_args.as_mut_ptr(),
                ptr::null_mut(),
            )
        })
        .expect("Kernel launch failed");
    }
    pub fn launch_name<Args>(
        &self,
        func_name: String,
        args: &Args,
        grid_dim: Dim3,
        block_dim: Dim3,
    ) -> CUfunction {
        // function
        let mut function: CUfunction = ptr::null_mut();
        let func_name_cstr =
            CString::new(func_name.as_str()).expect("Could not create function name");
        CuError::check(unsafe {
            cuModuleGetFunction(
                &mut function as *mut CUfunction,
                self.module,
                func_name_cstr.as_ptr(),
            )
        })
        .expect("Failed to find function");

        unsafe {
            self.launch(function, args, grid_dim, block_dim);
        }

        function
    }
    pub fn sync(&self) {
        CuError::check(unsafe { cuCtxSynchronize() }).expect("Failed to sync with device");
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            cuCtxDestroy_v2(self.context);
        }
    }
}
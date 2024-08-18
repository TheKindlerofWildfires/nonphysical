use super::{ffi, global::*};

use crate::cuda::error::CuError;
use crate::cuda::ffi::CUfunction;
use std::ffi::CString;
use std::ptr;
use std::string::String;
use std::vec;
use crate::cuda::runtime::pinned::CuPinnedBox;
use crate::cuda::runtime::host::CuGlobalBox;
static mut CUDA_INITIALIZED: bool = false;

pub struct Runtime {
    context: ffi::CUcontext,
    module: ffi::CUmodule,
}
pub struct Dim3 {
    x: usize,
    y: usize,
    z: usize,
}

impl Runtime {
    pub fn new(device_id: i32, kernel_str: &str) -> Runtime {
        // cuInit
        unsafe {
            if !CUDA_INITIALIZED {
                CuError::check(ffi::cuInit(0)).expect("Failed to initialize a cuda context");
                CUDA_INITIALIZED = true;
            }
        }
        // device check
        let mut device_num: i32 = 0;
        CuError::check(unsafe { ffi::cuDeviceGetCount(&mut device_num as *mut i32) })
            .expect("Failed to could the available devices");
        assert!(device_id < device_num);

        // context
        let mut context: ffi::CUcontext = ptr::null_mut();
        CuError::check(unsafe {
            ffi::cuCtxCreate_v2(
                &mut context as *mut ffi::CUcontext,
                ffi::CUctx_flags_enum::CU_CTX_SCHED_AUTO as u32,
                device_id,
            )
        })
        .expect("Failed to create a cuda context");

        // module
        let mut module: ffi::CUmodule = ptr::null_mut();
        let kernel_cstr = std::ffi::CString::new(kernel_str).unwrap();
        CuError::check(unsafe {
            ffi::cuModuleLoad(&mut module as *mut ffi::CUmodule, kernel_cstr.as_ptr())
        })
        .expect("Failed to load the module");
        // res
        Runtime { context, module }
    }
    pub fn launch<Args:Copy>(&self, function: CUfunction, args: &Args, grid_dim: Dim3, block_dim: Dim3) {
        // launch
        let mut pinned_args = CuPinnedBox::alloc(args);
        let mut device_args = CuGlobalBox::alloc(args);
        pinned_args.store(*args);
        device_args.store(&pinned_args);
        

        let mut launch_args = vec![&mut device_args.ptr as *mut &mut Args as *mut std::ffi::c_void];
        unsafe {
            use std::os::raw::c_uint;
            use std::ptr;
            let shared_mem_bytes = 0;
            CuError::check(ffi::cuLaunchKernel(
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
            )).expect("Kernel launch failed");
        }
    }
    pub fn launch_name<Args:Copy>(
        &self,
        func_name: String,
        args: &Args,
        grid_dim: Dim3,
        block_dim: Dim3,
    ) -> CUfunction {
        // function
        let mut function: ffi::CUfunction = ptr::null_mut();
        let func_name_cstr = CString::new(func_name.as_str()).unwrap();
        CuError::check(unsafe {
            ffi::cuModuleGetFunction(
                &mut function as *mut ffi::CUfunction,
                self.module,
                func_name_cstr.as_ptr(),
            )
        })
        .expect("Failed to find function");

        self.launch(function, args, grid_dim, block_dim);

        function
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            ffi::cuCtxDestroy_v2(self.context);
        }
    }
}

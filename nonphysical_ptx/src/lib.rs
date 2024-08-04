/*
    This is based on cuda-tools (https://github.com/spica314/cuda-tools/) with the following changes
        The FFI layer is statically compiled, to prevent breaking changes propagating and to remove the buildgen dependency
        Only nonphysical kernels are exposed, this isn't a cuda ffi interface, it's a nonphysical interface
        It enables SliceRef and BoxRef as passed types. These are unsafe around aliasing but the cuda aliasing model is messy anyway
        It fixes bugs in cuda-tools where the ecosystem changed since updates (notably to cuda loading and launching)
*/

#![no_std]
#![allow(internal_features,improper_ctypes_definitions)]
#![feature(abi_ptx,core_intrinsics)]

#![cfg_attr(target_arch="nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch="nvptx64", feature(asm_experimental_arch))]

#[cfg(not(target_arch = "nvptx64"))]
extern crate std;


extern crate alloc;
#[cfg(target_arch = "nvptx64")]
use crate::shared::primitive::F32;
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::primitive::Primitive;


#[cfg(target_arch = "nvptx64")]
pub mod shared;
//pub mod signal;
pub mod cuda;
use cuda::cu_slice::CuSlice;
use crate::cuda::cu_slice::CuSliceRef;
#[no_mangle]
#[cfg(target_arch = "nvptx64")]
pub extern "ptx-kernel"  fn sin(x: &mut CuSliceRef<F32>){
    x[0] = x[0].asin()
}

//-C llvm-args=--fp-contract=fast


//This is like a very minimized implementation of cuda-tools(https://github.com/spica314/cuda-tools/) for exactly nonphysical kernels
//This allows for (slightly) improved safety guarantees between host and device
//It also ditches the buildgen dependency in favor of a static build, and compiles/links the ptx at compile time not run time, as it's targets are static
//If I could (easily) force the ptx into the compiled assembly I would,

/*
    Has
        cuda slice/box for use
    Process (during build)
        build the .s file from cuda ptx
        find the link elements and link them out 

    Process during run 
        create the cuda device
        load the ptx into it

    Process for each kernel
        alloc memory for it
        launch the kernel
        return results

Alernatively, what if this is both ptx and exe mode
    Has the shared cuda slice interfaces/unsafe F32s for both modes
    each kernel has either a std mode or a cuda mode for compileing out 
        and an invoker that knows exactly how to use it
    statically create the cuda bind gens and store them here

    during compile first build out self as ptx format
    then load in the ptx, mesh up the functions (this could be hard)
    and now when you invoke the wrapper you get what you asked for 



*/
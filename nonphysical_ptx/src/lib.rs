/*
    This is based on cuda-tools (https://github.com/spica314/cuda-tools/) with the following changes
        The FFI layer is statically compiled, to prevent breaking changes propagating and to remove the buildgen dependency
        Only nonphysical kernels are exposed, this isn't a cuda ffi interface, it's a nonphysical interface
        It enables SliceRef and BoxRef as passed types. These are unsafe around aliasing but the cuda aliasing model is messy anyway
        It fixes bugs in cuda-tools where the ecosystem changed since updates (notably to cuda loading and launching)
*/

#![no_std]
#![allow(internal_features,improper_ctypes_definitions,dead_code)]
#![feature(abi_ptx,core_intrinsics)]

#![cfg_attr(target_arch="nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch="nvptx64", feature(asm_experimental_arch))]

#[cfg(not(target_arch = "nvptx64"))]
extern crate std;

#[cfg(target_arch = "nvptx64")]
extern crate alloc;

pub mod signal;
pub mod cuda;

#[cfg(target_arch = "nvptx64")]
pub mod shared;

//cargo rustc --target=nvptx64-nvidia-cuda --release -- --emit=llvm-ir -C llvm-args=--enable-approx-func-fp-math=1 -C llvm-args=--ffast-math -C llvm-args=--fp-contract=on -C llvm-args=--denormal-fp-math=preserve-sign
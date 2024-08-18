#![no_std]
#![allow(internal_features,improper_ctypes_definitions,dead_code)]
#![feature(abi_ptx,core_intrinsics)]
#![feature(link_llvm_intrinsics)]
#![cfg_attr(target_arch="nvptx64", feature(stdarch_nvptx))]
#![feature(asm_experimental_arch)]
#![feature(asm_const)]
#![feature(extern_types)]

//pub mod signal;
//pub mod cluster;
//pub mod graph;

extern crate alloc;
pub mod shared;
#[cfg(target_arch = "nvptx64")]
pub mod cuda;

#[cfg(not(target_arch = "nvptx64"))]
extern crate std;
//cargo rustc --target=nvptx64-nvidia-cuda --release -- --emit=llvm-ir -C llvm-args=--enable-approx-func-fp-math=1 -C llvm-args=--ffast-math -C llvm-args=--fp-contract=on -C llvm-args=--denormal-fp-math=preserve-sign
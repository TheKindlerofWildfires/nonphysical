#![allow(dead_code)]
#![no_std]
#![feature(core_intrinsics)]


#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

//core
pub mod shared;
pub mod random;
pub mod graph;
pub mod cluster;
pub mod signal;


//unstable
pub mod linear;

//stalled
//pub mod neural;
//pub mod filter;



fn main() {
}
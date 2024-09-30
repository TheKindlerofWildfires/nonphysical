#![allow(dead_code)]
#![no_std]
#![forbid(unsafe_code)]


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
pub mod neural;

//todo:
//-1: Validate cyclostationary
//0: Move cyclostationary to GPU
//1: Turn back on neural
//2: Identify types of layers needed
//3: Make very basic version work
//4: Impl all needed layers CPU side
//5: Do perf review specifically on gemm

//todo round 2:
/*
    1: DWT recon on CPU
    2: Cyclo spec on CPU
    3: GAB on GPU confirm
    4: DWT recon on GPU
    5: Cyclo spec on GPU
    6: Graph network
*/
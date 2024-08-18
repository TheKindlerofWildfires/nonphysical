pub mod global;

#[cfg(target_arch = "nvptx64")]
pub mod shared;
#[cfg(target_arch = "nvptx64")]
pub mod atomic;
#[cfg(target_arch = "nvptx64")]
pub mod intrinsic;

#[cfg(not(target_arch = "nvptx64"))]
pub mod ffi;
#[cfg(not(target_arch = "nvptx64"))]
pub mod runtime;
#[cfg(not(target_arch = "nvptx64"))]
pub mod link;
#[cfg(not(target_arch = "nvptx64"))]
pub mod error;
#[cfg(not(target_arch = "nvptx64"))]
pub mod stream;




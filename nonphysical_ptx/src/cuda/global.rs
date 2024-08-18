#[cfg(not(target_arch = "nvptx64"))]
pub mod host;
#[cfg(not(target_arch = "nvptx64"))]
pub mod pinned;

#[cfg(target_arch = "nvptx64")]
pub mod device;

#[cfg(target_arch = "nvptx64")]
pub mod atomic;



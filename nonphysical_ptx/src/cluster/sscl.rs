#[cfg(not(target_arch = "nvptx64"))]
pub mod driver;
#[cfg(target_arch = "nvptx64")]
pub mod ptx;



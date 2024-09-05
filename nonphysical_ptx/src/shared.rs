#[cfg(target_arch = "nvptx64")]
pub mod float;
#[cfg(target_arch = "nvptx64")]
pub mod primitive;
#[cfg(target_arch = "nvptx64")]
pub mod real;
#[cfg(target_arch = "nvptx64")]
pub mod unsigned;

pub mod vector;
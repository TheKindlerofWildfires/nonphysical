#[cfg(target_arch = "nvptx64")]
pub mod float;
#[cfg(target_arch = "nvptx64")]
pub mod primitive;
#[cfg(target_arch = "nvptx64")]
pub mod real;
#[cfg(target_arch = "nvptx64")]
pub mod unsigned;

#[cfg(feature = "vector")]
pub mod vector;

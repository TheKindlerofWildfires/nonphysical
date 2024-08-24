#[cfg(not(target_arch = "nvptx64"))]
pub mod sscl_driver;
#[cfg(target_arch = "nvptx64")]
pub mod sccl_ptx;

pub struct SSCLArguments<'a, P: Point> {
    pub data: CuGlobalSlice<'a, P>, //actual data
    pub marks: CuGlobalSliceRef<'a, u32>, //cluster marks
    pub clusters: CuGlobalSliceRef<'a, SSCluster<P>>, //actual clusters
}

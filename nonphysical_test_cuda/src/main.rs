use std::sync::Arc;

use nonphysical_core::shared::primitive::Primitive;
use nonphysical_cuda::cuda::runtime::{self, Runtime};
use nonphysical_ptx::shared::vector::vector_driver::CudaVectorHost;
use nonphysical_std::shared::primitive::F32;


pub fn main(){
    let runtime = Arc::new(Runtime::new(0, "nonphysical_ptx.ptx"));
    //Allocate a host buffer
    let numbers = (0..5).map(|i| F32::usize(i)).collect::<Vec<_>>();
    CudaVectorHost::sum(runtime,&numbers);
}
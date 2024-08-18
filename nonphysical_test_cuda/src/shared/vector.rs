#[cfg(test)]
mod vector_tests{
    use std::sync::Arc;
    use nonphysical_core::shared::primitive::Primitive;
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_ptx::shared::vector::vector_driver::*;
    use nonphysical_std::shared::primitive::F32;
    #[test]
    fn sum_host(){
        let runtime = Arc::new(Runtime::new(0, "../nonphysical_ptx.ptx"));
        //Allocate a host buffer
        let numbers = (0..5).map(|i| F32::usize(i)).collect::<Vec<_>>();
        CudaVectorHost::sum(runtime,&numbers);
    }
    #[test]
    fn sum_mixed(){

    }
    #[test]
    fn sum_device(){

    }
}
#[cfg(test)]
mod vector_tests {
    use nonphysical_core::shared::primitive::Primitive;
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_ptx::graph::hash_table::cuda_hash_table::CudaHashTable;
    use nonphysical_std::shared::primitive::F32;
    #[test]
    fn insert() {
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data = (0..32).rev();
        let data = data.into_iter().map(|i| F32::isize(i%16)).collect::<Vec<_>>();
        let mut arguments = CudaHashTable::<F32, F32>::create();
        CudaHashTable::<F32,F32>::insert(&data, &data, &mut arguments);
    }

    #[test]
    fn index() {
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data = (0..32).rev();
        let data = data.into_iter().map(|i| F32::isize(i%16)).collect::<Vec<_>>();
        dbg!(&data);
        let mut arguments = CudaHashTable::<F32, F32>::create();
        CudaHashTable::<F32,F32>::insert(&data, &data, &mut arguments);
        let (a,b) = CudaHashTable::<F32,F32>::iterate(&mut arguments);
        dbg!(a,b);
    }
}
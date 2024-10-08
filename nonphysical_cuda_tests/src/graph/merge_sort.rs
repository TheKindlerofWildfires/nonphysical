#[cfg(test)]
mod vector_tests {
    use nonphysical_core::shared::
        primitive::Primitive
    ;
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_ptx::graph::merge_sort::cuda_merge_sort::CudaMergeSort;
    use nonphysical_std::shared::primitive::F32;
    #[test]
    fn sort_tiny() {
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data = (0..512).rev();
        //let data = vec![9,7,5,3,8,4,6,5];
        let data = data.into_iter().map(|i| F32::isize(i)).collect::<Vec<_>>();
        let out = CudaMergeSort::<F32>::sort(&data);
        for i in 0..out.len() - 1 {
            assert!(out[i] <= out[i + 1]);
        }
    }
    #[test]
    fn sort_small() {
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data = vec![
            -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0,
            1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1,
            2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6,
            5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3,
            8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9,
            7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1,
            5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2,
            4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5,
            4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8,
            4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7,
            5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5,
            4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4,
            0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4,
            1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4,
            6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5,
            3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4,
            9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0,
            1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1,
            2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6,
            5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3,
            8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9,
            7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1,
            5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2,
            4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5,
            4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8,
            4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7,
            5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5,
            4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4,
            0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4,
            1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4,
            6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, -9, 7, 5,
            3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4,
            -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0,
            1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6, 5, 4, 1,
            2, 4, 0, 1, 5, 4, -9, 7, 5, 3, 8, 4, 6, 5, 4, 1, 2, 4, 0, 1, 5, 4, 9, 7, 5, 3, 8, 4, 6,
            5, 4, 1, 2, 4, 0, 1, 5, 4,
        ];
        //let data = vec![9,7,5,3,8,4,6,5];
        let data = data.into_iter().map(|i| F32::isize(i)).collect::<Vec<_>>();
        let out = CudaMergeSort::<F32>::sort(&data);
        for i in 0..out.len() - 1 {
            assert!(out[i] <= out[i + 1]);
        }
    }

    #[test]
    fn sort_reverse_small() {
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data = (0..1024).rev().collect::<Vec<_>>();
        //let data = vec![9,7,5,3,8,4,6,5];
        let data = data.into_iter().map(|i| F32::isize(i)).collect::<Vec<_>>();
        let out = CudaMergeSort::<F32>::sort(&data);
        for i in 0..out.len() - 1 {
            assert!(out[i] <= out[i + 1]);
        }
    }
    #[test]
    fn sort_reverse_medium() {
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data = (0..4096).rev().collect::<Vec<_>>();
        //let data = vec![9,7,5,3,8,4,6,5];
        let data = data.into_iter().map(|i| F32::isize(i)).collect::<Vec<_>>();
        let out = CudaMergeSort::<F32>::sort(&data);
        for i in 0..out.len() - 1 {
            assert!(out[i] <= out[i + 1]);
        }
    }
    #[test]
    fn sort_reverse_large() {
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let data = (0..8192).rev().collect::<Vec<_>>();
        //let data = vec![9,7,5,3,8,4,6,5];
        let data = data.into_iter().map(|i| F32::isize(i)).collect::<Vec<_>>();
        let out = CudaMergeSort::<F32>::sort(&data);
        for i in 0..out.len() - 1 {
            assert!(out[i] <= out[i + 1]);
        }
    }
}

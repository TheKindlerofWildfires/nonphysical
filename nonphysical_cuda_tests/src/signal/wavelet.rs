#[cfg(test)]
mod wavelet_tests {
    use nonphysical_core::{shared::{complex::{Complex, ComplexScaler}, primitive::Primitive}, signal::wavelet::{wavelet_heap::DaubechiesFirstWaveletHeap, DiscreteWavelet}};
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_std::shared::primitive::F32;
    use nonphysical_core::shared::float::Float;
    use nonphysical_ptx::signal::wavelet::cuda_wavelet::DaubechiesFirstWaveletCuda;
    #[test]
    fn forward_128(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 128;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.forward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.forward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_256(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 256;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.forward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.forward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_512(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 512;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.forward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.forward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }


    #[test]
    fn forward_1024(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 1024;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.forward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.forward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }


    #[test]
    fn forward_2048(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 2048;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.forward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.forward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_4096(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 4096;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.forward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.forward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn backward_128(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 128;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.backward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.backward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_256(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 256;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.backward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.backward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_512(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 512;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.backward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.backward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_1024(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 1024;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.backward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.backward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_2048(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 2048;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.backward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.backward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_4096(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let ndwt = 4096;
        let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
        let reference_dwt = DaubechiesFirstWaveletHeap::new(());
        let ref_out = reference_dwt.backward(&data_reference);    
        let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
        let out = dwt.backward(&data);
        out.chunks_exact(ndwt).for_each(|chunk|{
            chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
}
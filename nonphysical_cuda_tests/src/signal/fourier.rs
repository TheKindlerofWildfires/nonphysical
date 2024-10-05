#[cfg(test)]
mod fourier_tests {
    use nonphysical_core::{shared::{complex::{Complex, ComplexScaler}, primitive::Primitive}, signal::fourier::{fourier_heap::ComplexFourierTransformHeap, FourierTransform}};
    use nonphysical_cuda::cuda::runtime::Runtime;
    use nonphysical_std::shared::primitive::F32;
    use nonphysical_core::shared::float::Float;
    use nonphysical_ptx::signal::fourier::cuda_fourier::ComplexFourierTransformCuda;
    #[test]
    fn forward_128(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 128;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.forward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.forward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_256(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 256;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.forward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.forward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn forward_512(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 512;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.forward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.forward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_1024(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 1024;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.forward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.forward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_2048(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 2048;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.forward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.forward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_4096(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 4096;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.forward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.forward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn backward_128(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 128;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.backward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.backward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
        return;
    }

    #[test]
    fn backward_256(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 256;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.backward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.backward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_512(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 512;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.backward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.backward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn backward_1024(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 1024;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.backward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.backward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn backward_2048(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 2048;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.backward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.backward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn backward_4096(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 4096;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.backward(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.backward(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn forward_shifted_128(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 128;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.fft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.fft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn forward_shifted_256(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 256;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.fft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.fft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn forward_shifted_512(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 512;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.fft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.fft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn forward_shifted_1024(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 1024;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.fft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.fft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn forward_shifted_2048(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 2048;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.fft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.fft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn forward_shifted_4096(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 4096;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.fft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.fft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }

    #[test]
    fn backward_shifted_128(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 128;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.ifft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.ifft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_shifted_256(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 256;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.ifft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.ifft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_shifted_512(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 512;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.ifft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.ifft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_shifted_1024(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 1024;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.ifft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.ifft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_shifted_2048(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 2048;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.ifft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.ifft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
    #[test]
    fn backward_shifted_4096(){
        Runtime::init(0, "../nonphysical_ptx.ptx");
        let nfft = 4096;
        let mut data = (0..nfft*1024).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
        let reference_fft = ComplexFourierTransformHeap::new(nfft);
        reference_fft.ifft_shifted(&mut data_reference);    
        let fft = ComplexFourierTransformCuda::new(nfft);
        fft.ifft_shifted(&mut data);
        data.chunks_exact(nfft).for_each(|chunk|{
            chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
                assert!(a==b);
            });
        });
    }
   
}
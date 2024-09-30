use super::FourierArguments;
use nonphysical_core::{
    shared::complex::Complex,
    signal::fourier::{fourier_heap::ComplexFourierTransformHeap, FourierTransform},
};
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_cuda::cuda::runtime::Dim3;
use std::{cmp::min, marker::PhantomData};
use alloc::string::String;
use alloc::format;
use nonphysical_cuda::cuda::runtime::RUNTIME;

pub struct ComplexFourierTransformCuda<C: Complex> {
    phantom_data: PhantomData<C>,
    nfft: usize,
}

impl<C: Complex> FourierTransform<C> for ComplexFourierTransformCuda<C> {
    type FourierInit = usize;
    fn new(init: Self::FourierInit) -> Self {
        let nfft = init;
        Self {
            phantom_data: PhantomData,
            nfft,
        }
    }

    fn fft(&self, x: &mut [C]) {
        let reference_fft = ComplexFourierTransformHeap::<C>::new(self.nfft);

        let mut args = Self::fourier_alloc(x, &reference_fft.twiddles);

        self.fourier_transfer(&mut args, x,&reference_fft.twiddles,"forward")
    }

    fn ifft(&self, x: &mut [C]) {
        let reference_fft = ComplexFourierTransformHeap::<C>::new(self.nfft);

        let mut args = Self::fourier_alloc(x, &reference_fft.twiddles);

        self.fourier_transfer(&mut args, x,&reference_fft.twiddles,"backward")
    }

    fn fft_shifted(&self, x: &mut [C]) {
        let reference_fft = ComplexFourierTransformHeap::<C>::new(self.nfft);

        let mut args = Self::fourier_alloc(x, &reference_fft.twiddles);

        self.fourier_transfer(&mut args, x,&reference_fft.twiddles,"forward_shifted")
    }

    fn ifft_shifted(&self, x: &mut [C]) {
        let reference_fft = ComplexFourierTransformHeap::<C>::new(self.nfft);

        let mut args = Self::fourier_alloc(x, &reference_fft.twiddles);

        self.fourier_transfer(&mut args, x,&reference_fft.twiddles,"backward_shifted")
    }

}

impl<C: Complex> ComplexFourierTransformCuda<C> {
    fn launch<Args>(args: &mut Args, threads: usize, block_size: usize, kernel:String){

        let grid = Dim3 {
            x: block_size,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: threads,
            y: 1,
            z: 1,
        };
        match RUNTIME.get() {
            Some(rt) => {
                rt.launch_name(kernel, args, grid, block);
            }
            None => panic!("Cuda Runtime not initialized"),
        };
    }
    fn fourier_alloc<'a>(x: &[C], twiddles: &[C]) -> FourierArguments<'a, C> {
        let x = CuGlobalSliceRef::alloc(x);
        let twiddles = CuGlobalSlice::alloc(twiddles);

        FourierArguments { x, twiddles }
    }
    fn fourier_transfer<'a>(&self,args: &mut FourierArguments<'a, C>, x: &mut [C],twiddles: &[C], name: &str) {
        assert!(x.len() >= 4);
        let block_size = x.len() / self.nfft;
        assert!(x.len() == block_size * self.nfft);
        args.x.store(x);
        args.twiddles.store(twiddles);
        let threads = min(512, self.nfft / 2);
        let kernel = format!("fft_{}_{}_kernel",name,self.nfft);
        Self::launch(args,threads,block_size,kernel);
        args.x.load(x);
    }
}

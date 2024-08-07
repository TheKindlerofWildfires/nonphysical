use crate::cuda::runtime::Runtime;

use nonphysical_core::{shared::complex::Complex,signal::fourier::{FourierTransform,heap::ComplexFourierTransformHeap}};
use std::{rc::Rc, marker::PhantomData,cmp::min, borrow::ToOwned};
use super::FourierArguments;

#[cfg(not(target_arch = "nvptx64"))]
pub struct ComplexFourierTransformCuda<C: Complex> {
    runtime: Rc<Runtime>,
    phantom_data: PhantomData<C>,
    nfft: usize,
}

#[cfg(not(target_arch = "nvptx64"))]
impl<C: Complex> FourierTransform<C> for ComplexFourierTransformCuda<C> {
    type FourierInit = (Rc<Runtime>, usize);
    fn new(init: Self::FourierInit) -> Self {
        let (runtime, nfft) = init;
        Self {
            runtime,
            phantom_data: PhantomData,
            nfft,
        }
    }

    fn fft(&self, x: &mut [C]) {
        assert!(x.len() >= 4);
        let block_size = x.len()/self.nfft;
        assert!(x.len() ==block_size*self.nfft);
        let reference_fft = ComplexFourierTransformHeap::<C>::new(self.nfft);
        //let reference_fft = ComplexFourierTransformStack::<C,8>::new(());
        let x_device = self.runtime.alloc_slice_ref(&x).unwrap();
        let t_device = self.runtime.alloc_slice(&reference_fft.twiddles).unwrap();
        let args = FourierArguments {
            x: x_device,
            twiddles: t_device,
        };
        let threads = min(1024, self.nfft / 2);

        let _ = self.runtime.launch(
            "fft_forward_kernel".to_owned(),
            &args,
            block_size,
            1,
            1,
            threads,
            1,
            1,
        );
        let x_result = args.x.to_host().unwrap();
        x.copy_from_slice(&x_result);
    }

    fn ifft(&self, _x: &mut [C]) {
        todo!()
    }
}
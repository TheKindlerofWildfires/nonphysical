use crate::cuda::runtime::Runtime;

use nonphysical_core::{shared::complex::Complex,signal::fourier::{FourierTransform,fourier_heap::ComplexFourierTransformHeap}};
use std::{rc::Rc, marker::PhantomData,cmp::min, borrow::ToOwned};
use super::WaveletArguments;
use nonphysical_core::signal::wavelet::DiscreteWavelet;
use nonphysical_core::shared::primitive::Primitive;
use nonphysical_core::signal::wavelet::WaveletFamily;
pub struct ComplexFourierTransformCuda<C: Complex> {
    runtime: Rc<Runtime>,
    phantom_data: PhantomData<C>,
    nfft: usize,
}
use std::time::SystemTime;
use std::dbg;
pub struct DaubechiesFirstComplexWaveletCuda<C: Complex> {
    runtime: Rc<Runtime>,
    phantom_data: PhantomData<C>,
    ndwt: usize,
}

impl<C: Complex> DiscreteWavelet<C> for DaubechiesFirstComplexWaveletCuda<C> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type DiscreteWaveletInit = (Rc<Runtime>, usize);

    fn new(init: Self::DiscreteWaveletInit) -> Self{
        let (runtime, ndwt) = init;
        Self {
            runtime,
            phantom_data: PhantomData,
            ndwt,
        }
    }
    fn forward(&mut self, input: &mut [C]) {
        let now = SystemTime::now();
        let n = input.len();
        debug_assert!(n % 2 == 0);
        let x_device = self.runtime.alloc_slice_ref(&input).unwrap();
        let ndwt_device = self.runtime.alloc_slice(&[self.ndwt]).unwrap();
        dbg!(now.elapsed());
        let now = SystemTime::now();

        let threads = min(1024, self.ndwt / 2);
        let block_size = input.len()/self.ndwt;
        let args = WaveletArguments {
            x: x_device,
            ndwt: ndwt_device
        };
        let _ = self.runtime.launch(
            "dwt_forward_kernel".to_owned(),
            &args,
            block_size,
            1,
            1,
            threads,
            1,
            1,
        );
        dbg!(now.elapsed());

        let now = SystemTime::now();

        let x_result = args.x.to_host().unwrap();
        input.copy_from_slice(&x_result);
        dbg!(now.elapsed());
    }
    fn backward(&mut self, input: &mut [C]) {
        todo!();
    }

}
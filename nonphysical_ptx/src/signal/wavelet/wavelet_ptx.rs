use crate::{cuda::shared::CuShared, shared::primitive::F32};
use nonphysical_core::{
    shared::{
        complex::{Complex, ComplexScaler},
        float::Float,
        primitive::Primitive,
    },
    signal::wavelet::{DiscreteWavelet, WaveletFamily},
};
use crate::cuda::shared::Shared;
use super::WaveletArguments;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};

#[no_mangle]
pub extern "ptx-kernel" fn dwt_forward_kernel(args: &mut WaveletArguments<ComplexScaler<F32>>) {
    let mut cu_shared = CuShared::<F32,8192>::new();
    let block_idx = unsafe { _block_idx_x() } as usize;
    let idx = unsafe { _thread_idx_x() } as usize;
    let block_dim = unsafe { _block_dim_x() } as usize;
    //Get the section I'm supposed to work on
    let sub_data = args
        .x
        .chunks_exact_mut(args.ndwt[0])
        .nth(block_idx)
        .unwrap();

    let mut dwt = DaubechiesFirstComplexWaveletPtx::new(cu_shared);
    //let self_chunk = args.x_copy.chunks_exact_mut(args.ndwt[0]).nth(block_idx).unwrap();

    dwt.forward(sub_data);
    unsafe{_syncthreads()};

    //Get the writeable copy
    (0..sub_data.len()/2)
        .skip(idx)
        .step_by(block_dim)
        .for_each(|i| {
            sub_data[i] = ComplexScaler::new(
                cu_shared.load(4 * i),
                cu_shared.load(4 * i + 1),
            );
            sub_data[i + args.ndwt[0] / 2] = ComplexScaler::new(
                cu_shared.load(4 * i + 2),
                cu_shared.load(4 * i + 3),
            );
        });
}

/*
#[no_mangle]
pub extern "ptx-kernel" fn dwt_backward_kernel(args: &mut WaveletArguments<ComplexScaler<F32>>) {
    let block_idx = unsafe { _block_idx_x() } as usize;
    let idx = unsafe { _thread_idx_x() } as usize;
    let block_dim = unsafe { _block_dim_x() } as usize;
    //Get the section I'm supposed to work on
    let sub_data = &mut args
        .x
        .chunks_exact_mut(args.ndwt[0])
        .nth(block_idx)
        .unwrap();
    let self_chunk = args.x_copy.chunks_exact_mut(args.ndwt[0]).nth(block_idx).unwrap();
    let dwt = DaubechiesFirstComplexWaveletPtx::new(());
    sub_data.chunks_exact_mut(2).enumerate().skip(idx).step_by(block_dim).for_each(|(i,chunk)|{
        chunk[0]=self_chunk[i];
        chunk[1]=self_chunk[i+args.ndwt[0]/2];
    });
    dwt.backward(sub_data);
    //Get the writeable copy

}*/

pub struct DaubechiesFirstComplexWaveletPtx<C: Complex> {
    coefficients: [C; 2],
    shared: CuShared<F32,8192>,
}

impl<C: Complex<Primitive = F32>> DiscreteWavelet<C> for DaubechiesFirstComplexWaveletPtx<C> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type DiscreteWaveletInit = CuShared<F32,8192>;

    fn new(init: Self::DiscreteWaveletInit) -> Self {
        let first = C::new(C::Primitive::usize(2).sqrt().recip(), C::Primitive::ZERO);
        let coefficients = [first, first];
        Self {
            coefficients,
            shared: init,
        }
    }

    //This definition isn't literally correct, as the result is now interspersed
    fn forward(&mut self, input: &mut [C]) {
        let idx = unsafe { _thread_idx_x() } as usize;
        let block_dim = unsafe { _block_dim_x() } as usize;
        input
            .chunks_exact_mut(2)
            .enumerate()
            .skip(idx)
            .step_by(block_dim)
            .for_each(|(i, chunk)| {
                let cache_a = chunk[0] * self.coefficients[0];
                let cache_b = chunk[1] * self.coefficients[1];
                let low = cache_a + cache_b;
                let high = cache_a - cache_b;
                self.shared.store(4 * i, low.real());
                self.shared.store(4 * i + 1, low.imag());
                self.shared.store(4 * i + 2, high.real());
                self.shared.store(4 * i + 3, high.imag());
            });
    }
    //There's an expectation here that the vector was pre-interspersed
    fn backward(&mut self, input: &mut [C]) {
        let idx = unsafe { _thread_idx_x() } as usize;
        let block_dim = unsafe { _block_dim_x() } as usize;
        input
            .chunks_exact_mut(2)
            .skip(idx)
            .step_by(block_dim)
            .for_each(|chunk| {
                let cache_a = chunk[0] * self.coefficients[0];
                let cache_b = chunk[1] * self.coefficients[1];
                chunk[0] = cache_a + cache_b;
                chunk[1] = cache_a - cache_b;
            });
    }
}


/*

    Takes in the data as a mutable slice, 

*/
//#[cfg(target_arch = "nvptx64")]
//use core::arch::nvptx;

use nonphysical_core::{shared::{complex::{Complex, ComplexScaler}, matrix::{heap::MatrixHeap, stack::MatrixStack, Matrix}}, signal::{fourier::{stack::ComplexFourierTransformStack, FourierTransform}, gabor::GaborTransform}};
use crate::cuda::cu_slice::{CuSlice, CuSliceRef};


#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_std::{shared::primitive::F32};
#[cfg(not(target_arch = "nvptx64"))]
use crate::cuda::runtime::Runtime;
#[cfg(not(target_arch = "nvptx64"))]
use std::{borrow::ToOwned,rc::Rc,vec::Vec,marker::PhantomData};


#[cfg(target_arch = "nvptx64")]
use crate::shared::primitive::F32;


const NFFT_HALF: usize = 2048;
pub struct GaborArguments<'a, C: Complex> {
    pub gabor_data: CuSliceRef<'a, C>,
    pub twiddles: CuSlice<'a, C>,
}
#[cfg(not(target_arch = "nvptx64"))]
pub struct GaborTransformCuda<C: Complex>{
    runtime: Rc<Runtime>,
    phantom_data: PhantomData<C>,
}

#[cfg(not(target_arch = "nvptx64"))]
impl<C:Complex> GaborTransform<C> for GaborTransformCuda<C>{
    type GaborInit = (Rc<Runtime>,usize,Vec<C>);
    type GaborMatrix = MatrixHeap<C>;
    fn new(init: Self::GaborInit) -> Self {
        let (runtime,over_sample,window) = init;
        Self{runtime,phantom_data:PhantomData}
    }

    fn gabor(&self, x: &mut [C]) -> Self::GaborMatrix{
        let self_fft = ComplexFourierTransformStack::<C,NFFT_HALF>::new(());
        let twiddles = self_fft.twiddles;
        let x_device = self.runtime.alloc_slice_ref(&x).unwrap();
        let twiddles_device = self.runtime.alloc_slice(&twiddles).unwrap();
        let args = GaborArguments{gabor_data:x_device, twiddles: twiddles_device};
        let _ = self.runtime.launch("gabor_kernel".to_owned(), &args, 70, 1, 1,1024,1,1);
        let x_result = args.gabor_data.to_host().unwrap();
        x.copy_from_slice(&x_result);
        MatrixHeap::zero(1,1)
    }
}



//naive has big buffer in, no copy (global mem accesss), each twiddle
//Let start with
#[no_mangle]
#[cfg(target_arch = "nvptx64")]
pub extern "ptx-kernel" fn gabor_kernel(input: &mut GaborArguments<ComplexScaler<F32>>){
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _thread_idx_x};

    use nonphysical_core::shared::{float::Float, primitive::Primitive};
    //let window_len = 4096;
    //let win_step = window_len/1;
    //let win_count = input.len()/window_len;
    let NFFT = NFFT_HALF*2;

    let idx = unsafe { _thread_idx_x()+_block_idx_x()*_block_dim_x() } as usize;
    //input.chunks_exact_mut(16).filter(predicate)
    //copy.copy_from_slice(&input[idx*window_len..(idx+1)*window_len]);
    //let mut fft = ComplexFourierTransformStack{twiddles: [ComplexScaler::<F32>::ZERO;NFFT_HALF]};
    //input.gabor_data[0]= input.twiddles[0].sin();
    //fft.twiddles.copy_from_slice(&input.twiddles);
    //let chunk = &mut input.gabor_data[idx*NFFT..(idx+1)*NFFT];

    let chunk = input.gabor_data.chunks_exact_mut(NFFT).nth(0).unwrap();
    /* 
    let mut local_chunk = [ComplexScaler::<F32>::ZERO; NFFT_HALF*2];
    local_chunk.copy_from_slice(&chunk);

    local_chunk.iter_mut().for_each(|f|{
        f.real = F32(1.0);
    });*/
    let fft: ComplexFourierTransformStack<ComplexScaler<F32>, NFFT_HALF> = ComplexFourierTransformStack::new(());
    chunk[0..NFFT_HALF].copy_from_slice(&fft.twiddles);
    //chunk[0].real = F32(0.0);
    fft.fft(&mut input.gabor_data[idx*NFFT..(idx+1)*NFFT]);

    //input[idx].real = F32::usize(idx*2+2);

    //let fft: ComplexFourierTransformStack<ComplexScaler<F32>, NFFT_HALF> = ComplexFourierTransformStack::new(());
    //fft.fft(&mut input[idx*NFFT..(idx+1)*NFFT]);
    //input.gabor_data[idx*NFFT_HALF..(idx+1)*NFFT_HALF].copy_from_slice(&input.twiddles);
    //input.gabor_data[0..NFFT_HALF].copy_from_slice(&fft.twiddles);

}

/* 
pub struct Arguments<'a> {
    pub xs: CuSliceRef<'a, ComplexScaler<F32>>,
    pub ys: CuSliceRef<'a, ComplexScaler<F32>>,
}*/
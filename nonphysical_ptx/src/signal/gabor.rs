#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{_syncthreads, _thread_idx_x};

use nonphysical_core::{shared::{complex::{Complex, ComplexScaler}, float::Float}, signal::fourier::{stack::ComplexFourierTransformStack, FourierTransform}};

use crate::{shared::primitive::F32, cuda::cu_slice::{CuSlice,CuSliceRef}};


//This is a 'quick' test to see how much faster gabor is on the GPU, mostly because gabor doesn't have a stack version yet
pub struct Arguments<'a> {
    pub xs: CuSliceRef<'a, ComplexScaler<F32>>,
    pub ys: CuSliceRef<'a, ComplexScaler<F32>>,
}

#[no_mangle]
#[cfg(target_arch = "nvptx64")]
pub extern "ptx-kernel"  fn gabor_cheat(args: &mut Arguments){
    let x = &args.xs;
    let win_size = 8;
    
    //They can all have a fourier, that's fine
    let fourier = ComplexFourierTransformStack::<ComplexScaler<F32>,8>::new(win_size);

    //They can all get these values, that's fine
    let win_step = win_size;
    let win_count = x.len() / win_step;
    let size = win_count *win_size;

    //Use the thread id to know where in the data arrays to access
    //gabor_data[0].real = F32(100.0);
    args.xs[0].real = F32(52.0);

    args.ys[0].real = F32(104.0);

    /* 

    let idx = unsafe { _thread_idx_x() } as usize;
    gabor_data[(idx)*win_size..(idx+1)*win_size].copy_from_slice(&x[idx * win_step..idx * win_step + win_size]);

    //wait for all the threads to finish the copy
    unsafe { _syncthreads() };

    fourier.fft(&mut gabor_data[(idx)*win_size..(idx+1)*win_size]);*/
    

}

#[no_mangle]
#[cfg(target_arch = "nvptx64")]
pub extern "ptx-kernel"  fn gabor2(input: &mut CuSliceRef<ComplexScaler<F32>>){
    use nonphysical_core::shared::float::Float;
    //fft_data.copy_from_slice(&input);

    let fft: ComplexFourierTransformStack<ComplexScaler<F32>, 2047> = ComplexFourierTransformStack::new(4096);
    fft.fft(&mut input[0..2048]);
    //input.copy_from_slice(&fft_data);
}
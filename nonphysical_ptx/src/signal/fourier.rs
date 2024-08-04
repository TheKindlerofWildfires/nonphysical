
/*

    Takes in the data as a mutable slice, 

*/
//#[cfg(target_arch = "nvptx64")]
//use core::arch::nvptx;

use nonphysical_core::{shared::complex::{Complex, ComplexScaler}, signal::fourier::{stack::ComplexFourierTransformStack, FourierTransform}};

use crate::{shared::primitive::F32, cuda::cu_slice::CuSliceRef};



pub struct Arguments<'a> {
    pub x: CuSliceRef<'a, ComplexScaler<F32>>,
    pub twiddles: CuSliceRef<'a, ComplexScaler<F32>>,
}

#[no_mangle]
#[cfg(target_arch = "nvptx64")]
pub extern "ptx-kernel"  fn fourier(input: &mut CuSliceRef<ComplexScaler<F32>>){
    use nonphysical_core::shared::float::Float;

    //fft_data.copy_from_slice(&input);

    let fft: ComplexFourierTransformStack<ComplexScaler<F32>, 0> = ComplexFourierTransformStack::new(512);
    fft.fft(input);
    //input.copy_from_slice(&fft_data);
    input[0].real = F32(input.len() as f32);
}
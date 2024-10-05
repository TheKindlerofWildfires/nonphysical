use crate::{
    cuda::{
        grid::GridStride,
        shared::{CuShared, Shared},
    },
    shared::primitive::F32,
};
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use nonphysical_core::shared::complex::ComplexScaler;

use super::WaveletArguments;

#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_forward_128_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 128;
    daubechies_first_forward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_forward_256_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 256;
    daubechies_first_forward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_forward_512_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 512;
    daubechies_first_forward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_forward_1024_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 1024;
    daubechies_first_forward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_forward_2048_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 2048;
    daubechies_first_forward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_forward_4096_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 4096;
    daubechies_first_forward_kernel::<NDWT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_backward_128_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 128;
    daubechies_first_backward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_backward_256_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 256;
    daubechies_first_backward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_backward_512_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 512;
    daubechies_first_backward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_backward_1024_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 1024;
    daubechies_first_backward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_backward_2048_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 2048;
    daubechies_first_backward_kernel::<NDWT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn daubechies_first_backward_4096_kernel<'a>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    const NDWT: usize = 4096;
    daubechies_first_backward_kernel::<NDWT>(args);
}

fn daubechies_first_forward_kernel<'a, const NDWT: usize>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    let mut x = CuShared::<ComplexScaler<F32>, NDWT>::new();
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    //copy data into shared memory
    let slice = args.input.chunks_exact(NDWT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride(slice);
    thread_data
        .zip((0..NDWT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            x.store(i, *data);
        });

    //run dwt
    let dwt = DaubechiesFirstWaveletPtx::<NDWT>::new(&args.coefficients);
    unsafe { _syncthreads() };
    dwt.forward(thread_id, block_dim, &mut x);
    //copy data back into the slices
    let slice_ref = args.output.chunks_exact_mut(NDWT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride_ref(slice_ref);

    unsafe { _syncthreads() };
    let half_len = NDWT/2;

    let count = (0..half_len).skip(thread_id).step_by(block_dim).count();

    thread_data
        .take(count)
        .zip((0..half_len).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = i * 2 + 1;
            *data = x.load(idx);
        });
    let thread_data = GridStride::thread_stride_ref(slice_ref);

    thread_data
        .skip(count)
        .zip((0..half_len).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = i * 2;
            *data = x.load(idx);
        });
}

fn daubechies_first_backward_kernel<'a, const NDWT: usize>(
    args: &'a mut WaveletArguments<'a, ComplexScaler<F32>>,
) {
    let mut x = CuShared::<ComplexScaler<F32>, NDWT>::new();
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    //copy data into shared memory
    let slice = args.input.chunks_exact(NDWT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride(slice);
    thread_data
        .zip((0..NDWT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            x.store(i, *data);
        });

    //run dwt
    let dwt = DaubechiesFirstWaveletPtx::<NDWT>::new(&args.coefficients);
    unsafe { _syncthreads() };
    dwt.backward(thread_id, block_dim, &mut x);
    //copy data back into the slices
    let slice_ref = args.output.chunks_exact_mut(NDWT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride_ref(slice_ref);
    unsafe { _syncthreads() };
    let half_len = NDWT/2;
    thread_data
        .zip((0..NDWT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = if i%2==0{
                i/2
            }else{
                i/2+half_len
            };

            *data = x.load(idx);
        });
    /* 
    let thread_data = GridStride::thread_stride_ref(slice_ref);

    thread_data
        .skip(1)
        .step_by(2)
        .zip((0..half_len).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = i +half_len;
            *data = x.load(idx);
        });*/
}

pub struct DaubechiesFirstWaveletPtx<'a, const N: usize> {
    coefficients: &'a [ComplexScaler<F32>],
}
impl<'a, const N: usize> DaubechiesFirstWaveletPtx<'a, N> {
    fn new(coefficients: &'a [ComplexScaler<F32>]) -> Self {
        Self { coefficients }
    }
    fn forward(
        &self,
        thread_idx: usize,
        block_dim: usize,
        x: &mut CuShared<ComplexScaler<F32>, N>,
    ) {
        (0..x.len() / 2)
            .skip(thread_idx)
            .step_by(block_dim)
            .for_each(|responsible_idx| {
                let alt_idx = responsible_idx * 2;
                let cache_a = x.load(alt_idx) * self.coefficients[0];
                let cache_b = x.load(alt_idx + 1) * self.coefficients[0];
                x.store(alt_idx, cache_a - cache_b);
                x.store(alt_idx + 1, cache_a + cache_b);
            });
    }

    fn backward(
        &self,
        thread_idx: usize,
        block_dim: usize,
        x: &mut CuShared<ComplexScaler<F32>, N>,
    ) {
        let half_len = x.len()/2;
        (0..half_len)
            .skip(thread_idx)
            .step_by(block_dim)
            .for_each(|responsible_idx| {
                let alt_idx = responsible_idx ;
                let cache_a = x.load(alt_idx) * self.coefficients[0];
                let cache_b = x.load(alt_idx + half_len) * self.coefficients[0];
                x.store(alt_idx, cache_a + cache_b);
                x.store(alt_idx + half_len , cache_a - cache_b);
            });
    }
}

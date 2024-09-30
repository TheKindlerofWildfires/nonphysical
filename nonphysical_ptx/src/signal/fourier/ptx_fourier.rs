//fix plan
/*
    1: make it so the kernel calls a sub fn so iff is easy
    2: make it so a couple of kernels exist for various common sizes
    3: allocate shared memory for the block @ size
    4: make it so everything works on indices so we can shared memory index

*/
use crate::cuda::{
    grid::GridStride,
    shared::{CuShared, Shared},
};
use crate::shared::primitive::F32;
use crate::signal::fourier::FourierArguments;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use core::cmp::min;
use core::mem;
use nonphysical_core::shared::complex::{Complex, ComplexScaler};
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::primitive::Primitive;
#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_128_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 128;
    fft_forward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_256_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 256;
    fft_forward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_512_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 512;
    fft_forward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_1024_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 1024;
    fft_forward_kernel::<NFFT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_2048_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 2048;
    fft_forward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_4096_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 4096;
    fft_forward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_128_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 128;
    fft_backward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_256_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 256;
    fft_backward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_512_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 512;
    fft_backward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_1024_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 1024;
    fft_backward_kernel::<NFFT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_2048_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 2048;
    fft_backward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_4096_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 4096;
    fft_backward_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_shifted_128_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 128;
    fft_forward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_shifted_256_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 256;
    fft_forward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_shifted_512_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 512;
    fft_forward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_shifted_1024_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 1024;
    fft_forward_shifted_kernel::<NFFT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_shifted_2048_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 2048;
    fft_forward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_shifted_4096_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 4096;
    fft_forward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_shifted_128_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 128;
    fft_backward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_shifted_256_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 256;
    fft_backward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_shifted_512_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 512;
    fft_backward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_shifted_1024_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 1024;
    fft_backward_shifted_kernel::<NFFT>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_shifted_2048_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 2048;
    fft_backward_shifted_kernel::<NFFT>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn fft_backward_shifted_4096_kernel<'a>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    const NFFT: usize = 4096;
    fft_backward_shifted_kernel::<NFFT>(args);
}

fn fft_forward_kernel<'a, const NFFT: usize>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    let mut x = CuShared::<ComplexScaler<F32>, NFFT>::new();
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    //copy data into shared memory
    let slice = args.x.chunks_exact(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride(slice);
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            x.store(i, *data);
        });

    //run fft
    let fft = ComplexFourierTransformPtx::<NFFT>::new(&args.twiddles);
    unsafe { _syncthreads() };
    fft.core_fft(thread_id, block_dim, &mut x);
    //copy data back with reversal
    let slice_ref = args.x.chunks_exact_mut(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride_ref(slice_ref);

    unsafe { _syncthreads() };
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = ComplexFourierTransformPtx::<NFFT>::reverse_idx(i + NFFT);
            *data = x.load(idx);
        });
}

fn fft_forward_shifted_kernel<'a, const NFFT: usize>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    let mut x = CuShared::<ComplexScaler<F32>, NFFT>::new();
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    //copy data into shared memory
    let slice = args.x.chunks_exact(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride(slice);
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            x.store(i, *data);
        });

    //run fft
    let fft = ComplexFourierTransformPtx::<NFFT>::new(&args.twiddles);
    unsafe { _syncthreads() };
    fft.core_fft(thread_id, block_dim, &mut x);
    //copy data back with reversal
    let slice_ref = args.x.chunks_exact_mut(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride_ref(slice_ref);

    unsafe { _syncthreads() };
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = ComplexFourierTransformPtx::<NFFT>::reverse_idx((i + NFFT / 2) % NFFT + NFFT);
            *data = x.load(idx);
        });
}

fn fft_backward_kernel<'a, const NFFT: usize>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    let mut x = CuShared::<ComplexScaler<F32>, NFFT>::new();
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    //copy data into shared memory
    let slice = args.x.chunks_exact(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride(slice);
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            x.store(i, data.conjugate());
        });

    //run fft
    let fft = ComplexFourierTransformPtx::<NFFT>::new(&args.twiddles);
    unsafe { _syncthreads() };
    fft.core_fft(thread_id, block_dim, &mut x);
    //copy data back with reversal
    let slice_ref = args.x.chunks_exact_mut(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride_ref(slice_ref);
    let sf = F32::usize(NFFT).recip();
    unsafe { _syncthreads() };
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = ComplexFourierTransformPtx::<NFFT>::reverse_idx(i + NFFT);
            *data = x.load(idx).conjugate() * sf;
        });
}
fn fft_backward_shifted_kernel<'a, const NFFT: usize>(
    args: &'a mut FourierArguments<'a, ComplexScaler<F32>>,
) {
    let mut x = CuShared::<ComplexScaler<F32>, NFFT>::new();
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    //copy data into shared memory
    let slice = args.x.chunks_exact(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride(slice);
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            x.store(i, data.conjugate());
        });

    //run fft
    let fft = ComplexFourierTransformPtx::<NFFT>::new(&args.twiddles);
    unsafe { _syncthreads() };
    fft.core_fft(thread_id, block_dim, &mut x);
    //copy data back with reversal
    let slice_ref = args.x.chunks_exact_mut(NFFT).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride_ref(slice_ref);
    let sf = F32::usize(NFFT).recip();
    unsafe { _syncthreads() };
    thread_data
        .zip((0..NFFT).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            let idx = ComplexFourierTransformPtx::<NFFT>::reverse_idx((i + NFFT / 2) % NFFT + NFFT);
            *data = x.load(idx).conjugate() * sf;
        });
}
pub struct ComplexFourierTransformPtx<'a, const N: usize> {
    twiddles: &'a [ComplexScaler<F32>],
}
impl<'a, const N: usize> ComplexFourierTransformPtx<'a, N> {
    fn new(twiddles: &'a [ComplexScaler<F32>]) -> Self {
        Self { twiddles }
    }
    fn core_fft(
        &self,
        thread_idx: usize,
        block_dim: usize,
        x: &mut CuShared<ComplexScaler<F32>, N>,
    ) {
        let n = x.len().ilog2() as usize;
        let mut step = 1;
        (2..n).rev().for_each(|t| {
            (0..x.len() / 2)
                .skip(thread_idx)
                .step_by(block_dim)
                .for_each(|responsible_idx| {
                    Self::fft_chunk_n(x, &self.twiddles, step, t, responsible_idx);
                });
            step <<= 1;
            unsafe { _syncthreads() };
        });

        (0..x.len() / 2)
            .skip(thread_idx)
            .step_by(block_dim)
            .for_each(|responsible_idx| {
                Self::fft_chunk_4(x, responsible_idx);
            });

        unsafe { _syncthreads() };

        (0..x.len() / 2)
            .skip(thread_idx)
            .step_by(block_dim)
            .for_each(|responsible_idx| {
                Self::fft_chunk_2(x, responsible_idx);
            });
    }

    fn reverse_idx(n: usize) -> usize {
        let mut v = n;
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
        v = (v >> 16) | (v << 16);
        v = v >> min(31, v.trailing_zeros());
        v = (v - 1) / 2;
        v
    }
    fn fft_chunk_n(
        x: &mut CuShared<ComplexScaler<F32>, N>,
        twiddles: &[ComplexScaler<F32>],
        step: usize,
        t: usize,
        idx: usize,
    ) {
        let dist = 1 << t;
        let chunk_size = dist << 1;
        let sub_idx = idx >> t;
        let inner_idx = idx % dist;

        let alt_idx = sub_idx * chunk_size + inner_idx;

        let a = x.load(alt_idx);
        let b = x.load(alt_idx + dist);
        let w = twiddles.iter().step_by(step).nth(inner_idx).unwrap();

        x.store(alt_idx, a + b);
        x.store(alt_idx + dist, (a - b) * *w);
    }

    fn fft_chunk_4(x: &mut CuShared<ComplexScaler<F32>, N>, idx: usize) {
        let sub_idx = (idx >> 1) * 4;
        if idx & 1 == 0 {
            let a = x.load(sub_idx);
            let c = x.load(sub_idx + 2);
            x.store(sub_idx, a + c);
            x.store(sub_idx + 2, a - c);
        } else {
            let b = x.load(sub_idx + 1);
            let d = x.load(sub_idx + 3);
            x.store(sub_idx + 1, b + d);
            x.store(sub_idx + 3, (b - d).mul_ni());
        }
    }

    fn fft_chunk_2(x: &mut CuShared<ComplexScaler<F32>, N>, idx: usize) {
        let alt_idx = idx * 2;
        let a = x.load(alt_idx);
        let b = x.load(alt_idx + 1);
        x.store(alt_idx, a + b);
        x.store(alt_idx + 1, a - b);
    }
}

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
use crate::signal::{
    fourier::ptx_fourier::ComplexFourierTransformPtx,
    cyclostationary::{CyclostationaryIntermediateArguments,CyclostationaryCompleteArguments},
};
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use core::marker::PhantomData;
use nonphysical_core::shared::primitive::Primitive;
use nonphysical_core::shared::complex::{Complex, ComplexScaler};
use nonphysical_core::shared::float::Float;
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_intermediate_128_kernel<'a>(
    args: &'a mut CyclostationaryIntermediateArguments<'a, ComplexScaler<F32>>,
) {
    const NCST: usize = 128;
    cyclo_intermediate_kernel::<NCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_intermediate_256_kernel<'a>(
    args: &'a mut CyclostationaryIntermediateArguments<'a, ComplexScaler<F32>>,
) {
    const NCST: usize = 256;
    cyclo_intermediate_kernel::<NCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_intermediate_512_kernel<'a>(
    args: &'a mut CyclostationaryIntermediateArguments<'a, ComplexScaler<F32>>,
) {
    const NCST: usize = 512;
    cyclo_intermediate_kernel::<NCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_intermediate_1024_kernel<'a>(
    args: &'a mut CyclostationaryIntermediateArguments<'a, ComplexScaler<F32>>,
) {
    const NCST: usize = 1024;
    cyclo_intermediate_kernel::<NCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_intermediate_2048_kernel<'a>(
    args: &'a mut CyclostationaryIntermediateArguments<'a, ComplexScaler<F32>>,
) {
    const NCST: usize = 2048;
    cyclo_intermediate_kernel::<NCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_intermediate_4096_kernel<'a>(
    args: &'a mut CyclostationaryIntermediateArguments<'a, ComplexScaler<F32>>,
) {
    const NCST: usize = 4096;
    cyclo_intermediate_kernel::<NCST>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_complete_2_kernel<'a>(
    args: &'a mut CyclostationaryCompleteArguments<'a, ComplexScaler<F32>>,
) {
    const SCST: usize = 2;
    cyclo_complete_kernel::<SCST>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_complete_4_kernel<'a>(
    args: &'a mut CyclostationaryCompleteArguments<'a, ComplexScaler<F32>>,
) {
    const SCST: usize = 4;
    cyclo_complete_kernel::<SCST>(args);
}

#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_complete_8_kernel<'a>(
    args: &'a mut CyclostationaryCompleteArguments<'a, ComplexScaler<F32>>,
) {
    const SCST: usize = 8;
    cyclo_complete_kernel::<SCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_complete_16_kernel<'a>(
    args: &'a mut CyclostationaryCompleteArguments<'a, ComplexScaler<F32>>,
) {
    const SCST: usize = 16;
    cyclo_complete_kernel::<SCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_complete_32_kernel<'a>(
    args: &'a mut CyclostationaryCompleteArguments<'a, ComplexScaler<F32>>,
) {
    const SCST: usize = 32;
    cyclo_complete_kernel::<SCST>(args);
}
#[no_mangle]
pub extern "ptx-kernel" fn cyclostationary_complete_64_kernel<'a>(
    args: &'a mut CyclostationaryCompleteArguments<'a, ComplexScaler<F32>>,
) {
    const SCST: usize = 64;
    cyclo_complete_kernel::<SCST>(args);
}

fn cyclo_intermediate_kernel<'a, const NCST: usize>(
    args: &'a mut CyclostationaryIntermediateArguments<'a, ComplexScaler<F32>>,
) {
    let mut x = CuShared::<ComplexScaler<F32>, NCST>::new();
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    //copy data into shared memory
    let slice = args.x.chunks_exact(NCST).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride(slice);
    thread_data
        .zip((0..NCST).skip(thread_id).step_by(block_dim))
        .for_each(|(data, i)| {
            x.store(i, *data);
        });

    //run fft
    let fft = ComplexFourierTransformPtx::<NCST>::new(&args.twiddles);
    unsafe { _syncthreads() };
    fft.core_fft(thread_id, block_dim, &mut x);
    //copy data back with reversal
    let slice_ref = args.x.chunks_exact_mut(NCST).nth(block_id).unwrap();
    let thread_data = GridStride::thread_stride_ref(slice_ref);

    unsafe { _syncthreads() };
    thread_data
        .zip((0..NCST).skip(thread_id).step_by(block_dim))
        .zip(args.phase.iter().skip(thread_id).step_by(block_dim))
        .for_each(|((data, i),phase)| {
            let idx = ComplexFourierTransformPtx::<NCST>::reverse_idx((i + NCST / 2) % NCST + NCST);
            let update = (*phase*F32::usize(block_id)).exp();
            *data = x.load(idx)*update;
        });
}

fn cyclo_complete_kernel<'a, const SCST: usize>(
    args: &'a mut CyclostationaryCompleteArguments<'a, ComplexScaler<F32>>,
) {
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };
    //this could would be a *lot* faster in shared memory and there's probably room
    //let col = args.x.chunks_exact(SCST).nth(block_id).unwrap();
    let i_col = args.x.iter().skip(block_id).step_by(block_dim);

    //let personal_col = args.x.chunks_exact(SCST).nth(thread_id).unwrap();
    let j_col = args.x.iter().skip(thread_id).step_by(block_dim);
    //then each thread gets a small FFT to to do and copy in
    let result_row = args.result.chunks_exact_mut(block_dim*SCST).nth(block_id).unwrap();
    let mut result_chunk = result_row.chunks_exact_mut(SCST).nth(thread_id).unwrap();
    //move the base data into the output 
    result_chunk.iter_mut().zip(i_col).zip(j_col).for_each(|((rc,ic),jc)|{
        *rc = *ic*jc.conjugate()
    });
    //FFT the result_chunk
    
    CycloFourier::cyclo_fft_shift(&mut result_chunk, &args.twiddles);
    //result_chunk[0].real = F32::PI;
    return;
}

pub struct CycloFourier<C:Complex>{
    phantom_data: PhantomData<C>,
}

impl<C:Complex> CycloFourier<C>{
    fn cyclo_fft_shift(x: &mut[C], twiddles: &[C]){
        let n: usize = x.len().ilog2() as usize;
        (0..n).rev().for_each(|t| {
            let dist = 1 << t;
            let chunk_size = dist << 1;

            if chunk_size > 4 {
                Self::fft_chunk_n(x, twiddles, dist);
            } else if chunk_size == 2 {
                Self::fft_chunk_2(x);
            } else if chunk_size == 4 {
                Self::fft_chunk_4(x);
            }
        });
        Self::reverse(x, n);
        Self::shift(x);
    }
    #[inline]
    fn fft_chunk_n(x: &mut [C], twiddles: &[C], dist: usize) {
        let chunk_size = dist << 1;
        x.chunks_exact_mut(chunk_size).for_each(|chunk| {
            let (c_s0, c_s1) = chunk.split_at_mut(dist);
            c_s0
                .iter_mut()
                .zip(c_s1.iter_mut())
                .zip(twiddles.iter().step_by(dist))
                .for_each(|((c_s0, c_s1), w)| {
                    let temp = *c_s0 - *c_s1;
                    *c_s0 += *c_s1;
                    *c_s1 = temp * *w;
                });
        });
    }

    #[inline]
    fn fft_chunk_4(x: &mut [C]) {
        x.chunks_exact_mut(4).for_each(|chunk| {
            let (c_s0, c_s1) = chunk.split_at_mut(2);

            let temp = c_s0[0];
            c_s0[0] += c_s1[0];
            c_s1[0] = temp - c_s1[0];

            let temp = c_s0[1];
            c_s0[1] += c_s1[1];
            c_s1[1] = (temp - c_s1[1]).mul_ni();
        });
    }

    #[inline]
    fn fft_chunk_2(x: &mut [C]) {
        x.chunks_exact_mut(2).for_each(|chunk| {
            let temp = chunk[0];
            chunk[0] += chunk[1];
            chunk[1] = temp - chunk[1];
        });
    }
    #[inline]
    fn reverse(buf: &mut [C], log_n: usize) {
        let big_n = 1 << log_n;
        let half_n = big_n >> 1;
        let quart_n = big_n >> 2;
        let min = big_n - 1;

        let mut forward = half_n;
        let mut rev = 1;
        (0..quart_n).rev().for_each(|i: usize| {
            let zeros = i.trailing_ones();

            forward ^= 2 << zeros;
            rev ^= quart_n >> zeros;

            if forward < rev {
                
                buf.swap(forward, rev);
                buf.swap(min ^ forward, min ^ rev);
            }

            buf.swap(forward ^ 1, rev ^ half_n);
        });
    }
    fn shift(x: &mut [C]) {
        let half = x.len()/2;
        let mut chunks = x.chunks_exact_mut(half);
        let c1 = chunks.next().unwrap();
        let c2 = chunks.next().unwrap();
        c1.iter_mut().zip(c2.iter_mut()).for_each(|(a,b)|{
            core::mem::swap(a,b);
        });
    }
}


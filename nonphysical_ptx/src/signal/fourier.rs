/*

    Takes in the data as a mutable slice,

*/
//#[cfg(target_arch = "nvptx64")]
//use core::arch::nvptx;

use crate::cuda::cu_slice::{CuSlice, CuSliceRef};
use nonphysical_core::{
    shared::complex::{Complex, ComplexScaler},
    signal::fourier::{
        heap::ComplexFourierTransformHeap, stack::ComplexFourierTransformStack, FourierTransform,
    },
};

#[cfg(not(target_arch = "nvptx64"))]
use crate::cuda::runtime::Runtime;
#[cfg(not(target_arch = "nvptx64"))]
use nonphysical_std::shared::primitive::F32;

#[cfg(not(target_arch = "nvptx64"))]
use std::{borrow::ToOwned, dbg, marker::PhantomData, rc::Rc,cmp::min};

#[cfg(target_arch = "nvptx64")]
use crate::shared::primitive::F32;

#[cfg(not(target_arch = "nvptx64"))]
pub struct ComplexFourierTransformCuda<C: Complex> {
    runtime: Rc<Runtime>,
    phantom_data: PhantomData<C>,
    blocks: usize,
}

#[cfg(not(target_arch = "nvptx64"))]
impl<C: Complex> FourierTransform<C> for ComplexFourierTransformCuda<C> {
    type FourierInit = (Rc<Runtime>, usize);
    fn new(init: Self::FourierInit) -> Self {
        let (runtime, blocks) = init;
        Self {
            runtime,
            phantom_data: PhantomData,
            blocks,
        }
    }

    fn fft(&self, x: &mut [C]) {
        debug_assert!(x.len() > 4);
        debug_assert!(x.len() % 2 == 0);
        let nfft = 4096;
        let block_size = 2048;
        let reference_fft = ComplexFourierTransformHeap::<C>::new(nfft);
        //let reference_fft = ComplexFourierTransformStack::<C,8>::new(());
        let x_device = self.runtime.alloc_slice_ref(&x).unwrap();
        let t_device = self.runtime.alloc_slice(&reference_fft.twiddles).unwrap();
        let args = FourierArguments {
            x: x_device,
            twiddles: t_device,
        };
        let threads = min(1024,nfft/2);

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

    fn ifft(&self, x: &mut [C]) {
        todo!()
    }
}
pub struct FourierArguments<'a, C: Complex> {
    pub x: CuSliceRef<'a, C>,
    pub twiddles: CuSlice<'a, C>,
}
#[no_mangle]
#[cfg(target_arch = "nvptx64")]
pub extern "ptx-kernel" fn fft_forward_kernel(args: &mut FourierArguments<ComplexScaler<F32>>) {
    use core::arch::nvptx::_block_dim_x;

    use nonphysical_core::shared::{float::Float, primitive::Primitive};
    let fft = ComplexFourierTransformPtx {};
    let block_idx = unsafe { _block_idx_x() } as usize;
    let block_size = unsafe { _block_dim_x() } as usize;
    //let my_chunk = &mut args.x[block_idx*2048..(block_idx+1)*2048];
    args
        .x
        .chunks_exact_mut(args.twiddles.len() * 2)
        .skip(block_idx)
        .step_by(block_size)//this handles the case where we have a larger fft than threads
        .for_each(|chunk| {
            fft.fft(chunk, &args.twiddles);
        });
}
/*
#[no_mangle]
#[cfg(target_arch = "nvptx64")]
pub extern "ptx-kernel" fn fft_backward_kernel(input: &mut CuSliceRef<ComplexScaler<F32>>) {
    let fft: ComplexFourierTransformStack<ComplexScaler<F32>, 2048> =
        ComplexFourierTransformStack::new(());
    fft.ifft(input);
}*/

//premise is I'm going to operate on the fft with n/2 threads (lets say target is 2048)
//and threads will fall off as I stop being able to use them
#[cfg(target_arch = "nvptx64")]
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _syncthreads, _thread_idx_x},
};

#[cfg(target_arch = "nvptx64")]
use nonphysical_core::shared::float::Float;
#[cfg(target_arch = "nvptx64")]
pub struct ComplexFourierTransformPtx {}
#[cfg(target_arch = "nvptx64")]
impl ComplexFourierTransformPtx {
    fn fft(&self, x: &mut [ComplexScaler<F32>], twiddles: &[ComplexScaler<F32>]) {
        let idx = unsafe { _thread_idx_x() } as usize;
        let n = x.len().ilog2() as usize;
        let mut step = 1;
        (2..n).rev().for_each(|t| {
            Self::fft_chunk_n(x, &twiddles, step, t, idx);
            step <<= 1;
            unsafe { _syncthreads() };
        });
        Self::fft_chunk_4(x, idx);
        unsafe { _syncthreads() };
        Self::fft_chunk_2(x, idx);
        unsafe { _syncthreads() };
        //Self::reverse(x,n);
    }

    fn fft_chunk_n(
        x: &mut [ComplexScaler<F32>],
        twiddles: &[ComplexScaler<F32>],
        step: usize,
        t: usize,
        idx: usize,
    ) {
        let dist = 1 << t;
        let chunk_size = dist << 1;
        //Bad assumptions here around idx, pretending it's right for the first pass
        let sub_idx = idx >> t; //wrong for t=2,idx=4
        let inner_idx = idx % dist;

        let chunk = x.chunks_exact_mut(chunk_size).nth(sub_idx).unwrap(); //this can't be idx, it needs to be sub idx
        let (c_s0, c_s1) = chunk.split_at_mut(dist);

        let my_cs0 = c_s0.iter_mut().nth(inner_idx).unwrap();
        let my_cs1 = c_s1.iter_mut().nth(inner_idx).unwrap();
        let my_twiddle = twiddles.iter().step_by(step).nth(inner_idx).unwrap();

        let mut tmp1: F32 = F32(0.0);
        let mut tmp2: F32 = F32(0.0);

        unsafe {
            asm!(
                "sub.rn.ftz.f32 {tr}, {r1}, {r2};",
                "    sub.rn.ftz.f32 {ti}, {i1}, {i2};",
                "    add.rn.ftz.f32 {r1}, {r1}, {r2};", //set r1
                "    add.rn.ftz.f32 {i1}, {i1}, {i2};", //set i1
                "    mul.rn.ftz.f32 {r2}, {ti}, {wi};",
                "    neg.ftz.f32 {r2}, {r2};",
                "    fma.rn.ftz.f32 {r2}, {tr}, {wr}, {r2};", //set r2
                "    mul.rn.ftz.f32 {i2}, {ti}, {wr};",
                "    fma.rn.ftz.f32 {i2}, {tr}, {wi}, {i2};", //set i2
                r1 = inout(reg32) my_cs0.real.0,
                r2 = inout(reg32) my_cs1.real.0,
                i1 = inout(reg32) my_cs0.imag.0,
                i2 = inout(reg32) my_cs1.imag.0,
                wr = in(reg32) my_twiddle.real.0,
                wi = in(reg32) my_twiddle.imag.0,
                tr = out(reg32) tmp1.0,
                ti = out(reg32) tmp2.0,
            );
        }
    }

    fn fft_chunk_2(x: &mut [ComplexScaler<F32>], idx: usize) {
        let chunk = x.chunks_exact_mut(2).nth(idx).unwrap();

        let mut tmp: F32 = F32(0.0);

        unsafe {
            asm!(
                "add.rn.ftz.f32 {t}, {r1}, {r2};", //Add the reals into a temp value
                "    sub.rn.ftz.f32 {r2}, {r1}, {r2};", //do the simple add
                "    mov.f32 {r1}, {t};", //mov the tmp value into place
                "    add.rn.ftz.f32 {t}, {i1}, {i2};", //Add the reals into a temp value
                "    sub.rn.ftz.f32 {i2}, {i1}, {i2};", //do the simple add
                "    mov.f32 {i1}, {t};", //mov the tmp value into place
                r1 = inout(reg32) chunk[0].real.0,
                r2 = inout(reg32) chunk[1].real.0,
                i1 = inout(reg32) chunk[0].imag.0,
                i2 = inout(reg32) chunk[1].imag.0,
                t = out(reg32) tmp.0,
            );
        }
    }
    fn fft_chunk_4(x: &mut [ComplexScaler<F32>], idx: usize) {
        let sub_idx = idx >> 1;
        let chunk = x.chunks_exact_mut(4).nth(sub_idx).unwrap();
        let (c_s0, c_s1) = chunk.split_at_mut(2);
        let mut tmp_real: F32 = F32(0.0);
        let mut tmp_imag: F32 = F32(0.0);
        if idx % 2 == 0 {
            unsafe {
                asm!(
                    "add.rn.ftz.f32 {tr}, {r1}, {r2};",
                    "   add.rn.ftz.f32 {ti}, {i1}, {i2};",
                    "   sub.rn.ftz.f32 {r2}, {r1},{r2};",
                    "   mov.f32 {r1}, {tr};",
                    "   sub.rn.ftz.f32 {i2}, {i1},{i2};",
                    "   mov.f32 {i1}, {ti};",
                    r1 = inout(reg32) c_s0[0].real.0,
                    r2 = inout(reg32) c_s1[0].real.0,
                    i1 = inout(reg32) c_s0[0].imag.0,
                    i2 = inout(reg32) c_s1[0].imag.0,
                    tr = out(reg32) tmp_real.0,
                    ti = out(reg32) tmp_imag.0,
                );
            }
        } else {
            unsafe {
                asm!(
                    "   add.rn.ftz.f32 {tr}, {r1}, {r2};",
                    "   sub.rn.ftz.f32 {ti}, {r2},{r1};",
                    "   sub.rn.ftz.f32 {r2}, {i1},{i2};",
                    "   mov.f32 {r1}, {tr};",
                    "   add.rn.ftz.f32 {i1}, {i1}, {i2};",
                    "   mov.f32 {i2}, {ti};",
                    r1 = inout(reg32) c_s0[1].real.0,
                    r2 = inout(reg32) c_s1[1].real.0,
                    i1 = inout(reg32) c_s0[1].imag.0,
                    i2 = inout(reg32) c_s1[1].imag.0,
                    tr = out(reg32) tmp_real.0,
                    ti = out(reg32) tmp_imag.0,
                );
            }
        }
    }
    /*
    //this does not seem like a parra alg (at best I could precalc the trailing zeros)
    //maybe the cobra
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
    }*/
}

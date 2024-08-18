use crate::shared::primitive::F32;
#[cfg(target_arch = "nvptx64")]
use core::{
    arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x},
    cmp::min,
};
use nonphysical_core::{
    shared::
        complex::{Complex, ComplexScaler}
    ,
    signal::fourier::FourierTransform,
};
use super::FourierArguments;
#[no_mangle]
pub extern "ptx-kernel" fn fft_forward_kernel(args: &mut FourierArguments<ComplexScaler<F32>>) {
    let block_idx = unsafe { _block_idx_x() } as usize;
    //Get the section I'm supposed to work on
    let sub_data = &mut args
        .x
        .chunks_exact_mut(args.twiddles.len() * 2)
        .nth(block_idx)
        .unwrap();

    let fft = ComplexFourierTransformPtx::new(&args.twiddles);
    fft.fft(sub_data);
}

pub struct ComplexFourierTransformPtx<'a, C: Complex + 'a> {
    twiddles: &'a [C],
}

impl<'a, C: Complex + 'a> FourierTransform<C> for ComplexFourierTransformPtx<'a,C> {
    type FourierInit = &'a [C];

    fn new(init: Self::FourierInit) -> Self {
        let twiddles = init;
        Self { twiddles }
    }

    fn fft(&self, x: &mut [C]) {
        let idx = unsafe { _thread_idx_x() } as usize;
        let block_dim = unsafe { _block_dim_x() } as usize;
        let n = x.len().ilog2() as usize;
        let mut step = 1;
        (2..n).rev().for_each(|t| {
            (0..x.len() / 2)
                .skip(idx)
                .step_by(block_dim)
                .for_each(|responsible_idx| {
                    Self::fft_chunk_n(x, &self.twiddles, step, t, responsible_idx);
                });
            step <<= 1;
            unsafe { _syncthreads() };
        });

        (0..x.len() / 2)
            .skip(idx)
            .step_by(block_dim)
            .for_each(|responsible_idx| {
                Self::fft_chunk_4(x, responsible_idx);
            });
        unsafe { _syncthreads() };

        (0..x.len() / 2)
            .skip(idx)
            .step_by(block_dim)
            .for_each(|responsible_idx| {
                Self::fft_chunk_2(x, responsible_idx);
            });
        unsafe { _syncthreads() };
        //this will overflow if NFFT/threads>4, which starts at nfft 8192, this is also where shared memory ends
        assert!(x.len() / block_dim <= 4);
        let mut local = [C::ZERO; 8];
        (0..x.len())
            .skip(idx)
            .step_by(block_dim)
            .zip(local.iter_mut())
            .for_each(|(i, local_p)| {
                let n = i + x.len();
                *local_p = Self::reverse_value(x, n);
            });
        unsafe { _syncthreads() };
        (0..x.len())
            .skip(idx)
            .step_by(block_dim)
            .zip(local.iter())
            .for_each(|(i, local_p)| {
                x[i] = *local_p;
            });
    }

    fn ifft(&self, _x: &mut [C]) {
        todo!()
    }
}

impl<'a, C: Complex+'a> ComplexFourierTransformPtx<'a,C> {
    fn reverse_value(x: &mut [C], n: usize) -> C {
        let mut v = n;
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
        v = (v >> 16) | (v << 16);
        v = v >> min(31, v.trailing_zeros());
        v = (v - 1) / 2;
        x[v]
    }
    fn fft_chunk_n(
        x: &mut [C],
        twiddles: &[C],
        step: usize,
        t: usize,
        idx: usize,
    ) {
        let dist = 1 << t;
        let chunk_size = dist << 1;
        let sub_idx = idx >> t;
        let inner_idx = idx % dist;

        let chunk = x.chunks_exact_mut(chunk_size).nth(sub_idx).unwrap();
        let (c_s0, c_s1) = chunk.split_at_mut(dist);

        let i_cs0 = c_s0.iter_mut().nth(inner_idx).unwrap();
        let i_cs1 = c_s1.iter_mut().nth(inner_idx).unwrap();
        let w = twiddles.iter().step_by(step).nth(inner_idx).unwrap();

        let temp = *i_cs0 - *i_cs1;
        *i_cs0 += *i_cs1;
        *i_cs1 = temp * *w;

    }

    fn fft_chunk_4(x: &mut [C], idx: usize) {
        let sub_idx = idx >> 1;
        let chunk = x.chunks_exact_mut(4).nth(sub_idx).unwrap();
        let (c_s0, c_s1) = chunk.split_at_mut(2);
        if idx & 1 == 0 {
            let temp = c_s0[0];
            c_s0[0] += c_s1[0];
            c_s1[0] = temp - c_s1[0];

        } else {
            let temp = c_s0[1];
            c_s0[1] += c_s1[1];
            c_s1[1] = (temp - c_s1[1]).mul_ni();
        }
    }
    fn fft_chunk_2(x: &mut [C], idx: usize) {
        let chunk = x.chunks_exact_mut(2).nth(idx).unwrap();

        let temp = chunk[0];
        chunk[0] += chunk[1];
        chunk[1] = temp - chunk[1];
    }
}

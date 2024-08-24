use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use core::cmp::min;
use nonphysical_core::shared::{float::Float, primitive::Primitive};
use crate::shared::vector::vector_ptx::vector_index;
use crate::{
    cuda::{
        atomic::Reduce,
        shared::{CuShared, Shared},
    },
    shared::{primitive::F32, vector::*},
};


#[no_mangle]
pub extern "ptx-kernel" fn real_vector_mean_f32(args: &mut VectorArgumentsReduce<F32>) {
    let thread_idx = unsafe { _thread_idx_x() } as usize;
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    let mut block_acc = CuShared::<F32, 1>::new();
    stop = min(stop, args.data.len());
    let data = &args.data;

    if thread_idx == 0 {
        block_acc.store(0, F32::ZERO);
    }
    //Divide before the acc to reduce float errors
    let thread_acc =
        data[start..stop].iter().fold(F32::ZERO, |acc, d| acc + *d) / F32::usize(CYCLE_COMPARE);
    block_acc.reduce_add(0, thread_acc);
    unsafe { _syncthreads() };

    if thread_idx == 0 {
        let tmp = block_acc.load(0)*F32::usize(CYCLE_COMPARE)/F32::usize( args.data.len());
        args.acc.reduce_add(0,tmp) ;
    }
    return;
}

#[no_mangle]
pub extern "ptx-kernel" fn real_vector_variance_f32(args: &mut VectorArgumentsReduce<F32>) {
    let thread_idx = unsafe { _thread_idx_x() } as usize;
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    let mut block_acc = CuShared::<F32, 1>::new();
    stop = min(stop, args.data.len());
    let data = &args.data;

    if thread_idx == 0 {
        block_acc.store(0, F32::ZERO);
    }
    let mean = args.acc[0];
    //Divide before the acc to reduce float errors
    let thread_acc =
        data[start..stop].iter().fold(F32::ZERO, |acc, d| acc + (mean-*d).l2_norm()) / F32::usize(CYCLE_COMPARE);
    block_acc.reduce_add(0, thread_acc);
    unsafe { _syncthreads() };

    if thread_idx == 0 {
        let tmp = block_acc.load(0)*F32::usize(CYCLE_COMPARE)/F32::usize( args.data.len());
        args.acc.reduce_add(1,tmp);
    }
    return;
}
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use core::cmp::min;
use nonphysical_core::shared::{float::Float, primitive::Primitive};

use super::CYCLE_COMPARE;
use crate::{
    cuda::{
        atomic::Reduce,
        shared::{CuShared, Shared},
    },
    shared::{primitive::F32, vector::*},
};
fn vector_index() -> (usize, usize) {
    let thread_idx = unsafe { _thread_idx_x() } as usize;
    let block_idx = unsafe { _block_idx_x() } as usize;
    let block_size = unsafe { _block_dim_x() } as usize;

    let start = (thread_idx + block_idx * block_size) * CYCLE_COMPARE;
    let stop = start + CYCLE_COMPARE;
    return (start, stop);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sum_f32(args: &mut VectorArgumentsReduce<F32>) {
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
    //do in register sums
    let thread_acc = data[start..stop].iter().fold(F32::ZERO, |acc, d| acc + *d);
    block_acc.reduce_add(0, thread_acc);

    unsafe { _syncthreads() };
    if thread_idx == 0 {
        args.acc.reduce_add(0, block_acc.load(0));
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_l1_sum_f32(args: &mut VectorArgumentsReduce<F32>) {
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
    //do in register sums
    let thread_acc = data[start..stop]
        .iter()
        .fold(F32::ZERO, |acc, d| acc + d.l1_norm());
    block_acc.reduce_add(0, thread_acc);

    unsafe { _syncthreads() };
    if thread_idx == 0 {
        args.acc.reduce_add(0, block_acc.load(0));
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_l2_sum_f32(args: &mut VectorArgumentsReduce<F32>) {
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
    //do in register sums
    let thread_acc = data[start..stop]
        .iter()
        .fold(F32::ZERO, |acc, d| acc + d.l2_norm());
    block_acc.reduce_add(0, thread_acc);

    unsafe { _syncthreads() };
    if thread_idx == 0 {
        args.acc.reduce_add(0, block_acc.load(0));
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_add_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());

    let other = args.map[0];
    let data = &mut args.data;
    //add each cyle
    data[start..stop].iter_mut().for_each(|d| *d += other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sub_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());

    let other = args.map[0];
    let data = &mut args.data;
    //add each cyle
    data[start..stop].iter_mut().for_each(|d| *d -= other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_mul_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());

    let other = args.map[0];
    let data = &mut args.data;
    //add each cyle
    data[start..stop].iter_mut().for_each(|d| *d *= other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_div_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());

    let other = args.map[0];
    let data = &mut args.data;
    //add each cyle
    data[start..stop].iter_mut().for_each(|d| *d /= other);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_add_vec_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());
    let other = &args.map;
    let data = &mut args.data;
    data[start..stop]
        .iter_mut()
        .zip(other[start..stop].iter())
        .for_each(|(d, o)| *d += *o);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_sub_vec_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());
    let other = &args.map;
    let data = &mut args.data;
    data[start..stop]
        .iter_mut()
        .zip(other[start..stop].iter())
        .for_each(|(d, o)| *d -= *o);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_mul_vec_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());
    let other = &args.map;
    let data = &mut args.data;
    data[start..stop]
        .iter_mut()
        .zip(other[start..stop].iter())
        .for_each(|(d, o)| *d *= *o);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_div_vec_f32(args: &mut VectorArgumentsMap<F32>) {
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    stop = min(stop, args.data.len());
    let other = &args.map;
    let data = &mut args.data;
    data[start..stop]
        .iter_mut()
        .zip(other[start..stop].iter())
        .for_each(|(d, o)| *d /= *o);
}

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
/*
	ld.global.nc.u64 	%rd2, [%rd1+24];
	setp.lt.u64 	%p3, %rd2, 2;
	@%p3 bra 	$L__BB13_6;
	ld.global.nc.u64 	%rd11, [%rd1+16];
	st.u32 	[%rd11+4], %r9;
	// begin inline asm
	red.global.add.f32 [%rd11], %r9;
	// end inline asm*/
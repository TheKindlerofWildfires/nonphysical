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
}use crate::cuda::atomic::Atomic;
#[no_mangle]
pub extern "ptx-kernel" fn vector_l1_min_f32(args: &mut VectorArgumentsReducePrim<F32>) {
    let thread_idx = unsafe { _thread_idx_x() } as usize;
    let (start, mut stop) = vector_index();
    if start > args.data.len() {
        return;
    }
    let mut block_acc = CuShared::<F32, 1>::new();
    stop = min(stop, args.data.len());
    let data = &args.data;

    if thread_idx == 0 {
        block_acc.store(0, F32::MAX);
    }
    //do in register sums
    let thread_acc = data[start..stop].iter().fold(F32::MAX, |acc, d| d.lesser(acc));

    if thread_idx==0{
        let mut old = args.acc[0];
        let mut assumed: F32;
        let mut value = thread_acc;
        let assumed=old;
        old = block_acc.atomic_cas(0,assumed,value);
        /* 
        let mut old  = self.load(index);
        let mut assumed ;
        let mut value = value;
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.lesser(value);
            if old==assumed{
                break
            }
        }*/
        args.acc[0]=old;
    }

    return;
    block_acc.reduce_min(0, thread_acc);

    unsafe { _syncthreads() };
    if thread_idx == 0 {
        args.acc.reduce_min(0, block_acc.load(0));
        args.acc[0]=thread_acc;
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

pub extern "ptx-kernel" fn generic_1337<F:Float>(args: &mut VectorArgumentsMap<F>) {
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
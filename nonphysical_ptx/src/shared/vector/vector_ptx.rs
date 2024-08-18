use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x,_syncthreads};

use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::primitive::Primitive;
use crate::{
    cuda::shared::{CuShared, Shared},
    shared::primitive::F32,
};
use crate::cuda::atomic::Atomic;
use super::CYCLE_COMPARE;
use crate::shared::vector::VectorArgumentsReduce;
use crate::cuda::global::device::CuGlobalSliceRef;
#[no_mangle]
pub extern "ptx-kernel" fn vector_sum_f32(args: &mut CuGlobalSliceRef<u32>) {
    //args[0]=1;
    args[0]=1;
}
/* 
pub extern "ptx-kernel" fn vector_sum_f32(args: &mut VectorArgumentsReduce<F32>) {
    args.acc[0]=F32::ONE;
    return;
    let thread_idx = unsafe { _thread_idx_x() } as usize;
    let block_idx = unsafe { _block_idx_x() } as usize;
    let block_size = unsafe { _block_dim_x() } as usize;

    //stops buffer over reads
    if thread_idx * CYCLE_COMPARE + block_idx * block_size + CYCLE_COMPARE > args.data.len() {
        return;
    }

    let grid_acc = &mut args.acc;
    let data = &args.data;
    let mut block_acc = CuShared::<F32, 1>::new();
    let global_start = thread_idx * CYCLE_COMPARE + block_idx * block_size;

    //do in register sums
    let thread_acc =
        data[global_start..global_start + CYCLE_COMPARE].iter().fold(F32::ZERO, |acc, d| acc + *d);

    //before merging set up sub_fold
    if thread_idx == 0 {
        block_acc.store(0, F32::ZERO);
    }
    //combine the data atomically to the block
    block_acc.atomic_add(0,thread_acc);
    unsafe{_syncthreads();};
    if thread_idx == 0{
        //combine the data atomically to the grid
        grid_acc.atomic_add(0,block_acc.load(0));
    }


}
*/
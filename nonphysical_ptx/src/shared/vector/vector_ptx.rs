use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};

use nonphysical_core::shared::float::Float;

use crate::{
    cuda::{
        global::CuGlobalSlice,
        shared::{CuShared, Shared},
    },
    shared::primitive::F32,
};

use super::CYCLE_COMPARE;

#[no_mangle]
pub extern "ptx-kernel" fn vector_sum_f32(args: &mut VectorArgumentsReduce<F32>) {
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
        data[global_start..global_start + CYCLE_COMPARE].fold(F32::ZERO, |acc, d| acc + *d);

    //before merging set up sub_fold
    if thread_idx == 0 {
        block_acc.store(0, F32::ZERO);
    }
    //combine the data atomically to the block
    block_acc.atomic_add(0,thread_acc);
    unsafe{__sync_threads()};
    //combine the data atomically to the grid
    grid_acc.atomic_add(0,block_acc);

}

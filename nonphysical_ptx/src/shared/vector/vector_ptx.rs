pub mod vector_ptx;
pub mod real_vector_ptx;
pub mod point_vector_ptx;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};

use super::CYCLE_COMPARE;

fn vector_index() -> (usize, usize) {
    let thread_idx = unsafe { _thread_idx_x() } as usize;
    let block_idx = unsafe { _block_idx_x() } as usize;
    let block_size = unsafe { _block_dim_x() } as usize;

    let start = (thread_idx + block_idx * block_size) * CYCLE_COMPARE;
    let stop = start + CYCLE_COMPARE;
    return (start, stop);
}
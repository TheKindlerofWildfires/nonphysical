use crate::cuda::global::device::{CuGlobalSlice, CuGlobalSliceRef};
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _grid_dim_x, _thread_idx_x};
pub struct GridStride {}

impl GridStride {
    pub fn stride<'a, T: Copy + 'a>(
        data: &'a CuGlobalSlice<'a, T>,
    ) -> impl Iterator<Item = &'a T> + 'a {
        let (thread_id, block_dim, block_id, grid_dim) = unsafe {
            (
                _thread_idx_x() as usize,
                _block_dim_x() as usize,
                _block_idx_x() as usize,
                _grid_dim_x() as usize,
            )
        };
        data.iter()
            .skip(block_id * block_dim + thread_id)
            .step_by(block_dim * grid_dim)
    }

    pub fn stride_ref<'a, T: Copy + 'a>(
        data: &'a mut CuGlobalSliceRef<'a, T>,
    ) -> impl Iterator<Item = &'a mut T> + 'a {
        let (thread_id, block_dim, block_id, grid_dim) = unsafe {
            (
                _thread_idx_x() as usize,
                _block_dim_x() as usize,
                _block_idx_x() as usize,
                _grid_dim_x() as usize,
            )
        };
        data.iter_mut()
            .skip(block_id * block_dim + thread_id)
            .step_by(block_dim * grid_dim)
    }
    pub fn block_stride<'a, T: Copy + 'a>(
        data: &'a CuGlobalSlice<'a, T>,
        elements: usize,
        index: usize,
    ) -> impl Iterator<Item = &'a T> + 'a {
        let count = data.len().div_ceil(elements);
        data.iter().skip(index).step_by(count)
    }
    pub fn block_stride_deref<'a, T: Copy + 'a>(
        data: &'a CuGlobalSliceRef<'a, T>,
        elements: usize,
        index: usize,
    ) -> impl Iterator<Item = &'a T> + 'a {
        let count = data.len().div_ceil(elements);
        data.iter().skip(index).step_by(count)
    }

    pub fn block_stride_ref<'a, T: Copy + 'a>(
        data: &'a mut CuGlobalSliceRef<'a, T>,
        elements: usize,
        index: usize,
    ) -> impl Iterator<Item = &'a mut T> + 'a {
        let count = data.len().div_ceil(elements);
        data.iter_mut().skip(index).step_by(count)
    }

    pub fn thread_stride<'a, T: Copy + 'a>(
        data: &'a [T],
    ) -> impl Iterator<Item = &'a T> + 'a {
        let (thread_id, block_dim) = unsafe { (_thread_idx_x() as usize, _block_dim_x() as usize) };
        data.iter().skip(thread_id).step_by(block_dim)
    }

    pub fn thread_stride_ref<'a, T: Copy + 'a>(
        data: &'a mut [T],
    ) -> impl Iterator<Item = &'a mut T> + 'a {
        let (thread_id, block_dim) = unsafe { (_thread_idx_x() as usize, _block_dim_x() as usize) };
        data.iter_mut().skip(thread_id).step_by(block_dim)
    }
}

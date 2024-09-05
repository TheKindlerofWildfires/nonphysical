use super::{HashTableArguments, CAPACITY};
use crate::cuda::atomic::{Atomic, Reduce};
use crate::cuda::grid::GridStride;
use crate::shared::primitive::F32;
use nonphysical_core::shared::float::Float;
use core::num::Wrapping;
use core::arch::nvptx::{_thread_idx_x, _block_dim_x, _block_idx_x,_syncthreads};
use crate::cuda::shuffle::{Shuffle,Shuffler};
use crate::WARP_SIZE;
use crate::shared::unsigned::U32;
use nonphysical_core::shared::unsigned::Unsigned;
use crate::cuda::shared::{CuShared,Shared};

fn hash(key: F32) -> u32 {
    let mut value = Wrapping(key.0.to_bits());
    value ^= value >> 16;
    value *= 0x85ebca6b;
    value ^= value >> 13;
    value *= 0xc2b2ae35;
    value ^= value >> 16;
    value &= 1024 - 1;
    value.0
}
#[no_mangle]
pub extern "ptx-kernel" fn insert_hash_table_f32<'a>(
    args: &'a mut HashTableArguments<'a, F32, F32>,
) {
    let keys = GridStride::stride_ref(&mut args.keys);
    let values = GridStride::stride_ref(&mut args.values);
    keys.zip(values).for_each(|(key, value)| {
        let mut slot = hash(*key);
        let mut trying = true;
        while trying {
            let prev = args.table_keys.atomic_cas(slot as usize, F32::MAX, *key);
            if prev == F32::MAX || prev == *key {
                args.table_values[slot as usize] = *value;
                trying = false;
            } else {
                slot = (slot + 1) & (CAPACITY - 1);
            }
        }
    });
}

#[no_mangle]
pub extern "ptx-kernel" fn lookup_hash_table_f32<'a>(
    args: &'a mut HashTableArguments<'a, F32, F32>,
) {
    let keys = GridStride::stride_ref(&mut args.keys);
    let values = GridStride::stride_ref(&mut args.values);
    keys.zip(values).for_each(|(key, value)| {
        let mut slot = hash(*key);
        let mut trying = true;
        while trying {
            if args.table_keys[slot as usize] == *key {
                *value = args.table_values[slot as usize];
                trying = false;
            } else if args.table_keys[slot as usize] == F32::MAX {
                *value = F32::default();
                trying = false;
            } else {
                slot = (slot + 1) & (CAPACITY - 1)
            }
        }
    });
}

#[no_mangle]
pub extern "ptx-kernel" fn delete_hash_table_f32<'a>(
    args: &'a mut HashTableArguments<'a, F32, F32>,
) {
    let keys = GridStride::stride_ref(&mut args.keys);
    keys.for_each(|key| {
        let mut slot = hash(*key);
        let mut trying = true;
        while trying {
            if args.table_keys[slot as usize] == *key {
                args.table_values[slot as usize] = F32::MAX;
                trying = false;
            } else if args.table_keys[slot as usize] == F32::default() {
                trying = false;
            } else {
                slot = (slot + 1) & (CAPACITY - 1)
            }
        }
    });
}


//this could probably be a smarter / faster reduction kernel
#[no_mangle]
pub extern "ptx-kernel" fn count_hash_table_f32<'a>(
    args: &'a mut HashTableArguments<'a, F32, F32>,
) {
    let mut reduction = CuShared::<U32, 32>::new();
    let thread_id = unsafe { _thread_idx_x() } as usize;
    let block_dim = unsafe { _block_dim_x() } as usize;
    let lane = thread_id % WARP_SIZE;
    let wid = thread_id / WARP_SIZE;
    let hash_keys = GridStride::stride_ref(&mut args.table_keys);

    //Get the grid size result
    let mut result = hash_keys.fold(U32::ZERO,|acc,key| {
        if *key != F32::MAX{
            acc+U32::IDENTITY
        }else{
            acc
        }
    });

    let mut i = WARP_SIZE / 2;
    while i >= 1 {
        result += Shuffler::shuffle_bfly::<0xffffffff>(result, i, WARP_SIZE - 1);
        i >>= 1;
    }
    if lane == 0 {
        reduction.store(wid, result);
    }
    unsafe { _syncthreads() };
    result = if thread_id < block_dim / WARP_SIZE {
        reduction.load(lane)
    } else {
        U32::ZERO
    };

    if wid == 0 {
        let mut i = WARP_SIZE / 2;
        while i >= 1 {
            result += Shuffler::shuffle_bfly::<0xffffffff>(result, i, WARP_SIZE - 1);
            i >>= 1;
        }
        if lane == 0 {
            args.ctr.reduce_add(0, result);
        }
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn iterate_hash_table_f32<'a>(
    args: &'a mut HashTableArguments<'a, F32, F32>,
) {
    let hash_keys = GridStride::stride_ref(&mut args.table_keys);
    let hash_values = GridStride::stride_ref(&mut args.table_values);
    hash_keys.zip(hash_values).for_each(|(key, value)| {
        if *key != F32::MAX{
            let idx = args.ctr.atomic_inc(0, U32::usize(args.keys.len()));
            args.keys[idx.as_usize()] = *key;
            args.values[idx.as_usize()] = *value;
        }
    });

}

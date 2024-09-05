use std::{sync::Arc, time::SystemTime};
use core::num::Wrapping;
use nonphysical_core::shared::{primitive::Primitive, vector::{float_vector::FloatVector, Vector}};
use nonphysical_cuda::cuda::runtime::{Runtime, RUNTIME};
use nonphysical_ptx::{graph::{hash_table::cuda_hash_table::CudaHashTable, merge_sort::cuda_merge_sort::CudaMergeSort}, shared::vector::cuda_vector::CudaVector};
use nonphysical_std::shared::primitive::F32;
use nonphysical_core::shared::float::Float;

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

pub fn main() {
    let mut data = (0..1024*1024).map(|i| (F32::isize(i).sin()*F32::usize(100)).as_usize()).collect::<Vec<_>>();

    let now = SystemTime::now();
    data.sort();

    dbg!(now.elapsed());
    Runtime::init(0, "nonphysical_ptx.ptx");
    let data1 = (0..64).collect::<Vec<_>>();

    //let data = vec![9,7,5,3,8,4,6,5];
    let data1 = data1.into_iter().rev().map(|i| F32::isize(i)).collect::<Vec<_>>();
    let out = CudaMergeSort::merge_1024(&data1);
    dbg!(out.len(),&out[0..64]);
    //dbg!(out);
    return;
}

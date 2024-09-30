use core::num::Wrapping;
use std::ptr;
use nonphysical_core::shared::complex::{Complex, ComplexScaler};
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::{
    primitive::Primitive,
    vector::{float_vector::FloatVector, Vector},
};
use nonphysical_core::signal::fourier::fourier_heap::ComplexFourierTransformHeap;
use nonphysical_core::signal::fourier::FourierTransform;
use nonphysical_cuda::cuda::ffi::{cudaDeviceGetAttribute, cudaDeviceProp, cudaGetDeviceProperties_v2};
use nonphysical_cuda::cuda::runtime::{Runtime, RUNTIME};
use nonphysical_ptx::signal::fourier::cuda_fourier::ComplexFourierTransformCuda;
use nonphysical_ptx::{
    graph::{
        hash_table::cuda_hash_table::CudaHashTable
    },
    shared::vector::cuda_vector::CudaVector,
};
use nonphysical_std::shared::primitive::F32;
use std::{sync::Arc, time::SystemTime};

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
/* 
fn sort_test() {
    let mut data = (0..1024 * 1024)
        .map(|i| (F32::isize(i).sin() * F32::usize(100)).as_usize())
        .collect::<Vec<_>>();

    let now = SystemTime::now();
    data.sort();

    dbg!(now.elapsed());
    Runtime::init(0, "nonphysical_ptx.ptx");
    let data1 = (0..8 * 8).collect::<Vec<_>>();

    //let data = vec![9,7,5,3,8,4,6,5];
    let data1 = data1
        .into_iter()
        .rev()
        .map(|i| F32::isize(i))
        .collect::<Vec<_>>();
    let out = CudaMergeSort::merge_1024(&data1);
    dbg!(out.len(), &out);
}*/
pub fn main() {

    //dbg!(out);
    Runtime::init(0, "nonphysical_ptx.ptx");
    let nfft = 4096;
    let mut data = (0..nfft*2048).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
    let mut data_reference = (0..nfft).map(|i| ComplexScaler::new(F32::usize(i%nfft), F32::ZERO)).collect::<Vec<_>>();
    let reference_fft = ComplexFourierTransformHeap::new(nfft);
    reference_fft.fft(&mut data_reference);    
    let fft = ComplexFourierTransformCuda::new(nfft);
    fft.fft(&mut data);
    data.chunks_exact(nfft).for_each(|chunk|{
        chunk.iter().zip(data_reference.iter()).for_each(|(a,b)|{
            assert!(a==b);
        });
    });
    /*
    -511.99615 + 1246.8789i,
    -512.002 + -210.24414i,
    -511.9942 + 213.91992i,
    -512.00397 + -1225.4219i,
    -511.98737 + 515.1504i,
    -512.003 + -508.86914i,
    -511.99493 + 1.5703125i,
    -512.0535 + -166885.53i, */

}
fn reverse_idx(n: usize) -> usize {
    let mut v = n;
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    v = (v >> 16) | (v << 16);
    v = v >> std::cmp::min(31, v.trailing_zeros());
    v = (v - 1) / 2;
    v
}
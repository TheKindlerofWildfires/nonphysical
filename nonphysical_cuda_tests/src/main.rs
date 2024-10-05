use nonphysical_core::shared::complex::{Complex, ComplexScaler};
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::
    primitive::Primitive
;
use nonphysical_core::signal::wavelet::wavelet_heap::DaubechiesFirstWaveletHeap;
use nonphysical_core::signal::wavelet::DiscreteWavelet;
use nonphysical_cuda::cuda::runtime::{Runtime, RUNTIME};
use nonphysical_ptx::signal::wavelet::cuda_wavelet::DaubechiesFirstWaveletCuda;
use nonphysical_std::shared::primitive::F32;
use std::{sync::Arc, time::SystemTime};

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
    let ndwt = 4096;
    let data = (0..ndwt*1024).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
    let data_reference = (0..ndwt).map(|i| ComplexScaler::new(F32::usize(i%ndwt), F32::ZERO)).collect::<Vec<_>>();
    let reference_dwt = DaubechiesFirstWaveletHeap::new(());
    let ref_out = reference_dwt.forward(&data_reference);    
    let dwt = DaubechiesFirstWaveletCuda::new(ndwt);
    let out = dwt.forward(&data);
    out.chunks_exact(ndwt).for_each(|chunk|{
        chunk.iter().zip(ref_out.iter()).for_each(|(a,b)|{
            assert!(a==b);
        });
    });

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
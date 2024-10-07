use std::time::SystemTime;

use nonphysical_core::shared::complex::{Complex, ComplexScaler};
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::
    primitive::Primitive
;
use nonphysical_core::signal::cyclostationary::cyclostationary_heap::CycloStationaryTransformHeap;
use nonphysical_core::signal::cyclostationary::CycloStationaryTransform;
use nonphysical_core::signal::wavelet::wavelet_heap::DaubechiesFirstWaveletHeap;
use nonphysical_core::signal::wavelet::DiscreteWavelet;
use nonphysical_cuda::cuda::runtime::Runtime;
use nonphysical_ptx::signal::cyclostationary::cuda_cyclostationary::CycloStationaryTransformCuda;
use nonphysical_ptx::signal::wavelet::cuda_wavelet::DaubechiesFirstWaveletCuda;
use nonphysical_std::shared::primitive::F32;

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
    let ncst = 2048;
    let mut data = (0..ncst*128).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
    let cst = CycloStationaryTransformCuda::new(ncst);
    let now = SystemTime::now();

    let out = cst.fam(&mut data);
    dbg!(now.elapsed());
    //dbg!(out.rows,out.cols,&out.data);
}
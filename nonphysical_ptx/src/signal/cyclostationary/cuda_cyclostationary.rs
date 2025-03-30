use super::{CyclostationaryIntermediateArguments,CyclostationaryCompleteArguments};
use nonphysical_core::{
    shared::{complex::Complex, matrix::matrix_heap::MatrixHeap,primitive::Primitive,float::Float,matrix::Matrix},
    signal::{cyclostationary::CycloStationaryTransform, fourier::{fourier_heap::ComplexFourierTransformHeap, FourierTransform}},
};
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_cuda::cuda::runtime::Dim3;
use std::{cmp::min, vec::Vec};
use alloc::string::String;
use alloc::format;
use alloc::vec;

use nonphysical_cuda::cuda::runtime::RUNTIME;

pub struct CycloStationaryTransformCuda<C: Complex> {
    primary_twiddles: Vec<C>,
    ncst: usize,
}
/*
Best guess
    1 kernel to go to intermediate matrix like mass fourier
    1 kernel to complete the cycle

*/
impl<C: Complex> CycloStationaryTransform<C> for CycloStationaryTransformCuda<C> {
    type CycloInit = usize;
    type Matrix = MatrixHeap<C>;

    fn new(init: Self::CycloInit) -> Self {
        let ncst = init;
        let reference_fft = ComplexFourierTransformHeap::<C>::new(ncst);
        let primary_twiddles = reference_fft.twiddles;
        Self {
            primary_twiddles,
            ncst,
        }
    }
    
    
    fn fam(&self, x: &mut [C]) -> Self::Matrix {
        let win_count = x.len() / self.ncst;
        //Step 1 Precalculate the phase vector
        let phase_vec = (0..self.ncst).map(|i|{
            let temp_i = C::Primitive::usize(i)/C::Primitive::usize(self.ncst);
            let omega = C::Primitive::PI *(temp_i-C::Primitive::usize(2).recip());
            C::new(C::Primitive::ZERO, -omega*C::Primitive::usize(self.ncst))
        }).collect::<Vec<_>>();
        //Step 2: Allocate the vectors
        let mut intermediate = Self::cyclostationary_intermediate_alloc(x, &self.primary_twiddles, &phase_vec);
        //Step 3: Calculate the intermediate matrix
        self.cyclostationary_intermediate_transfer(&mut intermediate,x, &self.primary_twiddles, &phase_vec);
        //intermediate.x.load(x);
        //return Self::Matrix::new((win_count,x.to_vec()));

        //Step 4: Generate the secondary twiddles
        let secondary_fourier = ComplexFourierTransformHeap::new(win_count);
        let mut result = vec![C::ZERO; self.ncst*self.ncst*win_count];
        let mut complete = Self::cyclostationary_complete_alloc(intermediate,&secondary_fourier.twiddles,&result);
        self.cyclostationary_complete_transfer(&mut complete, &secondary_fourier.twiddles,&mut result);
        Self::Matrix::new((self.ncst,result))
    }

}

impl<C: Complex> CycloStationaryTransformCuda<C> {
    fn launch<Args>(args: &mut Args, threads: usize, block_size: usize, kernel:String){

        let grid = Dim3 {
            x: block_size,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: threads,
            y: 1,
            z: 1,
        };
        match RUNTIME.get() {
            Some(rt) => {
                rt.launch_name(kernel, args, grid, block);
            }
            None => panic!("Cuda Runtime not initialized"),
        };
    }
    fn cyclostationary_intermediate_alloc<'a>(x: &[C], twiddles: &[C], phase: &[C]) -> CyclostationaryIntermediateArguments<'a, C> {
        let x = CuGlobalSliceRef::alloc(x);
        let twiddles = CuGlobalSlice::alloc(twiddles);
        let phase = CuGlobalSlice::alloc(phase);
        CyclostationaryIntermediateArguments { x, twiddles,phase }
    }

    fn cyclostationary_complete_alloc<'a>(intermediate: CyclostationaryIntermediateArguments<'a, C>, twiddles: &[C], result: &[C]) -> CyclostationaryCompleteArguments<'a, C> {
        let x = intermediate.x;
        let twiddles = CuGlobalSlice::alloc(twiddles);
        let result = CuGlobalSliceRef::alloc(result);
        CyclostationaryCompleteArguments { x, twiddles,result }
    }

    fn cyclostationary_intermediate_transfer<'a>(&self,args: &mut CyclostationaryIntermediateArguments<'a, C>, x: &mut [C],twiddles: &[C], phase: &[C]) {
        assert!(x.len() >= 4);
        let block_size = x.len() / self.ncst;
        assert!(x.len() == block_size * self.ncst);
        args.x.store(x);
        args.twiddles.store(twiddles);
        args.phase.store(phase);
        let threads = min(512, self.ncst / 2);
        let kernel = format!("cyclostationary_intermediate_{}_kernel",self.ncst);
        Self::launch(args,threads,block_size,kernel);
    }

    fn cyclostationary_complete_transfer<'a>(&self,args: &mut CyclostationaryCompleteArguments<'a, C>,twiddles: &[C], result: &mut [C]) {
        args.twiddles.store(twiddles);
        args.result.store(result);
        let block_size = self.ncst;
        let threads = min(1024, self.ncst);
        let kernel = format!("cyclostationary_complete_{}_kernel",4);
        Self::launch(args,threads,block_size,kernel);
        args.result.load(result);

    }
}

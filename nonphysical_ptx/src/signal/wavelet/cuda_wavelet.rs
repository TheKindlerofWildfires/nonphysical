use super::WaveletArguments;
use nonphysical_core::{
    shared::matrix::matrix_heap::MatrixHeap,
    signal::wavelet::{DiscreteWavelet, WaveletFamily},
};
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_cuda::cuda::runtime::Dim3;
use std::cmp::min;
use alloc::string::String;
use alloc::format;
use alloc::vec::Vec;
use alloc::vec;
use nonphysical_cuda::cuda::runtime::RUNTIME;
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::matrix::Matrix;
pub struct DaubechiesFirstWaveletCuda<F: Float> {
    coefficients: [F;2],
    ndwt: usize,
}

impl<F: Float> DiscreteWavelet<F> for DaubechiesFirstWaveletCuda<F> {
    const SYMMETRY: usize = 0;
    const ORTHOGONAL: usize = 1;
    const BIORTHOGONAL: usize = 1;
    const FAMILY: WaveletFamily = WaveletFamily::Daubechies;
    type Matrix = MatrixHeap<F>;
    type DiscreteWaveletInit = usize; 
    fn new(init: Self::DiscreteWaveletInit) -> Self {
        let first = (F::IDENTITY + F::IDENTITY).sqrt().recip();
        let coefficients = [first, first];
        let ndwt = init;
        Self {
            coefficients,
            ndwt,
        }
    }
    fn forward(&self, input: &[F]) -> Vec<F> {
        let mut output = vec![F::ZERO; input.len()];
        let mut args = Self::wavelet_alloc(input, &output, &self.coefficients);

        self.wavelet_transfer(&mut args, input,&mut output,&self.coefficients,"forward");
        output
    }

    fn backward(&self, input: &[F]) -> Vec<F> {
        let mut output = vec![F::ZERO; input.len()];
        let mut args = Self::wavelet_alloc(input, &output, &self.coefficients);

        self.wavelet_transfer(&mut args, input,&mut output,&self.coefficients,"backward");
        output
    }
    //bugs known in these functions, need to compare to right answers

    //really more of a tensor output like (count/ndwt,ndwt,rows)
    fn decompose(&self, input: &[F]) -> Self::Matrix {
        todo!();
        let rows = self.ndwt.ilog2() as usize+1;
        let mut output = vec![F::ZERO; input.len()*rows];
        let mut args = Self::wavelet_alloc(input, &output, &self.coefficients);

        self.wavelet_transfer(&mut args, input,&mut output,&self.coefficients,"decompose");
        Self::Matrix::new((self.ndwt,output))
    }

    fn cis_detail(&self, _input: &[F]) -> Self::Matrix {
        todo!()
    }

    fn cis_approx(&self, _input: &[F]) -> Self::Matrix {
        todo!()
    }

    fn trans_detail(&self, _input: &[F]) -> Self::Matrix {
        todo!()
    }

    fn trans_approx(&self, _input: &[F]) -> Self::Matrix {
        todo!()
    }
}

impl<F: Float> DaubechiesFirstWaveletCuda<F> {
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
    fn wavelet_alloc<'a>(input: &[F], output: &[F], coefficients: &[F])-> WaveletArguments<'a,F>{
        let input = CuGlobalSlice::alloc(input);
        let output = CuGlobalSliceRef::alloc(output);
        let coefficients = CuGlobalSlice::alloc(coefficients);
        WaveletArguments{input,output,coefficients}
    }
    fn wavelet_transfer<'a>(&self, args: &mut WaveletArguments<'a,F>, input: &[F], output: &mut [F], coefficients: &[F], name: &str){
        args.input.store(input);
        args.coefficients.store(coefficients);
        let block_size = input.len()/self.ndwt;
        let threads = min(1024,self.ndwt/2);
        let kernel = format!("daubechies_first_{}_{}_kernel",name,self.ndwt);

        Self::launch(args,threads,block_size,kernel);
        args.output.load(output);
    }
}

use alloc::vec::Vec;

use crate::shared::complex::Complex;
use crate::shared::matrix::matrix_heap::MatrixHeap;
use crate::shared::matrix::Matrix;
use crate::shared::primitive::Primitive;
use super::fourier::fourier_heap::ComplexFourierTransformHeap;
use super::fourier::FourierTransform;
use crate::shared::float::Float;
use crate::linear::gemm::Gemm;
pub trait CycloStationaryTransform<C:Complex>{
    type CycloInit;
    type CycloMatrix: Matrix<C>;
    fn new(init: Self::CycloInit) -> Self;
    fn fam(&self, x: &mut [C]) -> Self::CycloMatrix;
}

pub struct CycloStationaryTransformHeap<C: Complex> {
    over_sample: usize,
    window: Vec<C>,
    fourier: ComplexFourierTransformHeap<C>,
}

impl<C: Complex> CycloStationaryTransform<C> for CycloStationaryTransformHeap<C> {
    type CycloInit = (usize, Vec<C>);
    type CycloMatrix = MatrixHeap<C>;
    fn new(init: Self::CycloInit) -> Self {
        let (over_sample, window) = init;
        let fourier = ComplexFourierTransformHeap::new(window.len());
        Self {
            over_sample,
            window,
            fourier,
        }
    }
    
    fn fam(&self, x: &mut [C]) -> Self::CycloMatrix {
        let win_step = self.window.len() / self.over_sample;
        let win_count = x.len() / win_step - self.over_sample + 1;
        let size = win_count * self.window.len();
        let mut cyclo_data = Vec::with_capacity(size);
        //Step 1: Windowing
        (0..win_count).for_each(|i| {
            cyclo_data.extend_from_slice(&x[i * win_step..i * win_step + self.window.len()])
        });

        //This is everything for the exp except for the window count
        let phase_vec = (0..self.window.len()).map(|i|{
            let omega = C::Primitive::PI - C::Primitive::PI*C::Primitive::float(2.0)/C::Primitive::usize(i);
            C::new(C::Primitive::ZERO, omega*C::Primitive::usize(win_step))
        }).collect::<Vec<_>>();
        cyclo_data
            .chunks_exact_mut(self.window.len())
            .enumerate().for_each(|(i,c_chunk)| {
                //Step 2: Tapering
                Self::convolve(c_chunk, &self.window);

                //Step 3a: Fourier and shift
                self.fourier.fft(c_chunk);
                ComplexFourierTransformHeap::shift(c_chunk);

                //Step 3b: Adjust the phase relationship
                Self::phase_correct(c_chunk, &phase_vec, i);

            });

        let intermediate_matrix = MatrixHeap::new((self.window.len(), cyclo_data));
        //Step 4: auto correlation
        let mut s = intermediate_matrix.data_rows().map(|row|{
            let j_mat = MatrixHeap::new((row.len(), row.to_vec()));
            let k_conj = row.iter().map(|rj|rj.conjugate()).collect::<Vec<_>>();
            let k_mat = MatrixHeap::new((1, k_conj.to_vec()));
            //<MatrixHeap<C> as Gemm<C>>::gemm(j_mat, k_mat) reference does a vector multiply operation not full gemm
            //but it also does it Np^2 times, it's possible gemm is better here

            //there's an FFT expectation here, but I think it's smaller? see it's value to write it to 
            //It decreases as overlap fraction increases -> it's the 'time dimension' 
            //complexity of NFFT (different S[i) * NFFT (boxes) * FFT(T) (time dimension) 
            //NFFT^2 * TlogT Cuda can do NFFT easily, esp if small FFTs
            //Time for 1024*1024 was 2ms 
            //then I'd need k,l to find i
        });
    }
}

impl<C:Complex> CycloStationaryTransformHeap<C>{
    #[inline(always)]
    fn convolve(x: &mut [C], y: &[C]) {
        x.iter_mut()
            .zip(y)
            .for_each(|(xi, yi)| *xi *= yi.conjugate());
    }
    fn phase_correct(x: &mut [C], y: &[C], i: usize){
        let iprim = C::Primitive::usize(i);
        x.iter_mut().zip(y).for_each(|(xi,yi)|{
            let factor = (*yi*iprim).exp();
            *xi *= factor
        })
    }
}
/*
    In short
        Create windows
        fft the first window and shift it back
        find demoulates mostly from constants
        multiply through by 


        Do the gabor transform to get matrix of N*P
        each row multiply by 

*/


use alloc::vec;
use alloc::vec::Vec;

use crate::shared::complex::Complex;
use crate::shared::matrix::matrix_heap::MatrixHeap;
use crate::shared::matrix::Matrix;
use crate::shared::primitive::Primitive;
use crate::shared::vector::float_vector::FloatVector;
use crate::shared::vector::Vector;
use crate::signal::fourier::fourier_heap::ComplexFourierTransformHeap;
use crate::signal::fourier::FourierTransform;
use crate::shared::float::Float;

use super::CycloStationaryTransform;
pub struct CycloStationaryTransformHeap<C: Complex> {
    window: Vec<C>,
    primary_fourier: ComplexFourierTransformHeap<C>,
}
/*
    convert x into an overlapped array ala gabor

    take a subsection of rows P, window and FFT it (If L != Np this drops data, don't do that)
    
    multiply through by a fix factor

    in a loop run fft on the conjugate pairs, placing them into the big matrix

*/
impl<C: Complex> CycloStationaryTransform<C> for CycloStationaryTransformHeap<C> {
    type CycloInit = Vec<C>;
    type Matrix = MatrixHeap<C>;
    fn new(init: Self::CycloInit) -> Self {
        let window = init;
        let primary_fourier = ComplexFourierTransformHeap::new(window.len());
        Self {
            window,
            primary_fourier,
        }
    }
    
    fn fam(&self, x: &mut [C]) -> Self::Matrix {
        let ncst = self.window.len();
        let win_step = ncst;
        let win_count = x.len() / ncst;
        let size = win_count * ncst;
        let mut cyclo_data = vec![C::ZERO;size];
        //Step 1: Windowing
        cyclo_data.copy_from_slice(x);

        let mut intermediate_matrix = Self::Matrix::new((win_count, cyclo_data));
        
        //This is everything for the exp except for the window count
        let phase_vec = (0..self.window.len()).map(|i|{
            let temp_i = C::Primitive::usize(i)/C::Primitive::usize(self.window.len());
            let omega = C::Primitive::PI *(temp_i-C::Primitive::usize(2).recip());
            C::new(C::Primitive::ZERO, -omega*C::Primitive::usize(win_step))
        }).collect::<Vec<_>>();
        intermediate_matrix.data_rows_ref()
            .enumerate().for_each(|(i,c_chunk)| {
                //Step 2: Tapering
                Self::convolve(c_chunk, &self.window);
                //Step 3a: Fourier and shift
                self.primary_fourier.forward_shifted(c_chunk);
                //Step 3b: Adjust the phase relationship
                Self::phase_correct(c_chunk, &phase_vec, i);
        });
        

        let mut result_matrix = Self::Matrix::zero(self.window.len(),self.window.len()*win_count);
        //Step 4: auto correlation
        let secondary_fourier = ComplexFourierTransformHeap::new(win_count);
        (0..intermediate_matrix.cols).zip(result_matrix.data_rows_ref()).for_each(|(i,result_row)|{
            (0..intermediate_matrix.cols).zip(result_row.chunks_exact_mut(win_count)).for_each(|(j, result_chunk)|{
                result_chunk.iter_mut().zip(intermediate_matrix.data_col(j)).for_each(|(rc,jc)|{
                    *rc =jc.conjugate()
                });
                FloatVector::mul_vec_ref(result_chunk.iter_mut(), intermediate_matrix.data_col(i));
                secondary_fourier.forward_shifted(result_chunk);
            });
        });
        result_matrix
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
        let prim = C::Primitive::usize(i);
        x.iter_mut().zip(y).for_each(|(xi,yi)|{
            let factor = (*yi*prim).exp();
            *xi *= factor
        });
    }
}
use core::marker::PhantomData;

use crate::shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector};

pub trait Predictor<T: Float> {
    fn predict(x: &Matrix<T>) -> Matrix<T>;
    fn loss(x: &Matrix<T>, y: &Matrix<T>) -> T;
    fn diff(x: &Matrix<T>, y: &Matrix<T>) -> Matrix<T>;
    
}

pub struct SoftPredictor<T: Float> {
    phantom_data: PhantomData<T>
}
//please go check the bad assumptions here, I think they are part of the problem
impl<T: Float> Predictor<T> for SoftPredictor<T> {

    fn predict(x: &Matrix<T>) -> Matrix<T> {
        let output = x.data().map(|c| c.exp());
        let mut output_mat = Matrix::new(x.rows, output.collect());
        output_mat.data_rows_ref().for_each(|row|{
            let sum:Complex<T> = <Vec<&'_ Complex<T>> as Vector<T>>::sum(row.iter());
            row.iter_mut().for_each(|c| *c = *c/sum);
        });
        output_mat
    }

    fn diff(x: &Matrix<T>,y:&Matrix<T>) -> Matrix<T> {
        let mut p = Self::predict(x);
        p.data_rows_ref().zip(y.data_rows()).for_each(|(p_row, y_row)|{
            //made a series of bad assumptions here
            p_row[y_row[0].real.to_usize()] = p_row[y_row[0].real.to_usize()] - Complex::<T>::one()
        });
        p
    }
    
    fn loss(x: &Matrix<T>, y: &Matrix<T>) -> T {
        let mut p = Self::predict(x);
        let mut sum = Complex::<T>::zero();
        p.data_rows_ref().zip(y.data_rows()).for_each(|(p_row, y_row)|{
            //made a series of bad assumptions here
            let log_loss = -p_row[y_row[0].real.to_usize()].ln();
            sum = sum + log_loss;
        });
        sum.real/T::usize(x.rows)
    }
}
use crate::{linear::gemm::Gemm, random::pcg::PermutedCongruentialGenerator, shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector}};


pub trait VonMises<T: Float> {
    fn vector(&self, iterations: usize) -> (Self,T) where Self: Sized;
    fn auto_vector(&self) -> (Self,T) where Self: Sized;
}
//Gemm issues really hurt this viability 
impl<T:Float> VonMises<T> for Matrix<T>{
    fn vector(&self, iterations: usize) -> (Self,T) {
        let mut pcg = PermutedCongruentialGenerator::<T>::new(0 as u32, 1 as u32+1);
        let b_data = pcg.uniform(T::ZERO, T::ONE, self.rows).iter().map(|t| Complex::new(*t,T::ZERO)).collect();
        let mut b = Matrix::new(self.rows, b_data);
        let mut norm = T::ZERO;
        (0..iterations).for_each(|_|{
            b =  <Matrix<T> as Gemm<T>>::gemm(self,&b);
            norm = <Vec<&Complex<T>> as Vector<T>>::square_norm_sum(b.data()).sqrt();
            <Vec<&Complex<T>> as Vector<T>>::scale(b.data_ref(),norm.recip());
        });

        (b,norm)
    }

    fn auto_vector(&self) -> (Self,T) {
        let mut pcg = PermutedCongruentialGenerator::<T>::new(0 as u32, 1 as u32+1);
        let b_data = pcg.uniform(T::ZERO, T::ONE, self.rows).iter().map(|t| Complex::new(*t,T::ZERO)).collect();
        let mut b = Matrix::new(self.rows, b_data);
        let mut norm = T::ZERO;
        let mut delta = T::MAX;
        let mut c = 0;
        while delta>T::EPSILON{
            
            b =  <Matrix<T> as Gemm<T>>::gemm(self,&b);
            delta = norm;
            norm = <Vec<&Complex<T>> as Vector<T>>::square_norm_sum(b.data()).sqrt();
            delta -= norm;
            <Vec<&Complex<T>> as Vector<T>>::scale(b.data_ref(),norm.recip());
            c+=1;
            dbg!(c);
        }

        (b,norm)
    }
}

#[cfg(test)]
mod von_mises_tests {
    use super::*;
    #[test]
    fn vector_3x3() {
        let mut m = Matrix::new(3, (0..9).map(|i| Complex::new(i as f32, 0.0)).collect());
        let (vector,value) = <Matrix<f32> as VonMises<f32>>::vector(&mut m,10);
        let known_values = vec![Complex::new(0.16476381,0.0),Complex::new(0.50577444,0.0),Complex::new(0.8467851,0.0)];
        vector.data().zip(known_values.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        assert!((value- 13.34847).square_norm()<f32::EPSILON);
    }

    #[test]
    fn vector_4x4() {
        let mut m = Matrix::new(4, (0..16).map(|i| Complex::new(i as f32, 0.0)).collect());
        let (vector,value) = <Matrix<f32> as VonMises<f32>>::vector(&mut m,10);

        let known_values = vec![Complex::new(0.11417645,0.0),Complex::new(0.3300046,0.0),Complex::new(0.54583275,0.0),Complex::new(0.76166089,0.0)];
        vector.data().zip(known_values.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        assert!((value- 32.46425).square_norm()<f32::EPSILON);

    }
    #[test]
    fn vector_5x5() {
        let mut m = Matrix::new(5, (0..25).map(|i| Complex::new(i as f32, 0.0)).collect());
        let (vector,value) = <Matrix<f32> as VonMises<f32>>::vector(&mut m,10);
        let known_values = vec![Complex::new(0.0851802,0.0),Complex::new(0.23825372,0.0),Complex::new(0.39132723,0.0),Complex::new(0.54440074,0.0),Complex::new(0.69747425,0.0)];
        vector.data().zip(known_values.iter()).for_each(|(c,k)|{
            assert!((*c-*k).square_norm()<f32::EPSILON);
        });
        assert!((value- 63.911655).square_norm()<f32::EPSILON);
    }
}
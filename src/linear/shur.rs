use crate::{linear::hessenberg::Hessenberg, shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector}};


pub trait Schur<T: Float> {
    fn schur(&mut self) -> (Vec<Complex<T>>, Self);

    fn reduce_triangular();

}
impl<T:Float> Schur<T> for Matrix<T>{
    fn schur(&mut self) -> (Vec<Complex<T>>, Self){
        debug_assert!(self.rows==self.columns);
        //hessenberg reduction
        let t = <Matrix<T> as Hessenberg<T>>::hess(self);
        //compute from hessenberg

        todo!()
    }

    fn reduce_triangular() {
        
    }
}
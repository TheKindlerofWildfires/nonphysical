use core::cmp::min;

use crate::shared::{float::Float, matrix::Matrix, real::Real};

use super::householder::{Householder, RealHouseholder};

pub trait Bidiagonal<'a,F:Float+'a>{
    const MBS: usize = 32;
    type H: Householder<'a,F>;
    fn new(matrix: &mut Matrix<F>) -> Self;
    fn blocked(matrix: &mut Matrix<F>, k: usize, brows: usize, bcols: usize, bs: usize);
    fn unblocked(matrix:&mut Matrix<F>, k: usize, brows: usize, bcols: usize);
    fn bidiagonal() -> Matrix<F>;
    fn householder() -> Self::H;
    fn householder_u() -> Self::H;
    fn householder_v() -> Self::H;
}

pub struct RealBidiagonal{

}

pub struct ComplexBidiagonal{

}

impl<'a,R: Real<Primitive = R>+'a>Bidiagonal<'a,R> for RealBidiagonal {
    type H = RealHouseholder<R>;

    fn new(matrix: &mut Matrix<R>) -> Self {
        let size= min(matrix.rows, matrix.cols);
        let block_size = min(size,<Self as Bidiagonal<R>>::MBS);
        (0..size).step_by(block_size).for_each(|k|{
            let bs = min(size-k, block_size);
            let brows = matrix.rows-k;
            let bcols = matrix.cols-k;
            if (k+bs == matrix.cols || bcols<48){
                Self::unblocked(matrix,k,brows, bcols);
            }else{
                todo!();
            }
        });
        todo!()
    }

    fn bidiagonal() -> Matrix<R> {
        todo!()
    }

    fn householder() -> Self::H {
        todo!()
    }

    fn householder_u() -> Self::H {
        todo!()
    }

    fn householder_v() -> Self::H {
        todo!()
    }
    
    fn blocked(matrix: &mut Matrix<R>, k: usize, brows: usize, bcols: usize, bs: usize) {
        todo!()
    }
    
    fn unblocked(matrix:&mut Matrix<R>, k: usize, brows: usize, bcols: usize) {
        let mut kk = 0; 
        dbg!(&matrix);
        loop{
            dbg!(kk);
            let remaining_rows = brows-kk;
            let remaining_cols = bcols-kk-1;
            //kinda feel like it's time to break out the matrix left/east/etc
            let prep = Self::H::make_householder_prep(&mut matrix.data_row(k+kk).skip(k+kk));
            let house = Self::H::make_householder_local(&mut matrix.data_row_ref(k+kk).skip(k+kk),prep);
            *matrix.coeff_ref(k+kk, k+kk) = house.tau;
            dbg!(&matrix);

            house.apply_left_local(matrix, k, k+kk..kk+brows, k+kk+1..kk+bcols);
            dbg!(&matrix);

            if k==bcols-1{
                break;
            }
            let prep = Self::H::make_householder_prep(&mut matrix.data_col(k+kk).skip(k+kk+1));
            dbg!(&prep);
            let house = Self::H::make_householder_local(&mut matrix.data_col_ref(k+kk).skip(k+kk+1),prep);
            dbg!(house.tau,house.beta);
            dbg!(&matrix);
            *matrix.coeff_ref(k+kk+1, k+kk) = house.tau;
            dbg!(&matrix);

            //problem here is that I assumed that it would invoke with col and that was foolish
            //I think I need to go back and reconsider householder 
            house.apply_right_local(matrix, k, k+kk+1..kk+brows, k+kk+1..kk+bcols);
            dbg!(&matrix);

            kk+=1;
            if kk==1{
                todo!();
            }
        }
    }
}

#[cfg(test)]
mod bidiagonal_tests {
    use super::*;

    #[test]
    fn unblocked_r_4x4(){
        let data = (0..4*4).map(|i| i as f32/15.0).collect();
        let mut m = Matrix::new(4,data);
        let b = RealBidiagonal::new(&mut m);
    }
}
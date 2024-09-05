use core::cmp::min;

use crate::shared::{float::Float, matrix::{heap::MatrixHeap, Matrix}, real::Real};

use super::householder::{Householder, RealHouseholder};

pub trait Bidiagonal<F:Float>{
    const MBS: usize = 32;
    type Matrix: Matrix<F>;
    type H: Householder<F>;
    fn new(matrix: &mut Self::Matrix) -> Self;
    fn blocked(matrix: &mut Self::Matrix, k: usize, brows: usize, bcols: usize, bs: usize);
    fn unblocked(matrix:&mut Self::Matrix, bidiagonal:&mut Self::Matrix, k: usize, brows: usize, bcols: usize);
    fn bidiagonal() -> Self::Matrix;
    fn householder() -> Self::H;
    fn householder_u() -> Self::H;
    fn householder_v() -> Self::H;
}

pub struct RealBidiagonal<R:Real>{
    pub bidiagonal: MatrixHeap<R>,
}

impl<R: Real<Primitive = R>>Bidiagonal<R> for RealBidiagonal<R> {
    type H = RealHouseholder<R>;
    type Matrix = MatrixHeap<R>;

    fn new(matrix: &mut Self::Matrix) -> Self {
        let mut bidiagonal = Self::Matrix::zero(matrix.rows, matrix.cols);
        let size= min(matrix.rows, matrix.cols);
        let block_size = min(size,<Self as Bidiagonal<R>>::MBS);
        (0..size).step_by(block_size).for_each(|k|{
            let bs = min(size-k, block_size);
            let brows = matrix.rows-k;
            let bcols = matrix.cols-k;
            if k+bs == matrix.cols || bcols<48{
                Self::unblocked(matrix,&mut bidiagonal, k,brows, bcols);
            }else{
                Self::blocked(matrix, k, brows, bcols, bs)
            }
        });
        Self{bidiagonal}
    }

    fn bidiagonal() -> Self::Matrix {
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
    
    fn blocked(matrix: &mut Self::Matrix, k: usize, brows: usize, bcols: usize, bs: usize) {
        //figure out how big
        //iterate up to block size
        //get a sub block around x/a which I don't know
        //get a column from A and adjust it
        //make a householder with the column of a
        //when k is less than the columns
        /*
            get two new submats from y/a
            do a little fixing to them

            do soome more fixing

            make a householder in uk
            do some more fixing
            even more fixing
            update a loop variable
         */
        //alternative just update A a little
        //update A a little
        //if the blocksize is smaller than the cols update A again
        /*
            update A22
            set up A11-A11-A01
            Update them a little
         */

    }
    
    fn unblocked(matrix:&mut Self::Matrix,bidiagonal:&mut Self::Matrix, k: usize, brows: usize, bcols: usize) {
        let mut kk = 0; 
        loop{
            let house = Self::H::make_householder_col(matrix, k+kk, k+kk);
            *bidiagonal.coeff_ref(k+kk, k+kk)=house.beta;
            *matrix.coeff_ref(k+kk, k+kk) = house.tau;

            let essential = matrix.data_row(k+kk).skip(k+kk+1).cloned().collect::<Vec<_>>();
            house.apply_left(matrix, &essential, [k+kk,k+bcols], [k+kk+1,k+brows]);
            
            if kk==bcols-1{
                break;
            }
            let house = Self::H::make_householder_row(matrix, k+kk+1, k+kk);
            *bidiagonal.coeff_ref(k+kk, k+kk+1)=house.beta;
            *matrix.coeff_ref(k+kk+1, k+kk) = house.tau;
            let essential = matrix.data_col(k+kk).skip(k+kk+2).cloned().collect::<Vec<_>>();
            house.apply_right(matrix, &essential, [k+kk+1,k+brows], [k+kk+1,k+bcols]);

            kk+=1;
        }
    }
}

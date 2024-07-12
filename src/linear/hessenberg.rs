use crate::shared::{float::Float, matrix::Matrix};


pub trait Hessenberg<T: Float> {
    fn hess(&mut self) -> Self;
}

impl<T:Float> Hessenberg<T> for Matrix<T>{
    fn hess(&mut self) -> Self{
        debug_assert!(self.rows==self.columns);
        //let householder_coefficients = vec![Complex::ZERO;self.rows];

        //interesting issue here. Need to modify row during hh in place, but then need to modify chunk opf matrix during right cols / bot right 
        //thats going to be a double mut access... unclear how that will work
        /* 
        self.data_rows_ref().enumerate().for_each(|(i,row)|{
            let remaining_size = self.rows - i - 1;

            let (h,beta) = house_holder_inplace(row[i..]);
            row[i+1] = beta;
            householder_coefficients[i] = h;

            apply_householder_left(self.bottom_right_corner(remaining_size, remaining_size),row[i+1..],h,temp);

            apply_householder_right(self.right_rows(remaining_size), row[i+1..].conjugate,h.conj(), temp);

        });
        */
        //probably need 
        // A) A householder trait for the apply householder functions

        // B) Some handlers for right rows / bottom_right corner etc 

        todo!()

    }
}
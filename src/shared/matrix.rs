use super::complex::{Complex, Float};

pub struct Matrix<T: Float> {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<Complex<T>>
}

impl<T: Float> Matrix<T>{
    pub fn new(rows: usize, data:Vec<Complex<T>>) -> Self{
        let columns = data.len()/rows;
        Matrix{
            rows,
            columns,
            data,
        }
    }
}
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

    pub fn norm_max(&self) -> T{
        self.data.iter().fold(T::maximum(),|max, &x| {
            let mag = x.magnitude();
            if max>mag{
                max
            }else{
                mag
            }
        })
    }

    pub fn norm_min(&self) -> T{
        self.data.iter().fold(T::minimum(),|min: T, &x| {
            let mag = x.magnitude();
            if min<mag{
                min
            }else{
                mag
            }
        })
    }
}
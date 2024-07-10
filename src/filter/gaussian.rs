use crate::shared::{complex::Complex, float::Float, matrix::Matrix};

pub trait GaussianFilter<T: Float> {
    fn window(size: Vec<usize>, sigma: Vec<T>) -> Self;

    fn filter(&mut self, filter: &Self);
}
impl<T: Float> GaussianFilter<T> for Vec<Complex<T>> {
    fn window(size: Vec<usize>, sigma: Vec<T>) -> Self {
        debug_assert!(size.len()==1);
        debug_assert!(sigma.len()==1);
        let sigma2 = *sigma.first().unwrap()*T::usize(2);
        let size = *size.first().unwrap();
        let sub = T::usize(size >> 1) - T::float(0.5);
        let mut window = vec![Complex::ZERO;size];
        window.iter_mut().enumerate().for_each(|(i,w)| {
            let value = ((T::usize(i) - sub) / sigma2).exp();
            *w = Complex::new(value, T::ZERO);
        });
        window
    }

    //this currently downsamples because it didn't extend the iterator by filter length on the edges I think
    fn filter(&mut self, filter: &Self){
        *self = self.windows(filter.len()).map(|selection| {
            selection.iter().zip(filter).map(|(s,f)| (*s*f.conj())).fold(Complex::ZERO, |acc,c| acc+c)
        }).collect();
    }
}

impl<T:Float> GaussianFilter<T> for Matrix<T>{
    fn window(size: Vec<usize>, sigma: Vec<T>) -> Self {
        debug_assert!(size.len()==2);
        debug_assert!(sigma.len()==2);

        let sigma_r2 = T::usize(2) * sigma[0];
        let sigma_c2 = T::usize(2) * sigma[1];
        let sub_r = T::usize(size[0] >> 1) - T::float(0.5);
        let sub_c = T::usize(size[1] >> 1) - T::float(0.5);
        let mut window = Matrix::zero(size[0],size[1]);
        window.data_rows_ref().enumerate().for_each(|(i,row)|{
            let x = (T::usize(i) - sub_r)/sigma_r2;
            row.iter_mut().enumerate().for_each(|(j,w)|{
                let y = (T::usize(j)-sub_c)/sigma_c2;
                let value = (x+y).exp();
                *w = Complex::new(value, T::ZERO)
            });
        });
        window
    }

    //this currently downsamples because it didn't extend the iterator by filter length on the edges I think
    //also there's no way this works correctly untested
    fn filter(&mut self, filter: &Self){
        let new_data = self.data_rows().collect::<Vec<_>>().windows(filter.rows).flat_map(|rows| {
            //get the row groups we need
            rows.windows(filter.columns).map(|selection|{
                //get the column groups we need
                selection.iter().zip(filter.data()).map(|(s,f) |s[0]*f.conj()).fold(Complex::ZERO, |acc,c| acc+c)
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        self.data = new_data;
    }
}

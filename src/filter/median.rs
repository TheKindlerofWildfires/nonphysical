use crate::shared::{complex::Complex, float::Float, matrix::Matrix};

pub trait MedianFilter<T: Float> {
    fn window(size: Vec<usize>, sigma: Vec<T>) -> Self;

    fn filter(&mut self, filter: &Self);
}

impl<T: Float> MedianFilter<T> for Vec<Complex<T>> {
    fn window(size: Vec<usize>, sigma: Vec<T>) -> Self {
        debug_assert!(size.len() == 1);
        debug_assert!(sigma.len() == 1);
        vec![Complex::ZERO; *size.first().unwrap()]
    }

    fn filter(&mut self, filter: &Self) {
        *self = self
            .windows(filter.len())
            .map(|selection| {
                let mut selection_copy = selection.iter().collect::<Vec<_>>();
                selection_copy.sort_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap());
                match selection.len() % 2 ==1 {
                    true => selection[selection.len()/2],
                    false => selection[selection.len()/2-1]+selection[selection.len()/2]
                }
            })
            .collect();
    }
}

impl<T: Float> MedianFilter<T> for Matrix<T> {
    fn window(size: Vec<usize>, sigma: Vec<T>) -> Self {
        debug_assert!(size.len() == 2);
        debug_assert!(sigma.len() == 2);
        Matrix::<T>::zero(size[0], size[1])
    }

    fn filter(&mut self, _filter: &Self) {
        todo!()
    }
}

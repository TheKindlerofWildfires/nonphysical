use super::{complex::Complex, float::Float};

pub trait Vector<'a, T: Float + 'a> {
    fn norm_max<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::minimum(), |acc, x| acc.greater(x.norm()))
    }

    fn norm_min<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::maximum(), |acc, x| acc.lesser(x.norm()))
    }

    fn scale<I>(iter: I, scaler: T)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
    {
        iter.for_each(|c| *c = *c * scaler)
    }

    fn mul<I>(iter: I, rhs: Complex<T>)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
    {
        iter.for_each(|c| *c = *c * rhs)
    }

    fn add<I>(iter: I, rhs: Complex<T>)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
    {
        iter.for_each(|c| *c = *c + rhs)
    }
}

impl<'a, T: Float> Vector<'a, T> for Vec<&'a Complex<T>> {}
impl<'a, T: Float> Vector<'a, T> for Vec<&'a mut Complex<T>> {}

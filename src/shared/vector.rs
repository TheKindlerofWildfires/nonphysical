<<<<<<< HEAD
use super::{complex::Complex, float::Float};

pub trait Vector<'a, T: Float + 'a> {
    fn norm_max<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::MIN, |acc, x| acc.greater(x.norm()))
    }

    fn norm_min<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::MAX, |acc, x| acc.lesser(x.norm()))
    }

    fn norm_sum<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::ZERO, |acc, x| acc+ x.norm())
    }

    fn sum<I>(iter: I) -> Complex<T>
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(Complex::<T>::ZERO, |acc, x| acc+ *x)
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

    fn acc<I,J>(iter: I, rhs:J)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
        J: Iterator<Item = &'a Complex<T>>,
    {
        iter.zip(rhs).for_each(|(i,j)| *i = *i + *j)
    }

    fn prod<I,J>(iter: I, rhs:J)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
        J: Iterator<Item = &'a Complex<T>>,
    {
        iter.zip(rhs).for_each(|(i,j)| *i = *i * *j)
    }
    
}

impl<'a, T: Float> Vector<'a, T> for Vec<&'a Complex<T>> {}
impl<'a, T: Float> Vector<'a, T> for Vec<&'a mut Complex<T>> {}
=======
use super::{complex::Complex, float::Float};

pub trait Vector<'a, T: Float + 'a> {
    fn norm_max<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::MIN, |acc, x| acc.greater(x.norm()))
    }

    fn norm_min<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::MAX, |acc, x| acc.lesser(x.norm()))
    }

    fn norm_sum<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(T::ZERO, |acc, x| acc+ x.norm())
    }

    fn sum<I>(iter: I) -> Complex<T>
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        iter.fold(Complex::<T>::ZERO, |acc, x| acc+ *x)
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

    fn acc<I,J>(iter: I, rhs:J)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
        J: Iterator<Item = &'a Complex<T>>,
    {
        iter.zip(rhs).for_each(|(i,j)| *i = *i + *j)
    }

    fn prod<I,J>(iter: I, rhs:J)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
        J: Iterator<Item = &'a Complex<T>>,
    {
        iter.zip(rhs).for_each(|(i,j)| *i = *i * *j)
    }
    
}

impl<'a, T: Float> Vector<'a, T> for Vec<&'a Complex<T>> {}
impl<'a, T: Float> Vector<'a, T> for Vec<&'a mut Complex<T>> {}
>>>>>>> master

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
        iter.for_each(|c| *c += rhs)
    }

    fn acc<I,J>(iter: I, rhs:J)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
        J: Iterator<Item = &'a Complex<T>>,
    {
        iter.zip(rhs).for_each(|(i,j)| *i += *j)
    }

    fn prod<I,J>(iter: I, rhs:J)
    where
        I: Iterator<Item = &'a mut Complex<T>>,
        J: Iterator<Item = &'a Complex<T>>,
    {
        iter.zip(rhs).for_each(|(i,j)| *i = *i * *j)
    }

    fn mean<I>(iter: I) -> T
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        let (_, mean) = iter.fold((0,T::ZERO), |acc, x| {
            let (mut count, mut mean) = acc;
            count +=1;
            let norm = x.norm();
            let delta = norm-mean;
            mean += delta/T::usize(count);
            (count,mean)
        });
        mean
    }

    fn variance<I>(iter: I) -> (T,T)
    where
        I: Iterator<Item = &'a Complex<T>>,
    {
        let (count, mean, square_distance) = iter.fold((0,T::ZERO,T::ZERO), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count +=1;
            let norm = x.norm();
            let delta = norm-mean;
            mean += delta/T::usize(count);
            let delta2 = norm - mean;
            square_distance+= delta*delta2;
            (count,mean,square_distance)
        });
        (mean,square_distance/T::usize(count))
    }
    
}

impl<'a, T: Float> Vector<'a, T> for Vec<&'a Complex<T>> {}
impl<'a, T: Float> Vector<'a, T> for Vec<&'a mut Complex<T>> {}

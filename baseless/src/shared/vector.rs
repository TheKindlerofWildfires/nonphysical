use super::{complex::Complex, float::Float, primitive::Primitive, real::Real};
use alloc::vec::Vec;

pub trait Vector<'a, F:Float+'a>{
    fn sum<I>(iter: I) -> F
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold(F::ZERO, |acc, x| acc+ *x)
    }

    fn dot<I>(iter: I,other: I) -> F
    where
        I: Iterator<Item = &'a F>,
    {
        iter.zip(other).fold(F::IDENTITY, |acc, (x,y)| acc+ (*x * *y))
    }

    fn add<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c += other)
    }

    fn sub<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c -= other)
    }

    fn mul<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c *= other)
    }

    fn div<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c /= other)
    }

    fn add_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i += *j)
    }

    fn sub_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i -= *j)
    }

    fn mul_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i *= *j)
    }

    fn div_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i /= *j)
    }
    fn l1_sum<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::ZERO, |acc, x| acc+ x.l1_norm())
    }

    fn l2_sum<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::ZERO, |acc, x| acc+ x.l2_norm())
    }
    fn l1_min<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::MAX, |acc, x| acc.lesser(x.l1_norm()))
    }

    fn l2_min<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::MAX, |acc, x| acc.lesser(x.l2_norm()))
    }

    fn l1_max<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::MIN, |acc, x| acc.greater(x.l1_norm()))
    }

    fn l2_max<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::MIN, |acc, x| acc.greater(x.l2_norm()))
    }

    fn l1_sum_ref<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.fold( F::Primitive::ZERO, |acc, x| acc+ x.l1_norm())
    }

    fn l2_sum_ref<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.fold( F::Primitive::ZERO, |acc, x| acc+ x.l2_norm())
    }

    fn l1_min_ref<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.fold( F::Primitive::MAX, |acc, x| acc.lesser(x.l1_norm()))
    }

    fn l2_min_ref<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.fold( F::Primitive::MAX, |acc, x| acc.lesser(x.l2_norm()))
    }
    fn l1_max_ref<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.fold( F::Primitive::MIN, |acc, x| acc.greater(x.l1_norm()))
    }

    fn l2_max_ref<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.fold( F::Primitive::MIN, |acc, x| acc.greater(x.l2_norm()))
    }

}

pub trait RealVector<'a, R:Real<Primitive = R>+'a>: Vector<'a,R>{
    fn mean<I>(iter: I) -> R::Primitive
    where
        I: Iterator<Item = &'a R>,
    {
        let (_, mean) = iter.fold((0,R::Primitive::ZERO), |acc, x| {
            let (mut count, mut mean) = acc;
            count +=1;
            let delta = *x-mean;
            mean += delta/R::usize(count);
            (count,mean)
        });
        mean
    }

    fn variance<I>(iter: I) -> (R::Primitive,R::Primitive)
    where
        I: Iterator<Item = &'a R>,
    {
        let (count, mean, square_distance) = iter.fold((0,R::Primitive::ZERO,R::Primitive::ZERO), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count +=1;
            let delta = *x-mean;
            mean += delta/R::usize(count);
            let delta2 = *x - mean;
            square_distance+= delta*delta2;
            (count,mean,square_distance)
        });
        (mean,square_distance/R::usize(count))
    }

    fn mean_ref<I>(iter: I) -> R::Primitive
    where
        I: Iterator<Item = &'a mut R>,
    {
        let (_, mean) = iter.fold((0,R::Primitive::ZERO), |acc, x| {
            let (mut count, mut mean) = acc;
            count +=1;
            let delta = *x-mean;
            mean += delta/R::usize(count);
            (count,mean)
        });
        mean
    }

    fn variance_ref<I>(iter: I) -> (R::Primitive,R::Primitive)
    where
        I: Iterator<Item = &'a mut R>,
    {
        let (count, mean, square_distance) = iter.fold((0,R::Primitive::ZERO,R::Primitive::ZERO), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count +=1;
            let delta = *x-mean;
            mean += delta/R::usize(count);
            let delta2 = *x - mean;
            square_distance+= delta*delta2;
            (count,mean,square_distance)
        });
        (mean,square_distance/R::usize(count))
    }

}


pub trait ComplexVector<'a, C:Complex+'a>: Vector<'a,C>{
    fn scale<I>(iter: I, scaler: C::Primitive)
    where
        I: Iterator<Item = &'a mut C>,
    {
        iter.for_each(|c| *c *= scaler)
    }

    fn descale<I>(iter: I, scaler:C::Primitive)
    where
        I: Iterator<Item = &'a mut C>,
    {
        iter.for_each(|c| *c /= scaler)
    }
    /* 
    fn mean<I>(iter: I) -> C
    where
        I: Iterator<Item = &'a C>,
    {
        let (_, mean) = iter.fold((0,C::ZERO), |acc, x| {
            let (mut count, mut mean) = acc;
            count +=1;
            let delta = *x-mean;
            mean += delta/C::Primitive::usize(count);
            (count,mean)
        });
        mean
    }

    fn variance<I>(iter: I) -> (C,C)
    where
     I: Iterator<Item = &'a C>,
    {
        let (count, mean, square_distance) = iter.fold((0,C::ZERO,C::ZERO), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count +=1;
            let delta = *x-mean;
            mean += delta/C::Primitive::usize(count);
            let delta2 = *x - mean;
            square_distance+= delta*delta2;
            (count,mean,square_distance)
        });
        (mean,square_distance/C::Primitive::usize(count))
    }*/
}



impl<'a, F: Float> Vector<'a, F> for Vec<&'a F> {}
impl<'a, F: Float> Vector<'a, F> for Vec<&'a mut F> {}
impl<'a, R: Real<Primitive = R>> RealVector<'a, R> for Vec<&'a R> {}
impl<'a, R: Real<Primitive = R>> RealVector<'a, R> for Vec<&'a mut R> {}
impl<'a, C: Complex> ComplexVector<'a, C> for Vec<&'a C> {}
impl<'a, C: Complex> ComplexVector<'a, C> for Vec<&'a mut C> {}
use core::marker::PhantomData;
use super::{complex::Complex, float::Float, point::Point, primitive::Primitive, real::Real};

pub struct Vector<F:Float>{
    phantom_data: PhantomData<F>
}


impl<'a, F:Float+'a> Vector<F>{
    pub fn sum<I>(iter: I) -> F
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold(F::ZERO, |acc, x| acc+ *x)
    }

    pub fn add<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c += other)
    }

    pub fn sub<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c -= other)
    }

    pub fn mul<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c *= other)
    }

    pub fn div<I>(iter: I, other: F)
    where
        I: Iterator<Item = &'a mut F>,
    {
        iter.for_each(|c| *c /= other)
    }

    pub fn add_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i += *j)
    }

    pub fn sub_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i -= *j)
    }

    pub fn mul_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i *= *j)
    }

    pub fn div_vec<I,J>(iter: I, other:J)
    where
        I: Iterator<Item = &'a mut F>,
        J: Iterator<Item = &'a F>,
    {
        iter.zip(other).for_each(|(i,j)| *i /= *j)
    }
    
    pub fn dot<I>(iter: I,other: I) -> F
    where
        I: Iterator<Item = &'a F>,
    {
        iter.zip(other).fold(F::IDENTITY, |acc, (x,y)| acc+ (*x * *y))
    }

    pub fn l1_sum<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::ZERO, |acc, x| acc+ x.l1_norm())
    }

    pub fn l2_sum<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::ZERO, |acc, x| acc+ x.l2_norm())
    }
    pub fn l1_min<I>(iter: I) -> F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::MAX, |acc, x| acc.lesser(x.l1_norm()))
    }

    pub fn l2_min<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::MAX, |acc, x| acc.lesser(x.l2_norm()))
    }

    pub fn l1_max<I>(iter: I) ->  F::Primitive
    where
        I: Iterator<Item = &'a F>,
    {
        iter.fold( F::Primitive::MIN, |acc, x| acc.greater(x.l1_norm()))
    }

    pub fn l2_max<I>(iter: I) ->  F::Primitive
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
pub struct RealVector<R:Real>{
    phantom_data: PhantomData<R>
}
impl<'a, R:Real<Primitive=R>+'a> RealVector<R>{
    pub fn mean<I>(iter: I) -> R::Primitive
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

    pub fn variance<I>(iter: I) -> (R::Primitive,R::Primitive)
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

pub struct ComplexVector<C:Complex>{
    phantom_data: PhantomData<C>
}
impl<'a, C:Complex+'a> ComplexVector<C>{
    pub fn scale<I>(iter: I, scaler: C::Primitive)
    where
        I: Iterator<Item = &'a mut C>,
    {
        iter.for_each(|c| *c *= scaler)
    }

    pub fn descale<I>(iter: I, scaler:C::Primitive)
    where
        I: Iterator<Item = &'a mut C>,
    {
        iter.for_each(|c| *c /= scaler)
    }
    /* 
    pub fn mean<I>(iter: I) -> C
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

    pub fn variance<I>(iter: I) -> (C,C)
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

pub struct PointVector<P:Point>{
    phantom_data: PhantomData<P>
}

impl<'a, R: Real<Primitive = R>, P:Point<Primitive=R>+'a> PointVector<P>{
    pub fn mean<I>(iter: I) -> P
    where
        I: Iterator<Item = &'a P>,
    {
        let (_, mean) = iter.fold((0,P::ORIGIN), |acc, x| {
            let (mut count, mut mean) = acc;
            count +=1;
            let mut delta = *x-mean;
            delta.scale(R::usize(count).recip());
            mean += delta;
            (count,mean)
        });
        mean
    }

    pub fn variance<I>(iter: I) -> (P,P)
    where
     I: Iterator<Item = &'a P>,
    {
        let (count, mean, mut square_distance) = iter.fold((0,P::ORIGIN,P::ORIGIN), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count +=1;
            let mut delta = *x-mean;
            delta.scale(R::usize(count).recip());
            mean += delta;
            let delta2 = *x - mean;
            square_distance+= delta*delta2;
            (count,mean,square_distance)
        });
        square_distance.scale(P::Primitive::usize(count).recip());
        (mean,square_distance)
    }
}
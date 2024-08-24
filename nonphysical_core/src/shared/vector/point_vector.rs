use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::shared::{point::Point, real::Real};

pub struct PointVector<P: Point> {
    phantom_data: PhantomData<P>,
}

impl<'a, R: Real<Primitive = R>, P: Point<Primitive = R> + 'a> PointVector<P> {
    pub fn sum<I>(iter: I) -> P
    where
        I: Iterator<Item = &'a P>,
    {
        iter.fold(P::ORIGIN, |acc, x| acc + *x)
    }

    pub fn add<I>(iter: I, other: P)
    where
        I: Iterator<Item = &'a mut P>,
    {
        iter.for_each(|c| *c += other)
    }

    pub fn sub<I>(iter: I, other: P)
    where
        I: Iterator<Item = &'a mut P>,
    {
        iter.for_each(|c| *c -= other)
    }

    pub fn mul<I>(iter: I, other: P)
    where
        I: Iterator<Item = &'a mut P>,
    {
        iter.for_each(|c| *c *= other)
    }

    pub fn div<I>(iter: I, other: P)
    where
        I: Iterator<Item = &'a mut P>,
    {
        iter.for_each(|c| *c /= other)
    }

    pub fn add_vec<I, J>(iter: I, other: J)
    where
        I: Iterator<Item = &'a mut P>,
        J: Iterator<Item = &'a P>,
    {
        iter.zip(other).for_each(|(i, j)| *i += *j)
    }

    pub fn sub_vec<I, J>(iter: I, other: J)
    where
        I: Iterator<Item = &'a mut P>,
        J: Iterator<Item = &'a P>,
    {
        iter.zip(other).for_each(|(i, j)| *i -= *j)
    }

    pub fn mul_vec<I, J>(iter: I, other: J)
    where
        I: Iterator<Item = &'a mut P>,
        J: Iterator<Item = &'a P>,
    {
        iter.zip(other).for_each(|(i, j)| *i *= *j)
    }

    pub fn div_vec<I, J>(iter: I, other: J)
    where
        I: Iterator<Item = &'a mut P>,
        J: Iterator<Item = &'a P>,
    {
        iter.zip(other).for_each(|(i, j)| *i /= *j)
    }

    pub fn dot<I>(iter: I, other: I) -> P
    where
        I: Iterator<Item = &'a P>,
    {
        iter.zip(other)
            .fold(P::IDENTITY, |acc, (x, y)| acc + (*x * *y))
    }

    pub fn l1_sum<I>(iter: I) -> P::Primitive
    where
        I: Iterator<Item = &'a P>,
    {
        iter.fold(P::Primitive::ZERO, |mut acc, x| {
            acc.data_ref()
                .zip(x.data())
                .for_each(|(ap, xp)| *ap += xp.l1_norm());
            acc
        })
    }

    pub fn l2_sum<I>(iter: I) -> P::Primitive
    where
        I: Iterator<Item = &'a P>,
    {
        iter.fold(P::Primitive::ZERO, |mut acc, x| {
            acc.data_ref()
                .zip(x.data())
                .for_each(|(ap, xp)| *ap += xp.l2_norm());
            acc
        })
    }
    pub fn l1_min<I>(iter: I) -> P::Primitive
    where
        I: Iterator<Item = &'a P>,
    {
        iter.fold(P::Primitive::ZERO, |mut acc, x| {
            acc.data_ref()
                .zip(x.data())
                .for_each(|(ap, xp)| *ap = ap.lesser(xp.l1_norm()));
            acc
        })
    }

    pub fn l2_min<I>(iter: I) -> P::Primitive
    where
        I: Iterator<Item = &'a P>,
    {
        iter.fold(P::Primitive::ZERO, |mut acc, x| {
            acc.data_ref()
                .zip(x.data())
                .for_each(|(ap, xp)| *ap = ap.lesser(xp.l2_norm()));
            acc
        })
    }

    pub fn l1_max<I>(iter: I) -> P::Primitive
    where
        I: Iterator<Item = &'a P>,
    {
        iter.fold(P::Primitive::ZERO, |mut acc, x| {
            acc.data_ref()
                .zip(x.data())
                .for_each(|(ap, xp)| *ap = ap.greater(xp.l1_norm()));
            acc
        })
    }

    pub fn l2_max<I>(iter: I) -> P::Primitive
    where
        I: Iterator<Item = &'a P>,
    {
        iter.fold(P::Primitive::ZERO, |mut acc, x| {
            acc.data_ref()
                .zip(x.data())
                .for_each(|(ap, xp)| *ap = ap.greater(xp.l1_norm()));
            acc
        })
    }

    pub fn mean<I>(iter: I) -> P
    where
        I: Iterator<Item = &'a P>,
    {
        let (_, mean) = iter.fold((0, P::ORIGIN), |acc, x| {
            let (mut count, mut mean) = acc;
            count += 1;
            let mut delta = *x - mean;
            delta.scale(R::usize(count).recip());
            mean += delta;
            (count, mean)
        });
        mean
    }

    pub fn variance<I>(iter: I) -> (P, P)
    where
        I: Iterator<Item = &'a P>,
    {
        let (count, mean, mut square_distance) = iter.fold((0, P::ORIGIN, P::ORIGIN), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count += 1;
            let mut delta = *x - mean;
            delta.scale(R::usize(count).recip());
            mean += delta;
            let delta2 = *x - mean;
            square_distance += delta * delta2;
            (count, mean, square_distance)
        });
        square_distance.scale(P::Primitive::usize(count).recip());
        (mean, square_distance)
    }

    //This probably should have been a HashSet for categories
    pub fn marked_sum<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![P::ORIGIN;categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            buckets[idx] +=*ip;

        });
        buckets
    }

    pub fn marked_l1_sum<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![P::ORIGIN;categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            buckets[idx].data_ref().zip(ip.data()).for_each(|(bp,ipp)|{
                *bp+=ipp.l1_norm()
            })

        });
        buckets
    }

    pub fn marked_l2_sum<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![P::ORIGIN;categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            buckets[idx].data_ref().zip(ip.data()).for_each(|(bp,ipp)|{
                *bp+=ipp.l2_norm()
            })

        });
        buckets
    }
    pub fn marked_l1_min<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![P::ORIGIN;categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            buckets[idx].data_ref().zip(ip.data()).for_each(|(bp,ipp)|{
                *bp=bp.lesser(ipp.l1_norm())
            })

        });
        buckets
    }

    pub fn marked_l2_min<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![P::ORIGIN;categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            buckets[idx].data_ref().zip(ip.data()).for_each(|(bp,ipp)|{
                *bp=bp.lesser(ipp.l2_norm())
            })

        });
        buckets
    }

    pub fn marked_l1_max<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![P::ORIGIN;categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            buckets[idx].data_ref().zip(ip.data()).for_each(|(bp,ipp)|{
                *bp=bp.greater(ipp.l1_norm())
            })

        });
        buckets
    }

    pub fn marked_l2_max<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![P::ORIGIN;categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            buckets[idx].data_ref().zip(ip.data()).for_each(|(bp,ipp)|{
                *bp=bp.greater(ipp.l2_norm())
            })

        });
        buckets
    }

    pub fn marked_mean<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<P>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![(0,P::ORIGIN);categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            let (mut count, mut mean) = buckets[idx];
            count +=1;
            let mut delta = *ip-mean;
            delta.scale(R::usize(count).recip());
            mean +=delta;
            buckets[idx] = (count,mean)

        });
        buckets.iter().map(|(_,m)|*m).collect()
    }

    pub fn marked_variance<I, M>(iter: I, mark: M, categories: Vec<usize>) -> Vec<(P,P)>
    where
        I: Iterator<Item = &'a P>,
        M: Iterator<Item = &'a usize>,
    {
        let mut buckets = vec![(0,P::ORIGIN, P::ORIGIN);categories.len()];
        iter.zip(mark).for_each(|(ip,mp)|{
            let idx = categories.iter().position(|c| *c==*mp).unwrap();
            let (mut count, mut mean, mut square_distance) = buckets[idx];
            count += 1;
            let mut delta = *ip - mean;
            delta.scale(R::usize(count).recip());
            mean += delta;
            let delta2 = *ip - mean;
            square_distance += delta * delta2;
            buckets[idx] = (count, mean, square_distance)

        });
        buckets.iter().map(|(_,m,v)|(*m,*v)).collect()
    }
}

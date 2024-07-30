use crate::random::pcg::PermutedCongruentialGenerator;
use alloc::vec::Vec;

use super::{real::Real, vector::Vector};

#[derive(Clone, Debug)]
pub struct StaticPoint<R: Real, const N: usize> {
    pub data: [R; N],
}

pub trait Point: Clone {
    type Primitive: Real;
    type Precursor;
    const ZERO: Self;
    const MAX: Self;
    const MIN: Self;
    fn new(data: Self::Precursor) -> Self;
    fn l2_distance(&self, other: &Self) -> Self::Primitive;
    fn l1_distance(&self, other: &Self) -> Self::Primitive;
    fn l1_farthest(&self, other: &Self) -> (Self::Primitive, usize);
    fn l2_farthest(&self, other: &Self) -> (Self::Primitive, usize);
    fn greater(&self, other: &Self) -> Self;
    fn lesser(&self, other: &Self) -> Self;
    fn ordered_farthest(&self, lesser: &Self) -> (Self::Primitive, usize);
    fn distance_to_range(&self, lesser: &Self, greater: &Self) -> Self::Primitive;

    fn max_data(&self) -> Self::Primitive;
    fn min_data(&self) -> Self::Primitive;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn dot(&self, other: &Self) -> Self::Primitive;
    fn scale(&mut self, other: Self::Primitive);
    fn data(&self, index: usize) -> Self::Primitive;
    fn uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self;
    fn partial_random(data:Vec<Self::Primitive>, rng: &mut PermutedCongruentialGenerator) -> Self;
}

impl<R: Real<Primitive = R>, const N: usize> Point for StaticPoint<R, N> {
    type Primitive = R;
    type Precursor = [R; N];

    const ZERO: Self = Self { data: [R::ZERO; N] };
    const MAX: Self = Self { data: [R::MAX; N] };
    const MIN: Self = Self { data: [R::MIN; N] };

    #[inline(always)]
    fn new(data: [R; N]) -> Self {
        Self { data }
    }

    #[inline(always)]
    fn l2_distance(&self, other: &Self) -> R {
        self.data
            .into_iter()
            .zip(other.data.into_iter())
            .fold(R::ZERO, |acc, (p1, p2)| acc + (p1 - p2).l2_norm())
    }

    #[inline(always)]
    fn l1_distance(&self, other: &Self) -> R {
        self.data
            .into_iter()
            .zip(other.data.into_iter())
            .fold(R::ZERO, |acc, (p1, p2)| acc + (p1 - p2).l1_norm())
    }

    #[inline(always)]
    fn greater(&self, other: &Self) -> Self {
        let mut ret = Self::ZERO;
        ret.data
            .iter_mut()
            .zip(self.data)
            .zip(other.data)
            .for_each(|((r, s), o)| {
                *r = s.greater(o);
            });
        ret
    }

    #[inline(always)]
    fn lesser(&self, other: &Self) -> Self {
        let mut ret = Self::ZERO;
        ret.data
            .iter_mut()
            .zip(self.data)
            .zip(other.data)
            .for_each(|((r, s), o)| {
                *r = s.lesser(o);
            });
        ret
    }
    #[inline(always)]
    fn ordered_farthest(&self, lesser: &Self) -> (Self::Primitive, usize) {
        self.data.into_iter().zip(lesser.data.into_iter()).enumerate().fold(
            (Self::Primitive::MIN, 0),
            |acc, (i, (s, o))| {
                let (distance, _) = acc;
                let new_distance = s - o;
                if new_distance > distance {
                    (new_distance, i)
                } else {
                    acc
                }
            },
        )
    }

    #[inline(always)]
    fn l1_farthest(&self, other: &Self) -> (Self::Primitive, usize) {
        self.data.into_iter().zip(other.data.into_iter()).enumerate().fold(
            (Self::Primitive::MIN, 0),
            |acc, (i, (s, o))| {
                let (distance, _) = acc;
                let new_distance = s.l1_distance(&o);
                if new_distance > distance {
                    (distance, i)
                } else {
                    acc
                }
            },
        )
    }

    #[inline(always)]
    fn l2_farthest(&self, other: &Self) -> (Self::Primitive, usize) {
        self.data.into_iter().zip(other.data.into_iter()).enumerate().fold(
            (Self::Primitive::MIN, 0),
            |acc, (i, (s, o))| {
                let (distance, _) = acc;
                let new_distance = s.l2_distance(&o);
                if new_distance > distance {
                    (distance, i)
                } else {
                    acc
                }
            },
        )
    }

    #[inline(always)]
    fn data(&self, index: usize) -> Self::Primitive {
        self.data[index]
    }

    #[inline(always)]
    fn distance_to_range(&self, lesser: &Self, greater: &Self) -> Self::Primitive {
        self.data
            .into_iter()
            .zip(lesser.data.into_iter())
            .zip(greater.data.into_iter())
            .fold(Self::Primitive::ZERO, |acc, ((sp, lp), gp)| {
                if sp > gp {
                    acc + sp - gp
                } else if sp < lp {
                    acc + lp - sp
                } else {
                    acc
                }
            })
    }
    #[inline(always)]
    fn max_data(&self) -> Self::Primitive{
        self.data.into_iter().fold(Self::Primitive::MIN, |acc, dp| dp.greater(acc))
    }
    #[inline(always)]
    fn min_data(&self) -> Self::Primitive{
        self.data.into_iter().fold(Self::Primitive::MAX, |acc, dp| dp.lesser(acc))
    }
    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        let mut ret = self.clone();
        <Vec<&Self::Primitive> as Vector<Self::Primitive>>::add_vec(
            ret.data.iter_mut(),
            other.data.iter(),
        );
        ret
    }
    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        let mut ret = self.clone();
        <Vec<&Self::Primitive> as Vector<Self::Primitive>>::sub_vec(
            ret.data.iter_mut(),
            other.data.iter(),
        );
        ret
    }
    #[inline(always)]
    fn dot(&self, other: &Self) -> Self::Primitive{
        <Vec<&Self::Primitive> as Vector<Self::Primitive>>::dot(
            self.data.iter(),
            other.data.iter(),
        )
    }

    #[inline(always)]
    fn uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self {

        let mut ret = Self::ZERO;
        let baseline = rng.interval::<Self::Primitive>(N);
        ret.data.iter_mut().zip(baseline.into_iter()).zip(min.data.into_iter()).zip(max.data.into_iter()).for_each(|(((r,b),mn),mx)|{
            *r = b*(mx-mn)+mn;

        });
        ret
    }

    #[inline(always)]
    fn partial_random(data:Vec<Self::Primitive>, rng: &mut PermutedCongruentialGenerator) -> Self {
        let mut ret = Self::ZERO;
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        rng.shuffle_usize(&mut indices);
        let reordered = indices.into_iter().take(N).map(|i| data[i]);
        ret.data.iter_mut().zip(reordered).for_each(|(rp,r)|{
            *rp = r;
        });

        ret
    }

    #[inline(always)]
    fn scale(&mut self, other: Self::Primitive) {
        <Vec<&Self::Primitive> as Vector<Self::Primitive>>::mul(
            self.data.iter_mut(),other
        );
    }
}

impl<R: Real<Primitive = R>> Point for R {
    type Primitive = Self;
    type Precursor = Self;

    const ZERO: Self = R::ZERO;
    const MAX: Self = R::MAX;
    const MIN: Self = R::MIN;

    #[inline(always)]
    fn new(data: Self::Precursor) -> Self {
        data
    }

    #[inline(always)]
    fn l2_distance(&self, other: &Self) -> Self::Primitive {
        (*self - *other).l2_norm()
    }

    #[inline(always)]
    fn l1_distance(&self, other: &Self) -> Self::Primitive {
        (*self - *other).l1_norm()
    }

    #[inline(always)]
    fn greater(&self, other: &Self) -> Self {
        self.greater(*other)
    }

    #[inline(always)]
    fn lesser(&self, other: &Self) -> Self {
        self.lesser(*other)
    }

    #[inline(always)]
    fn ordered_farthest(&self, lesser: &Self) -> (Self::Primitive, usize) {
        (*self - *lesser, 0)
    }

    #[inline(always)]
    fn l1_farthest(&self, other: &Self) -> (Self::Primitive, usize) {
        (self.l1_distance(other), 0)
    }
    #[inline(always)]
    fn l2_farthest(&self, other: &Self) -> (Self::Primitive, usize) {
        (self.l1_distance(other), 0)
    }

    #[inline(always)]
    fn data(&self, _index: usize) -> Self::Primitive {
        *self
    }

    #[inline(always)]
    fn distance_to_range(&self, lesser: &Self, greater: &Self) -> Self::Primitive {
        if self > greater {
            *self - *greater
        } else if self < lesser {
            *lesser - *self
        } else {
            Self::Primitive::ZERO
        }
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }
    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }

    #[inline(always)]
    fn max_data(&self) -> Self::Primitive{
        *self
    }
    #[inline(always)]
    fn min_data(&self) -> Self::Primitive{
        *self
    }

    #[inline(always)]
    fn dot(&self, other: &Self) -> Self::Primitive {
        *self * *other
    }

    #[inline(always)]
    fn uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self {
        rng.uniform_singular(*min, *max)
    }

    #[inline(always)]
    fn partial_random(data:Vec<Self::Primitive>, rng: &mut PermutedCongruentialGenerator) -> Self {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        rng.shuffle_usize(&mut indices);
        let mut reordered = indices.into_iter().take(1).map(|i| data[i]);
        reordered.next().unwrap_or(Self::ZERO)
    }

    #[inline(always)]
    fn scale(&mut self, other: Self::Primitive) {
       *self /=other;
    }
}

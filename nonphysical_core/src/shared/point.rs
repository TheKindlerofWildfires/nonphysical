use core::iter;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use core::fmt::Debug;

use crate::random::pcg::PermutedCongruentialGenerator;
use alloc::vec::Vec;

use super::vector::Vector;
use super::{primitive::Primitive, real::Real, vector::float_vector::FloatVector};

#[derive(Clone,Copy, Debug)]
pub struct StaticPoint<P: Primitive, const N: usize> {
    pub data: [P; N],
}

pub trait Point:
    Clone + Copy+Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
    + AddAssign+SubAssign+ MulAssign+ DivAssign+Debug
{
    type Primitive: Primitive;
    type Precursor;
    const ORIGIN: Self;
    const IDENTITY: Self;
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
    fn dot(&self, other: &Self) -> Self::Primitive;
    fn scale(&mut self, other: Self::Primitive);
    fn coeff(&self, index: usize) -> Self::Primitive;
    fn coeff_ref(&mut self, _index: usize) -> &mut Self::Primitive;
    fn uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self;
    fn random_uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self;
    fn partial_random(data: Vec<Self::Primitive>, rng: &mut PermutedCongruentialGenerator) -> Self;
    fn dimension(&self) -> usize;
    fn data<'a>(&'a self) -> impl Iterator<Item = &'a Self::Primitive>
    where
        Self::Primitive: 'a;
    fn data_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Primitive>
    where
        Self::Primitive: 'a;

}

impl<P: Primitive<Primitive = P>, const N: usize> Point for StaticPoint<P, N> {
    type Primitive = P;
    type Precursor = [P; N];

    const ORIGIN: Self = Self { data: [P::ZERO; N] };
    const IDENTITY: Self = Self{data: [P::IDENTITY;N]};
    const MAX: Self = Self { data: [P::MAX; N] };
    const MIN: Self = Self { data: [P::MIN; N] };

    #[inline(always)]
    fn new(data: [P; N]) -> Self {
        Self { data }
    }

    #[inline(always)]
    fn l2_distance(&self, other: &Self) -> P {
        self.data
            .into_iter()
            .zip(other.data)
            .fold(P::ZERO, |acc, (p1, p2)| acc + (p1 - p2).l2_norm())
    }

    #[inline(always)]
    fn l1_distance(&self, other: &Self) -> P {
        self.data
            .into_iter()
            .zip(other.data)
            .fold(P::ZERO, |acc, (p1, p2)| acc + (p1 - p2).l1_norm())
    }

    #[inline(always)]
    fn greater(&self, other: &Self) -> Self {
        let mut ret = Self::ORIGIN;
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
        let mut ret = Self::ORIGIN;
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
        self.data.into_iter().zip(lesser.data).enumerate().fold(
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
        self.data.into_iter().zip(other.data).enumerate().fold(
            (Self::Primitive::MIN, 0),
            |acc, (i, (s, o))| {
                let (distance, _) = acc;
                let new_distance = (s - o).l1_norm();
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
        self.data.into_iter().zip(other.data).enumerate().fold(
            (Self::Primitive::MIN, 0),
            |acc, (i, (s, o))| {
                let (distance, _) = acc;
                let new_distance = (s - o).l1_norm();
                if new_distance > distance {
                    (distance, i)
                } else {
                    acc
                }
            },
        )
    }

    #[inline(always)]
    fn coeff(&self, index: usize) -> Self::Primitive {
        self.data[index]
    }
    #[inline(always)]
    fn coeff_ref(&mut self, index: usize) -> &mut Self::Primitive {
        &mut self.data[index]
    }

    #[inline(always)]
    fn distance_to_range(&self, lesser: &Self, greater: &Self) -> Self::Primitive {
        self.data
            .into_iter()
            .zip(lesser.data)
            .zip(greater.data)
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
    fn max_data(&self) -> Self::Primitive {
        self.data
            .into_iter()
            .fold(Self::Primitive::MIN, |acc, dp| dp.greater(acc))
    }
    #[inline(always)]
    fn min_data(&self) -> Self::Primitive {
        self.data
            .into_iter()
            .fold(Self::Primitive::MAX, |acc, dp| dp.lesser(acc))
    }

    #[inline(always)]
    fn dot(&self, other: &Self) -> Self::Primitive {
        FloatVector::dot(self.data.iter(), other.data.iter())
    }

    #[inline(always)]
    fn uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self {
        let mut ret = Self::ORIGIN;
        let baseline = rng.interval::<Self::Primitive>(N);
        ret.data
            .iter_mut()
            .zip(baseline)
            .zip(min.data)
            .zip(max.data)
            .for_each(|(((r, b), mn), mx)| {
                *r = b * (mx - mn) + mn;
            });
        ret
    }

    #[inline(always)]
    fn partial_random(data: Vec<Self::Primitive>, rng: &mut PermutedCongruentialGenerator) -> Self {
        let mut ret = Self::ORIGIN;
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        rng.shuffle_usize(&mut indices);
        let reordered = indices.into_iter().take(N).map(|i| data[i]);
        ret.data.iter_mut().zip(reordered).for_each(|(rp, r)| {
            *rp = r;
        });

        ret
    }

    #[inline(always)]
    fn scale(&mut self, other: Self::Primitive) {
        FloatVector::scale_ref(self.data.iter_mut(), other);
    }
    #[inline(always)]
    fn dimension(&self) -> usize {
        N
    }

    fn data<'a>(&'a self) -> impl Iterator<Item = &'a Self::Primitive>
    where
        Self::Primitive: 'a,
    {
        self.data.iter()
    }
    fn data_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Primitive>
    where
        Self::Primitive: 'a,
    {
        self.data.iter_mut()
    }

    fn random_uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self {
        let mut ret = Self::ORIGIN;
        ret.data
            .iter_mut()
            .zip(min.data.iter())
            .zip(max.data.iter())
            .for_each(|((r, n), x)| {
                *r = rng.uniform_singular(*n, *x);
            });
        ret
    }
}
impl<P: Primitive<Primitive = P>, const N: usize> Add for StaticPoint<P,N> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut ret = Self::ORIGIN;
        ret.data.iter_mut().zip(self.data).zip(other.data).for_each(|((r,s),o)|{
            *r = s+o
        });
        ret
    }
}

impl<P: Primitive<Primitive = P>, const N: usize> Sub for StaticPoint<P,N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut ret = Self::ORIGIN;
        ret.data.iter_mut().zip(self.data).zip(other.data).for_each(|((r,s),o)|{
            *r = s-o
        });
        ret
    }
}
impl<P: Primitive<Primitive = P>, const N: usize> Mul for StaticPoint<P,N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let mut ret = Self::ORIGIN;
        ret.data.iter_mut().zip(self.data).zip(other.data).for_each(|((r,s),o)|{
            *r = s*o
        });
        ret
    }
}
impl<P: Primitive<Primitive = P>, const N: usize> Div for StaticPoint<P,N> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let mut ret = Self::ORIGIN;
        ret.data.iter_mut().zip(self.data).zip(other.data).for_each(|((r,s),o)|{
            *r = s/o
        });
        ret
    }
}

impl<P: Primitive<Primitive = P>, const N: usize> AddAssign for StaticPoint<P,N> {
    fn add_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data).for_each(|(s,o)|{
            *s+=o
        });
    }
}
impl<P: Primitive<Primitive = P>, const N: usize> SubAssign for StaticPoint<P,N> {
    fn sub_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data).for_each(|(s,o)|{
            *s-=o
        });
    }
}
impl<P: Primitive<Primitive = P>, const N: usize> MulAssign for StaticPoint<P,N> {
    fn mul_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data).for_each(|(s,o)|{
            *s*=o
        });
    }
}
impl<P: Primitive<Primitive = P>, const N: usize> DivAssign for StaticPoint<P,N> {
    fn div_assign(&mut self, other: Self) {
        self.data.iter_mut().zip(other.data).for_each(|(s,o)|{
            *s/=o
        });
    }
}


impl<P: Real<Primitive = P>> Point for P {
    type Primitive = Self;
    type Precursor = Self;

    const ORIGIN: Self = Self::ZERO;
    const IDENTITY: Self = Self::IDENTITY;
    const MAX: Self = P::MAX;
    const MIN: Self = P::MIN;

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
        (*self as P).greater(*other)
    }

    #[inline(always)]
    fn lesser(&self, other: &Self) -> Self {
        (*self as P).lesser(*other)
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
    fn coeff(&self, _index: usize) -> Self::Primitive {
        *self
    }

    #[inline(always)]
    fn coeff_ref(&mut self, _index: usize) -> &mut Self::Primitive {
        self
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
    fn max_data(&self) -> Self::Primitive {
        *self
    }
    #[inline(always)]
    fn min_data(&self) -> Self::Primitive {
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
    fn partial_random(data: Vec<Self::Primitive>, rng: &mut PermutedCongruentialGenerator) -> Self {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        rng.shuffle_usize(&mut indices);
        let mut reordered = indices.into_iter().take(1).map(|i| data[i]);
        reordered.next().unwrap_or(Self::ZERO)
    }

    #[inline(always)]
    fn scale(&mut self, other: Self::Primitive) {
        *self /= other;
    }
    #[inline(always)]
    fn dimension(&self) -> usize {
        1
    }

    fn data<'a>(&'a self) -> impl Iterator<Item = &'a Self::Primitive>
    where
        Self::Primitive: 'a,
    {
        iter::once(self)
    }
    fn data_ref<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Primitive>
    where
        Self::Primitive: 'a,
    {
        iter::once(self)
    }

    fn random_uniform(min: &Self, max: &Self, rng: &mut PermutedCongruentialGenerator) -> Self {
        rng.uniform_singular(*min, *max)
    }
}

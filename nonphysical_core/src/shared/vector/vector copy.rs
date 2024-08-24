use crate::shared::{float::Float, primitive::Primitive};
use core::marker::PhantomData;

use super::Vector;

struct FloatVector {
}

impl<'a, F: Float + 'a> Vector<'a,F> for FloatVector {
    /*
        Reduction style operations where [F] -> F

    */
    #[inline(always)]
    fn sum<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::ZERO, |acc, x| acc.add(*x))
    }
    #[inline(always)]
    fn product<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::IDENTITY, |acc, x| acc.mul(*x))
    }
    #[inline(always)]
    fn greater<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::MIN, |acc, x| acc.greater(*x))
    }
    #[inline(always)]
    fn lesser<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::MIN, |acc, x| acc.lesser(*x))
    }
    #[inline(always)]
    fn mean<I: Iterator<Item = &'a F>>(iter: I) -> F {
        let (_, mean) = iter.fold((0, F::ZERO), |acc, x| {
            let (mut count, mut mean) = acc;
            count += 1;
            let delta = *x - mean;
            mean += delta / F::Primitive::usize(count);
            (count, mean)
        });
        mean
    }
    #[inline(always)]
    fn variance<I: Iterator<Item = &'a F>>(iter: I) -> (F, F) {
        let (count, mean, square_distance) = iter.fold((0, F::ZERO, F::ZERO), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count += 1;
            let delta = *x - mean;
            mean += delta / F::Primitive::usize(count);
            let delta2 = *x - mean;
            square_distance += delta * delta2;
            (count, mean, square_distance)
        });
        (mean, square_distance / F::Primitive::usize(count))
    }

    #[inline(always)]
    fn deviation<I: Iterator<Item = &'a F>>(iter: I) -> (F, F) {
        let (mean, variance) = Self::variance(iter);
        (mean, variance.sqrt())
    }

    /*
        Single Map style operations where [F], F -> [F]
    */

    #[inline(always)]
    fn add<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).add(other))
    }

    #[inline(always)]
    fn sub<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).sub(other))
    }

    #[inline(always)]
    fn mul<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).mul(other))
    }

    #[inline(always)]
    fn div<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).div(other))
    }

    #[inline(always)]
    fn scale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: F::Primitive,
    ) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).mul(other))
    }

    #[inline(always)]
    fn descale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: F::Primitive,
    ) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).div(other))
    }

    #[inline(always)]
    fn fma<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        mul: F,
        add: F,
    ) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).fma(mul, add))
    }

    #[inline(always)]
    fn powf<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).powf(other))
    }

    #[inline(always)]
    fn powi<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: i32,
    ) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).powi(other))
    }

    #[inline(always)]
    fn ln<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).ln())
    }

    #[inline(always)]
    fn log2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).log2())
    }

    #[inline(always)]
    fn exp<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).exp())
    }

    #[inline(always)]
    fn exp2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).exp2())
    }

    #[inline(always)]
    fn recip<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).recip())
    }

    #[inline(always)]
    fn sin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).sin())
    }

    #[inline(always)]
    fn cos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).cos())
    }

    #[inline(always)]
    fn tan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).tan())
    }

    #[inline(always)]
    fn asin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).asin())
    }

    #[inline(always)]
    fn acos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).acos())
    }

    #[inline(always)]
    fn atan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).atan())
    }

    #[inline(always)]
    fn sinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).sinh())
    }

    #[inline(always)]
    fn cosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).cosh())
    }

    #[inline(always)]
    fn tanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).tanh())
    }

    #[inline(always)]
    fn asinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).asinh())
    }

    #[inline(always)]
    fn acosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).acosh())
    }

    #[inline(always)]
    fn atanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).atanh())
    }
    /*
        Transformation map operation where [F] -> F::Primitive

    */
    #[inline(always)]
    fn l1_norm<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
    ) -> impl Iterator<Item = F::Primitive> + 'a {
        iter.map(move |x| (*x).l1_norm())
    }

    #[inline(always)]
    fn l2_norm<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
    ) -> impl Iterator<Item = F::Primitive> + 'a {
        iter.map(move |x| (*x).l2_norm())
    }

    /*
        Single in place style operations where [mut F], F -> [F]
    */
    #[inline(always)]
    fn add_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).add_assign(other))
    }

    #[inline(always)]
    fn sub_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).sub_assign(other))
    }

    #[inline(always)]
    fn mul_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).mul_assign(other))
    }

    #[inline(always)]
    fn div_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).div_assign(other))
    }

    #[inline(always)]
    fn scale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F::Primitive) {
        iter.for_each(|x| (*x).mul_assign(other))
    }

    #[inline(always)]
    fn descale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F::Primitive) {
        iter.for_each(|x| (*x).div_assign(other))
    }

    #[inline(always)]
    fn fma_ref<I: Iterator<Item = &'a mut F>>(iter: I, mul: F, add: F) {
        iter.for_each(|x| *x = (*x).fma(mul, add))
    }
    #[inline(always)]
    fn powf_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| *x = (*x).powf(other))
    }
    #[inline(always)]
    fn powi_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: i32) {
        iter.for_each(|x| *x = (*x).powi(other))
    }

    #[inline(always)]
    fn ln_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).ln())
    }

    #[inline(always)]
    fn log2_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).log2())
    }

    #[inline(always)]
    fn exp_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).exp())
    }

    #[inline(always)]
    fn exp2_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).exp2())
    }

    #[inline(always)]
    fn recip_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).recip())
    }

    #[inline(always)]
    fn sin_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).sin())
    }

    #[inline(always)]
    fn cos_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).cos())
    }

    #[inline(always)]
    fn tan_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).tan())
    }

    #[inline(always)]
    fn asin_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).asin())
    }

    #[inline(always)]
    fn acos_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).acos())
    }

    #[inline(always)]
    fn atan_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).atan())
    }

    #[inline(always)]
    fn sinh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).sinh())
    }

    #[inline(always)]
    fn cosh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).cosh())
    }

    #[inline(always)]
    fn tanh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).tanh())
    }

    #[inline(always)]
    fn asinh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).asinh())
    }

    #[inline(always)]
    fn acosh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).acosh())
    }

    #[inline(always)]
    fn atanh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).atanh())
    }

    /*
        Multi Map style operations where [F], [F] -> [F]
    */
    #[inline(always)]
    fn add_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).add(*other))
    }

    #[inline(always)]
    fn sub_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).sub(*other))
    }

    #[inline(always)]
    fn mul_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).mul(*other))
    }

    #[inline(always)]
    fn div_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).div(*other))
    }

    #[inline(always)]
    fn fma_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        mul: I,
        add: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(mul)
            .zip(add)
            .map(move |((x, mul), add)| (*x).fma(*mul, *add))
    }

    #[inline(always)]
    fn powf_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).powf(*other))
    }

    #[inline(always)]
    fn powi_vec<I: Iterator<Item = &'a F> + 'a, J: Iterator<Item = &'a i32> + 'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).powi(*other))
    }

    #[inline(always)]
    fn greater_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).greater(*other))
    }

    #[inline(always)]
    fn lesser_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).lesser(*other))
    }

    /*
        Multi in place style operations where [mut F], [F] -> [F]
    */

    #[inline(always)]
    fn add_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).add_assign(*other))
    }

    #[inline(always)]
    fn sub_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).sub_assign(*other))
    }

    #[inline(always)]
    fn mul_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).mul_assign(*other))
    }

    #[inline(always)]
    fn div_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).div_assign(*other))
    }

    #[inline(always)]
    fn fma_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, mul: I, add: I) {
        iter.zip(mul)
            .zip(add)
            .for_each(move |((x, mul), add)| *x = (*x).fma(*mul, *add))
    }

    #[inline(always)]
    fn powf_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| *x = (*x).powf(*other))
    }

    #[inline(always)]
    fn powi_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a i32>>(
        iter: I,
        other: J,
    ) {
        iter.zip(other)
            .for_each(move |(x, other)| *x = (*x).powi(*other))
    }

    #[inline(always)]
    fn greater_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| *x = (*x).greater(*other))
    }

    #[inline(always)]
    fn lesser_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| *x = (*x).lesser(*other))
    }
}

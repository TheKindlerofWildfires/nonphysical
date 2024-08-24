use crate::shared::{float::Float, primitive::Primitive};

use super::Vector;

pub struct FloatVector {}

impl<'a, F: Float + 'a> Vector<'a, F> for FloatVector {
    fn sum<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::ZERO, |acc, x| acc.add(*x))
    }

    fn product<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::IDENTITY, |acc, x| acc.mul(*x))
    }

    fn greater<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::MIN, |acc, x| acc.greater(*x))
    }

    fn lesser<I: Iterator<Item = &'a F>>(iter: I) -> F {
        iter.fold(F::MIN, |acc, x| acc.lesser(*x))
    }

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

    fn deviation<I: Iterator<Item = &'a F>>(iter: I) -> (F, F) {
        let (mean, variance) = Self::variance(iter);
        (mean, variance.sqrt())
    }

    fn add<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).add(other))
    }

    fn sub<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).sub(other))
    }

    fn mul<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).mul(other))
    }

    fn div<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).div(other))
    }

    fn scale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).mul(other))
    }

    fn descale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).div(other))
    }

    fn fma<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        mul: F,
        add: F,
    ) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).fma(mul, add))
    }

    fn powf<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).powf(other))
    }

    fn powi<I: Iterator<Item = &'a F> + 'a>(iter: I, other: i32) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).powi(other))
    }

    fn ln<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).ln())
    }

    fn log2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).log2())
    }

    fn exp<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).exp())
    }

    fn exp2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).exp2())
    }

    fn recip<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).recip())
    }

    fn sin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).sin())
    }

    fn cos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).cos())
    }

    fn tan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).tan())
    }

    fn asin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).asin())
    }

    fn acos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).acos())
    }

    fn atan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).atan())
    }

    fn sinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).sinh())
    }

    fn cosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).cosh())
    }

    fn tanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).tanh())
    }

    fn asinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).asinh())
    }

    fn acosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).acosh())
    }

    fn atanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        iter.map(move |x| (*x).atanh())
    }

    fn l1_norm<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> + 'a {
        iter.map(move |x| (*x).l1_norm())
    }

    fn l2_norm<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> + 'a {
        iter.map(move |x| (*x).l2_norm())
    }

    fn add_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).add_assign(other))
    }

    fn sub_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).sub_assign(other))
    }

    fn mul_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).mul_assign(other))
    }

    fn div_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| (*x).div_assign(other))
    }

    fn scale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: <F as Float>::Primitive) {
        iter.for_each(|x| (*x).mul_assign(other))
    }

    fn descale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: <F as Float>::Primitive) {
        iter.for_each(|x| (*x).div_assign(other))
    }

    fn fma_ref<I: Iterator<Item = &'a mut F>>(iter: I, mul: F, add: F) {
        iter.for_each(|x| *x = (*x).fma(mul, add))
    }

    fn powf_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        iter.for_each(|x| *x = (*x).powf(other))
    }

    fn powi_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: i32) {
        iter.for_each(|x| *x = (*x).powi(other))
    }

    fn ln_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).ln())
    }

    fn log2_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).log2())
    }

    fn exp_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).exp())
    }

    fn exp2_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).exp2())
    }

    fn recip_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).recip())
    }

    fn sin_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).sin())
    }

    fn cos_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).cos())
    }

    fn tan_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).tan())
    }

    fn asin_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).asin())
    }

    fn acos_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).acos())
    }

    fn atan_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).atan())
    }

    fn sinh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).sinh())
    }

    fn cosh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).cosh())
    }

    fn tanh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).tanh())
    }

    fn asinh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).asinh())
    }

    fn acosh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).acosh())
    }

    fn atanh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        iter.for_each(|x| *x = (*x).atanh())
    }

    fn add_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).add(*other))
    }

    fn sub_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).sub(*other))
    }

    fn mul_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).mul(*other))
    }

    fn div_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).div(*other))
    }

    fn fma_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        mul: I,
        add: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(mul)
            .zip(add)
            .map(move |((x, mul), add)| (*x).fma(*mul, *add))
    }

    fn powf_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).powf(*other))
    }

    fn powi_vec<I: Iterator<Item = &'a F> + 'a, J: Iterator<Item = &'a i32> + 'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).powi(*other))
    }

    fn greater_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).greater(*other))
    }

    fn lesser_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        iter.zip(other).map(move |(x, other)| (*x).lesser(*other))
    }

    fn add_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).add_assign(*other))
    }

    fn sub_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).sub_assign(*other))
    }

    fn mul_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).mul_assign(*other))
    }

    fn div_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| (*x).div_assign(*other))
    }

    fn fma_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, mul: I, add: I) {
        iter.zip(mul)
            .zip(add)
            .for_each(move |((x, mul), add)| *x = (*x).fma(*mul, *add))
    }

    fn powf_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| *x = (*x).powf(*other))
    }

    fn powi_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a i32>>(iter: I, other: J) {
        iter.zip(other)
            .for_each(move |(x, other)| *x = (*x).powi(*other))
    }

    fn greater_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| *x = (*x).greater(*other))
    }

    fn lesser_vec_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: I) {
        iter.zip(other)
            .for_each(|(x, other)| *x = (*x).lesser(*other))
    }

    fn dot<I: Iterator<Item = &'a F>>(iter: I, other: I) -> F {
        iter.zip(other)
            .fold(F::ZERO, |acc, (x, other)| acc.add(x.mul(*other)))
    }

    fn quote<I: Iterator<Item = &'a F>>(iter: I, other: I) -> F {
        iter.zip(other)
            .fold(F::ZERO, |acc, (x, other)| acc.add(x.div(*other)))
    }

    fn add_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        iter.map(move |x| x.add(other))
    }

    fn sub_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        iter.map(move |x| x.sub(other))
    }

    fn mul_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        iter.map(move |x| x.mul(other))
    }

    fn div_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        iter.map(move |x| x.div(other))
    }

    fn scale_direct<I: Iterator<Item = F>>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> {
        iter.map(move |x| x.mul(other))
    }

    fn descale_direct<I: Iterator<Item = F>>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> {
        iter.map(move |x| x.div(other))
    }

    fn fma_direct<I: Iterator<Item = F>>(iter: I, mul: F, add: F) -> impl Iterator<Item = F> {
        iter.map(move |x| x.fma(mul, add))
    }

    fn powf_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        iter.map(move |x| x.powf(other))
    }

    fn powi_direct<I: Iterator<Item = F>>(iter: I, other: i32) -> impl Iterator<Item = F> {
        iter.map(move |x| x.powi(other))
    }

    fn ln_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.ln())
    }

    fn log2_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.log2())
    }

    fn exp_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.exp())
    }

    fn exp2_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.exp2())
    }

    fn recip_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.recip())
    }

    fn sin_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.sin())
    }

    fn cos_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.cos())
    }

    fn tan_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.tan())
    }

    fn asin_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.asin())
    }

    fn acos_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.acos())
    }

    fn atan_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.atan())
    }

    fn sinh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.sinh())
    }

    fn cosh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.cosh())
    }

    fn tanh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.tanh())
    }

    fn asinh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.asinh())
    }

    fn acosh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.acosh())
    }

    fn atanh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        iter.map(move |x| x.atanh())
    }

    fn l1_norm_direct<I: Iterator<Item = F>>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> {
        iter.map(move |x| x.l1_norm())
    }

    fn l2_norm_direct<I: Iterator<Item = F>>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> {
        iter.map(move |x| x.l2_norm())
    }

    fn add_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.add(other))
    }

    fn sub_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.sub(other))
    }

    fn mul_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.mul(other))
    }

    fn div_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.div(other))
    }

    fn fma_vec_direct<I: Iterator<Item = F>>(iter: I, mul: I, add: I) -> impl Iterator<Item = F> {
        iter.zip(mul)
            .zip(add)
            .map(move |((x, mul), add)| x.fma(mul, add))
    }

    fn powf_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.powf(other))
    }

    fn powi_vec_direct<I: Iterator<Item = &'a F> + 'a, J: Iterator<Item = i32> + 'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.powi(other))
    }

    fn greater_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.greater(other))
    }

    fn lesser_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        iter.zip(other).map(move |(x, other)| x.lesser(other))
    }

    fn sum_direct<I: Iterator<Item = F>>(iter: I) -> F {
        iter.fold(F::ZERO, |acc, x| acc.add(x))
    }

    fn product_direct<I: Iterator<Item = F>>(iter: I) -> F {
        iter.fold(F::IDENTITY, |acc, x| acc.mul(x))
    }

    fn greater_direct<I: Iterator<Item = F>>(iter: I) -> F {
        iter.fold(F::MIN, |acc, x| acc.greater(x))
    }

    fn lesser_direct<I: Iterator<Item = F>>(iter: I) -> F {
        iter.fold(F::MIN, |acc, x| acc.lesser(x))
    }

    fn mean_direct<I: Iterator<Item = F>>(iter: I) -> F {
        let (_, mean) = iter.fold((0, F::ZERO), |acc, x| {
            let (mut count, mut mean) = acc;
            count += 1;
            let delta = x - mean;
            mean += delta / F::Primitive::usize(count);
            (count, mean)
        });
        mean
    }

    fn variance_direct<I: Iterator<Item = F>>(iter: I) -> (F, F) {
        let (count, mean, square_distance) = iter.fold((0, F::ZERO, F::ZERO), |acc, x| {
            let (mut count, mut mean, mut square_distance) = acc;
            count += 1;
            let delta = x - mean;
            mean += delta / F::Primitive::usize(count);
            let delta2 = x - mean;
            square_distance += delta * delta2;
            (count, mean, square_distance)
        });
        (mean, square_distance / F::Primitive::usize(count))
    }

    fn deviation_direct<I: Iterator<Item = F>>(iter: I) -> (F, F) {
        let (mean, variance) = Self::variance_direct(iter);
        (mean, variance.sqrt())
    }
}

pub mod float_vector;
pub mod point_vector;

use super::float::Float;

pub trait Vector<'a, F: Float + 'a> {
    // Reduction operations: [F] -> F
    fn sum<I: Iterator<Item = &'a F>>(iter: I) -> F;

    fn product<I: Iterator<Item = &'a F>>(iter: I) -> F;

    fn greater<I: Iterator<Item = &'a F>>(iter: I) -> F;

    fn lesser<I: Iterator<Item = &'a F>>(iter: I) -> F;

    fn mean<I: Iterator<Item = &'a F>>(iter: I) -> F;

    fn variance<I: Iterator<Item = &'a F>>(iter: I) -> (F, F);

    fn deviation<I: Iterator<Item = &'a F>>(iter: I) -> (F, F);

    // Map style operations: [F], F -> [F]
    fn add<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a;

    fn sub<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a;

    fn mul<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a;

    fn div<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a;

    fn neg<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn scale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: F::Primitive,
    ) -> impl Iterator<Item = F> + 'a;

    fn descale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: F::Primitive,
    ) -> impl Iterator<Item = F> + 'a;

    fn fma<I: Iterator<Item = &'a F> + 'a>(iter: I, mul: F, add: F)
        -> impl Iterator<Item = F> + 'a;

    fn powf<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a;

    fn ln<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn log2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn exp<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn exp2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn recip<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn sin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn cos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn tan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn asin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn acos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn atan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn sinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn cosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn tanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn asinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn acosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    fn atanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a;

    // Transformation map operation: [F] -> F::Primitive
    fn l1_norm<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F::Primitive> + 'a;

    fn l2_norm<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F::Primitive> + 'a;

    // In-place operations: [mut F], F -> [F]
    fn add_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F);

    fn sub_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F);

    fn mul_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F);

    fn div_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F);

    fn neg_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn scale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F::Primitive);

    fn descale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F::Primitive);

    fn fma_ref<I: Iterator<Item = &'a mut F>>(iter: I, mul: F, add: F);

    fn powf_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F);

    fn ln_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn log2_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn exp_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn exp2_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn recip_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn sin_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn cos_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn tan_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn asin_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn acos_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn atan_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn sinh_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn cosh_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn tanh_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn asinh_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn acosh_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    fn atanh_ref<I: Iterator<Item = &'a mut F>>(iter: I);

    /*
        Multi Map style operations where [F], [F] -> [F]
    */

    fn add_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a;

    fn sub_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a;

    fn mul_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a;

    fn div_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a;

    fn scale_vec<I: Iterator<Item = &'a F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> + 'a;

    fn descale_vec<I: Iterator<Item = &'a F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> + 'a;

    fn fma_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        mul: I,
        add: I,
    ) -> impl Iterator<Item = F> + 'a;

    fn powf_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a;

    fn greater_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a;

    fn lesser_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a;

    fn add_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, other: J);

    fn sub_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, other: J);

    fn mul_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, other: J);

    fn div_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, other: J);

    fn scale_vec_ref<I: Iterator<Item = &'a mut F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    );

    fn descale_vec_ref<I: Iterator<Item = &'a mut F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    );

    fn fma_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, mul: J, add: J);

    fn powf_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, other: J);

    fn greater_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, other: J);

    fn lesser_vec_ref<I: Iterator<Item = &'a mut F>, J:Iterator<Item = &'a F> >(iter: I, other: J);

    /*
        Common Map -> Reduce operations for ease of use
    */

    fn dot<I: Iterator<Item = &'a F>>(iter: I, other: I) -> F;

    fn quote<I: Iterator<Item = &'a F>>(iter: I, other: I) -> F;

    // Direct map style operations: [F], F -> [F]
    fn add_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F>;

    fn sub_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F>;

    fn mul_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F>;

    fn div_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F>;

    fn neg_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn scale_direct<I: Iterator<Item = F>>(iter: I, other: F::Primitive)
        -> impl Iterator<Item = F>;

    fn descale_direct<I: Iterator<Item = F>>(
        iter: I,
        other: F::Primitive,
    ) -> impl Iterator<Item = F>;

    fn fma_direct<I: Iterator<Item = F>>(iter: I, mul: F, add: F) -> impl Iterator<Item = F>;

    fn powf_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F>;

    fn ln_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn log2_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn exp_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn exp2_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn recip_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn sin_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn cos_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn tan_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn asin_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn acos_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn atan_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn sinh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn cosh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn tanh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn asinh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn acosh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    fn atanh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F>;

    // Direct map operation: [F] -> F::Primitive
    fn l1_norm_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F::Primitive>;

    fn l2_norm_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F::Primitive>;

    fn add_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F>;

    fn sub_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F>;

    fn mul_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F>;

    fn div_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F>;

    fn scale_vec_direct<I: Iterator<Item = F> + 'a, J: Iterator<Item=F::Primitive>+'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F>;

    fn descale_vec_direct<I: Iterator<Item = F> + 'a, J: Iterator<Item=F::Primitive>+'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F>;

    fn fma_vec_direct<I: Iterator<Item = F>>(iter: I, mul: I, add: I) -> impl Iterator<Item = F>;

    fn powf_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F>;

    fn greater_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F>;

    fn lesser_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F>;

    fn add_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, other: J);

    fn sub_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, other: J);

    fn mul_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, other: J);

    fn div_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, other: J);

    fn scale_vec_ref_direct<I: Iterator<Item = &'a mut F> + 'a, J: Iterator<Item=F::Primitive>+'a>(
        iter: I,
        other: J,
    );

    fn descale_vec_ref_direct<I: Iterator<Item = &'a mut F> + 'a, J: Iterator<Item=F::Primitive>+'a>(
        iter: I,
        other: J,
    );

    fn fma_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, mul: J, add: J);

    fn powf_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, other: J);

    fn greater_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, other: J);

    fn lesser_vec_ref_direct<I: Iterator<Item = &'a mut F>, J:Iterator<Item = F> >(iter: I, other: J);
    // Reduction operations: [F] -> F
    fn sum_direct<I: Iterator<Item = F>>(iter: I) -> F;

    fn product_direct<I: Iterator<Item = F>>(iter: I) -> F;

    fn greater_direct<I: Iterator<Item = F>>(iter: I) -> F;

    fn lesser_direct<I: Iterator<Item = F>>(iter: I) -> F;

    fn mean_direct<I: Iterator<Item = F>>(iter: I) -> F;

    fn variance_direct<I: Iterator<Item = F>>(iter: I) -> (F, F);

    fn deviation_direct<I: Iterator<Item = F>>(iter: I) -> (F, F);

    
    /*
        Common Map -> Reduce operations for ease of use
    */

    fn dot_direct<I: Iterator<Item = F>>(iter: I, other: I) -> F;

    fn quote_direct<I: Iterator<Item = F>>(iter: I, other: I) -> F;
}

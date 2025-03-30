use core::cmp::min;
use std::string::String;

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use nonphysical_core::shared::{float::Float, vector::Vector};
use nonphysical_cuda::cuda::{
    global::host::{CuGlobalSlice, CuGlobalSliceRef},
    runtime::{Dim3, RUNTIME},
    stream::CuStream,
    
};

use super::{
    VectorArgumentsApply, VectorArgumentsMap, VectorArgumentsMapReduce, VectorArgumentsReduce,
};
use crate::WARP_SIZE;
pub struct CudaVector {}

/*Theory of every op is
    Make sure runtime is viable
    Allocate memory -> function for this per vector operation
    Call data function collects the vector to allocated memory (Visible for use) -> only unique per vector operation
    That calls the kernel dispatch function, which handles launching the kernel (And relaunches as necessary) -> only unique per vector operation
    Inside PTX land in a landing pad function which exposes the generics
        Call the generic function with a gridstride /warpshuffle algorithm occurs to compute the results,repeat as needed
        Final answer always goes into device memory
    kernel dispatch returns without any memory copy
    memory copy function surfaces relevant memory to host
    Function returns the expected type, dropping device memory

    When I get as far as streams there will be thread local streams/contexts (#[thread_local])
*/

impl<'a, F: Float + 'a> Vector<'a, F> for CudaVector {
    fn sum<I: Iterator<Item = &'a F>>(iter: I) -> F {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "sum");
        acc[0]
    }

    fn product<I: Iterator<Item = &'a F>>(iter: I) -> F {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::IDENTITY];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "product");
        acc[0]
    }

    fn max<I: Iterator<Item = &'a F>>(iter: I) -> F {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "max");
        acc[0]
    }

    fn min<I: Iterator<Item = &'a F>>(iter: I) -> F {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "min");
        acc[0]
    }

    fn mean<I: Iterator<Item = &'a F>>(iter: I) -> F {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "mean");
        acc[0]
    }

    fn variance<I: Iterator<Item = &'a F>>(iter: I) -> (F, F) {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO; 2];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "variance");
        (acc[0], acc[1])
    }

    fn deviation<I: Iterator<Item = &'a F>>(iter: I) -> (F, F) {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO; 2];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "deviation");
        (acc[0], acc[1])
    }

    /*
    fn add<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        use std::time::SystemTime;
        use std::dbg;
        let now = SystemTime::now();

        let stream = CuStream::non_blocking();
        dbg!(now.elapsed());
        let now = SystemTime::now();

        let mut arguments = Self::map_alloc_async(&vector, &output, &[other],&stream);
        dbg!(now.elapsed());

        let now = SystemTime::now();

        Self::map_transfer_async(&mut arguments, &vector, &mut output, &[other], "add",&stream);
        dbg!(now.elapsed());


        output.into_iter()
    }*/
    
    fn add<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "add");
        output.into_iter()
    }

    fn sub<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "sub");
        output.into_iter()
    }

    fn mul<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "mul");
        output.into_iter()
    }

    fn div<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "div");
        output.into_iter()
    }

    fn neg<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[F::ZERO]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[F::ZERO], "neg");
        output.into_iter()
    }
    /*
    fn scale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> + 'a {
        //let vector = iter.map(|x| *x).collect::<Vec<_>>();

        let mut data = CuPinnedSlice::from_iter(iter);
        let mut output = CuPinnedSliceRef::alloc(data.len());
        let mut map = CuPinnedSlice::alloc(1);
        let mut arguments = VectorArgumentsMap { data:data.to_global(), output:output.to_global(), map:map.to_global() };
        let mut output = vec![F::ZERO; data.len()];
        Self::map_transfer_pinned(&mut arguments, &mut output, &[other], "scale");
        output.into_iter()
    }*/

    fn scale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "scale");
        output.into_iter()
    }

    fn descale<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "descale");
        output.into_iter()
    }

    fn fma<I: Iterator<Item = &'a F> + 'a>(
        _iter: I,
        _mul: F,
        _add: F,
    ) -> impl Iterator<Item = F> + 'a {
        todo!();
        /*
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[&[mul], &[add]]);
        Self::map_transfer(
            &mut arguments,
            &vector,
            &mut output,
            &[&[mul], &[add]],
            "fma",
        );
        output.into_iter()*/
        [F::ZERO].into_iter()
    }

    fn powf<I: Iterator<Item = &'a F> + 'a>(iter: I, other: F) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "powf");
        output.into_iter()
    }

    fn ln<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "ln");
        output.into_iter()
    }

    fn log2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "log2");
        output.into_iter()
    }

    fn exp<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "exp");
        output.into_iter()
    }

    fn exp2<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "exp2");
        output.into_iter()
    }

    fn recip<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "recip");
        output.into_iter()
    }

    fn sin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "sin");
        output.into_iter()
    }

    fn cos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "cos");
        output.into_iter()
    }

    fn tan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "tan");
        output.into_iter()
    }

    fn asin<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "asin");
        output.into_iter()
    }

    fn acos<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "acos");
        output.into_iter()
    }

    fn atan<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "atan");
        output.into_iter()
    }

    fn sinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "sinh");
        output.into_iter()
    }

    fn cosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "cosh");
        output.into_iter()
    }

    fn tanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "tanh");
        output.into_iter()
    }

    fn asinh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "asinh");
        output.into_iter()
    }

    fn acosh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "acosh");
        output.into_iter()
    }

    fn atanh<I: Iterator<Item = &'a F> + 'a>(iter: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "atanh");
        output.into_iter()
    }

    fn l1_norm<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::Primitive::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "l1_norm");
        output.into_iter()
    }

    fn l2_norm<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::Primitive::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "l2_norm");
        output.into_iter()
    }

    fn add_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[other]);
        Self::apply_transfer(&mut arguments, &mut vector, &[other], "add_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn sub_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[other]);
        Self::apply_transfer(&mut arguments, &mut vector, &[other], "sub_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn mul_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[other]);
        Self::apply_transfer(&mut arguments, &mut vector, &[other], "mul_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn div_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[other]);
        Self::apply_transfer(&mut arguments, &mut vector, &[other], "div_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn neg_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[F::ZERO]);
        Self::apply_transfer(&mut arguments, &mut vector, &[F::ZERO], "neg_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn scale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: <F as Float>::Primitive) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[other]);
        Self::apply_transfer(&mut arguments, &mut vector, &[other], "scale_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn descale_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: <F as Float>::Primitive) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[other]);
        Self::apply_transfer(&mut arguments, &mut vector, &[other], "descale_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn fma_ref<I: Iterator<Item = &'a mut F>>(_iter: I, _mul: F, _add: F) {
        todo!();
        /*
         let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[&[mul], &[add]]);
        Self::apply_transfer(&mut arguments, &mut vector, &[&[mul], &[add]], "fma_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
        */
    }

    fn powf_ref<I: Iterator<Item = &'a mut F>>(iter: I, other: F) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[other]);
        Self::apply_transfer(&mut arguments, &mut vector, &[other], "powf_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn ln_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "ln_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn log2_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "log2_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn exp_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "exp_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn exp2_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "exp2_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn recip_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "recip_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn sin_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "sin_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn cos_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "cos_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn tan_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "tan_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn asin_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "asin_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn acos_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "acos_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn atan_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "atan_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn sinh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "sinh_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn cosh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "cosh_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn tanh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "atan_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn asinh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "asinh_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn acosh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "acosh_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn atanh_ref<I: Iterator<Item = &'a mut F>>(iter: I) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let apply = [F::ZERO];
        let mut arguments = Self::apply_alloc(&vector, &apply);
        Self::apply_transfer(&mut arguments, &mut vector, &apply, "atanh_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn add_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "add_vec");
        output.into_iter()
    }

    fn sub_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "sub_vec");
        output.into_iter()
    }

    fn mul_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "mul_vec");
        output.into_iter()
    }

    fn div_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "div_vec");
        output.into_iter()
    }

    fn scale_vec<I: Iterator<Item = &'a F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> + 'a{
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "scale_vec");
        output.into_iter()
    }

    fn descale_vec<I: Iterator<Item = &'a F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> + 'a{
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "descale_vec");
        output.into_iter()
    }

    fn fma_vec<I: Iterator<Item = &'a F> + 'a>(
        _iter: I,
        _mul: I,
        _add: I,
    ) -> impl Iterator<Item = F> + 'a {
        todo!();
        /*
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let mul = mul.map(|x| *x).collect::<Vec<_>>();
        let add = add.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output,&[&mul, &add]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[&mul, &add], "div_vec");
        output.into_iter()*/
        [F::ZERO].into_iter()
    }

    fn powf_vec<I: Iterator<Item = &'a F> + 'a>(iter: I, other: I) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "powf_vec");
        output.into_iter()
    }

    fn max_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "max_vec");
        output.into_iter()
    }

    fn min_vec<I: Iterator<Item = &'a F> + 'a>(
        iter: I,
        other: I,
    ) -> impl Iterator<Item = F> + 'a {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "min_vec");
        output.into_iter()
    }

    fn add_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "add_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn sub_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "sub_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn mul_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "mul_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn div_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "div_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    
    fn scale_vec_ref<I: Iterator<Item = &'a mut F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    ){
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "scale_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn descale_vec_ref<I: Iterator<Item = &'a mut F> + 'a, J: Iterator<Item=&'a F::Primitive>+'a>(
        iter: I,
        other: J,
    ){
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "scale_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn fma_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(
        _iter: I,
        _mul: J,
        _add: J,
    ) {
        todo!();
        /*
         let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let mul = mul.map(|x|*x).collect::<Vec<_>>();
        let add = add.map(|x|*x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &[&mul,&add]);
        Self::apply_transfer(&mut arguments, &mut vector, &[&mul,&add], "fma_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
        */
    }

    fn powf_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "powf_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn max_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(
        iter: I,
        other: J,
    ) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "max_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn min_vec_ref<I: Iterator<Item = &'a mut F>, J: Iterator<Item = &'a F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "min_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn dot<I: Iterator<Item = &'a F>>(iter: I, other: I) -> F {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::map_reduce_alloc(&vector, &acc, &other);
        Self::map_reduce_transfer(&mut arguments, &vector, &mut acc, &other, "dot");
        acc[0]
    }

    fn quote<I: Iterator<Item = &'a F>>(iter: I, other: I) -> F {
        let vector = iter.map(|x| *x).collect::<Vec<_>>();
        let other = other.map(|x| *x).collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::map_reduce_alloc(&vector, &acc, &other);
        Self::map_reduce_transfer(&mut arguments, &vector, &mut acc, &other, "quote");
        acc[0]
    }

    fn add_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "add");
        output.into_iter()
    }

    fn sub_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "sub");
        output.into_iter()
    }

    fn mul_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "mul");
        output.into_iter()
    }

    fn div_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "div");
        output.into_iter()
    }

    fn neg_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[F::ZERO]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[F::ZERO], "neg");
        output.into_iter()
    }

    fn scale_direct<I: Iterator<Item = F>>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "scale");
        output.into_iter()
    }

    fn descale_direct<I: Iterator<Item = F>>(
        iter: I,
        other: <F as Float>::Primitive,
    ) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "descale");
        output.into_iter()
    }

    fn fma_direct<I: Iterator<Item = F>>(_iter: I, _mul: F, _add: F) -> impl Iterator<Item = F> {
        todo!();
        /*
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[&[mul], &[add]]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[&[mul], &[add]], "fma");
        output.into_iter()*/
        [F::ZERO].into_iter()
    }

    fn powf_direct<I: Iterator<Item = F>>(iter: I, other: F) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &vector, &[other]);
        Self::map_transfer(&mut arguments, &vector, &mut output, &[other], "powf");
        output.into_iter()
    }

    fn ln_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "ln");
        output.into_iter()
    }

    fn log2_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "log2");
        output.into_iter()
    }

    fn exp_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "exp");
        output.into_iter()
    }

    fn exp2_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "exp2");
        output.into_iter()
    }

    fn recip_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "recip");
        output.into_iter()
    }

    fn sin_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "sin");
        output.into_iter()
    }

    fn cos_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "cos");
        output.into_iter()
    }

    fn tan_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "tan");
        output.into_iter()
    }

    fn asin_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "asin");
        output.into_iter()
    }

    fn acos_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "acos");
        output.into_iter()
    }

    fn atan_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "atan");
        output.into_iter()
    }

    fn sinh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "sinh");
        output.into_iter()
    }

    fn cosh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "cosh");
        output.into_iter()
    }

    fn tanh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "tanh");
        output.into_iter()
    }

    fn asinh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "asinh");
        output.into_iter()
    }

    fn acosh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "acosh");
        output.into_iter()
    }

    fn atanh_direct<I: Iterator<Item = F>>(iter: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "atanh");
        output.into_iter()
    }

    fn l1_norm_direct<I: Iterator<Item = F>>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::Primitive::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "l1_norm");
        output.into_iter()
    }

    fn l2_norm_direct<I: Iterator<Item = F>>(
        iter: I,
    ) -> impl Iterator<Item = <F as Float>::Primitive> {
        let vector = iter.collect::<Vec<_>>();
        let mut output = vec![F::Primitive::ZERO; vector.len()];
        let map = [F::ZERO];
        let mut arguments = Self::map_alloc(&vector, &output, &map);
        Self::map_transfer(&mut arguments, &vector, &mut output, &map, "l2_norm");
        output.into_iter()
    }

    fn add_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "add_vec");
        output.into_iter()
    }

    fn sub_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "sub_vec");
        output.into_iter()
    }

    fn mul_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "mul_vec");
        output.into_iter()
    }

    fn div_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "div_vec");
        output.into_iter()
    }
    fn scale_vec_direct<I: Iterator<Item = F> + 'a, J: Iterator<Item = F::Primitive> + 'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "scale_vec");
        output.into_iter()
    }

    fn descale_vec_direct<I: Iterator<Item = F> + 'a, J: Iterator<Item = F::Primitive> + 'a>(
        iter: I,
        other: J,
    ) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "descale_vec");
        output.into_iter()
    }

    fn fma_vec_direct<I: Iterator<Item = F>>(_iter: I, _mul: I, _add: I) -> impl Iterator<Item = F> {
        todo!();
        /*
        let vector = iter.collect::<Vec<_>>();
        let mul = mul.collect::<Vec<_>>();
        let add = add.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &[&mul, &add]);
        Self::map_transfer(&mut arguments, &vector, &mut output,  &[&mul, &add], "fma_vec");
        output.into_iter()*/
        [F::ZERO].into_iter()
    }

    fn powf_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "powf_vec");
        output.into_iter()
    }

    fn max_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "max_vec");
        output.into_iter()
    }

    fn min_vec_direct<I: Iterator<Item = F>>(iter: I, other: I) -> impl Iterator<Item = F> {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut output = vec![F::ZERO; vector.len()];
        let mut arguments = Self::map_alloc(&vector, &output, &other);
        Self::map_transfer(&mut arguments, &vector, &mut output, &other, "min_vec");
        output.into_iter()
    }

    fn add_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "add_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn sub_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "sub_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn mul_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "mul_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn div_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(iter: I, other: J) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "div_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn scale_vec_ref_direct<
        I: Iterator<Item = &'a mut F> + 'a,
        J: Iterator<Item = F::Primitive> + 'a,
    >(
        iter: I,
        other: J,
    ) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "scale_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn descale_vec_ref_direct<
        I: Iterator<Item = &'a mut F> + 'a,
        J: Iterator<Item = F::Primitive> + 'a,
    >(
        iter: I,
        other: J,
    ) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "descale_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn fma_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(
        _iter: I,
        _mul: J,
        _add: J,
    ) {
        todo!()
    }

    fn powf_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(
        iter: I,
        other: J,
    ) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "powf_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn max_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(
        iter: I,
        other: J,
    ) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "max_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn min_vec_ref_direct<I: Iterator<Item = &'a mut F>, J: Iterator<Item = F>>(
        iter: I,
        other: J,
    ) {
        let mut ref_vector = iter.collect::<Vec<_>>();
        let mut vector = ref_vector.iter().map(|x| **x).collect::<Vec<_>>();
        let other = other.map(|x| x).collect::<Vec<_>>();
        let mut arguments = Self::apply_alloc(&vector, &other);
        Self::apply_transfer(&mut arguments, &mut vector, &other, "min_vec_ref");
        ref_vector
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(r, v)| **r = *v);
    }

    fn sum_direct<I: Iterator<Item = F>>(iter: I) -> F {
        let vector = iter.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "sum");
        acc[0]
    }

    fn product_direct<I: Iterator<Item = F>>(iter: I) -> F {
        let vector = iter.collect::<Vec<_>>();
        let mut acc = vec![F::IDENTITY];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "product");
        acc[0]
    }

    fn max_direct<I: Iterator<Item = F>>(iter: I) -> F {
        let vector = iter.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "max");
        acc[0]
    }

    fn min_direct<I: Iterator<Item = F>>(iter: I) -> F {
        let vector = iter.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "min");
        acc[0]
    }

    fn mean_direct<I: Iterator<Item = F>>(iter: I) -> F {
        let vector = iter.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "mean");
        acc[0]
    }

    fn variance_direct<I: Iterator<Item = F>>(iter: I) -> (F, F) {
        let vector = iter.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO; 2];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "variance");
        (acc[0], acc[1])
    }

    fn deviation_direct<I: Iterator<Item = F>>(iter: I) -> (F, F) {
        let vector = iter.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO; 2];
        let mut arguments = Self::reduce_alloc(&vector, &acc);
        Self::reduce_transfer(&mut arguments, &vector, &mut acc, "deviation");
        (acc[0], acc[1])
    }

    fn dot_direct<I: Iterator<Item = F>>(iter: I, other: I) -> F {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::map_reduce_alloc(&vector, &acc, &other);
        Self::map_reduce_transfer(&mut arguments, &vector, &mut acc, &other, "dot");
        acc[0]
    }

    fn quote_direct<I: Iterator<Item = F>>(iter: I, other: I) -> F {
        let vector = iter.collect::<Vec<_>>();
        let other = other.collect::<Vec<_>>();
        let mut acc = vec![F::ZERO];
        let mut arguments = Self::map_reduce_alloc(&vector, &acc, &other);
        Self::map_reduce_transfer(&mut arguments, &vector, &mut acc, &other, "quote");
        acc[0]
    }
}

impl CudaVector {
    fn launch<Args>(args: &mut Args, len: usize, kernel: String) {
        let threads = min(1024, len.div_ceil(WARP_SIZE / 2));
        let block_size = len.div_ceil(threads * WARP_SIZE / 2); //Half a warp was optimal in testing
        let grid = Dim3 {
            x: block_size,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: threads,
            y: 1,
            z: 1,
        };
        match RUNTIME.get() {
            Some(rt) => {
                rt.launch_name(kernel, args, grid, block);
            }
            None => panic!("Cuda Runtime not initialized"),
        };
    }

    fn launch_async<Args>(args: &mut Args, len: usize, kernel: String,stream:&CuStream) {
        let threads = min(1024, len.div_ceil(WARP_SIZE / 2));
        let block_size = len.div_ceil(threads * WARP_SIZE / 2); //Half a warp was optimal in testing
        let grid = Dim3 {
            x: block_size,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: threads,
            y: 1,
            z: 1,
        };
        match RUNTIME.get() {
            Some(rt) => {
                rt.launch_name_async(kernel, args, grid, block,stream);
            }
            None => panic!("Cuda Runtime not initialized"),
        };
    }
    fn reduce_alloc<'a, Fa: Float + 'a, Fb: Float + 'a>(
        vector: &[Fa],
        acc: &[Fb],
    ) -> VectorArgumentsReduce<'a, Fa, Fb> {
        let data = CuGlobalSlice::alloc(&vector);
        let acc = CuGlobalSliceRef::alloc(&acc);
        VectorArgumentsReduce { data, acc }
    }
    fn map_alloc<'a, Fa: Float + 'a, Fb: Float + 'a, Fc: Float + 'a>(
        vector: &[Fa],
        output: &[Fb],
        other: &[Fc],
    ) -> VectorArgumentsMap<'a, Fa, Fb, Fc> {
        let data = CuGlobalSlice::alloc(&vector);
        let output = CuGlobalSliceRef::alloc(&output);
        let map = CuGlobalSlice::alloc(other);
        VectorArgumentsMap { data, output, map }
    }

    fn map_alloc_async<'a, Fa: Float + 'a, Fb: Float + 'a, Fc: Float + 'a>(
        vector: &[Fa],
        output: &[Fb],
        other: &[Fc],
        stream: &CuStream,
    ) -> VectorArgumentsMap<'a, Fa, Fb, Fc> {
        let data = CuGlobalSlice::alloc_async(&vector,stream);
        let output = CuGlobalSliceRef::alloc_async(&output,stream);
        let map = CuGlobalSlice::alloc_async(other,stream);
        VectorArgumentsMap { data, output, map }
    }

    fn apply_alloc<'a, Fa: Float + 'a, Fb: Float + 'a>(
        vector: &[Fa],
        other: &[Fb],
    ) -> VectorArgumentsApply<'a, Fa, Fb> {
        let data = CuGlobalSliceRef::alloc(&vector);
        let map = CuGlobalSlice::alloc(other);
        VectorArgumentsApply { data, map }
    }

    fn map_reduce_alloc<'a, Fa: Float + 'a, Fb: Float + 'a, Fc: Float + 'a>(
        vector: &[Fa],
        acc: &[Fb],
        other: &[Fc],
    ) -> VectorArgumentsMapReduce<'a, Fa, Fb, Fc> {
        let data = CuGlobalSlice::alloc(&vector);
        let acc = CuGlobalSliceRef::alloc(&acc);
        let map = CuGlobalSlice::alloc(other);
        VectorArgumentsMapReduce { data, acc, map }
    }

    fn reduce_transfer<'a, Fa: Float + 'a, Fb: Float + 'a>(
        arguments: &mut VectorArgumentsReduce<'a, Fa, Fb>,
        vector: &[Fa],
        acc: &mut [Fb],
        name: &str,
    ) {
        let len = arguments.data.ptr.len();
        arguments.data.store(vector);
        let kernel = format!("vector_{}_{}", name, Fa::type_id());
        Self::launch(arguments, len, kernel);
        arguments.acc.load(acc);
    }

    fn map_transfer<'a, Fa: Float + 'a, Fb: Float + 'a, Fc: Float + 'a>(
        arguments: &mut VectorArgumentsMap<'a, Fa, Fb, Fc>,
        vector: &[Fa],
        output: &mut [Fb],
        map: &[Fc],
        name: &str,
    ) {
        let len = arguments.data.ptr.len();
        assert!(len == arguments.output.ptr.len());
        arguments.data.store(&vector);
        arguments.map.store(map);
        let kernel = format!("vector_{}_{}", name, Fa::type_id());
        Self::launch(arguments, len, kernel);
        arguments.output.load(output);
    }

    fn map_transfer_async<'a, Fa: Float + 'a, Fb: Float + 'a, Fc: Float + 'a>(
        arguments: &mut VectorArgumentsMap<'a, Fa, Fb, Fc>,
        vector: &[Fa],
        output: &mut [Fb],
        map: &[Fc],
        name: &str,
        stream: &CuStream,
    ) {
        let len = arguments.data.ptr.len();
        assert!(len == arguments.output.ptr.len());
        stream.sync();
        arguments.data.store_async(&vector,stream);
        arguments.map.store_async(map,stream);
        stream.sync();
        let kernel = format!("vector_{}_{}", name, Fa::type_id());
        Self::launch_async(arguments, len, kernel,stream);
        stream.sync();
        arguments.output.load_async(output,stream);
        stream.sync();

    }
    fn map_transfer_pinned<'a, Fa: Float + 'a, Fb: Float + 'a, Fc: Float + 'a>(
        arguments: &mut VectorArgumentsMap<'a, Fa, Fb, Fc>,
        output: &mut [Fb],
        map: &[Fc],
        name: &str,
    ) {
        let len = arguments.data.ptr.len();
        assert!(len == arguments.output.ptr.len());
        arguments.map.store(map);
        let kernel = format!("vector_{}_{}", name, Fa::type_id());
        Self::launch(arguments, len, kernel);
        arguments.output.load(output);
    }
    

    fn apply_transfer<'a, Fa: Float + 'a, Fb: Float + 'a>(
        arguments: &mut VectorArgumentsApply<'a, Fa, Fb>,
        vector: &mut [Fa],
        map: &[Fb],
        name: &str,
    ) {
        let len = arguments.data.ptr.len();
        arguments.data.store(&vector);
        arguments.map.store(map);
        let kernel = format!("vector_{}_{}", name, Fa::type_id());
        Self::launch(arguments, len, kernel);
        arguments.data.load(vector);
    }

    fn map_reduce_transfer<'a, Fa: Float + 'a, Fb: Float + 'a, Fc: Float + 'a>(
        arguments: &mut VectorArgumentsMapReduce<'a, Fa, Fb, Fc>,
        vector: &[Fa],
        acc: &mut [Fb],
        map: &[Fc],
        name: &str,
    ) {
        let len = arguments.data.ptr.len();
        arguments.data.store(&vector);
        arguments.map.store(map);
        let kernel = format!("vector_{}_{}", name, Fa::type_id());
        Self::launch(arguments, len, kernel);
        arguments.acc.load(acc);
    }
}

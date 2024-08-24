use core::marker::PhantomData;
use crate::shared::complex::Complex;

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
}
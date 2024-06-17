use std::{marker::PhantomData, num::Wrapping, time::SystemTime};

use crate::shared::{complex::Complex, float::Float};

pub struct PermutedCongruentialGenerator<T> {
    state: usize,
    inc: usize,
    phantom: PhantomData<T>,
}

//this is probably not cryptographically secure
impl<T:Float> PermutedCongruentialGenerator<T> {
    pub fn new(state: usize, inc: usize) -> Self {
        Self { state, inc, phantom: PhantomData::default() }
    }

    //not safe at all, reasonably easy to extract time
    pub fn new_timed() -> Self {
        let time = SystemTime::now();

        //do a few calculations here just to allow some time
        let state = (!time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()) as usize;

        let inc = (!time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()) as usize;

        Self { state, inc,phantom: PhantomData::default() }
    }

    pub fn next_usize(&mut self) -> usize {
        let old_state = Wrapping(self.state);
        // I suspect there is a bug here when usize=u32 and the multiplicand to state wraps. 
        // Could fix this by breaking the large number to smaller chunks, but the prime decomposition is too large
        self.state = (old_state * Wrapping(6_364_136_223_846_793_005) + Wrapping(self.inc | 1)).0;  
        let shift = ((old_state >> 18) ^ old_state >> 26).0;
        let rot = (old_state >> 59).0;
        (shift >> rot) | (shift << (-(rot as isize) & 31))
    }

    pub fn normal(&mut self, mean: Complex<T>, scale: T, size: usize) -> Vec<Complex<T>>{
        let mut samples = Vec::with_capacity(size);
        (0..size).for_each(|_| {
            let u1 = T::usize(self.next_usize())/T::usize(usize::MAX);
            let u2 = T::usize(self.next_usize())/T::usize(usize::MAX);
            let z0 = (-T::usize(2)*u1.ln()).sqrt() * (T::usize(2)*T::pi()*u2).cos();
            samples.push(mean+Complex::new(z0*scale,T::zero()));
        });
        samples
    }

    pub fn shuffle_usize(&mut self, x: &mut Vec<usize>) {
        (1..x.len()).rev().for_each(|i|{
            let j = self.next_usize() % i; //this introduces bias when x.len() approaches usize, I find this to be a non problem
            x.swap(i, j);
        })
    }
}

use std::{marker::PhantomData, num::Wrapping, time::SystemTime};

use crate::shared::{complex::Complex, float::Float};

pub struct PermutedCongruentialGenerator<T> {
    state: u32,
    inc: u32,
    phantom: PhantomData<T>,
}

//this is probably not cryptographically secure
impl<T:Float> PermutedCongruentialGenerator<T> {
    pub fn new(state: u32, inc: u32) -> Self {
        debug_assert!(state!=0 || inc!=0);
        Self { state, inc, phantom: PhantomData }
    }

    //not safe at all, reasonably easy to extract time
    pub fn new_timed() -> Self {
        let time = SystemTime::now();

        //do a few calculations here just to allow some time
        let state = (!time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()) as u32;

        let inc = state<<1 |1;

        Self { state, inc,phantom: PhantomData }
    }

    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = (Wrapping(old_state)*Wrapping(747796405)+Wrapping(self.inc)).0;
        let word = Wrapping((old_state >>((old_state>>28)+4))^old_state) *Wrapping(277803737);
        ((word>>22) ^word).0
    }

    pub fn normal(&mut self, mean: Complex<T>, scale: T, size: usize) -> Vec<Complex<T>>{        
        let mut samples = Vec::with_capacity(size);
        (0..size).for_each(|_| {
            let mut u1 = T::usize(self.next_u32() as usize)/T::usize(u32::MAX as usize);
            while u1==T::ZERO{
                u1 = T::usize(self.next_u32()as usize)/T::usize(u32::MAX as usize);
            }
            let u2 = T::usize(self.next_u32()as usize)/T::usize(u32::MAX as usize);
            let z0 = (-T::usize(2)*u1.ln()).sqrt() * (T::usize(2)*T::PI*u2).cos();
            samples.push(mean+Complex::new(z0*scale,T::ZERO));
        });
        samples
    }

    pub fn shuffle_usize(&mut self, x: &mut [usize]) {
        (1..x.len()).rev().for_each(|i|{
            let j = (self.next_u32() as usize) % i; //this introduces bias when x.len() approaches usize, I find this to be a non problem
            x.swap(i, j);
        })
    }
}

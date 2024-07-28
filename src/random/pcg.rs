use core::num::Wrapping;
use std::time::SystemTime;
use crate::shared::real::Real;
use alloc::vec::Vec;

pub struct PermutedCongruentialGenerator {
    state: u32,
    inc: u32,
}

//this is probably not cryptographically secure
impl PermutedCongruentialGenerator{
    pub fn new(state: u32, inc: u32) -> Self {
        debug_assert!(state!=0 || inc!=0);
        Self { state, inc }
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

        Self { state, inc }
    }

    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = (Wrapping(old_state)*Wrapping(747796405)+Wrapping(self.inc)).0;
        let word = Wrapping((old_state >>((old_state>>28)+4))^old_state) *Wrapping(277803737);
        ((word>>22) ^word).0
    }

    pub fn uniform_singular<R:Real>(&mut self, start: R, end:R)-> R{
        R::u32(self.next_u32())/R::u32(u32::MAX)*(end-start)+start
    }

    pub fn uniform<R:Real>(&mut self, start: R, end:R, size: usize)-> Vec<R>{
        (0..size).map(|_|{
            R::u32(self.next_u32())/R::u32(u32::MAX)*(end-start)+start
        }).collect()
    }

    pub fn interval<R:Real>(&mut self, size: usize)-> Vec<R>{
        (0..size).map(|_|{
            R::u32(self.next_u32())/R::u32(u32::MAX)
        }).collect()
    }

    pub fn normal<R:Real>(&mut self, mean: R, scale: R, size: usize) -> Vec<R>{        
        let mut samples = Vec::with_capacity(size);
        (0..size).for_each(|_| {
            let mut u1 = R::u32(self.next_u32())/R::u32(u32::MAX);
            while u1==R::ZERO{
                u1 = R::u32(self.next_u32())/R::u32(u32::MAX);
            }
            let u2 = R::u32(self.next_u32())/R::u32(u32::MAX);
            let z0 = (-R::usize(2)*u1.ln()).sqrt() * (R::u8(2)*R::PI*u2).cos();
            samples.push(mean+z0*scale);
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

use core::num::Wrapping;
use crate::shared::primitive::Primitive;
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

    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = (Wrapping(old_state)*Wrapping(747796405)+Wrapping(self.inc)).0;
        let word = Wrapping((old_state >>((old_state>>28)+4))^old_state) *Wrapping(277803737);
        ((word>>22) ^word).0
    }

    pub fn uniform_singular<P:Primitive>(&mut self, start: P, end:P)-> P{
        P::u32(self.next_u32())/P::u32(u32::MAX)*(end-start)+start
    }

    pub fn uniform<P:Primitive>(&mut self, start: P, end:P, size: usize)-> Vec<P>{
        (0..size).map(|_|{
            P::u32(self.next_u32())/P::u32(u32::MAX)*(end-start)+start
        }).collect()
    }

    pub fn interval<P:Primitive>(&mut self, size: usize)-> Vec<P>{
        (0..size).map(|_|{
            P::u32(self.next_u32())/P::u32(u32::MAX)
        }).collect()
    }

    pub fn normal<P:Primitive>(&mut self, mean: P, scale: P, size: usize) -> Vec<P>{        
        let mut samples = Vec::with_capacity(size);
        (0..size).for_each(|_| {
            let mut u1 = P::u32(self.next_u32())/P::u32(u32::MAX);
            while u1==P::ZERO{
                u1 = P::u32(self.next_u32())/P::u32(u32::MAX);
            }
            let u2 = P::u32(self.next_u32())/P::u32(u32::MAX);
            let z0 = (-P::usize(2)*u1.ln()).sqrt() * (P::u8(2)*P::PI*u2).cos();
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

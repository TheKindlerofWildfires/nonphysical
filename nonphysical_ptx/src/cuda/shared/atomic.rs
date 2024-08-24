use crate::cuda::atomic::{Atomic, Reduce};
use crate::cuda::shared::CuShared;
use crate::shared::primitive::F32;
use core::arch::asm;
use nonphysical_core::shared::float::Float;
use crate::cuda::shared::Shared;
use nonphysical_core::shared::primitive::Primitive;
impl<const N: usize> Atomic<F32> for CuShared<F32, N> {
    fn atomic_add(&mut self, index: usize, value: F32) -> F32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.shared.add.f32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }


    fn atomic_exch(&mut self, index: usize, value: F32) -> F32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.shared.exch.f32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
    fn atomic_max(&mut self, index: usize, value: F32) -> F32 {
        let mut old  = self.load(index);
        let mut assumed ;
        let mut value = value;
        if old>=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.greater(value);
            if old==assumed{
                break
            }
        }
        return old;
    }

    fn atomic_min(&mut self, index: usize, value: F32) -> F32 {
        let mut old  = self.load(index);
        let mut assumed ;
        let mut value = value;

        if old<=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.lesser(value);
            if old==assumed{
                break
            }
        }
        return old;
    }

    fn atomic_inc(&mut self, index: usize, value: F32) -> F32 {
        let mut old  = self.load(index);
        let mut assumed;
        if old>=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed+F32::ONE);
            if old==assumed || old>=value{
                break
            }
        }
        return old;
    }

    fn atomic_dec(&mut self, index: usize, value: F32) -> F32 {
        let mut old  = self.load(index);
        let mut assumed;
        if old<=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed-F32::ONE);
            if old==assumed || old<=value{
                break
            }
        }
        return old;
    }
    fn atomic_cas(&mut self, index: usize, compare: F32, value: F32) -> F32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.shared.cas.b32 {o},[{idx}], {v}, {c};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                c = in(reg32) compare.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_and(&mut self, index: usize, value: F32) -> F32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.shared.and.b32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_or(&mut self, index: usize, value: F32) -> F32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.shared.or.b32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_xor(&mut self, index: usize, value: F32) -> F32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.shared.xor.b32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
}

impl<const N: usize> Reduce<F32> for CuShared<F32, N> {
    fn reduce_add(&mut self, index: usize, value: F32) {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        unsafe {
            asm!(
                "red.shared.add.f32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }
    fn reduce_min(&mut self, index: usize, value: F32) {
        let mut old  = self.load(index);
        let mut assumed ;
        let mut value = value;
        if old<=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.lesser(value);
            if old==assumed{
                break
            }
        }
    }
    fn reduce_max(&mut self, index: usize, value: F32) {
        let mut old  = self.load(index);
        let mut assumed ;
        let mut value = value;
        if old>=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.greater(value);
            if old==assumed{
                break
            }
        }
    }
    fn reduce_inc(&mut self, index: usize, value: F32) {
        let mut old  = self.load(index);
        let mut assumed;
        if old>=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed+F32::ONE);
            if old==assumed || old>=value{
                break
            }
        }
    }

    fn reduce_dec(&mut self, index: usize, value: F32) {
        let mut old  = self.load(index);
        let mut assumed;
        if old<=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed-F32::ONE);
            if old==assumed|| old<=value{
                break
            }
        }
    }

    fn reduce_and(&mut self, index: usize, value: F32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        unsafe {
            asm!(
                "red.shared.and.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }

    fn reduce_or(&mut self, index: usize, value: F32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        unsafe {
            asm!(
                "red.shared.or.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }

    fn reduce_xor(&mut self, index: usize, value: F32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<F32>()) as u32;
        unsafe {
            asm!(
                "red.shared.xor.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }
}

impl<const N: usize> Atomic<u32> for CuShared<u32, N> {
    fn atomic_add(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.add.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_exch(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.exch.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }
    fn atomic_max(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out:u32;
        unsafe {
            asm!(
                "atom.shared.max.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_min(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.min.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_inc(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.inc.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_dec(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.dec.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_cas(&mut self, index: usize, compare: u32, value: u32) -> u32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.cas.b32 {o},[{idx}], {c}, {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                c = in(reg32) compare,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_and(&mut self, index: usize, value: u32) -> u32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.or.b32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_or(&mut self, index: usize, value: u32) -> u32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.or.b32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_xor(&mut self, index: usize, value: u32) -> u32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.shared.xor.b32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }
}

impl<const N: usize> Reduce<u32> for CuShared<u32, N> {
    fn reduce_add(&mut self, index: usize, value: u32) {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.add.u32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_max(&mut self, index: usize, value: u32) {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.max.u32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_min(&mut self, index: usize, value: u32) {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.min.u32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_inc(&mut self, index: usize, value: u32) {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.inc.u32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }
    fn reduce_dec(&mut self, index: usize, value: u32) {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.dec.u32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_and(&mut self, index: usize, value: u32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.or.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_or(&mut self, index: usize, value: u32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.or.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_xor(&mut self, index: usize, value: u32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.shared.xor.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value,
            );
        }
    }
}

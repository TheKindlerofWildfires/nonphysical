use crate::{
    cuda::{
        atomic::{Atomic, Reduce},
        global::device::CuGlobalSliceRef,
    },
    shared::primitive::F32,
};
use core::arch::asm;
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::primitive::Primitive;

impl<'a> Atomic<F32> for CuGlobalSliceRef<'a, F32> {
    fn atomic_add(&mut self, index: usize, value: F32) -> F32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.global.add.f32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_exch(&mut self, index: usize, value: F32) -> F32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.global.exch.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_max(&mut self, index: usize, value: F32) -> F32 {
        let mut old = self[index];
        let mut assumed;
        let mut value = value;
        if old >= value {
            return old;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.greater(value);
            if old == assumed {
                break;
            }
        }
        return old;
    }

    fn atomic_min(&mut self, index: usize, value: F32) -> F32 {
        let mut old = self[index];
        let mut assumed;
        let mut value = value;
        if old <= value {
            return old;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.lesser(value);
            if old == assumed {
                break;
            }
        }
        return old;
    }

    fn atomic_inc(&mut self, index: usize, value: F32) -> F32 {
        let mut old = self[index];
        let mut assumed;
        if old >= value {
            return old;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed + F32::ONE);
            if old == assumed || old >= value {
                break;
            }
        }
        return old;
    }

    fn atomic_dec(&mut self, index: usize, value: F32) -> F32 {
        let mut old = self[index];
        let mut assumed;
        if old <= value {
            return old;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed - F32::ONE);
            if old == assumed || old <= value {
                break;
            }
        }
        return old;
    }

    fn atomic_cas(&mut self, index: usize, compare: F32, value: F32) -> F32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.global.cas.b32 {o},[{idx}], {c}, {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                c = in(reg32) compare.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_and(&mut self, index: usize, value: F32) -> F32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.global.and.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_or(&mut self, index: usize, value: F32) -> F32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.global.or.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn atomic_xor(&mut self, index: usize, value: F32) -> F32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "atom.global.xor.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
}

impl<'a> Reduce<F32> for CuGlobalSliceRef<'a, F32> {
    fn reduce_add(&mut self, index: usize, value: F32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        unsafe {
            asm!(
                "red.global.add.f32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }

    fn reduce_max(&mut self, index: usize, value: F32) {
        let mut old = self[index];
        let mut assumed;
        let mut value = value;
        if old >= value {
            return;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.greater(value);
            if old == assumed {
                break;
            }
        }
    }

    fn reduce_min(&mut self, index: usize, value: F32) {
        let mut old = self[index];
        let mut assumed;
        let mut value = value;
        if old <= value {
            return;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, value);
            value = old.lesser(value);
            if old == assumed {
                break;
            }
        }
    }

    fn reduce_inc(&mut self, index: usize, value: F32) {
        let mut old = self[index];
        let mut assumed;
        if old >= value {
            return;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed + F32::ONE);
            if old == assumed || old >= value {
                break;
            }
        }
    }

    fn reduce_dec(&mut self, index: usize, value: F32) {
        let mut old = self[index];
        let mut assumed;
        if old <= value {
            return;
        }
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed - F32::ONE);
            if old == assumed || old <= value {
                break;
            }
        }
    }

    fn reduce_and(&mut self, index: usize, value: F32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        unsafe {
            asm!(
                "red.global.and.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }

    fn reduce_or(&mut self, index: usize, value: F32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        unsafe {
            asm!(
                "red.global.or.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }

    fn reduce_xor(&mut self, index: usize, value: F32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<F32>()) as u64;
        unsafe {
            asm!(
                "red.global.xor.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
}

impl<'a> Atomic<u32> for CuGlobalSliceRef<'a, u32> {
    fn atomic_add(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.add.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_exch(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.exch.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_max(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.max.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_min(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.min.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_inc(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.inc.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_dec(&mut self, index: usize, value: u32) -> u32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.dec.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_cas(&mut self, index: usize, compare: u32, value: u32) -> u32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.cas.b32 {o}, [{idx}], {v}, {c};",
                idx = in(reg64) index,
                v = in(reg32) value,
                c = in(reg32) compare,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_and(&mut self, index: usize, value: u32) -> u32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.or.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_or(&mut self, index: usize, value: u32) -> u32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.or.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }

    fn atomic_xor(&mut self, index: usize, value: u32) -> u32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "atom.global.xor.b32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
                o = out(reg32) out,
            );
        }
        out
    }
}

impl<'a> Reduce<u32> for CuGlobalSliceRef<'a, u32> {
    fn reduce_add(&mut self, index: usize, value: u32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.add.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_max(&mut self, index: usize, value: u32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.max.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_min(&mut self, index: usize, value: u32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.min.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_inc(&mut self, index: usize, value: u32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.inc.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_dec(&mut self, index: usize, value: u32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.dec.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_and(&mut self, index: usize, value: u32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.or.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_or(&mut self, index: usize, value: u32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.or.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }

    fn reduce_xor(&mut self, index: usize, value: u32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u32 + (index * size_of::<u32>()) as u32;
        unsafe {
            asm!(
                "red.global.xor.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value,
            );
        }
    }
}

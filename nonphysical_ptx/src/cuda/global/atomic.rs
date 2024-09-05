use crate::{
    cuda::{
        atomic::{Atomic, Reduce},
        global::device::CuGlobalSliceRef,
    },
    shared::primitive::F32,
};
use core::arch::asm;
use nonphysical_core::shared::{float::Float, unsigned::Unsigned};
use nonphysical_core::shared::primitive::Primitive;
use crate::shared::unsigned::U32;

impl<'a> Atomic<F32> for CuGlobalSliceRef<'a, F32> {

    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn atomic_mul(&mut self, index: usize, value: F32) -> F32 {
        let mut old = self[index];
        let mut assumed;
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed*value);
            if old == assumed {
                break;
            }
        }
        return old;
    }
}

impl<'a> Reduce<F32> for CuGlobalSliceRef<'a, F32> {
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn reduce_mul(&mut self, _index: usize, _value: F32) {
        todo!()
    }
}

impl<'a> Atomic<U32> for CuGlobalSliceRef<'a, U32> {

    #[inline(always)]
    fn atomic_add(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_exch(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_max(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "atom.global.max.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
    #[inline(always)]
    fn atomic_min(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "atom.global.min.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
    #[inline(always)]
    fn atomic_inc(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "atom.global.inc.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
    #[inline(always)]
    fn atomic_dec(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "atom.global.dec.u32 {o},[{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
    #[inline(always)]
    fn atomic_cas(&mut self, index: usize, compare: U32, value: U32) -> U32 {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_and(&mut self, index: usize, value: U32) -> U32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_or(&mut self, index: usize, value: U32) -> U32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_xor(&mut self, index: usize, value: U32) -> U32 {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_mul(&mut self, index: usize, value: U32) -> U32 {
        let mut old = self[index];
        let mut assumed;
        loop {
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed*value);
            if old == assumed {
                break;
            }
        }
        return old;
    }
}

impl<'a> Reduce<U32> for CuGlobalSliceRef<'a, U32> {
    #[inline(always)]
    fn reduce_add(&mut self, index: usize, value: U32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.add.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_max(&mut self, index: usize, value: U32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.max.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_min(&mut self, index: usize, value: U32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.min.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_inc(&mut self, index: usize, value: U32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.inc.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_dec(&mut self, index: usize, value: U32) {
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.dec.u32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_and(&mut self, index: usize, value: U32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.and.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_or(&mut self, index: usize, value: U32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.or.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_xor(&mut self, index: usize, value: U32) {
        //this operation seems poorly defined for floats
        assert!(index < self.ptr.len());
        let index = self.ptr.as_ptr() as u64 + (index * size_of::<U32>()) as u64;
        unsafe {
            asm!(
                "red.global.xor.b32 [{idx}], {v};",
                idx = in(reg64) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_mul(&mut self, _index: usize, _value: U32) {
        todo!()
    }
}

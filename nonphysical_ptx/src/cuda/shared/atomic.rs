use crate::cuda::atomic::{Atomic, Reduce};
use crate::cuda::shared::CuShared;
use crate::shared::primitive::F32;
use core::arch::asm;
use nonphysical_core::shared::float::Float;
use crate::cuda::shared::Shared;
use crate::cuda::shared::U32;
use nonphysical_core::shared::unsigned::Unsigned;


impl<const N: usize> Atomic<F32> for CuShared<F32, N> {
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn atomic_inc(&mut self, index: usize, value: F32) -> F32 {
        let mut old  = self.load(index);
        let mut assumed;
        if old>=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed+F32::IDENTITY);
            if old==assumed || old>=value{
                break
            }
        }
        return old;
    }
    #[inline(always)]
    fn atomic_dec(&mut self, index: usize, value: F32) -> F32 {
        let mut old  = self.load(index);
        let mut assumed;
        if old<=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed-F32::IDENTITY);
            if old==assumed || old<=value{
                break
            }
        }
        return old;
    }
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn atomic_mul(&mut self, _index: usize, _value: F32) -> F32 {
        todo!()
    }
}

impl<const N: usize> Reduce<F32> for CuShared<F32, N> {
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn reduce_inc(&mut self, index: usize, value: F32) {
        let mut old  = self.load(index);
        let mut assumed;
        if old>=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed+F32::IDENTITY);
            if old==assumed || old>=value{
                break
            }
        }
    }
    #[inline(always)]
    fn reduce_dec(&mut self, index: usize, value: F32) {
        let mut old  = self.load(index);
        let mut assumed;
        if old<=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed-F32::IDENTITY);
            if old==assumed|| old<=value{
                break
            }
        }
    }
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn reduce_mul(&mut self, _index: usize, _value: F32) {
        todo!()
    }
}


impl<const N: usize> Atomic<U32> for CuShared<U32, N> {
    #[inline(always)]
    fn atomic_add(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "atom.shared.add.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
    #[inline(always)]
    fn atomic_exch(&mut self, index: usize, value: U32) -> U32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "atom.shared.exch.u32 {o},[{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
                o = out(reg32) out.0,
            );
        }
        out
    }
    #[inline(always)]
    fn atomic_max(&mut self, index: usize, value: U32) -> U32 {
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
    #[inline(always)]
    fn atomic_min(&mut self, index: usize, value: U32) -> U32 {
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
    #[inline(always)]
    fn atomic_inc(&mut self, index: usize, value: U32) -> U32 {
        let mut old  = self.load(index);
        let mut assumed;
        if old>=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed+U32::IDENTITY);
            if old==assumed || old>=value{
                break
            }
        }
        return old;
    }
    #[inline(always)]
    fn atomic_dec(&mut self, index: usize, value: U32) -> U32 {
        let mut old  = self.load(index);
        let mut assumed;
        if old<=value{
            return old;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed-U32::IDENTITY);
            if old==assumed || old<=value{
                break
            }
        }
        return old;
    }
    #[inline(always)]
    fn atomic_cas(&mut self, index: usize, compare: U32, value: U32) -> U32 {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_and(&mut self, index: usize, value: U32) -> U32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_or(&mut self, index: usize, value: U32) -> U32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_xor(&mut self, index: usize, value: U32) -> U32 {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        let mut out = U32::ZERO;
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
    #[inline(always)]
    fn atomic_mul(&mut self, _index: usize, _value: U32) -> U32 {
        todo!()
    }
}

impl<const N: usize> Reduce<U32> for CuShared<U32, N> {
    #[inline(always)]
    fn reduce_add(&mut self, index: usize, value: U32) {
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        unsafe {
            asm!(
                "red.shared.add.u32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_min(&mut self, index: usize, value: U32) {
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
    #[inline(always)]
    fn reduce_max(&mut self, index: usize, value: U32) {
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
    #[inline(always)]
    fn reduce_inc(&mut self, index: usize, value: U32) {
        let mut old  = self.load(index);
        let mut assumed;
        if old>=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed+U32::IDENTITY);
            if old==assumed || old>=value{
                break
            }
        }
    }
    #[inline(always)]
    fn reduce_dec(&mut self, index: usize, value: U32) {
        let mut old  = self.load(index);
        let mut assumed;
        if old<=value{
            return;
        }
        loop{
            assumed = old;
            old = self.atomic_cas(index, assumed, assumed-U32::IDENTITY);
            if old==assumed|| old<=value{
                break
            }
        }
    }
    #[inline(always)]
    fn reduce_and(&mut self, index: usize, value: U32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        unsafe {
            asm!(
                "red.shared.and.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_or(&mut self, index: usize, value: U32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        unsafe {
            asm!(
                "red.shared.or.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_xor(&mut self, index: usize, value: U32) {
        //this operation seems poorly defined for floats
        assert!(index < N);
        let index = self.ptr + (index * size_of::<U32>()) as u32;
        unsafe {
            asm!(
                "red.shared.xor.b32 [{idx}], {v};",
                idx = in(reg32) index,
                v = in(reg32) value.0,
            );
        }
    }
    #[inline(always)]
    fn reduce_mul(&mut self, _index: usize, _value: U32) {
        todo!()
    }
}

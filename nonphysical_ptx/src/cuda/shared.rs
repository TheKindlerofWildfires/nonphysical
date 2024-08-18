use crate::shared::primitive::F32;
use core::{arch::asm, marker::PhantomData};

pub mod atomic;

#[derive(Clone, Copy)]
pub struct CuShared<T, const N: usize> {
    pub ptr: u32,
    phatom_data: PhantomData<T>,
}

pub trait Shared<T> {
    fn new() -> Self;
    fn load(&mut self, index: usize) -> T;
    fn store(&mut self, index: usize, data: T);
}

impl<const N: usize> Shared<F32> for CuShared<F32, N> {
    fn new() -> Self {
        let mut ptr;
        const ALIGN: usize = size_of::<F32>();
        unsafe {
            asm!(
                ".shared .align {a} .b8 nonphysical[{b}];",
                "    mov.u32 {pt}, nonphysical;",
                a = const ALIGN,
                b = const ALIGN*N,
                pt = out(reg32) ptr,
            );
        }
        Self {
            ptr,
            phatom_data: PhantomData,
        }
    }

    fn load(&mut self, index: usize) -> F32 {
        assert!(index < N);
        let offset = index * 4;
        let index = self.ptr + offset as u32;
        let mut out = F32(0.0);
        unsafe {
            asm!(
                "ld.shared.f32 {o}, [{idx}];",
                idx = in(reg32) index,
                o = out(reg32) out.0,
            );
        }
        out
    }

    //this function might as well be called 'shoot yourself in the foot v2'
    fn store(&mut self, index: usize, data: F32) {
        assert!(index < N);
        let offset = index * 4;
        let index = self.ptr + offset as u32;
        unsafe {
            asm!(
                "st.shared.f32 [{idx}], {d};",
                idx = in(reg32) index,
                d = in(reg32) data.0,
            );
        }
    }
}

impl<const N: usize> Shared<u32> for CuShared<u32, N> {
    fn new() -> Self {
        let mut ptr;
        const ALIGN: usize = size_of::<u32>();
        unsafe {
            asm!(
                ".shared .align {a} .b8 nonphysical[{b}];",
                "    mov.u32 {pt}, nonphysical;",
                a = const ALIGN,
                b = const ALIGN*N,
                pt = out(reg32) ptr,
            );
        }
        Self {
            ptr,
            phatom_data: PhantomData,
        }
    }

    fn load(&mut self, index: usize) -> u32 {
        assert!(index < N);
        let offset = index * 4;
        let index = self.ptr + offset as u32;
        let mut out: u32;
        unsafe {
            asm!(
                "ld.shared.u32 {o}, [{idx}];",
                idx = in(reg32) index,
                o = out(reg32) out,
            );
        }
        out
    }

    fn store(&mut self, index: usize, data: u32) {
        assert!(index < N);
        let offset = index * 4;
        let index = self.ptr + offset as u32;
        unsafe {
            asm!(
                "st.shared.u32 [{idx}], {d};",
                idx = in(reg32) index,
                d = in(reg32) data,
            );
        }
    }
}

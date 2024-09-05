use crate::shared::primitive::F32;
use core::{arch::asm, marker::PhantomData};
pub mod atomic;
use crate::shared::unsigned::U32;

#[derive(Clone, Copy)]
pub struct CuShared<T, const N: usize> {
    pub ptr: u32,
    pub phatom_data: PhantomData<T>,
}

pub trait Shared<T> {
    fn new() -> Self;
    fn load(&self, index: usize) -> T;
    fn store(&mut self, index: usize, data: T);
    fn len(&self)->usize;
}
#[macro_export]
macro_rules! named_share {
    ($type:ty, $size:expr, $literal:expr) => {
        {
            let mut ptr;
            const ALIGN: usize = size_of::<$type>();
        
            unsafe {
                asm!(
                    concat!(".shared .align {a} .b8 ", $literal, "[{b}];"),
                    concat!("    mov.u32 {pt}, ", $literal,";"),
                    a = const ALIGN,
                    b = const ALIGN*$size,
                    pt = out(reg32) ptr,
                );
            }
            CuShared::<$type,$size> {
                ptr,
                phatom_data: PhantomData,
            }
        }
    };
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

    fn load(&self, index: usize) -> F32 {
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
    
    fn len(&self)->usize {
        N
    }

    
    
}

impl<const N: usize> Shared<U32> for CuShared<U32, N> {
    fn new() -> Self {
        let mut ptr;
        const ALIGN: usize = size_of::<U32>();
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

    fn load(&self, index: usize) -> U32 {
        assert!(index < N);
        let offset = index * 4;
        let index = self.ptr + offset as u32;
        let mut out = U32(0);
        unsafe {
            asm!(
                "ld.shared.u32 {o}, [{idx}];",
                idx = in(reg32) index,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn store(&mut self, index: usize, data: U32) {
        assert!(index < N);
        let offset = index * 4;
        let index = self.ptr + offset as u32;
        unsafe {
            asm!(
                "st.shared.u32 [{idx}], {d};",
                idx = in(reg32) index,
                d = in(reg32) data.0,
            );
        }
    }

    fn len(&self)->usize {
        N
    }
}

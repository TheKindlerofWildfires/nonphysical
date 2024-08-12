use core::arch::asm;

pub trait Intrinsic {
    fn clz(&self) -> Self;
}

impl Intrinsic for u32 {
    fn clz(&self) -> Self {
        let mut out = 0;
        unsafe {
            asm!(
                "clz.b32 {o}, {i};",
                i = in(reg32) *self,
                o = out(reg32) out,
            );
        }
        out
    }
}

impl Intrinsic for u64 {
    fn clz(&self) -> Self {
        let mut out = 0;
        unsafe {
            asm!(
                "clz.b64 {o}, {i};",
                i = in(reg64) *self,
                o = out(reg64) out,
            );
        }
        out
    }
}

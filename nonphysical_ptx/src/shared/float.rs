use core::{arch::asm, intrinsics};

use alloc::boxed::Box;
use nonphysical_core::shared::{float::Float,primitive::Primitive};
use super::primitive::F32;

impl Float for F32 {
    const ZERO: Self = F32(0.0);
    const IDENTITY: Self = F32(1.0);
    type Primitive = F32;
    #[inline(always)]
    fn l1_norm(self) -> Self::Primitive {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "abs.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret    
    }

    #[inline(always)]
    fn l2_norm(self) -> Self::Primitive {
        //Keeps it from jumping to float registers
        self*self
    }

    #[inline(always)]
    fn fma(self, mul: Self, add: Self) -> Self {
        //F32(unsafe {core::intrinsics::fmaf32(self.0, mul.0, add.0)})
        
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "fma.rn.ftz.f32 {r}, {i}, {m}, {a};",
                i = in(reg32) self.0,
                m = in(reg32) mul.0,
                a = in(reg32) add.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn powf(self, other: Self) -> Self {
        F32(unsafe { intrinsics::powf32(self.0, other.0) })
    }

    #[inline(always)]
    fn powi(self, other: i32) -> Self {
        F32(unsafe { intrinsics::powif32(self.0, other) })
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "sqrt.approx.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret    
    }

    #[inline(always)]
    fn ln(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "lg2.approx.ftz.f32 {r}, {i};",
                "    mul.ftz.f32 {r}, {r}, 0f3F317218;",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn log2(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "lg2.approx.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "mul.ftz.f32 {r}, {i}, 0f3FB8AA3B;",
                "    ex2.approx.ftz.f32 {r}, {r};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "ex2.approx.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "rcp.approx.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret    
    }

    #[inline(always)]
    fn sin(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "sin.approx.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret    
    }

    #[inline(always)]
    fn cos(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "cos.approx.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn tan(self) -> Self {
        let mut ret: F32 = F32(0.0);
        let mut tmp: F32 = F32(0.0);
        unsafe {
            asm!(
                "sin.approx.ftz.f32 {r}, {i};",
                "    cos.approx.ftz.f32 {t}, {i};",
                "    div.approx.ftz.f32 {r}, {r}, {t};",
                i = in(reg32) self.0,
                t = out(reg32) tmp.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn asin(self) -> Self {
        todo!();
        //lib c port
        let mut y = Self::ZERO;
        loop {
            let (ys, yc) = y.sin_cos();
            if y > Self::FRAC_PI_2 || y < -Self::FRAC_PI_2 {
                y %= Self::PI;
            }
            if ys + Self::EPSILON >= self && ys - Self::EPSILON <= self {
                break;
            }
            y -= (ys - self) / yc
        }
        y
    }

    #[inline(always)]
    fn acos(self) -> Self {
        Self::FRAC_PI_2 - self.asin()
    }

    #[inline(always)]
    fn atan(self) -> Self {
        (self / (self * self + F32::ONE).sqrt()).asin()
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        let tmp = self.exp();
        (tmp - tmp.recip()) / F32::usize(2)
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        let tmp = self.exp();
        (tmp + tmp.recip()) / F32::usize(2)
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "tanh.approx.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn to_be_bytes(self) -> Box<[u8]> {
        Box::new((self.0).to_be_bytes())
    }

    #[inline(always)]
    fn to_le_bytes(self) -> Box<[u8]> {
        Box::new((self.0).to_le_bytes())
    }
}
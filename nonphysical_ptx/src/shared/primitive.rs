use core::{arch::asm, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign}};

use nonphysical_core::shared::{float::Float, primitive::Primitive};

#[derive(PartialOrd, PartialEq, Debug, Copy, Clone)]
pub struct F32(pub f32);

impl Primitive for F32 {
    const PI: Self = F32(core::f32::consts::PI);
    const FRAC_PI_2: Self = F32(core::f32::consts::FRAC_PI_2);
    const ONE: Self = F32(1.0);
    const NEGATIVE_ONE: Self = F32(-1.0);
    const MAX: Self = F32(f32::MAX);
    const MIN: Self = F32(f32::MIN);
    const SMALL: Self = F32(f32::MIN_POSITIVE);
    const EPSILON: Self = F32(f32::EPSILON);
    const GAMMA: Self = F32(0.577_215_7);

    #[inline(always)]
    fn u8(u: u8) -> Self {
        F32(u as f32)
    }

    #[inline(always)]
    fn u16(u: u16) -> Self {
        F32(u as f32)
    }

    #[inline(always)]
    fn u32(u: u32) -> Self {
        F32(u as f32)
    }

    #[inline(always)]
    fn u64(u: u64) -> Self {
        F32(u as f32)
    }

    #[inline(always)]
    fn usize(u: usize) -> Self {
        F32(u as f32)
    }

    #[inline(always)]
    fn as_u8(self) -> u8 {
        self.0 as u8
    }

    #[inline(always)]
    fn as_u16(self) -> u16 {
        self.0 as u16
    }

    #[inline(always)]
    fn as_u32(self) -> u32 {
        self.0 as u32
    }

    #[inline(always)]
    fn as_u64(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    fn i8(i: i8) -> Self {
        F32(i as f32)
    }

    #[inline(always)]
    fn i16(i: i16) -> Self {
        F32(i as f32)
    }

    #[inline(always)]
    fn i32(i: i32) -> Self {
        F32(i as f32)
    }

    #[inline(always)]
    fn i64(i: i64) -> Self {
        F32(i as f32)
    }

    #[inline(always)]
    fn isize(i: isize) -> Self {
        F32(i as f32)
    }

    #[inline(always)]
    fn as_i8(self) -> i8 {
        self.0 as i8
    }

    #[inline(always)]
    fn as_i16(self) -> i16 {
        self.0 as i16
    }

    #[inline(always)]
    fn as_i32(self) -> i32 {
        self.0 as i32
    }

    #[inline(always)]
    fn as_i64(self) -> i64 {
        self.0 as i64
    }

    #[inline(always)]
    fn as_isize(self) -> isize {
        self.0 as isize
    }

    #[inline(always)]
    fn float(f: f32) -> Self {
        F32(f)
    }

    #[inline(always)]
    fn double(f: f64) -> Self {
        F32(f as f32)
    }

    #[inline(always)]
    fn as_float(self) -> f32 {
        self.0
    }

    #[inline(always)]
    fn as_double(self) -> f64 {
        self.0 as f64
    }

    #[inline(always)]
    fn floor(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "cvt.rmi.f32.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret   
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "cvt.rpi.f32.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret   
    }


    //intrinsic worked here, but was questionable
    #[inline(always)]
    fn round(self) -> Self {
        todo!()
    }
    
    #[inline(always)]
    fn sign(self) -> Self {
        Self::ONE.copy_sign(self)
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        if self == Self::ZERO{
            if other >= Self::ZERO{
                Self::ZERO
            }else{
                Self::PI
            }
        }else if self > Self::ZERO{
            if other==Self::ZERO{
                Self::FRAC_PI_2
            }else if other>Self::ZERO{
                (self/other).atan()
            }else{
                Self::PI-(self/other).atan()
            }
        }else {
            if other==Self::ZERO{
                Self::PI+Self::FRAC_PI_2
            }else if other>Self::ZERO{
                Self::usize(2)*Self::PI-(self/other).atan()
            }else{
                Self::PI+(self/other).atan()
            }
        }  
    }

    #[inline(always)]
    fn copy_sign(self, other: Self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "copysign.f32 {r}, {o}, {i};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) ret.0,
            );
        }
        ret  
    }
}

impl AddAssign for F32{
    fn add_assign(&mut self, other: Self) {
        unsafe {
            asm!(
                "add.rn.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) self.0,
            );
        }
    }
}
impl SubAssign for F32{
    fn sub_assign(&mut self, other: Self) {
        unsafe {
            asm!(
                "sub.rn.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) self.0,
            );
        }
    }
}
impl MulAssign for F32{
    fn mul_assign(&mut self, other: Self) {
        unsafe {
            asm!(
                "mul.rn.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) self.0,
            );
        }
    }
}
impl DivAssign for F32{
    fn div_assign(&mut self, other: Self) {
        unsafe {
            asm!(
                "div.approx.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) self.0,
            );
        }
    }
}

impl RemAssign for F32{
    fn rem_assign(&mut self, other: Self) {
        self.0%=other.0
    }
}

impl Add for F32{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "add.rn.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) ret.0,
            );
        }
        ret  
    }
}
impl Sub for F32{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "sub.rn.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) ret.0,
            );
        }
        ret  
    }
}
impl Mul for F32{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "mul.rn.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) ret.0,
            );
        }
        ret  
    }
}
impl Div for F32{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "div.approx.ftz.f32 {r}, {i}, {o};",
                i = in(reg32) self.0,
                o = in(reg32) other.0,
                r = out(reg32) ret.0,
            );
        }
        ret  
    }
}
impl Rem for F32{
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        F32(self.0%other.0)
    }
}
impl Neg for F32{
    type Output = Self;

    fn neg(self) -> Self {
        let mut ret: F32 = F32(0.0);
        unsafe {
            asm!(
                "neg.ftz.f32 {r}, {i};",
                i = in(reg32) self.0,
                r = out(reg32) ret.0,
            );
        }
        ret  
    }
}
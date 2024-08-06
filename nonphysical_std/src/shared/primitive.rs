use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use std::fmt::Debug;

use nonphysical_core::shared::primitive::Primitive;

#[derive(PartialOrd, PartialEq, Copy, Clone)]
pub struct F32(pub f32);

#[derive(PartialOrd, PartialEq, Copy, Clone)]

pub struct F64(pub f64);

impl Debug for F32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0).unwrap();
        Ok(())
    }
}

impl Debug for F64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0).unwrap();
        Ok(())
    }
}

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
        F32(self.0.floor())
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        F32(self.0.ceil())
    }

    #[inline(always)]
    fn round(self) -> Self {
        F32(self.0.round())
    }

    #[inline(always)]
    fn sign(self) -> Self {
        F32(self.0.signum())
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        let out = self.0.sin_cos();
        (F32(out.0), F32(out.1))
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        F32(self.0.atan2(other.0))
    }

    #[inline(always)]
    fn copy_sign(self, other: Self) -> Self {
        F32(self.0.copysign(other.0))
    }
}

impl AddAssign for F32{
    fn add_assign(&mut self, other: Self) {
        self.0+=other.0
    }
}
impl SubAssign for F32{
    fn sub_assign(&mut self, other: Self) {
        self.0-=other.0
    }
}
impl MulAssign for F32{
    fn mul_assign(&mut self, other: Self) {
        self.0*=other.0
    }
}
impl DivAssign for F32{
    fn div_assign(&mut self, other: Self) {
        self.0/=other.0
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
        F32(self.0+other.0)
    }
}
impl Sub for F32{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        F32(self.0-other.0)
    }
}
impl Mul for F32{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        F32(self.0*other.0)
    }
}
impl Div for F32{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        F32(self.0/other.0)
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
        F32(-self.0)
    }
}
impl Primitive for F64 {
    const PI: Self = F64(core::f64::consts::PI);
    const FRAC_PI_2: Self = F64(core::f64::consts::FRAC_PI_2);
    const ONE: Self = F64(1.0);
    const NEGATIVE_ONE: Self = F64(-1.0);
    const MAX: Self = F64(f64::MAX);
    const MIN: Self = F64(f64::MIN);
    const SMALL: Self = F64(f64::MIN_POSITIVE);
    const EPSILON: Self = F64(f64::EPSILON);
    const GAMMA: Self = F64(0.577_215_664_901_532_9);

    #[inline(always)]
    fn u8(u: u8) -> Self {
        F64(u as f64)
    }

    #[inline(always)]
    fn u16(u: u16) -> Self {
        F64(u as f64)
    }

    #[inline(always)]
    fn u32(u: u32) -> Self {
        F64(u as f64)
    }

    #[inline(always)]
    fn u64(u: u64) -> Self {
        F64(u as f64)
    }

    #[inline(always)]
    fn usize(u: usize) -> Self {
        F64(u as f64)
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
        F64(i as f64)
    }

    #[inline(always)]
    fn i16(i: i16) -> Self {
        F64(i as f64)
    }

    #[inline(always)]
    fn i32(i: i32) -> Self {
        F64(i as f64)
    }

    #[inline(always)]
    fn i64(i: i64) -> Self {
        F64(i as f64)
    }

    #[inline(always)]
    fn isize(i: isize) -> Self {
        F64(i as f64)
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
        F64(f as f64)
    }

    #[inline(always)]
    fn double(f: f64) -> Self {
        F64(f)
    }

    #[inline(always)]
    fn as_float(self) -> f32 {
        self.0 as f32
    }

    #[inline(always)]
    fn as_double(self) -> f64 {
        self.0 
    }

    #[inline(always)]
    fn floor(self) -> Self {
        F64(self.0.floor())
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        F64(self.0.ceil())
    }

    #[inline(always)]
    fn round(self) -> Self {
        F64(self.0.round())
    }

    #[inline(always)]
    fn sign(self) -> Self {
        F64(self.0.signum())
    }

    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        let out = self.0.sin_cos();
        (F64(out.0), F64(out.1))
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        F64(self.0.atan2(other.0))
    }

    #[inline(always)]
    fn copy_sign(self, other: Self) -> Self {
        F64(self.0.copysign(other.0))
    }
}

impl AddAssign for F64{
    fn add_assign(&mut self, other: Self) {
        self.0+=other.0
    }
}
impl SubAssign for F64{
    fn sub_assign(&mut self, other: Self) {
        self.0-=other.0
    }
}
impl MulAssign for F64{
    fn mul_assign(&mut self, other: Self) {
        self.0*=other.0
    }
}
impl DivAssign for F64{
    fn div_assign(&mut self, other: Self) {
        self.0/=other.0
    }
}
impl RemAssign for F64{
    fn rem_assign(&mut self, other: Self) {
        self.0%=other.0
    }
}
impl Add for F64{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        F64(self.0+other.0)
    }
}
impl Sub for F64{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        F64(self.0-other.0)
    }
}
impl Mul for F64{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        F64(self.0*other.0)
    }
}
impl Div for F64{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        F64(self.0/other.0)
    }
}

impl Neg for F64{
    type Output = Self;

    fn neg(self) -> Self {
        F64(-self.0)
    }
}

impl Rem for F64{
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        F64(self.0%other.0)
    }
}
use super::float::Float;

/*
    This Trait defines functions which are specific to real numbers
*/
pub trait Real: Float + PartialOrd {
    const PI: Self;
    const ONE: Self;
    const NEGATIVE_ONE: Self;
    const MAX: Self;
    const MIN: Self;
    const SMALL: Self;
    const EPSILON: Self;
    const GAMMA: Self;
    //real specific functions

    fn floor(&self) -> Self;
    fn ceil(&self) -> Self;
    fn round(&self) -> Self;
    fn sign(&self) -> Self;
    fn exp_m1(&self) -> Self;
    fn log_p1(&self) -> Self;
    fn sin_cos(&self) -> (Self, Self);
    fn atan2(&self, other: Self) -> Self;
    fn copy_sign(&self, other: Self) -> Self;
    fn div_euclid(&self, other: Self) -> Self;
    fn rem_euclid(&self, other: Self) -> Self;

    //conversion functions
    fn u8(u: u8) -> Self;
    fn u16(u: u16) -> Self;
    fn u32(u: u32) -> Self;
    fn u64(u: u64) -> Self;
    fn usize(u: usize) -> Self;

    fn as_u8(&self) -> u8;
    fn as_u16(&self) -> u16;
    fn as_u32(&self) -> u32;
    fn as_u64(&self) -> u64;
    fn as_usize(&self) -> usize;

    fn i8(i: i8) -> Self;
    fn i16(i: i16) -> Self;
    fn i32(i: i32) -> Self;
    fn i64(i: i64) -> Self;
    fn isize(i: isize) -> Self;

    fn as_i8(&self) -> i8;
    fn as_i16(&self) -> i16;
    fn as_i32(&self) -> i32;
    fn as_i64(&self) -> i64;
    fn as_isize(&self) -> isize;

    fn float(f: f32) -> Self;
    fn double(f: f64) -> Self;

    fn as_float(&self) -> f32;
    fn as_double(&self) -> f64;

    fn greater(&self, other: Self) -> Self {
        if *self > other {
            *self
        } else {
            other
        }
    }

    fn lesser(&self, other: Self) -> Self {
        if *self < other {
            *self
        } else {
            other
        }
    }
}

impl Real for f32 {
    const PI: Self = core::f32::consts::PI;
    const ONE: Self = 1.0;
    const NEGATIVE_ONE: Self = -1.0;
    const MAX: Self = f32::MAX;
    const MIN: Self = f32::MIN;
    const SMALL: Self = f32::MIN_POSITIVE;
    const EPSILON: Self = f32::EPSILON;
    const GAMMA: Self = 0.577_215_7;


    #[inline(always)]
    fn floor(&self) -> Self {
        (*self).floor()
    }

    #[inline(always)]
    fn ceil(&self) -> Self {
        (*self).ceil()
    }

    #[inline(always)]
    fn round(&self) -> Self {
        (*self).round()
    }

    #[inline(always)]
    fn sign(&self) -> Self {
        (*self).signum()
    }

    #[inline(always)]
    fn exp_m1(&self) -> Self {
        (*self).exp_m1()
    }

    #[inline(always)]
    fn log_p1(&self) -> Self {
        (*self).ln_1p()
    }

    #[inline(always)]
    fn sin_cos(&self) -> (Self, Self) {
        (*self).sin_cos()
    }

    #[inline(always)]
    fn atan2(&self, other: Self) -> Self {
        (*self).atan2(other)
    }

    #[inline(always)]
    fn copy_sign(&self, other: Self) -> Self {
        (*self).copysign(other)
    }

    #[inline(always)]
    fn div_euclid(&self, other: Self) -> Self {
        (*self).div_euclid(other)
    }

    #[inline(always)]
    fn rem_euclid(&self, other: Self) -> Self {
        (*self).rem_euclid(other)
    }

    #[inline(always)]
    fn u8(u: u8) -> Self {
        u as f32
    }

    #[inline(always)]
    fn u16(u: u16) -> Self {
        u as f32
    }

    #[inline(always)]
    fn u32(u: u32) -> Self {
        u as f32
    }

    #[inline(always)]
    fn u64(u: u64) -> Self {
        u as f32
    }

    #[inline(always)]
    fn usize(u: usize) -> Self {
        u as f32
    }

    #[inline(always)]
    fn as_u8(&self) -> u8 {
        (*self) as u8
    }

    #[inline(always)]
    fn as_u16(&self) -> u16 {
        (*self) as u16
    }

    #[inline(always)]
    fn as_u32(&self) -> u32 {
        (*self) as u32
    }

    #[inline(always)]
    fn as_u64(&self) -> u64 {
        (*self) as u64
    }

    #[inline(always)]
    fn as_usize(&self) -> usize {
        (*self) as usize
    }

    #[inline(always)]
    fn i8(i: i8) -> Self {
        i as f32
    }

    #[inline(always)]
    fn i16(i: i16) -> Self {
        i as f32
    }

    #[inline(always)]
    fn i32(i: i32) -> Self {
        i as f32
    }

    #[inline(always)]
    fn i64(i: i64) -> Self {
        i as f32
    }

    #[inline(always)]
    fn isize(i: isize) -> Self {
        i as f32
    }

    #[inline(always)]
    fn as_i8(&self) -> i8 {
        (*self) as i8
    }

    #[inline(always)]
    fn as_i16(&self) -> i16 {
        (*self) as i16
    }

    #[inline(always)]
    fn as_i32(&self) -> i32 {
        (*self) as i32
    }

    #[inline(always)]
    fn as_i64(&self) -> i64 {
        (*self) as i64
    }

    #[inline(always)]
    fn as_isize(&self) -> isize {
        (*self) as isize
    }

    #[inline(always)]
    fn float(f: f32) -> Self {
        f
    }

    #[inline(always)]
    fn double(f: f64) -> Self {
        f as f32
    }

    #[inline(always)]
    fn as_float(&self) -> f32 {
        *self
    }

    #[inline(always)]
    fn as_double(&self) -> f64 {
        (*self) as f64
    }
}

impl Real for f64 {
    const PI: Self = core::f64::consts::PI;
    const ONE: Self = 1.0;
    const NEGATIVE_ONE: Self = -1.0;
    const MAX: Self = f64::MAX;
    const MIN: Self = f64::MIN;
    const SMALL: Self = f64::MIN_POSITIVE;
    const EPSILON: Self = f64::EPSILON;
    const GAMMA: Self = 0.577_215_664_901_532_9;

    #[inline(always)]
    fn floor(&self) -> Self {
        (*self).floor()
    }

    #[inline(always)]
    fn ceil(&self) -> Self {
        (*self).ceil()
    }

    #[inline(always)]
    fn round(&self) -> Self {
        (*self).round()
    }

    #[inline(always)]
    fn sign(&self) -> Self {
        (*self).signum()
    }

    #[inline(always)]
    fn exp_m1(&self) -> Self {
        (*self).exp_m1()
    }

    #[inline(always)]
    fn log_p1(&self) -> Self {
        (*self).ln_1p()
    }

    #[inline(always)]
    fn sin_cos(&self) -> (Self, Self) {
        (*self).sin_cos()
    }

    #[inline(always)]
    fn atan2(&self, other: Self) -> Self {
        (*self).atan2(other)
    }

    #[inline(always)]
    fn copy_sign(&self, other: Self) -> Self {
        (*self).copysign(other)
    }

    #[inline(always)]
    fn div_euclid(&self, other: Self) -> Self {
        (*self).div_euclid(other)
    }

    #[inline(always)]
    fn rem_euclid(&self, other: Self) -> Self {
        (*self).rem_euclid(other)
    }

    #[inline(always)]
    fn u8(u: u8) -> Self {
        u as f64
    }

    #[inline(always)]
    fn u16(u: u16) -> Self {
        u as f64
    }

    #[inline(always)]
    fn u32(u: u32) -> Self {
        u as f64
    }

    #[inline(always)]
    fn u64(u: u64) -> Self {
        u as f64
    }

    #[inline(always)]
    fn usize(u: usize) -> Self {
        u as f64
    }

    #[inline(always)]
    fn as_u8(&self) -> u8 {
        (*self) as u8
    }

    #[inline(always)]
    fn as_u16(&self) -> u16 {
        (*self) as u16
    }

    #[inline(always)]
    fn as_u32(&self) -> u32 {
        (*self) as u32
    }

    #[inline(always)]
    fn as_u64(&self) -> u64 {
        (*self) as u64
    }

    #[inline(always)]
    fn as_usize(&self) -> usize {
        (*self) as usize
    }
    #[inline(always)]
    fn i8(i: i8) -> Self {
        i as f64
    }

    #[inline(always)]
    fn i16(i: i16) -> Self {
        i as f64
    }

    #[inline(always)]
    fn i32(i: i32) -> Self {
        i as f64
    }

    #[inline(always)]
    fn i64(i: i64) -> Self {
        i as f64
    }

    #[inline(always)]
    fn isize(i: isize) -> Self {
        i as f64
    }

    #[inline(always)]
    fn as_i8(&self) -> i8 {
        (*self) as i8
    }

    #[inline(always)]
    fn as_i16(&self) -> i16 {
        (*self) as i16
    }

    #[inline(always)]
    fn as_i32(&self) -> i32 {
        (*self) as i32
    }

    #[inline(always)]
    fn as_i64(&self) -> i64 {
        (*self) as i64
    }

    #[inline(always)]
    fn as_isize(&self) -> isize {
        (*self) as isize
    }
    #[inline(always)]
    fn float(f: f32) -> Self {
        f as f64
    }

    #[inline(always)]
    fn double(f: f64) -> Self {
        f
    }

    #[inline(always)]
    fn as_float(&self) -> f32 {
        (*self) as f32
    }

    #[inline(always)]
    fn as_double(&self) -> f64 {
        *self
    }
}

use core::ops::{Rem, RemAssign};

use super::float::Float;
pub trait Primitive: Float + PartialOrd + Rem + RemAssign {
    const PI: Self;
    const FRAC_PI_2: Self;
    const ONE: Self;
    const NEGATIVE_ONE: Self;
    const SMALL: Self;
    const EPSILON: Self;
    const GAMMA: Self;

    //conversion functions
    fn u8(u: u8) -> Self;
    fn u16(u: u16) -> Self;
    fn u32(u: u32) -> Self;
    fn u64(u: u64) -> Self;
    fn usize(u: usize) -> Self;

    fn as_u8(self) -> u8;
    fn as_u16(self) -> u16;
    fn as_u32(self) -> u32;
    fn as_u64(self) -> u64;
    fn as_usize(self) -> usize;

    fn i8(i: i8) -> Self;
    fn i16(i: i16) -> Self;
    fn i32(i: i32) -> Self;
    fn i64(i: i64) -> Self;
    fn isize(i: isize) -> Self;

    fn as_i8(self) -> i8;
    fn as_i16(self) -> i16;
    fn as_i32(self) -> i32;
    fn as_i64(self) -> i64;
    fn as_isize(self) -> isize;

    fn float(f: f32) -> Self;
    fn double(f: f64) -> Self;

    fn as_float(self) -> f32;
    fn as_double(self) -> f64;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn sign(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn atan2(self, other: Self) -> Self;
    fn copy_sign(self, other: Self) -> Self;
}

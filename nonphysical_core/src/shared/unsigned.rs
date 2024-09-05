use core::fmt::Debug;
use core::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign, Not, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};
pub trait Unsigned:
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + Clone
    + Copy
    + Sized
    + Sync
    + Send
    + Debug
    + PartialEq
    +Not
    +Shr
    +Shl
    +ShrAssign
    +ShlAssign
    +BitAnd
    +BitAndAssign
    +BitOr
    +BitOrAssign
    +BitXor
    +BitXorAssign
    +PartialOrd
{
    const ZERO: Self;
    const IDENTITY: Self;
    const MIN: Self;
    const MAX: Self;
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

    fn greater(self,other: Self)->Self;
    fn lesser(self,other: Self)->Self;
}

use nonphysical_core::shared::unsigned::Unsigned;
use std::fmt::Debug;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Not, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
pub struct U8(pub u8);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
pub struct U16(pub u16);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
pub struct U32(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
pub struct U64(pub u64);

impl Unsigned for U8 {
    const ZERO: Self = U8(0);

    const IDENTITY: Self = U8(1);

    const MIN: Self = U8(u8::MIN);

    const MAX: Self = U8(u8::MAX);
    #[inline(always)]
    fn u8(u: u8) -> Self {
        Self(u as u8)
    }
    #[inline(always)]
    fn u16(u: u16) -> Self {
        Self(u as u8)
    }
    #[inline(always)]
    fn u32(u: u32) -> Self {
        Self(u as u8)
    }
    #[inline(always)]
    fn u64(u: u64) -> Self {
        Self(u as u8)
    }
    #[inline(always)]
    fn usize(u: usize) -> Self {
        Self(u as u8)
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
        Self(i as u8)
    }
    #[inline(always)]
    fn i16(i: i16) -> Self {
        Self(i as u8)
    }
    #[inline(always)]
    fn i32(i: i32) -> Self {
        Self(i as u8)
    }
    #[inline(always)]
    fn i64(i: i64) -> Self {
        Self(i as u8)
    }
    #[inline(always)]
    fn isize(i: isize) -> Self {
        Self(i as u8)
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
    fn greater(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
    #[inline(always)]
    fn lesser(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
}
impl Add for U8 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}
impl Sub for U8 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}
impl Mul for U8 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}
impl Div for U8 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self(self.0 / other.0)
    }
}

impl AddAssign for U8 {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0
    }
}
impl SubAssign for U8 {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for U8 {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl DivAssign for U8 {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl Shr for U8 {
    type Output = Self;

    fn shr(self, other: Self) -> Self::Output {
        Self(self.0 >> other.0)
    }
}
impl Shl for U8 {
    type Output = Self;

    fn shl(self, other: Self) -> Self::Output {
        Self(self.0 << other.0)
    }
}

impl ShrAssign for U8 {
    fn shr_assign(&mut self, other: Self) {
        self.0 >>= other.0
    }
}

impl ShlAssign for U8 {
    fn shl_assign(&mut self, other: Self) {
        self.0 <<= other.0
    }
}

impl Not for U8 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl BitAnd for U8 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        Self(self.0 & other.0)
    }
}
impl BitOr for U8 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        Self(self.0 | other.0)
    }
}
impl BitXor for U8 {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        Self(self.0 ^ other.0)
    }
}
impl BitAndAssign for U8 {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0
    }
}
impl BitOrAssign for U8 {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0
    }
}
impl BitXorAssign for U8 {
    fn bitxor_assign(&mut self, other: Self) {
        self.0 ^= other.0
    }
}
impl Debug for U8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0).unwrap();
        Ok(())
    }
}
impl Unsigned for U16 {
    const ZERO: Self = U16(0);

    const IDENTITY: Self = U16(1);

    const MIN: Self = U16(u16::MIN);

    const MAX: Self = U16(u16::MAX);
    #[inline(always)]
    fn u8(u: u8) -> Self {
        Self(u as u16)
    }
    #[inline(always)]
    fn u16(u: u16) -> Self {
        Self(u as u16)
    }
    #[inline(always)]
    fn u32(u: u32) -> Self {
        Self(u as u16)
    }
    #[inline(always)]
    fn u64(u: u64) -> Self {
        Self(u as u16)
    }
    #[inline(always)]
    fn usize(u: usize) -> Self {
        Self(u as u16)
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
        Self(i as u16)
    }
    #[inline(always)]
    fn i16(i: i16) -> Self {
        Self(i as u16)
    }
    #[inline(always)]
    fn i32(i: i32) -> Self {
        Self(i as u16)
    }
    #[inline(always)]
    fn i64(i: i64) -> Self {
        Self(i as u16)
    }
    #[inline(always)]
    fn isize(i: isize) -> Self {
        Self(i as u16)
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
    fn greater(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
    #[inline(always)]
    fn lesser(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
}
impl Add for U16 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}
impl Sub for U16 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}
impl Mul for U16 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}
impl Div for U16 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self(self.0 / other.0)
    }
}

impl AddAssign for U16 {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0
    }
}
impl SubAssign for U16 {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for U16 {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl DivAssign for U16 {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl Shr for U16 {
    type Output = Self;

    fn shr(self, other: Self) -> Self::Output {
        Self(self.0 >> other.0)
    }
}
impl Shl for U16 {
    type Output = Self;

    fn shl(self, other: Self) -> Self::Output {
        Self(self.0 << other.0)
    }
}

impl ShrAssign for U16 {
    fn shr_assign(&mut self, other: Self) {
        self.0 >>= other.0
    }
}

impl ShlAssign for U16 {
    fn shl_assign(&mut self, other: Self) {
        self.0 <<= other.0
    }
}

impl Not for U16 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl BitAnd for U16 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        Self(self.0 & other.0)
    }
}
impl BitOr for U16 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        Self(self.0 | other.0)
    }
}
impl BitXor for U16 {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        Self(self.0 ^ other.0)
    }
}
impl BitAndAssign for U16 {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0
    }
}
impl BitOrAssign for U16 {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0
    }
}
impl BitXorAssign for U16 {
    fn bitxor_assign(&mut self, other: Self) {
        self.0 ^= other.0
    }
}
impl Debug for U16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0).unwrap();
        Ok(())
    }
}
impl Unsigned for U32 {
    const ZERO: Self = U32(0);

    const IDENTITY: Self = U32(1);

    const MIN: Self = U32(u32::MIN);

    const MAX: Self = U32(u32::MAX);

    fn u8(u: u8) -> Self {
        Self(u as u32)
    }

    fn u16(u: u16) -> Self {
        Self(u as u32)
    }

    fn u32(u: u32) -> Self {
        Self(u as u32)
    }

    fn u64(u: u64) -> Self {
        Self(u as u32)
    }

    fn usize(u: usize) -> Self {
        Self(u as u32)
    }

    fn as_u8(self) -> u8 {
        self.0 as u8
    }

    fn as_u16(self) -> u16 {
        self.0 as u16
    }

    fn as_u32(self) -> u32 {
        self.0 as u32
    }

    fn as_u64(self) -> u64 {
        self.0 as u64
    }

    fn as_usize(self) -> usize {
        self.0 as usize
    }

    fn i8(i: i8) -> Self {
        Self(i as u32)
    }

    fn i16(i: i16) -> Self {
        Self(i as u32)
    }

    fn i32(i: i32) -> Self {
        Self(i as u32)
    }

    fn i64(i: i64) -> Self {
        Self(i as u32)
    }

    fn isize(i: isize) -> Self {
        Self(i as u32)
    }

    fn as_i8(self) -> i8 {
        self.0 as i8
    }

    fn as_i16(self) -> i16 {
        self.0 as i16
    }

    fn as_i32(self) -> i32 {
        self.0 as i32
    }

    fn as_i64(self) -> i64 {
        self.0 as i64
    }

    fn as_isize(self) -> isize {
        self.0 as isize
    }

    fn greater(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }

    fn lesser(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
}
impl Add for U32 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}
impl Sub for U32 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}
impl Mul for U32 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}
impl Div for U32 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self(self.0 / other.0)
    }
}

impl AddAssign for U32 {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0
    }
}
impl SubAssign for U32 {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for U32 {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl DivAssign for U32 {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl Shr for U32 {
    type Output = Self;

    fn shr(self, other: Self) -> Self::Output {
        Self(self.0 >> other.0)
    }
}
impl Shl for U32 {
    type Output = Self;

    fn shl(self, other: Self) -> Self::Output {
        Self(self.0 << other.0)
    }
}

impl ShrAssign for U32 {
    fn shr_assign(&mut self, other: Self) {
        self.0 >>= other.0
    }
}

impl ShlAssign for U32 {
    fn shl_assign(&mut self, other: Self) {
        self.0 <<= other.0
    }
}

impl Not for U32 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl BitAnd for U32 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        Self(self.0 & other.0)
    }
}
impl BitOr for U32 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        Self(self.0 | other.0)
    }
}
impl BitXor for U32 {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        Self(self.0 ^ other.0)
    }
}
impl BitAndAssign for U32 {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0
    }
}
impl BitOrAssign for U32 {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0
    }
}
impl BitXorAssign for U32 {
    fn bitxor_assign(&mut self, other: Self) {
        self.0 ^= other.0
    }
}
impl Debug for U32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0).unwrap();
        Ok(())
    }
}
impl Unsigned for U64 {
    const ZERO: Self = U64(0);

    const IDENTITY: Self = U64(1);

    const MIN: Self = U64(u64::MIN);

    const MAX: Self = U64(u64::MAX);

    fn u8(u: u8) -> Self {
        Self(u as u64)
    }

    fn u16(u: u16) -> Self {
        Self(u as u64)
    }

    fn u32(u: u32) -> Self {
        Self(u as u64)
    }

    fn u64(u: u64) -> Self {
        Self(u as u64)
    }

    fn usize(u: usize) -> Self {
        Self(u as u64)
    }

    fn as_u8(self) -> u8 {
        self.0 as u8
    }

    fn as_u16(self) -> u16 {
        self.0 as u16
    }

    fn as_u32(self) -> u32 {
        self.0 as u32
    }

    fn as_u64(self) -> u64 {
        self.0 as u64
    }

    fn as_usize(self) -> usize {
        self.0 as usize
    }

    fn i8(i: i8) -> Self {
        Self(i as u64)
    }

    fn i16(i: i16) -> Self {
        Self(i as u64)
    }

    fn i32(i: i32) -> Self {
        Self(i as u64)
    }

    fn i64(i: i64) -> Self {
        Self(i as u64)
    }

    fn isize(i: isize) -> Self {
        Self(i as u64)
    }

    fn as_i8(self) -> i8 {
        self.0 as i8
    }

    fn as_i16(self) -> i16 {
        self.0 as i16
    }

    fn as_i32(self) -> i32 {
        self.0 as i32
    }

    fn as_i64(self) -> i64 {
        self.0 as i64
    }

    fn as_isize(self) -> isize {
        self.0 as isize
    }
    fn greater(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }

    fn lesser(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
}
impl Add for U64 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}
impl Sub for U64 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}
impl Mul for U64 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}
impl Div for U64 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self(self.0 / other.0)
    }
}

impl AddAssign for U64 {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0
    }
}
impl SubAssign for U64 {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for U64 {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl DivAssign for U64 {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl Shr for U64 {
    type Output = Self;

    fn shr(self, other: Self) -> Self::Output {
        Self(self.0 >> other.0)
    }
}
impl Shl for U64 {
    type Output = Self;

    fn shl(self, other: Self) -> Self::Output {
        Self(self.0 << other.0)
    }
}

impl ShrAssign for U64 {
    fn shr_assign(&mut self, other: Self) {
        self.0 >>= other.0
    }
}

impl ShlAssign for U64 {
    fn shl_assign(&mut self, other: Self) {
        self.0 <<= other.0
    }
}

impl Not for U64 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl BitAnd for U64 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        Self(self.0 & other.0)
    }
}
impl BitOr for U64 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        Self(self.0 | other.0)
    }
}
impl BitXor for U64 {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        Self(self.0 ^ other.0)
    }
}
impl BitAndAssign for U64 {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0
    }
}
impl BitOrAssign for U64 {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0
    }
}
impl BitXorAssign for U64 {
    fn bitxor_assign(&mut self, other: Self) {
        self.0 ^= other.0
    }
}
impl Debug for U64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0).unwrap();
        Ok(())
    }
}

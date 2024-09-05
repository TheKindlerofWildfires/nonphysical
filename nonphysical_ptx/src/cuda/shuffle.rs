use core::arch::asm;

use nonphysical_core::shared::float::Float;
use crate::shared::primitive::F32;
use crate::shared::unsigned::U32;
use nonphysical_core::shared::unsigned::Unsigned;
pub trait Shuffle<T>{
    fn shuffle_up<const MASK: usize>(value:T, index: usize, warp:usize) -> T;
    fn shuffle_down<const MASK: usize>(value:T, index: usize, warp:usize) -> T;
    fn shuffle_bfly<const MASK: usize>(value:T, index: usize, warp:usize) -> T;
    fn shuffle_idx<const MASK: usize>( value:T, index: usize, warp:usize) -> T;
}
pub struct Shuffler{}

impl Shuffle<F32> for Shuffler{
    fn shuffle_up<const MASK: usize>( value:F32, index: usize, warp:usize) -> F32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.up.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn shuffle_down<const MASK: usize>( value:F32, index: usize, warp:usize) -> F32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.down.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn shuffle_bfly<const MASK: usize>(value:F32, index: usize,  warp:usize) -> F32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.bfly.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn shuffle_idx<const MASK: usize>(value:F32, index: usize, warp:usize) -> F32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = F32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.idx.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }
}


impl Shuffle<U32> for Shuffler{
    fn shuffle_up<const MASK: usize>( value:U32, index: usize, warp:usize) -> U32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.up.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn shuffle_down<const MASK: usize>( value:U32, index: usize, warp:usize) -> U32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.down.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn shuffle_bfly<const MASK: usize>(value:U32, index: usize,  warp:usize) -> U32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.bfly.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }

    fn shuffle_idx<const MASK: usize>(value:U32, index: usize, warp:usize) -> U32{
        let index = index as u32;
        let warp = warp as u32;
        let mut out = U32::ZERO;
        unsafe {
            asm!(
                "shfl.sync.idx.b32 {o},{v}, {i}, {w}, {m};",
                i = in(reg32) index,
                w = in(reg32) warp,
                v = in(reg32) value.0,
                m = const MASK,
                o = out(reg32) out.0,
            );
        }
        out
    }
}
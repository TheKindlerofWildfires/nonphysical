use super::HashArguments;
use alloc::format;
use alloc::string::String;
use nonphysical_core::shared::unsigned::Unsigned;
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_cuda::cuda::runtime::Dim3;
use nonphysical_cuda::cuda::runtime::RUNTIME;
use nonphysical_std::shared::unsigned::U8;
use std::vec::Vec;
use std::{print, vec};

pub struct Hasher {
    target: Vec<U8>,
}

impl Hasher {
    pub fn new(target: &[u8]) -> Self {
        let target = target.iter().map(|t| U8(*t)).collect::<Vec<_>>();
        Self { target }
    }
    fn lookup_convert(pattern: &str) -> (u128, fn(u128) -> [u8; 16]) {
        match pattern {
            "all" => (0x100, convert_all),
            "ascii" => (0x80, convert_ascii),
            "password" => (0x5d, convert_password),
            "alphanumeric" => (0x3e, convert_alphanumeric),
            "alpha" => (0x34, convert_alpha),
            "numeric" => (0xa, convert_numeric),
            "lower" => (0x1a, convert_lower),
            "upper" => (0x1a, convert_upper),
            "hex" => (0x10, convert_hex),
            _ => panic!("Bad pattern"),
        }
    }
    pub fn crack(&self, alg: &str, pattern: &str) -> Vec<U8> {
        let (mut result, threads, block_size) = match alg {
            "md4" => (vec![U8::ZERO; 16], 768, 70),
            "md5" => (vec![U8::ZERO; 16], 768, 70),
            "sha1" => (vec![U8::ZERO; 20], 768, 70),
            "sha256" => (vec![U8::ZERO; 32], 768, 70),
            "sha512" => (vec![U8::ZERO; 64], 512, 105),
            _ => {
                panic!("Invalid alg")
            }
        };
        let mut args = Self::hash_alloc(&self.target, &result);

        let (log, f) = Self::lookup_convert(pattern);
        std::dbg!(log, f);

        self.hash_transfer(
            f,
            log,
            threads,
            block_size,
            &mut args,
            &self.target,
            &mut result,
            alg,
            pattern,
        );

        result
    }

    fn launch<Args>(args: &mut Args, threads: usize, block_size: usize, kernel: String) {
        let grid = Dim3 {
            x: block_size,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: threads,
            y: 1,
            z: 1,
        };
        match RUNTIME.get() {
            Some(rt) => {
                rt.launch_name(kernel, args, grid, block);
            }
            None => panic!("Cuda Runtime not initialized"),
        };
    }
    fn hash_alloc<'a>(target: &[U8], hit: &[U8]) -> HashArguments<'a> {
        let target = CuGlobalSlice::alloc(target);
        let base = CuGlobalSlice::alloc(&[U8::ZERO; 14]);
        let hit = CuGlobalSliceRef::alloc(&hit);

        HashArguments { target, base, hit }
    }
    fn hash_transfer<'a, F: Fn(u128) -> [u8; 16]>(
        &self,
        f: F,
        log: u128,
        threads: usize,
        block_size: usize,
        args: &mut HashArguments<'a>,
        target: &[U8],
        hit: &mut [U8],
        hash_type: &str,
        hash_list: &str,
    ) {
        args.target.store(target);

        let mut base = [U8::ZERO; 14];
        base[13] = U8(4);
        let kernel = format!("{}_check_{}", hash_type, hash_list);
        let mut offset = 0u128;
        let mut old_count = 0;
        while hit[0] == U8::ZERO {
            args.base.store(&base);

            Self::launch(args, threads, block_size, kernel.clone());
            args.hit.load(hit);
            offset += 1;
            let bytes = f(offset);
            let count = offset.ilog(log); //(128 - offset.leading_zeros()).div_ceil(8);
            base.iter_mut()
                .zip(bytes.iter().take(count as usize))
                .for_each(|(b, o)| *b = U8(*o));

            base[13] = U8(4 + count as u8);
            if count != old_count {
                std::println!("Increased from {} to {}", old_count + 4, count + 4);
                old_count = count;
            }
        }
    }
}
fn convert_base(mut number: u128, base: u128, add: u128) -> [u8; 16] {
    let mut out = [0; 16];
    let mut ptr = 0;
    while ptr < 15 {
        out[ptr] = (number % base + add) as u8;
        number /= base;
        ptr += 1
    }
    out
}
fn convert_ascii(number: u128) -> [u8; 16] {
    convert_base(number, 0x80, 0)
}
fn convert_all(number: u128) -> [u8; 16] {
    number.to_le_bytes()
}
fn convert_password(number: u128) -> [u8; 16] {
    convert_base(number, 0x5d, 0x21)
}
fn convert_alphanumeric(number: u128) -> [u8; 16] {
    let mut out = convert_base(number, 0x3e, 0x31);
    out.iter_mut().for_each(|o| {
        if *o > 0x39 {
            *o += 7;
        }
        if *o > 0x5a {
            *o += 6;
        }
    });
    out
}
fn convert_alpha(number: u128) -> [u8; 16] {
    let mut out = convert_base(number, 0x34, 0x41);
    out.iter_mut().for_each(|o| {
        if *o > 0x5a {
            *o += 6;
        }
    });
    out
}
fn convert_numeric(number: u128) -> [u8; 16] {
    convert_base(number, 0xa, 0x31)
}
fn convert_upper(number: u128) -> [u8; 16] {
    convert_base(number, 0x1a, 0x41)
}
fn convert_lower(number: u128) -> [u8; 16] {
    convert_base(number, 0x1a, 0x61)
}
fn convert_hex(number: u128) -> [u8; 16] {
    let mut out = convert_base(number, 0x10, 0x11);
    out.iter_mut().for_each(|o| {
        if *o > 0x39 {
            *o += 7;
        }
    });
    out
}

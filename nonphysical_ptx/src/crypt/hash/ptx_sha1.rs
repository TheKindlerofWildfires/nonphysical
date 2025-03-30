use crate::crypt::hash::ptx::*;
use crate::crypt::hash::HashArguments;
use crate::shared::unsigned::U8;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use nonphysical_core::crypt::hash::sha1::SHA1;
use nonphysical_core::shared::unsigned::Unsigned;
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_all<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<78062, _>(convert_all, args);
}

#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_ascii<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<4879, _>(convert_ascii, args);
}
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_password<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<1360, _>(convert_password, args);
}
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_alphanumeric<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<267, _>(convert_alphanumeric, args);
}
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_alpha<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<133, _>(convert_alpha, args);
}
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_numeric<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<1, _>(convert_numeric, args);
}
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_lower<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<9, _>(convert_lower, args);
}
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_upper<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<9, _>(convert_upper, args);
}
#[no_mangle]
pub extern "ptx-kernel" fn sha1_check_hex<'a>(args: &'a mut HashArguments<'a>) {
    sha1_check_internal::<2, _>(convert_hex, args);
}

fn sha1_check_internal<'a, const COUNT: u32, F: Fn(u32) -> [u8; 4]>(
    f: F,
    args: &'a mut HashArguments<'a>,
) {
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as u32,
            _block_dim_x() as u32,
            _block_idx_x() as u32,
        )
    };
    //How many we do per cycle, I've assume 1024*1024*16 = 2^24
    //let carve_out = block_dim*grid_dim*COUNT;
    let num_base = (block_id * block_dim * COUNT + thread_id * COUNT) as u32;
    let mut input = [0u8; 16]; //capped at 16
    let mut target = [0u8; 20];
    let len = args.base[13].as_usize();
    args.target
        .iter()
        .zip(target.iter_mut())
        .for_each(|(i, t)| {
            *t = i.as_u8();
        });

    //set up the input beforehand

    input
        .iter_mut()
        .skip(4)
        .zip(args.base.iter())
        .for_each(|(t, b)| {
            *t = b.as_u8();
        });

    unsafe { _syncthreads() };
    //Check 16 repeats (because that's my easy number, can optimize here)
    (num_base..num_base + COUNT).for_each(|i| {
        //i is now a number 0->80^4 rep as u32
        let part = f(i);
        //overwrite my input, offset into ascii
        input[0] = part[0];
        input[1] = part[1];
        input[2] = part[2];
        input[3] = part[3];
        let mut hasher = SHA1::new();
        let digest = SHA1::digest(&mut hasher, &input[..len]);
        //Compare with target
        let hit = digest
            .iter()
            .zip(target.iter())
            .fold(true, |acc, (d, t)| acc & (d == t));
        if hit {
            args.hit.iter_mut().zip(input).for_each(|(h, i)| *h = U8(i));
        }
    });
}

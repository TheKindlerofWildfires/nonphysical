use nonphysical_core::shared;
use nonphysical_core::shared::unsigned::Unsigned;

use super::{INSERT_SIZE, THREAD_MAX};
use crate::cuda::grid::GridStride;
use crate::cuda::shared::CuShared;
use crate::cuda::shared::Shared;
use crate::graph::merge_sort::MergeSortArgumentsGlobal;
use crate::graph::merge_sort::MergeSortArgumentsLocal;
use crate::named_share;
use crate::shared::primitive::F32;
use crate::shared::unsigned::U32;
use core::arch::asm;
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use core::cmp::min;
use core::marker::PhantomData;
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::primitive::Primitive;

/*
#[no_mangle]
pub extern "ptx-kernel" fn merge_path_stride<'a>(args: &'a mut MergeSortArgumentsGlobal<'a, F32>) {
    let mut a_temp = named_share!(F32, 1024, "a");
    let mut b_temp = named_share!(F32, 1024, "b");
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    let expected_size = args.params[0].0 as usize;
    let d = expected_size * 2;
    let thread_d = thread_id % d;
    let qt = (thread_id - thread_d) / d;
    let gbx = qt + block_id * (block_dim / d);

    //use gbx to find the size/
    //something is off here
    let mut block_a = args.src.iter().skip(gbx).step_by(8).take(4);
    let mut block_b = args.src.iter().skip(gbx+32).step_by(8).take(4);
    let mut block_c = args.dst.iter_mut().skip(gbx).step_by(8).take(8);
    //let block_c = args.dst.chunks_mut(d).nth(gbx).unwrap();

    //fully committed to power of two requirement
    let size_a = expected_size;
    let size_b = expected_size;
    let size_c = d;
    let qtd = qt * d;

    let alt_a = thread_id % size_a;
    let alt_b = thread_id % size_b;
    let alt_d = thread_id % size_c;

    a_temp.store(qtd + thread_d, *block_a.nth(alt_a).unwrap());
    b_temp.store(qtd + thread_d, *block_b.nth(alt_b).unwrap());
    unsafe { _syncthreads() };
    let mut c = block_c.nth(alt_d).unwrap();
    let (mut k, mut p) = if thread_d > size_a {
        let tmp = thread_d - size_a;
        ([tmp, size_a], [size_a, tmp])
    } else {
        ([0, thread_d], [thread_d, 0])
    };

    loop {
        //observations: k[0] only goes up probably and k[1] only goes down
        let offset = ((k[1] as i32 - p[1] as i32).abs() / 2) as usize;
        let q = [k[0] + offset, k[1] - offset];

        if q[1] >= 0
            && q[0] <= size_b
            && (q[1] == size_a
                || q[0] == 0
                || a_temp.load(qtd + q[1]) > b_temp.load(qtd + q[0] - 1))
        {
            if q[0] == size_b || q[1] == 0 || a_temp.load(qtd + q[1] - 1) <= b_temp.load(qtd + q[0])
            {
                if q[1] < size_a
                    && (q[0] == size_b || a_temp.load(qtd + q[1]) <= b_temp.load(qtd + q[0]))
                {
                    *c = a_temp.load(qtd + q[1]);
                } else {
                    *c = b_temp.load(qtd + q[0]);
                }
                break;
            } else {
                k[0] = q[0] + 1;
                k[1] = q[1] - 1;
            }
        } else {
            p[0] = q[0] - 1;
            p[1] = q[1] + 1;
        }
    }
}*/
#[no_mangle]
pub extern "ptx-kernel" fn merge_path<'a>(args: &'a mut MergeSortArgumentsGlobal<'a, F32>) {
    let mut a_temp = named_share!(F32, 1024, "a");
    let mut b_temp = named_share!(F32, 1024, "b");
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };

    let expected_size = args.params[0].0 as usize;
    let d = expected_size * 2;
    let thread_d = thread_id % d;
    let qt = (thread_id - thread_d) / d;
    let gbx = qt + block_id * (block_dim / d);

    //use gbx to find the size/
    let block_a = args.src.chunks(expected_size).nth(gbx * 2).unwrap();
    let block_b = args.src.chunks(expected_size).nth(gbx * 2 + 1).unwrap();
    let dst_block = args.dst.chunks_mut(d).nth(gbx).unwrap();

    let size_a = block_a.len();
    let size_b = block_b.len();
    let qtd = qt * d;

    //badness here
    //block_a.iter().nth(thread_id)
    let alt_a = thread_id % size_a; //
    let alt_b = thread_id % size_b;
    let alt_d = thread_id % dst_block.len();

    a_temp.store(qtd + thread_d, block_a[alt_a]);
    b_temp.store(qtd + thread_d, block_b[alt_b]);
    unsafe { _syncthreads() };

    let (mut k, mut p) = if thread_d > size_a {
        let tmp = thread_d - size_a;
        ([tmp, size_a], [size_a, tmp])
    } else {
        ([0, thread_d], [thread_d, 0])
    };

    loop {
        //observations: k[0] only goes up probably and k[1] only goes down
        let offset = ((k[1] as i32 - p[1] as i32).abs() / 2) as usize;
        let q = [k[0] + offset, k[1] - offset];

        if q[1] >= 0
            && q[0] <= size_b
            && (q[1] == size_a
                || q[0] == 0
                || a_temp.load(qtd + q[1]) > b_temp.load(qtd + q[0] - 1))
        {
            if q[0] == size_b || q[1] == 0 || a_temp.load(qtd + q[1] - 1) <= b_temp.load(qtd + q[0])
            {
                if q[1] < size_a
                    && (q[0] == size_b || a_temp.load(qtd + q[1]) <= b_temp.load(qtd + q[0]))
                {
                    dst_block[alt_d] = a_temp.load(qtd + q[1]);
                } else {
                    dst_block[alt_d] = b_temp.load(qtd + q[0]);
                }
                break;
            } else {
                k[0] = q[0] + 1;
                k[1] = q[1] - 1;
            }
        } else {
            p[0] = q[0] - 1;
            p[1] = q[1] + 1;
        }
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn insert_skip<'a>(args: &'a mut MergeSortArgumentsGlobal<'a, F32>) {
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };
    let effective_id = thread_id + block_id * block_dim;
    let block_src_iter = GridStride::block_stride_ref(&mut args.src, INSERT_SIZE, effective_id);
    let mut block_dst_iter = args
        .dst
        .chunks_mut(INSERT_SIZE)
        .nth(effective_id)
        .unwrap()
        .iter_mut(); //GridStride::block_stride_ref(&mut args.dst, INSERT_SIZE,effective_id);

    let mut thread_memory = [F32::ZERO; INSERT_SIZE];
    block_src_iter
        .zip(thread_memory.iter_mut())
        .for_each(|(bsi, tmi)| {
            *tmi = *bsi;
        });

    insert_sort(&mut thread_memory);
    block_dst_iter
        .zip(thread_memory.iter())
        .for_each(|(bdi, tmi)| {
            *bdi = *tmi;
        });
}

#[no_mangle]
pub extern "ptx-kernel" fn insert_skip_stride<'a>(args: &'a mut MergeSortArgumentsGlobal<'a, F32>) {
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };
    let effective_id = thread_id + block_id * block_dim;
    let block_src_iter = GridStride::block_stride_ref(&mut args.src, INSERT_SIZE, effective_id);
    //let mut block_dst_iter = GridStride::block_stride_ref(&mut args.dst, INSERT_SIZE,effective_id);
    let alt_id = thread_id + (effective_id / THREAD_MAX);

    let mut block_dst_iter = args
        .dst
        .chunks_mut(INSERT_SIZE)
        .nth(alt_id)
        .unwrap()
        .iter_mut();
    let mut thread_memory = [F32::ZERO; INSERT_SIZE];

    block_src_iter
        .zip(thread_memory.iter_mut())
        .for_each(|(bsi, tmi)| {
            *tmi = *bsi;
        });

    insert_sort(&mut thread_memory);
    block_dst_iter
        .zip(thread_memory.iter())
        .for_each(|(bdi, tmi)| {
            *bdi = *tmi;
        });
}

fn insert_sort(arr: &mut [F32]) {
    for i in 0..INSERT_SIZE {
        let mut j = i;
        let key = arr[i];
        loop {
            if j <= 0 {
                break;
            }
            let cmp = arr[j - 1];
            if cmp < key {
                break;
            }
            arr[j] = cmp;
            j -= 1;
        }
        arr[j] = key;
    }
}

//perf is better when it actually sorts
//All of the work is in the branches I think (ALU), can any of this be branchless
//If I was doing multiple at once the stalls would be less bad
#[no_mangle]
pub extern "ptx-kernel" fn matrix_sort<'a>(args: &'a mut MergeSortArgumentsGlobal<'a, F32>) {
    //need to find P/Q
    let (thread_id, block_dim, block_id) = unsafe {
        (
            _thread_idx_x() as usize,
            _block_dim_x() as usize,
            _block_idx_x() as usize,
        )
    };
    //I've essentially said thread_id = p, block_id = q, which doesn't scale right as thread id caps at 1024, but so does merge path
    let m = THREAD_MAX;
    let nm = THREAD_MAX;
    let mut p = thread_id;
    let mut q = block_id;

    let x = args.src[p + q * block_dim];

    let mut lpq = (p + 1) * (q + 1) - 1; //p*q;
    //bounds checking here is probably ugly

    //only really need to check conditions when they change
    /*if p>= 1 && q<nm-1 {
        loop {
            if x < args.src[p + 1 + (q - 1) * block_dim] {
                p -= 1;
                if p>= 1{
                    continue;
                }else{
                    break;
                }
            } else {
                lpq += q;
                q += 1;
                if q<nm-1{
                    continue;
                }else{
                    break;
                }
            }
        }
    }*/
    //unroll the first check
    /* 
    if p >= 1 && q < nm - 1 {
        //only check one condition per loop (takes check from 3 cmp per loop to 2, but it's slower)
        loop{
            if x < args.src[p - 1 + (q + 1) * block_dim] {
                p -= 1;
                if p==0{
                    break;
                }
                
            } else {
                lpq += q;
                q += 1;
                if q==nm-1{
                    break;
                }
            }
        }
    }*/
    /* 
    while p >= 1 && q < nm - 1 {
        if x < args.src[p - 1 + (q + 1) * block_dim] {
            p -= 1;
        } else {
            lpq += p;
            q += 1;
        }
    }*/

    /* 
    while p >= 1 && q < nm - 1 {
        let idx = p-1+(q+1)*block_dim;
        let cond = (x<args.src[idx]) as usize;
        p-= cond;
        q+= (1-cond);

    }*/
    /* 
    let mut idx =p - 1 + (q + 1) * block_dim; 
    let mut cnt = 0;
    while p >= 1 && q < nm - 1 {
        if x < args.src[idx] {
            p -= 1;
            idx-=1;
        } else {
            lpq += p;
            idx+=block_dim;
        }
        cnt+=1;
    }*/
    
    //The mutating idx helped a fair bit (40%) but seems fishy
    //conditional reduction may have helped a little (5%)
    //I'm interested in the multi idx theory but it seems far fetched (balance divergent workloads)
    //alternatively I could dedicate two threads to do the two parts and then combine(but if one is short the other will be long always)
    //I'd like a read on how many times the loop runs, if there's a pattern maybe I could deep ballot -> it's N-1, 
    //could try raw deref
    //I could cheat on the edges 
    //args.dst[0] = F32::PI;
    let mut cnt = 0;
    while p >= 1 && q < nm - 1 {
        if x < args.src[p - 1 + (q + 1) * block_dim] {
            p -= 1;
        } else {
            lpq += p;
            q += 1;
        }
        cnt+=1;
    }
    let mut p = thread_id;
    let mut q = block_id;
    args.dst[p + q * block_dim]=F32::usize(cnt);
    return;
    /* 
    while p >= 1 && q < nm - 1 {
        if x < args.src[p - 1 + (q + 1) * block_dim] {
            p -= 1;
        } else {
            lpq += p;
            q += 1;
        }
    }*/
    let mut p = thread_id;
    let mut q = block_id;
    while p < m - 1 && q >= 1 {
        if x < args.src[p + 1 + (q - 1) * block_dim] {
            q -= 1;
        } else {
            lpq += q;
            p += 1;
        }
    }

    let mut p = thread_id;
    let mut q = block_id;

    args.dst[lpq] = x;
    args.dst[0] = F32::PI;
}

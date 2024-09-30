//Basic idea is to use a macro to generate entry functions for each of the generic functions
//I think to keep the runtime global and sig sane I'll need to start using global memory
//in ptx land I don't think I can do signiture matching so we are on our own
use crate::cuda::atomic::Reduce;
use crate::cuda::grid::GridStride;
use crate::cuda::shared::CuShared;
use crate::cuda::shared::Shared;
use crate::cuda::shuffle::Shuffle;
use crate::shared::primitive::F32;
use crate::shared::vector::VectorArgumentsMap;
use core::arch::nvptx::{_block_dim_x, _syncthreads, _thread_idx_x};
use nonphysical_core::shared::float::Float;
use nonphysical_core::shared::vector::float_vector::FloatVector;
use nonphysical_core::shared::vector::Vector;
//basic idea is that I grid stride the iterator then call actual vector, then clean up results
use super::VectorArgumentsReduce;
use crate::cuda::shuffle::Shuffler;
use crate::shared::vector::VectorArgumentsApply;
use crate::WARP_SIZE;

#[no_mangle]
pub extern "ptx-kernel" fn vector_sum_f32<'a>(args: &'a mut VectorArgumentsReduce<'a, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let mut reduction = CuShared::<F32, 32>::new();
    let mut result = FloatVector::sum(iter); //Handles x16

    let thread_id = unsafe { _thread_idx_x() } as usize;
    let block_dim = unsafe { _block_dim_x() } as usize;
    let lane = thread_id % WARP_SIZE;
    let wid = thread_id / WARP_SIZE;

    let mut i = WARP_SIZE / 2;
    while i >= 1 {
        result += Shuffler::shuffle_bfly::<0xffffffff>(result, i, WARP_SIZE - 1);
        i >>= 1;
    }
    if lane == 0 {
        reduction.store(wid, result);
    }
    unsafe { _syncthreads() };
    result = if thread_id < block_dim / WARP_SIZE {
        reduction.load(lane)
    } else {
        F32::ZERO
    };

    if wid == 0 {
        let mut i = WARP_SIZE / 2;
        while i >= 1 {
            result += Shuffler::shuffle_bfly::<0xffffffff>(result, i, WARP_SIZE - 1);
            i >>= 1;
        }
        if lane == 0 {
            args.acc.reduce_add(0, result);
        }
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_product_f32<'a>(
    _args: &'a mut VectorArgumentsReduce<'a, F32, F32>,
) {
    todo!();//Problem where result for non-grid threads is 0 instead of 1
    /*
    let iter = GridStride::stride(&_args.data);
    let mut reduction = CuShared::<F32, 32>::new();
    let mut result = FloatVector::product(iter); //Handles x16

    let thread_id = unsafe { _thread_idx_x() } as usize;
    let block_dim = unsafe { _block_dim_x() } as usize;
    let lane = thread_id % WARP_SIZE;
    let _wid = thread_id / WARP_SIZE;
    if lane == 1 {
        _args.acc[0] = result;
    }
    return;

    let mut i = WARP_SIZE / 2;
    while i >= 1 {
        result *= Shuffler::shuffle_bfly::<0xffffffff>(result, i, WARP_SIZE - 1);
        i >>= 1;
    }
    if lane == 0 {
        reduction.store(_wid, result);
        _args.acc[0] = F32::PI;
    }
    return;
    unsafe { _syncthreads() };

    result = if thread_id < block_dim / WARP_SIZE {
        reduction.load(lane)
    } else {
        <F32 as Float>::IDENTITY
    };
    if _wid == 0 {
        let mut i = WARP_SIZE / 2;
        while i >= 1 {
            result *= Shuffler::shuffle_bfly::<0xffffffff>(result, i, WARP_SIZE - 1);
            i >>= 1;
        }
        if lane == 0 {
            _args.acc.atomic_mul(0, result);
        }
    } */
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_greater_f32<'a>(
    _args: &'a mut VectorArgumentsReduce<'a, F32, F32>,
) {
    todo!();
    /* 
    let iter = GridStride::stride(&_args.data);
    let mut reduction = CuShared::<F32, 32>::new();
    let mut result = FloatVector::greater(iter); //Handles x16

    let thread_id = unsafe { _thread_idx_x() } as usize;
    let block_dim = unsafe { _block_dim_x() } as usize;
    let lane = thread_id % WARP_SIZE;
    let wid = thread_id / WARP_SIZE;

    let mut i = WARP_SIZE / 2;
    while i >= 1 {
        result = result.greater(Shuffler::shuffle_bfly::<0xffffffff>(
            result,
            i,
            WARP_SIZE - 1,
        ));
        i >>= 1;
    }
    if lane == 0 {
        reduction.store(wid, result);
    }
    unsafe { _syncthreads() };
    result = if thread_id < block_dim / WARP_SIZE {
        reduction.load(lane)
    } else {
        <F32 as Float>::MIN
    };

    if wid == 0 {
        let mut i = WARP_SIZE / 2;
        while i >= 1 {
            result = result.greater(Shuffler::shuffle_bfly::<0xffffffff>(
                result,
                i,
                WARP_SIZE - 1,
            ));
            i >>= 1;
        }
        if lane == 0 {
            _args.acc.atomic_max(0, result);
        }
    }*/
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_lesser_f32<'a>(
    _args: &'a mut VectorArgumentsReduce<'a, F32, F32>,
) {
    todo!();
    /* 
    let iter = GridStride::stride(&_args.data);
    let mut reduction = CuShared::<F32, 32>::new();
    let mut result = FloatVector::lesser(iter); //Handles x16

    let thread_id = unsafe { _thread_idx_x() } as usize;
    let block_dim = unsafe { _block_dim_x() } as usize;
    let lane = thread_id % WARP_SIZE;
    let wid = thread_id / WARP_SIZE;

    let mut i = WARP_SIZE / 2;
    while i >= 1 {
        result = result.lesser(Shuffler::shuffle_bfly::<0xffffffff>(
            result,
            i,
            WARP_SIZE - 1,
        ));
        i >>= 1;
    }
    if lane == 0 {
        reduction.store(wid, result);
    }
    unsafe { _syncthreads() };
    result = if thread_id < block_dim / WARP_SIZE {
        reduction.load(lane)
    } else {
        <F32 as Float>::MAX
    };

    if wid == 0 {
        let mut i = WARP_SIZE / 2;
        while i >= 1 {
            result = result.lesser(Shuffler::shuffle_bfly::<0xffffffff>(
                result,
                i,
                WARP_SIZE - 1,
            ));
            i >>= 1;
        }
        if lane == 0 {
            _args.acc.atomic_min(0, result);
        }
    }*/
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_mean_f32<'a>(_args: &'a mut VectorArgumentsReduce<'a, F32, F32>) {
    todo!();
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_variance_f32<'a>(
    _args: &'a mut VectorArgumentsReduce<'a, F32, F32>,
) {
    todo!()
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_deviation_f32<'a>(
    _args: &'a mut VectorArgumentsReduce<'a, F32, F32>,
) {
    todo!()
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_add_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let other = args.map[0];
    let out_iter = FloatVector::add(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sub_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let other = args.map[0];
    let out_iter = FloatVector::sub(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_mul_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let other = args.map[0];
    let out_iter = FloatVector::mul(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_div_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let other = args.map[0];
    let out_iter = FloatVector::div(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_neg_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::neg(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_scale_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, <F32 as Float>::Primitive>,
) {
    let iter = GridStride::stride(&args.data);
    let other = args.map[0];
    let out_iter = FloatVector::scale(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_descale_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, <F32 as Float>::Primitive>,
) {
    let iter = GridStride::stride(&args.data);
    let other = args.map[0];
    let out_iter = FloatVector::descale(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_powf_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = args.map[0];
    let out_iter = FloatVector::powf(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_ln_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::ln(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_log2_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::log2(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_exp_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::exp(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_exp2_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::exp2(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_recip_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::recip(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sin_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::sin(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_cos_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::cos(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_tan_f32<'a>(args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::tan(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_asin_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::asin(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_acos_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::acos(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_atan_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::atan(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sinh_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::sinh(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_cosh_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::cosh(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_tanh_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::tanh(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_asinh_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::asinh(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_acosh_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::acosh(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_atanh_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::atanh(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_l1_norm_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, <F32 as Float>::Primitive>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::l1_norm(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_l2_norm_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, <F32 as Float>::Primitive>,
) {
    let iter = GridStride::stride(&args.data);
    let out_iter = FloatVector::l2_norm(iter);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_add_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = args.map[0];
    FloatVector::add_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sub_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = args.map[0];
    FloatVector::sub_ref(iter, other);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_mul_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = args.map[0];
    FloatVector::mul_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_div_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = args.map[0];
    FloatVector::div_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_neg_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::neg_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_scale_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, <F32 as Float>::Primitive>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = args.map[0];
    FloatVector::mul_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_descale_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, <F32 as Float>::Primitive>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = args.map[0];
    FloatVector::div_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_powf_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = args.map[0];
    FloatVector::powf_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_ln_ref_f32<'a>(args: &'a mut VectorArgumentsApply<'a, F32, F32>) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::ln_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_log2_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::log2_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_exp_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::exp_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_exp2_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::exp2_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_recip_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::recip_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sin_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::sin_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_cos_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::cos_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_tan_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::tan_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_asin_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::asin_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_acos_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::acos_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_atan_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::atan_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sinh_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::sinh_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_cosh_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::cosh_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_tanh_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::tanh_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_asinh_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::asinh_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_acosh_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::acosh_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_atanh_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    FloatVector::atanh_ref(iter);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_add_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::add_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sub_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::sub_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_mul_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::mul_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_div_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::div_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_scale_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::scale_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_descale_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::descale_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_powf_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::powf_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_greater_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::greater_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_lesser_vec_f32<'a>(
    args: &'a mut VectorArgumentsMap<'a, F32, F32, F32>,
) {
    let iter = GridStride::stride(&args.data);
    let other = GridStride::stride(&args.map);
    let out_iter = FloatVector::lesser_vec(iter, other);
    let out_global = GridStride::stride_ref(&mut args.output);
    out_global
        .zip(out_iter)
        .for_each(|(global, local)| *global = local);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_dot_f32<'a>() {
    todo!()
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_quote_f32<'a>() {
    todo!()
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_add_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::add_vec_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_sub_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::sub_vec_ref(iter, other);
}
#[no_mangle]
pub extern "ptx-kernel" fn vector_mul_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::mul_vec_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_div_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::div_vec_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_scale_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::scale_vec_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_descale_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::descale_vec_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_powf_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::powf_vec_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_greater_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::greater_vec_ref(iter, other);
}

#[no_mangle]
pub extern "ptx-kernel" fn vector_lesser_vec_ref_f32<'a>(
    args: &'a mut VectorArgumentsApply<'a, F32, F32>,
) {
    let iter = GridStride::stride_ref(&mut args.data);
    let other = GridStride::stride(&args.map);
    FloatVector::lesser_vec_ref(iter, other);
}

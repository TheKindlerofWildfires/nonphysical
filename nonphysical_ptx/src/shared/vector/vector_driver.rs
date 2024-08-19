//The best way to vector operate is to have each thread process N Max operations and then combine them across thread and across block
//there's a 'best' way to do this based on transfer time / clock speed / atomic cost but that's hard

use core::marker::PhantomData;

use crate::shared::vector::*;
use alloc::format;
use alloc::sync::Arc;
use core::cmp::min;
use nonphysical_core::shared::{float::Float, real::Real};
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_cuda::cuda::runtime::Dim3;
use nonphysical_cuda::cuda::runtime::Runtime;
use std::string::String;
/*
pub struct VectorArgumentsPinnedReduce<'a, F: Float> {
    pub data: CuPinnedSlice<'a, F>,
    pub acc: CuPinnedSliceRef<'a, F>,
}

pub struct VectorArgumentsPinnedMap<'a, F: Float> {
    pub data: CuPinnedSliceRef<'a, F>,
    pub map: CuPinnedSlice<'a, F>,
}

pub struct VectorArgumentsPinnedMapReduce<'a, F: Float> {
    pub data: CuPinnedSliceRef<'a, F>,
    pub map: CuPinnedSlice<'a, F>,
    pub acc: CuPinnedSliceRef<'a, F>,
}*/

pub struct CudaVectorHost<F: Float> {
    phantom_data: PhantomData<F>,
}
pub struct CudaRealVectorHost<R: Real> {
    phantom_data: PhantomData<R>,
}

pub struct CudaVectorDevice<F: Float> {
    phantom_data: PhantomData<F>,
}

pub struct CudaRealVectorDevice<R: Real> {
    phantom_data: PhantomData<R>,
}

impl<F: Float> CudaVectorHost<F> {
    pub fn reduce_alloc(host_data: &[F], out: F)->VectorArgumentsReduce<'_, F>{
        let mut global_data = CuGlobalSlice::alloc(host_data);
        let mut global_acc = CuGlobalSliceRef::alloc(&[out]);
        global_data.store(host_data);
        global_acc.store(&[out]);

        VectorArgumentsReduce {
            data: global_data,
            acc: global_acc,
        }
    }
    /*
    pub fn map_single_alloc<'a>(host_data: &'a mut [F], other: F)->VectorArgumentsMap<'a, F>{
        let mut global_data = CuGlobalSliceRef::alloc(&host_data);
        let mut global_other = CuGlobalSlice::alloc(&[other]);
        global_data.store(host_data);
        global_other.store(&[other]);
        VectorArgumentsMap {
            data: global_data,
            map: global_other,
        }
    }*/

    pub fn sum(runtime: Arc<Runtime>, host_data: &[F]) -> F {
        let mut out = [F::ZERO];
        let mut global_memory = Self::reduce_alloc(host_data, F::ZERO);

        CudaVectorDevice::sum(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.acc.load(&mut out);
        out[0]
    }

    pub fn l1_sum(runtime: Arc<Runtime>, host_data: &[F]) -> F {
        let mut out = [F::ZERO];
        let mut global_memory = Self::reduce_alloc(host_data, F::ZERO);

        CudaVectorDevice::l1_sum(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.acc.load(&mut out);
        out[0]
    }

    pub fn l2_sum(runtime: Arc<Runtime>, host_data: &[F]) -> F {
        let mut out = [F::ZERO];
        let mut global_memory = Self::reduce_alloc(host_data, F::ZERO);

        CudaVectorDevice::l2_sum(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.acc.load(&mut out);
        out[0]
    }

    pub fn add(runtime: Arc<Runtime>, host_data: &mut [F], other: F) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(&[other]);
        global_data.store(host_data);
        global_other.store(&[other]);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::add(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }

    pub fn sub(runtime: Arc<Runtime>, host_data: &mut [F], other: F) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(&[other]);
        global_data.store(host_data);
        global_other.store(&[other]);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::sub(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }

    pub fn mul(runtime: Arc<Runtime>, host_data: &mut [F], other: F) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(&[other]);
        global_data.store(host_data);
        global_other.store(&[other]);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::mul(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }

    pub fn div(runtime: Arc<Runtime>, host_data: &mut [F], other: F) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(&[other]);
        global_data.store(host_data);
        global_other.store(&[other]);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::div(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }

    pub fn add_vec(runtime: Arc<Runtime>, host_data: &mut [F], other: &[F]) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(other);
        global_data.store(host_data);
        global_other.store(other);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::add_vec(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }

    pub fn sub_vec(runtime: Arc<Runtime>, host_data: &mut [F], other: &[F]) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(other);
        global_data.store(host_data);
        global_other.store(other);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::sub_vec(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }

    pub fn mul_vec(runtime: Arc<Runtime>, host_data: &mut [F], other: &[F]) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(other);
        global_data.store(host_data);
        global_other.store(other);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::mul_vec(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }

    pub fn div_vec(runtime: Arc<Runtime>, host_data: &mut [F], other: &[F]) {
        let mut global_data = CuGlobalSliceRef::alloc(host_data);
        let mut global_other = CuGlobalSlice::alloc(other);
        global_data.store(host_data);
        global_other.store(other);
        let mut global_memory = VectorArgumentsMap {
            data: global_data,
            map: global_other,
        };

        CudaVectorDevice::div_vec(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.data.load(host_data);
    }
}

impl<F: Float> CudaVectorDevice<F> {
    fn vector_launch<Args>(runtime: Arc<Runtime>, args: &mut Args, len: usize, kernel: String) {
        let threads = min(1024, len.div_ceil(CYCLE_COMPARE));
        let block_size = len.div_ceil(threads * CYCLE_COMPARE);
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
        runtime.launch_name(kernel, args, grid, block);
    }
    pub fn sum(runtime: Arc<Runtime>, args: &mut VectorArgumentsReduce<F>) {
        let kernel = format!("vector_sum_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }
    pub fn l1_sum(runtime: Arc<Runtime>, args: &mut VectorArgumentsReduce<F>) {
        let kernel = format!("vector_l1_sum_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }
    pub fn l2_sum(runtime: Arc<Runtime>, args: &mut VectorArgumentsReduce<F>) {
        let kernel = format!("vector_l2_sum_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn add(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_add_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn sub(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_sub_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn mul(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_mul_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn div(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_div_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn add_vec(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_add_vec_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn sub_vec(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_sub_vec_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn mul_vec(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_mul_vec_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }

    pub fn div_vec(runtime: Arc<Runtime>, args: &mut VectorArgumentsMap<F>) {
        let kernel = format!("vector_div_vec_{}", F::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }
}
impl<R: Real> CudaRealVectorHost<R> {
    pub fn reduce_alloc(host_data: &[R], out: R)->VectorArgumentsReduce<'_, R> {
        let mut global_data = CuGlobalSlice::alloc(host_data);
        let mut global_acc = CuGlobalSliceRef::alloc(&[out]);
        global_data.store(host_data);
        global_acc.store(&[out]);

        VectorArgumentsReduce {
            data: global_data,
            acc: global_acc,
        }
    }
    pub fn mean(runtime: Arc<Runtime>, host_data: &[R]) -> R {
        let mut out = [R::ZERO];
        let mut global_memory = Self::reduce_alloc(host_data, R::ZERO);

        CudaRealVectorDevice::mean(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.acc.load(&mut out);
        out[0]
    }

    pub fn variance(runtime: Arc<Runtime>, host_data: &[R]) -> (R, R) {
        let mut out = [R::ZERO, R::ZERO];
        let mut global_data = CuGlobalSlice::alloc(host_data);
        let mut global_acc = CuGlobalSliceRef::alloc(&out);
        global_data.store(host_data);
        global_acc.store(&out);
        let mut global_memory = VectorArgumentsReduce {
            data: global_data,
            acc: global_acc,
        };

        CudaRealVectorDevice::mean(runtime.clone(), &mut global_memory);
        CudaRealVectorDevice::variance(runtime.clone(), &mut global_memory);

        runtime.sync();
        global_memory.acc.load(&mut out);
        (out[0], out[1])
    }
}
impl<R: Real> CudaRealVectorDevice<R> {
    fn vector_launch<Args>(runtime: Arc<Runtime>, args: &mut Args, len: usize, kernel: String) {
        let threads = min(1024, len.div_ceil(CYCLE_COMPARE));
        let block_size = len.div_ceil(threads * CYCLE_COMPARE);
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
        runtime.launch_name(kernel, args, grid, block);
    }
    pub fn mean(runtime: Arc<Runtime>, args: &mut VectorArgumentsReduce<R>) {
        let kernel = format!("real_vector_mean_{}", R::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }
    pub fn variance(runtime: Arc<Runtime>, args: &mut VectorArgumentsReduce<R>) {
        let kernel = format!("real_vector_variance_{}", R::type_id());
        Self::vector_launch(runtime, args, args.data.ptr.len(), kernel);
    }
}

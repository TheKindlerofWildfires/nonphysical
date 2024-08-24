//The best way to vector operate is to have each thread process N Max operations and then combine them across thread and across block
//there's a 'best' way to do this based on transfer time / clock speed / atomic cost but that's hard

use core::marker::PhantomData;

use crate::shared::vector::*;
use alloc::format;
use alloc::sync::Arc;
use core::cmp::min;
use nonphysical_core::shared::real::Real;
use nonphysical_cuda::cuda::global::host::{CuGlobalSlice, CuGlobalSliceRef};
use nonphysical_cuda::cuda::runtime::Dim3;
use nonphysical_cuda::cuda::runtime::Runtime;
use std::string::String;

pub struct CudaRealVectorHost<R: Real> {
    phantom_data: PhantomData<R>,
}

pub struct CudaRealVectorDevice<R: Real> {
    phantom_data: PhantomData<R>,
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

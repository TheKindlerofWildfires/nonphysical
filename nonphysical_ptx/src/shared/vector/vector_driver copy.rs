//The best way to vector operate is to have each thread process N Max operations and then combine them across thread and across block
//there's a 'best' way to do this based on transfer time / clock speed / atomic cost but that's hard

use core::marker::PhantomData;

use crate::shared::vector::*;
use alloc::sync::Arc;
use nonphysical_core::shared::float::Float;
use nonphysical_cuda::cuda::global::{pinned::{CuPinnedSlice,CuPinnedSliceRef},host::{CuGlobalSlice, CuGlobalSliceRef}};
use nonphysical_cuda::cuda::runtime::Runtime;
use nonphysical_cuda::cuda::stream::CuStream;
use nonphysical_cuda::cuda::runtime::Dim3;
use alloc::format;
use core::cmp::min;
use alloc::vec;
use core::ffi::c_void;
use std::dbg;
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
}

/*
    This is for when there is only host data and the alloc / copy / invoke needs to happen
    It alloc's it's own memory, then calls children to do the rest of the work
    When it it done it free's the memory
*/
pub struct CudaVectorHost<F: Float> {
    phantom_data: PhantomData<F>,
}

/*
    This is for when there is only host data and allocated memory on the device
    It copies to the memory, then calls children to do the rest of the work
    When it's done it doesn't free the memory, making it unsafe
    It copies back the result to the host asynchronously
*/
pub struct CudaVectorMixed<F: Float> {
    phantom_data: PhantomData<F>,
}

/*
    This is for when the data is already on device
    It invokes the kernel with any necessary arguments
    When it's done it doesn't free the memory, making it unsafe
    It does not copy the result back to the host
*/
pub struct CudaVectorDevice<F: Float> {
    phantom_data: PhantomData<F>,
}

impl<F: Float> CudaVectorHost<F> {
    pub fn sum(runtime: Arc<Runtime>, host_data: &[F]) -> F {
        dbg!("Started host side");
        
        let mut out = [F::ZERO];
        let stream = CuStream::non_blocking();


        let device_data = CuGlobalSlice::<F>::alloc_async(&host_data, &stream);
        let device_acc = CuGlobalSliceRef::alloc_async(&out, &stream);
        let mut device_memory = VectorArgumentsReduce { data:device_data, acc:device_acc };

        let mut pinned_data = CuPinnedSlice::alloc(&host_data);
        let mut pinned_acc = CuPinnedSliceRef::alloc(&out);
        pinned_data.store(host_data);
        pinned_acc.store(&out);

        let mut pinned_memory = VectorArgumentsPinnedReduce{data: pinned_data, acc: pinned_acc};
        dbg!("Did device and pinned memory");

        CudaVectorMixed::sum(runtime, stream, &mut device_memory, &mut pinned_memory);
        Runtime::sync();
        pinned_memory.acc.load(&mut out);
        out[0]
    }
}

impl<F: Float> CudaVectorMixed<F> {
    pub fn sum(
        runtime: Arc<Runtime>, stream: CuStream, device_memory: &mut VectorArgumentsReduce<F>, pinned_memory: &mut VectorArgumentsPinnedReduce<F>
    ) {
        device_memory.data.store_async(&mut pinned_memory.data,&stream);
        device_memory.acc.store_async(&mut pinned_memory.acc,&stream);

        CudaVectorDevice::sum(runtime, device_memory);
        device_memory.acc.load_async(&mut pinned_memory.acc,&stream);
    }
}

impl<F: Float> CudaVectorDevice<F> {
    pub fn sum(runtime: Arc<Runtime>, args: &mut VectorArgumentsReduce<F>) {
        let threads = min(1024, args.data.ptr.len());
        let block_size = args.data.ptr.len().div_ceil(threads * CYCLE_COMPARE);
        let grid = Dim3{x:block_size, y:1, z: 1};
        let block = Dim3{x:threads, y:1, z:1};
        let kernel = format!("vector_sum_{}", F::type_id());
        let mut pointers = vec![
            args.data.ptr as *mut [F] as *mut c_void,
            args.acc.ptr as *mut [F] as *mut c_void
        ];

        //should be able to avoid the lookup by cache
        runtime
            .launch_name(kernel, &mut pointers, grid, block);
    }
}

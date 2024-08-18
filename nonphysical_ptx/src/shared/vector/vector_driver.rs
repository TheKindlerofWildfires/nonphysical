//The best way to vector operate is to have each thread process N Max operations and then combine them across thread and across block
//there's a 'best' way to do this based on transfer time / clock speed / atomic cost but that's hard

use core::marker::PhantomData;

use crate::cuda::global::{CuGlobalSlice, CuGlobalSliceRef};


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
        let mut out = [F::ZERO];
        let data = CuGlobalSlice::<F>::alloc_async(runtime.clone(), &data)
            .expect("Failed to allocate data for vector operation");
        let mut acc = CuGlobalSliceRef::alloc_async(runtime.clone(), &out)
            .expect("Failed to allocate output for vector operation");
        let mut args = VectorArgumentsReduce { data, acc };
        CudaVectorMixed::sum(runtime, &mut args, host_data, &mut out);
        self.runtime.sync();
        out[0]
    }
}
impl<F: Float> CudaVectorMixed<F> {
    pub fn sum(
        runtime: Arc<Runtime>,
        args: &mut VectorArgumentsReduce<F32>,
        data: &[F],
        out: &mut [F; 1],
    ) {
        args.data
            .copy_async(runtime.clone(), data, HostToDevice)
            .expect("Failed to copy data for vector operation");
        CudaVectorDevice::sum(runtime, args);
        args.acc
            .copy_async(runtime.clone(), out, DeviceToHost)
            .expect("Failed to output data for vector operation");
    }
}

impl<F: Float> CudaVectorDevice<F> {
    pub fn sum(runtime: Arc<Runtime>, args: &mut VectorArgumentsReduce<F32>) {
        let threads = min(1024, args.data.len());
        let block_size = div_ceil(args.data.len(), threads * CYCLE_COMPARE);
        let kernel = format!("vector_sum_{}", F::type_id());

        self.runtime
            .launch(kernel, &mut args, block_size, 1, 1, threads, 1, 1).expect("Failed to launch vector operation");
    }
}

/*
    as follows
        takes in array, puts out sorted array
        allocates a source array, count array, dst array
            pointers to hist/histscan
        computes block/grid size for some stuff
        mallocs hist/hist scan
        creates a bunch of streams
        does a mem copy of the data per stream (this is clever)
        applies local kernel sort
        enters a kbit loop
            in not first pass sorts local again (probably because we did this in the copy)
            transposes
            creates a scan array
            reduces something
            scatters something
            reads data back to host

*/
use core::cmp::min;
use core::marker::PhantomData;
use crate::std::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use nonphysical_core::shared::float::Float;
use nonphysical_cuda::cuda::{
    global::host::{CuGlobalSlice, CuGlobalSliceRef},
    runtime::{Dim3, RUNTIME},
};
use crate::graph::merge_sort::INSERT_SIZE;
use crate::graph::merge_sort::{THREAD_MAX};

use nonphysical_std::shared::unsigned::U32;
use nonphysical_core::shared::unsigned::Unsigned;
use crate::graph::merge_sort::MergeSortArgumentsGlobal;
use std::dbg;
use alloc::string::String;
//wrong, this is supposed to be points, but thats going to be scary
//will have to impl compare for poitns
pub struct CudaMergeSort<F:Float> {
    phantom_float: PhantomData<F>,
}

//Might as well call this trash sort for how poorly it's implemented (slower than CPU by x100, only accepts power of two inputs)
//Using https://davidbader.net/publication/2012-gm-ba/2012-gm-ba.pdf would probably fix 90% of the problems
//But I have KDT to build and MST to find 
//When this becomes the limiting issue I'll rework it, starting with 
// 1) merge_launch could not be worse, use the bader paper (95% of perf issues - 256 out of 269 ms)
// 2) add the bader elements to the internal loop of local_sort_launch (78% of remaining perf issues based on barrier stalls)
// 3) Address the power of two problem
// 4) Optimize shared memory (66% of remaining perf based on limits)
impl<F:Float> CudaMergeSort<F>{
    pub fn merge_1024(data: &[F]) ->Vec<F>{
        let mut output = vec![F::ZERO; data.len()];
        let params_host = [U32::ZERO];
        let src = CuGlobalSliceRef::alloc(data);
        let dst = CuGlobalSliceRef::alloc(&output);
        let params = CuGlobalSlice::alloc(&params_host);
        let mut args = MergeSortArgumentsGlobal{
            src,
            dst,
            params
        };
        args.src.store(data);

        //clean up on the small section (takes 491 us to 120 us)
        Self::launch_insert_skip(&mut args, "insert_skip".to_string());
        
        let mut expected_size = INSERT_SIZE; 
         
        //merge together sorted arrays, up to size 1024 (this has some global access issues)
        while expected_size<THREAD_MAX &&expected_size < data.len(){
            std::mem::swap(&mut args.src,&mut args.dst);
            Self::launch_merge_path(&mut args, expected_size,"merge_path".to_string());
            expected_size<<=1;
        }
        
        //Cross index the sort along the new axis
        std::mem::swap(&mut args.src,&mut args.dst);
        Self::launch_insert_skip(&mut args, "insert_skip_stride".to_string());
        
        //sort this by blocks again
        let mut expected_size = INSERT_SIZE;
        
        //merge together sorted arrays, up to size 1024 (this has some global access issues)
        while expected_size<THREAD_MAX &&expected_size < data.len(){
            std::mem::swap(&mut args.src,&mut args.dst);
            Self::launch_merge_path(&mut args, expected_size,"merge_path".to_string());
            expected_size<<=1;
        }
        //An assumption was broken here, it's possible for the wrong order to occur here in the columns
        //apply cleanup
        std::mem::swap(&mut args.src,&mut args.dst);
        Self::launch_matrix_sort(&mut args);
        
        args.dst.load(&mut output);
        output
    }

    fn launch_insert_skip<'a>(args: &mut MergeSortArgumentsGlobal<'a,F>,kernel: String){
        let threads = min(THREAD_MAX,args.src.ptr.len().div_ceil(INSERT_SIZE));
        let block_size = args.src.ptr.len().div_ceil(threads*INSERT_SIZE);
        dbg!(threads,block_size);
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

    fn launch_matrix_sort<'a>(args: &mut MergeSortArgumentsGlobal<'a,F>){
        let threads = min(THREAD_MAX,args.src.ptr.len());
        let block_size = args.src.ptr.len().div_ceil(threads);
        dbg!(threads,block_size);
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
                rt.launch_name("matrix_sort".to_string(), args, grid, block);
            }
            None => panic!("Cuda Runtime not initialized"),
        };
    }

    fn launch_merge_path<'a>(args: &mut MergeSortArgumentsGlobal<'a,F>, expected_size:usize,kernel:String){
        args.params.store(&[U32::usize(expected_size)]);
        let threads = min(1024,args.src.ptr.len());
        let count = args.src.ptr.len().div_ceil(expected_size*2);
        let block_size = (threads-1+count*expected_size*2)/threads;
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
}
//TLDR is this can be used to create a hash table with key -> F32 and value ??? and will reduce it down to
//the non duplicates
//It will have the four methods
//And it's main use is currently to remove duplicates pre-kdtree
use crate::graph::hash_table::{HashTableArguments, CAPACITY};
use crate::WARP_SIZE;
use alloc::vec;
use nonphysical_core::shared::float::Float;
use nonphysical_cuda::cuda::global::host::CuGlobalSliceRef;
use nonphysical_cuda::cuda::runtime::{Dim3, RUNTIME};
use core::cmp::min;
use std::marker::PhantomData;
use std::string::{String, ToString};
use std::vec::Vec;
use nonphysical_core::shared::unsigned::Unsigned;
use nonphysical_std::shared::unsigned::U32;
pub struct CudaHashTable<F: Float, T: Copy> {
    phantom_float: PhantomData<F>,
    phantom_t: PhantomData<T>,
}

impl<'a, F: Float + 'a, T: Copy + Default> CudaHashTable<F, T> {
    pub fn create() -> HashTableArguments<'a, F, T> {
        let keys = vec![F::MAX];
        let values = vec![T::default()];
        let hash_keys = vec![F::MAX; CAPACITY as usize];
        let hash_values = vec![T::default(); CAPACITY as usize];
        let ctr = vec![U32::ZERO];
        let mut arguments = Self::hash_alloc(&keys, &values, &hash_keys, &hash_values, &ctr);
        Self::transfer_table(&hash_keys, &hash_values, &mut arguments);
        arguments
    }

    pub fn insert(keys: &[F], values: &[T], arguments: &mut HashTableArguments<'a, F, T>) {
        let (device_keys, device_values) = Self::hash_alloc_kvp(keys,values);
        arguments.keys = device_keys;
        arguments.values = device_values;
        Self::transfer_keys(keys, values, arguments);
        Self::hash_launch(arguments,keys.len(), "insert_hash_table_f32".to_string());
    }


    pub fn lookup(keys: &[F], values: &mut [T], arguments: &mut HashTableArguments<'a, F, T>) {
        let (device_keys, device_values) = Self::hash_alloc_kvp(keys,values);
        arguments.keys = device_keys;
        arguments.values = device_values;
        Self::transfer_keys(keys, values, arguments);
        Self::hash_launch(arguments,keys.len(),"lookup_hash_table_f32".to_string());
        arguments.values.load(values);
    }
    pub fn delete(keys: &[F], arguments: &mut HashTableArguments<'a, F, T>) {
        arguments.keys = CuGlobalSliceRef::alloc(keys);
        arguments.keys.store(keys);
        Self::hash_launch(arguments,keys.len(),"delete_hash_table_f32".to_string());
    }

    pub fn count(
        arguments: &mut HashTableArguments<'a, F, T>,
    ) -> U32{
        Self::hash_launch(arguments,CAPACITY as usize, "count_hash_table_f32".to_string());
        let mut ctr = vec![U32::ZERO];
        arguments.ctr.load(&mut ctr);
        ctr[0]
    }
    pub fn iterate(
        arguments: &mut HashTableArguments<'a, F, T>,
    ) ->(Vec<F>, Vec<T>){

        let mut keys = vec![F::MAX;CAPACITY as usize];
        let mut values = vec![T::default();CAPACITY as usize];
        let (device_keys, device_values) = Self::hash_alloc_kvp(&keys,&values);
        arguments.keys = device_keys;
        arguments.values = device_values;
        let mut ctr = [U32::ZERO];
        arguments.ctr.store(&ctr);
        Self::hash_launch(arguments,CAPACITY as usize, "iterate_hash_table_f32".to_string());
        arguments.ctr.load(&mut ctr);
        arguments.keys.load(&mut keys);
        arguments.values.load(&mut values);
        (keys[..ctr[0].as_usize()].to_vec(),values[..ctr[0].as_usize()].to_vec())
    }

    fn hash_alloc(
        keys: &[F],
        values: &[T],
        hash_keys: &[F],
        hash_values: &[T],
        counter: &[U32],
    ) -> HashTableArguments<'a, F, T> {
        let (keys, values) = Self::hash_alloc_kvp(keys, values);
        let (table_keys, table_values) = Self::hash_alloc_kvp(hash_keys, hash_values);
        let ctr = CuGlobalSliceRef::alloc(counter);

        HashTableArguments {
            keys,
            values,
            table_keys,
            table_values,
            ctr,
        }
    }
    fn hash_alloc_kvp(
        keys: &[F],
        values: &[T],
    ) -> (CuGlobalSliceRef<'a, F>, CuGlobalSliceRef<'a, T>) {
        let keys = CuGlobalSliceRef::alloc(keys);
        let values = CuGlobalSliceRef::alloc(values);
        (keys, values)
    }
    fn transfer_table(
        hash_keys: &[F],
        hash_values: &[T],
        arguments: &mut HashTableArguments<F, T>,
    ) {
        arguments.table_keys.store(hash_keys);
        arguments.table_values.store(hash_values);
    }
    fn transfer_keys(keys: &[F], values: &[T], arguments: &mut HashTableArguments<F, T>) {
        arguments.keys.store(keys);
        arguments.values.store(values);
    }

    pub fn hash_launch(args: &mut HashTableArguments<'a, F, T>, len: usize, kernel: String) {
        let threads = min(1024, len.div_ceil(WARP_SIZE / 2));
        let block_size = len.div_ceil(threads * WARP_SIZE / 2); //Half a warp was optimal in testing
        use std::dbg;
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
        if kernel != "count_hash_table_f322"{
            match RUNTIME.get() {
                Some(rt) => {
                    rt.launch_name(kernel, args, grid, block);
                }
                None => panic!("Cuda Runtime not initialized"),
            };
        }

    }
}
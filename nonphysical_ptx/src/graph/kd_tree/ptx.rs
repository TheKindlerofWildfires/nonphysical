

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};
use core::marker::PhantomData;
use crate::cuda::shared::CuShared;
use nonphysical_core::shared::point::StaticPoint;
use nonphysical_core::shared::{point::Point, primitive::Primitive};
use nonphysical_core::graph::kd_tree::KdShared;
use crate::cuda::cu_slice::CuSliceRef;
use crate::shared::primitive::F32;
use crate::graph::kd_tree::KdTreeArguments;

use crate::graph::kd_tree::{KdLeaf1337, KdNode1337, KdTree1337,KdBranch1337};

use super::KdShared1337;
use crate::cuda::intrinsic::Intrinsic;

#[no_mangle]
pub extern "ptx-kernel" fn build_kd_tree(args: &mut KdTreeArguments<F32>) {
    let shared_bounds = KdTreeCuda::<StaticPoint<F32,4>,8>::compute_bounds(&args.stat);
    KdTreeCuda::<StaticPoint<F32,4>,8>::build_tree(&args.stat, &mut args.tags, &shared_bounds);
}

pub struct KdTreeCuda<P:Point, const N:usize>{
    phantom_data: PhantomData<P>,

}
impl<P:Point<Primitive = F32>, const N:usize> KdTreeCuda<P,N>{
    pub fn compute_bounds(points: &[P]) -> CuShared{
        let shared_bounds = CuShared::new::<N,4>();
        let idx = unsafe { _thread_idx_x() +_block_idx_x()*_block_dim_x()} as usize;
        (0..N).for_each(|i|{
            shared_bounds.store_f32(2*i, <P::Primitive as Primitive>::MIN);
            shared_bounds.store_f32(2*i+1, <P::Primitive as Primitive>::MAX);

        });
        
        points[idx].data().enumerate().for_each(|(i,p)|{
            unsafe { shared_bounds.atomic_max_f32(2*i, *p) };
            unsafe { shared_bounds.atomic_min_f32(2*i+1, *p) };
        });

        shared_bounds
    }   
    pub fn build_tree(points: &[P], tags: &mut CuSliceRef<u32>, bounds: & CuShared) {
        //assumes tags are zeroed

        //figure out the levels
        let num_levels = Self::level((points.len()-1) as u32)+1;
        let deepest_level = num_levels-1;
        (0..deepest_level).for_each(|level|{
            //sort the data
            //update the tags

        });
        //sort the data
    }
    pub fn level(idx: u32) -> u32 {
        63-(idx+1).clz()
    } 
}

/*
    new version
    does a checka bout explicit dim

    computes the bounds 
        seems ok

    builds the actual tree
        allocs memory for the tags and sets it to zero
        figures out the number of levels 
        computes the bounds again... 
        for each level 
            sorts the data by the tags
            cuda call update tags
        sorts the data by the tags
        drops the tags


*/
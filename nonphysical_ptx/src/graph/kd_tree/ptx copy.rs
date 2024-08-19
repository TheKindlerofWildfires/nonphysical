

use core::arch::nvptx::{_syncthreads, _thread_idx_x};
use crate::cuda::atomic::AtomicGlobal;
use nonphysical_core::shared::{point::Point, primitive::Primitive};
use nonphysical_core::graph::kd_tree::KdShared;
use crate::cuda::cu_slice::CuSliceRef;
use crate::shared::primitive::F32;
use crate::graph::kd_tree::KdTreeArguments;

use crate::graph::kd_tree::{KdLeaf1337, KdNode1337, KdTree1337,KdBranch1337};

use super::KdShared1337;

/*

pub struct KdTreeParameters<'a>{
    capacity: CuSliceRef<'a, usize>,
    tree_idx: CuSliceRef<'a, usize>,
}
pub struct KdTreeArguments<'a, P:Point>{
    parameters: KdTreeParameters<'a>,
    points: CuSliceRef<'a, P>, 
    trees: CuSliceRef<'a, KdTree1337<P>>,
}
*/

#[no_mangle]
pub extern "ptx-kernel" fn build_kd_tree(args: &mut KdTreeArguments<F32>) {
    KdTree1337::construct(args.parameters.capacity[0], &mut args.points,&mut args.trees, 0,&mut args.parameters.tree_idx[0]);

}


impl<P:Point<Primitive = F32>> KdTree1337<P>{

    //this needs an array of KdTrees of length 2*n-1
    //I've assumed threads=points
    pub fn construct(capacity: usize, points: &mut CuSliceRef<P>, trees: &mut CuSliceRef<KdTree1337<P>>, tree_idx: u32, tree_size: &mut u32){

        //Take in all of the points at once
        let idx = unsafe { _thread_idx_x() } as usize;
        assert!(trees.len() == points.len()*2-1);

        //Find the max and the min bounds
        let self_tree= &mut trees[tree_idx as usize];
        let kd_shared = &mut self_tree.shared;
        kd_shared.size = points.len();
        Self::get_bounding_box(points, idx, kd_shared);
        unsafe {_syncthreads()};

        if points.len()>capacity{
            //Find the split dimension/split point (this could be done in para, but it's not needed)
            let (_, split_dimension) = kd_shared.max_bounds.ordered_farthest(&kd_shared.min_bounds);
            let min = kd_shared.min_bounds.data(split_dimension);
            let max = kd_shared.max_bounds.data(split_dimension);
            let split_value = min + (max - min) / P::Primitive::usize(2);
            let right_ptr =  unsafe { AtomicGlobal::inc_u32(*tree_size,u32::MAX) };
            let left_ptr =  unsafe { AtomicGlobal::inc_u32(*tree_size,u32::MAX) };
            self_tree.node = KdNode1337::Branch(KdBranch1337{
                split_dimension,
                split_value,
                left_ptr,
                right_ptr,
            });
            unsafe {_syncthreads()};
            //figure out where my point is supposed to go
            let my_point = &points[idx];
            
            let my_workspace_idx = if my_point.data(split_dimension)>split_value{
                unsafe { AtomicGlobal::inc_u32(kd_shared.low_inc,u32::MAX) }
            }else{
                unsafe { AtomicGlobal::inc_u32(kd_shared.high_inc,u32::MAX) }
            };
            unsafe {_syncthreads()};
            let split = kd_shared.low_inc;

            let offset = if my_point.data(split_dimension)>split_value{
                0
            }else{
                split
            };
            //put it there
            points[(offset+my_workspace_idx) as usize]=my_point.clone();
            unsafe {_syncthreads()};

            //build new kd trees 
            //Self::construct(capacity, points[..split as usize], workspace, trees, right_ptr, tree_size);
            //Self::construct(capacity, points[split as usize..], workspace, trees, left_ptr, tree_size);

        }


    }

    //atomic not impl yet
    fn get_bounding_box(points: &mut [P],idx:usize, kd_shared: &mut KdShared1337<P>){
        //there are N dimensions, so I need to find 2N parameters
        let modulus = points[0].dimension();

        //figure out what parameter this thread is working on
        let p_idx = idx%modulus;
        let div = idx/modulus;
        if idx &1 ==0 {
            //get the min of the points I'm responsible for
            let thread_min = points.iter().skip(div).step_by(modulus).fold(<P::Primitive as Primitive>::MAX, |acc,p|{
                acc.lesser(p.data(p_idx))
            });
            //use atomics to vote on the minimum
            unsafe { AtomicGlobal::min_f32(kd_shared.min_bounds.data_ref(p_idx),thread_min) };

        }else{
            //get the max of the points I'm responsible for
            let thread_max = points.iter().skip(div).step_by(modulus).fold(<P::Primitive as Primitive>::MIN, |acc,p|{
                acc.greater(p.data(p_idx))
            });
            //use atomics to vote on the minimum
            unsafe { AtomicGlobal::max_f32(kd_shared.min_bounds.data_ref(p_idx),thread_max) };
        }
    }
}

use nonphysical_core::shared::point::{Point, StaticPoint};

use crate::{cuda::cu_slice::CuSliceRef, shared::primitive::F32};

#[cfg(not(target_arch = "nvptx64"))]
pub mod cuda;
#[cfg(target_arch = "nvptx64")]
pub mod ptx;

pub struct KdTreeParameters<'a>{
    capacity: CuSliceRef<'a, usize>,
    workspace: CuSliceRef<'a, u32>,
    tree_idx: CuSliceRef<'a, u32>,
}
pub struct KdTreeArguments<'a, P:Point>{
    parameters: KdTreeParameters<'a>,
    points: CuSliceRef<'a, P>, 
    trees: CuSliceRef<'a, KdTree1337<P>>,
    stat: CuSliceRef<'a, StaticPoint<F32,4>>,
    tags: CuSliceRef<'a, u32>,
}

pub struct KdTree1337<P:Point>{
    pub shared: KdShared1337<P>,
    pub node: KdNode1337<P>,
}

pub struct KdLeaf1337 {
}
pub struct KdBranch1337<P: Point> {
    pub left_ptr: u32,
    pub right_ptr: u32,
    pub split_value: P::Primitive,
    pub split_dimension: usize,
}
pub enum KdNode1337<P: Point> {
    Leaf(KdLeaf1337),
    Branch(KdBranch1337<P>),
}
pub struct KdShared1337<P: Point>{
    pub size: usize,
    pub min_bounds: P,
    pub max_bounds: P,
    pub low_inc: u32,
    pub high_inc: u32,
}
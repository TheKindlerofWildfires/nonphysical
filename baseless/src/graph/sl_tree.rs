use crate::shared::point::Point;
use alloc::vec::Vec;
use alloc::vec;
use super::ms_tree::MSTree;

pub struct SLTreeNode<P: Point> {
    pub left_node_idx: usize,
    pub right_node_idx: usize,
    pub distance: P::Primitive,
    pub size: usize,
}

pub struct SLTree<P: Point> {
    pub sl_tree_vec: Vec<SLTreeNode<P>>,
}

impl<P: Point> SLTreeNode<P> {
    fn new(left_node_idx: usize, right_node_idx: usize, distance: P::Primitive, size: usize) -> Self {
        Self {
            left_node_idx,
            right_node_idx,
            distance,
            size,
        }
    }
}

impl<P: Point> SLTree<P> {
    pub fn new(input: &MSTree<P>) -> Self {
        let samples = input.ms_tree_vec.len()+1;
        let mut sl_tree_vec = Vec::with_capacity(samples-1);
        let double_length = 2 * samples-1;
        let mut parent = vec![double_length; double_length];
        let mut next_label = samples;
        let mut size_vec = ((0..samples).map(|_| 1))
            .chain((samples..double_length).map(|_| 0))
            .collect::<Vec<_>>();
        for i in 0..(samples-1){
            let ms_tree_edge = &input.ms_tree_vec[i];
            let left_child = Self::find(ms_tree_edge.left_node_idx, &mut parent);
            let right_child = Self::find(ms_tree_edge.right_node_idx, &mut parent);
            let distance = ms_tree_edge.distance;

            let size = size_vec[left_child] + size_vec[right_child];
            sl_tree_vec.push(SLTreeNode::new(left_child, right_child, distance, size));

            parent[left_child] = next_label;
            parent[right_child] = next_label;
            size_vec[next_label] = size;
            next_label += 1;
        };
        Self { sl_tree_vec }
    }

    fn find(node_idx: usize, parent: &mut [usize]) -> usize {
        let mut n = node_idx;
        let mut p = node_idx;

        while parent[n] != parent.len() {
            n = parent[n];
        }
        while parent[p] != n {
            p = if p == parent.len() {
                parent.len() - 1
            } else {
                p
            };
            p = parent[p];
            p = if p == parent.len() {
                parent.len() - 1
            } else {
                p
            };
            parent[p] = n;
        }
        n
    }
}

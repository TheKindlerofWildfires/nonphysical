use crate::shared::{float::Float,primitive::Primitive, point::Point};
use alloc::vec::Vec;
use alloc::vec;
pub struct MSTreeNode<P: Point> {
    pub left_node_idx: usize,
    pub right_node_idx: usize,
    pub distance: P::Primitive,
}

pub struct MSTree<P: Point> {
    pub ms_tree_vec: Vec<MSTreeNode<P>>,
}

impl<P: Point> MSTreeNode<P> {
    fn new(left_node_idx: usize, right_node_idx: usize, distance: P::Primitive) -> Self {
        Self {
            left_node_idx,
            right_node_idx,
            distance,
        }
    }
}

//prim's algorithm
impl<P: Point> MSTree<P> {
    pub fn new(input: &[P],distance_overrides: &[P::Primitive]) -> Self{
        let samples = input.len();
        let mut in_tree = vec![false; samples];
        let mut distances = vec![P::Primitive::MAX; samples];

        distances[0] = P::Primitive::ZERO;

        let mut ms_tree_vec = Vec::with_capacity(samples);
        let mut left_node_idx = 0;
        let mut right_node_idx = 0;

        (1..samples).for_each(|_| {
            in_tree[left_node_idx] = true;
            let mut current_min_dist = P::Primitive::MAX;
            (0..samples).for_each(|i| {
                if !in_tree[i] {
                    let mutual_reach = Self::mutual_reach(left_node_idx, i, input,distance_overrides);
                    if mutual_reach < distances[i] {
                        distances[i] = mutual_reach;
                    }
                    if distances[i] < current_min_dist {
                        right_node_idx = i;
                        current_min_dist = distances[i];
                    }
                }
            });
            ms_tree_vec.push(MSTreeNode::new(
                left_node_idx,
                right_node_idx,
                current_min_dist,
            ));
            left_node_idx = right_node_idx;
        });

        let mut output = MSTree { ms_tree_vec };
        output.sort();
        output
    }

    fn mutual_reach(node_a_idx: usize, node_b_idx: usize, input: &[P],distances: &[P::Primitive]) -> P::Primitive{
        let dist_a = distances[node_a_idx];
        let dist_b = distances[node_b_idx];

        let dist = input[node_a_idx].l1_distance(&input[node_b_idx]);

        dist.greater(dist_a).greater(dist_b)
    }

    fn sort(&mut self){
        self.ms_tree_vec.sort_by(|a,b| a.distance.partial_cmp(&b.distance).unwrap())
    }
}
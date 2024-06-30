/* 
use std::{char::MAX, collections::VecDeque};

use crate::{
    cluster::Classification::{Core, Edge, Noise},
    geometric::{kd_tree::KdTree, ms_tree::MSTree, sl_tree::SLTree},
    shared::float::Float,
};

use super::Classification;

trait HDBSCAN<T: Float> {
    fn cluster(input: &Self, epsilon: &T, min_points: usize) -> Vec<Classification>
    where
        Self: Sized;

    fn kd_core_distances(input: &Self, k: usize) -> Vec<T>;

    fn condense_tree(sl_tree: &SLTree<T>, size: usize, min_points: usize) -> Vec<CondensedNode<T>>;

    fn search_sl_tree(sl_tree: &SLTree<T>, root: usize, size: usize) -> Vec<usize>;

    fn add_children(
        node_id: usize,
        new_node_id: usize,
        sl_tree: &SLTree<T>,
        condensed_tree: &mut Vec<CondensedNode<T>>,
        visited: &mut Vec<bool>,
        lambda: T,
        size: usize,
    );
}

struct CondensedNode<T: Float> {
    node_idx: usize,
    parent_node_idx: usize,
    lambda: T,
    size: usize,
}

impl<T: Float> CondensedNode<T> {
    pub fn new(node_idx: usize, parent_node_idx: usize, lambda: T, size: usize) -> Self {
        Self {
            node_idx,
            parent_node_idx,
            lambda,
            size,
        }
    }
}

impl<T: Float> HDBSCAN<T> for Vec<Vec<T>> {
    fn cluster(input: &Self, epsilon: &T, min_points: usize) -> Vec<Classification>
    where
        Self: Sized,
    {
        let core_distances = Self::kd_core_distances(input, min_points);
        let ms_tree = MSTree::new(input, &core_distances);
        let sl_tree = SLTree::new(&ms_tree);
        let condensed_tree = Self::condense_tree(&sl_tree, input.len(), min_points);
        let winning_clusters = Self::winning_clusters(&condensed_tree,input.len());

        Vec::new()
    }

    fn kd_core_distances(input: &Self, k: usize) -> Vec<T> {
        let dimensions = input[0].len();
        let capacity = (input.len() as f32).sqrt() as usize;
        let mut kd_tree = KdTree::<T>::new(dimensions, capacity);
        input.iter().enumerate().for_each(|(i, point)| {
            let mag_point = point.iter().map(|p| p.square_norm()).collect();
            kd_tree.add(mag_point, i)
        });
        input
            .iter()
            .map(|point| kd_tree.nearest(point, k).iter().last().unwrap().0)
            .collect()
    }

    fn condense_tree(sl_tree: &SLTree<T>, size: usize, min_points: usize) -> Vec<CondensedNode<T>> {
        let top_node = (size - 1) * 2;
        let node_indices = Self::search_sl_tree(sl_tree, top_node, size);

        let mut new_node_indices = vec![0; top_node + 1];
        new_node_indices[top_node] = size;
        let mut next_parent_id = size + 1;

        let mut visited = vec![false; node_indices.len()];
        let mut condensed_tree = Vec::new();
        node_indices.iter().for_each(|&node_idx| {
            if !visited[node_idx] && !(node_idx < size) {
                let left_child_idx = sl_tree.sl_tree_vec[node_idx - size].left_node_idx;
                let right_child_idx = sl_tree.sl_tree_vec[node_idx - size].right_node_idx;
                let lambda = sl_tree.sl_tree_vec[node_idx - size].distance.recip();
                let left_child_size = if left_child_idx < size {
                    1
                } else {
                    sl_tree.sl_tree_vec[left_child_idx - size].size
                };
                let right_child_size = if right_child_idx < size {
                    1
                } else {
                    sl_tree.sl_tree_vec[right_child_idx - size].size
                };

                let is_left_cluster = left_child_size > min_points;
                let is_right_cluster = right_child_size > min_points;

                match (is_left_cluster, is_right_cluster) {
                    (true, true) => {
                        [left_child_idx, right_child_idx]
                            .iter()
                            .zip([left_child_size, right_child_size].iter())
                            .for_each(|(child_idx, &child_size)| {
                                new_node_indices[*child_idx] = next_parent_id;
                                next_parent_id += 1;
                                condensed_tree.push(CondensedNode::new(
                                    new_node_indices[*child_idx],
                                    new_node_indices[node_idx],
                                    lambda,
                                    child_size,
                                ));
                            });
                    }
                    (true, false) => {
                        new_node_indices[left_child_idx] = new_node_indices[node_idx];
                        Self::add_children(
                            right_child_idx,
                            new_node_indices[node_idx],
                            sl_tree,
                            &mut condensed_tree,
                            &mut visited,
                            lambda,
                            size,
                        );
                    }
                    (false, true) => {
                        new_node_indices[right_child_idx] = new_node_indices[node_idx];
                        Self::add_children(
                            left_child_idx,
                            new_node_indices[node_idx],
                            sl_tree,
                            &mut condensed_tree,
                            &mut visited,
                            lambda,
                            size,
                        );
                    }
                    (false, false) => {
                        let new_node_id = new_node_indices[node_idx];
                        Self::add_children(
                            left_child_idx,
                            new_node_id,
                            sl_tree,
                            &mut condensed_tree,
                            &mut visited,
                            lambda,
                            size,
                        );
                        Self::add_children(
                            right_child_idx,
                            new_node_id,
                            sl_tree,
                            &mut condensed_tree,
                            &mut visited,
                            lambda,
                            size,
                        );
                    }
                }
            }
        });
        condensed_tree
    }

    fn search_sl_tree(sl_tree: &SLTree<T>, root: usize, size: usize) -> Vec<usize> {
        let mut process_queue = VecDeque::from([root]);
        let mut child_nodes = Vec::new();

        while !process_queue.is_empty() {
            let mut current_node_id = match process_queue.pop_front() {
                Some(node_id) => node_id,
                None => break,
            };
            child_nodes.push(current_node_id);
            if current_node_id < size {
                continue;
            }
            current_node_id -= size;
            let current_node = &sl_tree.sl_tree_vec[current_node_id];
            process_queue.push_back(current_node.left_node_idx);
            process_queue.push_back(current_node.right_node_idx);
        }

        child_nodes
    }

    fn add_children(
        node_idx: usize,
        new_node_idx: usize,
        sl_tree: &SLTree<T>,
        condensed_tree: &mut Vec<CondensedNode<T>>,
        visited: &mut Vec<bool>,
        lambda: T,
        size: usize,
    ) {
        Self::search_sl_tree(sl_tree, node_idx, size)
            .iter()
            .for_each(|&child_id| {
                if child_id < size {
                    condensed_tree.push(CondensedNode::new(child_id, new_node_idx, lambda, 1));
                }
                visited[child_id] = true;
            });
    }

    fn winning_clusters(condensed_tree: &Vec<CondensedNode<T>>, size: usize){
        let n_clusters = condensed_tree.len() - size +1;
        let stabilities =         (0..size)
        .map(|n| self.n_samples + n)
        .map(|cluster_id| (
            cluster_id, RefCell::new(self.calc_stability(cluster_id, &condensed_tree))
        ))
        .collect()
    }
}
*/
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::{cell::RefCell, marker::PhantomData};
use std::collections::HashMap;
use std::dbg;
use std::time::SystemTime;

use crate::{
    cluster::Classification::{Core, Noise},
    graph::{kd_tree::KdTree, ms_tree::MSTree, sl_tree::SLTree},
    shared::{float::Float, point::Point, primitive::Primitive},
};

use super::Classification;
pub struct Hdbscan<P: Point> {
    min_points: usize,
    max_points: usize,
    single_cluster: bool,
    min_samples: usize,
    phantom_data: PhantomData<P>,
}

impl<P: Point> Hdbscan<P> {
    pub fn new(
        min_points: usize,
        max_points: usize,
        single_cluster: bool,
        min_samples: usize,
    ) -> Self {
        let phantom_data = PhantomData;
        Self {
            min_points,
            max_points,
            single_cluster,
            min_samples,
            phantom_data,
        }
    }

    pub fn cluster(&self, data: &[P]) -> Vec<Classification> {
        let now = SystemTime::now();
        let core_distances = self.kd_core_distances(data);
        dbg!(now.elapsed());
        let now = SystemTime::now();

        let ms_tree = MSTree::new(data, &core_distances);
        dbg!(now.elapsed());
        let now = SystemTime::now();
        let sl_tree = SLTree::new(&ms_tree);
        let condensed_tree = self.condense_tree(&sl_tree, data.len());
        dbg!(now.elapsed());
        let now = SystemTime::now();
        let winning_clusters = self.winning_clusters(&condensed_tree, data.len());
        dbg!(now.elapsed());
        let now = SystemTime::now();

        let out = self.label_data(&winning_clusters, &condensed_tree, data.len());
        dbg!(now.elapsed());
        out
    }

    fn kd_core_distances(&self, data: &[P]) -> Vec<P::Primitive> {
        let capacity = (data.len() as f32).sqrt() as usize;
        let mut kd_tree = KdTree::new(capacity);
        data.into_iter()
            .enumerate()
            .for_each(|(i, point)| kd_tree.add(point.clone(), i));
        data.into_iter()
            .map(|point| {
                kd_tree
                    .nearest(point, self.min_samples)
                    .into_iter()
                    .last()
                    .unwrap()
                    .0
            })
            .collect()
    }

    fn condense_tree(&self, sl_tree: &SLTree<P>, size: usize) -> Vec<CondensedNode<P>> {
        let top_node = (size - 1) * 2;
        let node_indices = Self::search_sl_tree(sl_tree, top_node, size);

        let mut new_node_indices = vec![0; top_node + 1];
        new_node_indices[top_node] = size;
        let mut next_parent_id = size + 1;

        let mut visited = vec![false; node_indices.len()];
        let mut condensed_tree = Vec::new();
        node_indices.into_iter().for_each(|node_idx| {
            if !visited[node_idx] && (node_idx >= size) {
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

                let is_left_cluster = left_child_size > self.min_points;
                let is_right_cluster = right_child_size > self.min_points;

                match (is_left_cluster, is_right_cluster) {
                    (true, true) => {
                        [left_child_idx, right_child_idx]
                            .into_iter()
                            .zip([left_child_size, right_child_size].into_iter())
                            .for_each(|(child_idx, child_size)| {
                                new_node_indices[child_idx] = next_parent_id;
                                next_parent_id += 1;
                                condensed_tree.push(CondensedNode::new(
                                    new_node_indices[child_idx],
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

    fn search_sl_tree(sl_tree: &SLTree<P>, root: usize, size: usize) -> Vec<usize> {
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
        sl_tree: &SLTree<P>,
        condensed_tree: &mut Vec<CondensedNode<P>>,
        visited: &mut [bool],
        lambda: P::Primitive,
        size: usize,
    ) {
        Self::search_sl_tree(sl_tree, node_idx, size)
            .into_iter()
            .for_each(|child_id| {
                if child_id < size {
                    condensed_tree.push(CondensedNode::new(child_id, new_node_idx, lambda, 1));
                }
                visited[child_id] = true;
            });
    }

    fn winning_clusters(&self, condensed_tree: &[CondensedNode<P>], size: usize) -> Vec<usize> {
        let n_clusters = condensed_tree.len() - size + 1;
        let stabilities = (0..size)
            .filter(|n| self.single_cluster || *n != 0)
            .map(|n| size + n)
            .map(|cluster_id| {
                (
                    cluster_id,
                    RefCell::new(Self::calc_stability(cluster_id, condensed_tree, n_clusters)),
                )
            })
            .collect::<BTreeMap<usize, RefCell<P::Primitive>>>();
        let mut selected_clusters = stabilities
            .keys()
            .map(|id| (*id, false))
            .collect::<HashMap<usize, bool>>();
        for (cluster_idx, stability) in stabilities.iter().rev() {
            let combined_child_stability =
                Self::intermediate_child_clusters(*cluster_idx, condensed_tree, size)
                    .iter()
                    .map(|node| {
                        *stabilities
                            .get(&node.node_idx)
                            .unwrap_or(&RefCell::new(P::Primitive::ZERO))
                            .borrow()
                    })
                    .fold(P::Primitive::ZERO, |acc, n| acc + n);
            if *stability.borrow() > combined_child_stability
                && self.get_cluster_size(*cluster_idx, condensed_tree, size) < self.max_points
            {
                *selected_clusters.get_mut(&cluster_idx).unwrap() = true;

                Self::find_child_clusters(&cluster_idx, condensed_tree, size)
                    .into_iter()
                    .for_each(|node_idx| {
                        let is_selected = selected_clusters.get(&node_idx);
                        if let Some(true) = is_selected {
                            *selected_clusters.get_mut(&node_idx).unwrap() = false;
                        }
                    });
            } else {
                stabilities
                    .get(&cluster_idx)
                    .unwrap()
                    .replace(combined_child_stability);
            }
        }
        selected_clusters
            .into_iter()
            .filter(|(_x, keep)| *keep)
            .map(|(id, _)| id)
            .collect()
    }

    fn calc_stability(
        cluster_idx: usize,
        condensed_tree: &[CondensedNode<P>],
        size: usize,
    ) -> P::Primitive {
        let lambda = if cluster_idx == size {
            P::Primitive::ZERO
        } else {
            condensed_tree
                .into_iter()
                .find(|node| node.node_idx == cluster_idx)
                .map(|node| node.lambda)
                .unwrap_or(P::Primitive::ZERO)
        };

        condensed_tree
            .into_iter()
            .filter(|node| node.parent_node_idx == cluster_idx)
            .map(|node| (node.lambda - lambda) * P::Primitive::usize(node.size))
            .fold(P::Primitive::ZERO, |acc, n| acc + n)
    }

    fn intermediate_child_clusters(
        cluster_idx: usize,
        condensed_tree: &[CondensedNode<P>],
        size: usize,
    ) -> Vec<&CondensedNode<P>> {
        condensed_tree
            .into_iter()
            .filter(|node| node.parent_node_idx == cluster_idx)
            .filter(|node| node.node_idx >= size)
            .collect()
    }

    fn find_child_clusters(
        root_node_idx: &usize,
        condensed_tree: &[CondensedNode<P>],
        size: usize,
    ) -> Vec<usize> {
        let mut process_queue = VecDeque::from([root_node_idx]);
        let mut child_clusters = Vec::new();

        while !process_queue.is_empty() {
            let current_node_id = match process_queue.pop_front() {
                Some(node_id) => node_id,
                None => break,
            };

            for node in condensed_tree {
                if node.node_idx < size {
                    continue;
                }
                if node.parent_node_idx == *current_node_id {
                    child_clusters.push(node.node_idx);
                    process_queue.push_back(&node.node_idx);
                }
            }
        }
        child_clusters
    }

    fn label_data(
        &self,
        wining_clusters: &[usize],
        condensed_tree: &[CondensedNode<P>],
        size: usize,
    ) -> Vec<Classification> {
        let mut labels = vec![-1; size];

        for (current_cluster_idx, cluster_idx) in wining_clusters.into_iter().enumerate() {
            let node_size = self.get_cluster_size(*cluster_idx, condensed_tree, size);
            self.find_child_samples(*cluster_idx, node_size, condensed_tree, size)
                .into_iter()
                .for_each(|id| labels[id] = current_cluster_idx as isize);
        }
        labels
            .into_iter()
            .map(|l| if l == -1 { Noise } else { Core(l as usize) })
            .collect()
    }

    fn get_cluster_size(
        &self,
        cluster_idx: usize,
        condensed_tree: &[CondensedNode<P>],
        size: usize,
    ) -> usize {
        if self.single_cluster && cluster_idx == size {
            condensed_tree
                .into_iter()
                .filter(|node| node.node_idx >= size)
                .filter(|node| node.parent_node_idx == cluster_idx)
                .map(|node| node.size)
                .sum()
        } else {
            condensed_tree
                .into_iter()
                .find(|node| node.node_idx == cluster_idx)
                .map(|node| node.size)
                .unwrap_or(1)
        }
    }

    fn find_child_samples(
        &self,
        root_node_idx: usize,
        node_size: usize,
        condensed_tree: &[CondensedNode<P>],
        size: usize,
    ) -> Vec<usize> {
        let mut process_queue = VecDeque::from([root_node_idx]);
        let mut child_nodes = Vec::with_capacity(node_size);

        while !process_queue.is_empty() {
            let current_node_idx = match process_queue.pop_front() {
                Some(node_idx) => node_idx,
                None => break,
            };
            for node in condensed_tree {
                if node.parent_node_idx == current_node_idx {
                    if node.node_idx < size {
                        //this line is concerning
                        if !self.single_cluster && current_node_idx == size {
                            continue;
                        }
                        child_nodes.push(node.node_idx);
                    } else {
                        process_queue.push_back(node.node_idx);
                    }
                }
            }
        }
        child_nodes
    }
}
struct CondensedNode<P: Point> {
    node_idx: usize,
    parent_node_idx: usize,
    lambda: P::Primitive,
    size: usize,
}

impl<P: Point> CondensedNode<P> {
    pub fn new(node_idx: usize, parent_node_idx: usize, lambda: P::Primitive, size: usize) -> Self {
        Self {
            node_idx,
            parent_node_idx,
            lambda,
            size,
        }
    }
}

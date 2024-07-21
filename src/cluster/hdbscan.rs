use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, VecDeque},
};

use crate::{
    cluster::Classification::{Core, Noise},
    graph::{
        kd_tree::KdTree,
        ms_tree::MSTree,
        sl_tree::SLTree,
    },
    shared::{float::Float, point::Point},
};

use super::Classification;
pub struct HdbscanParameters {
    min_points: usize,
    max_points: usize,
    single_cluster: bool,
    min_samples: usize,
}
trait Hdbscan<T: Float, const N: usize> {
    fn cluster(input: &Self, parameters: &HdbscanParameters) -> Vec<Classification>
    where
        Self: Sized;

    fn kd_core_distances(input: &Self, parameters: &HdbscanParameters) -> Vec<T>;

    fn condense_tree(
        sl_tree: &SLTree<T>,
        size: usize,
        parameters: &HdbscanParameters,
    ) -> Vec<CondensedNode<T>>;

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

    fn calc_stability(cluster_idx: usize, condensed_tree: &[CondensedNode<T>], size: usize) -> T;

    fn winning_clusters(
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
    ) -> Vec<usize>;

    fn intermediate_child_clusters(
        cluster_idx: usize,
        condensed_tree: &[CondensedNode<T>],
        size: usize,
    ) -> Vec<&CondensedNode<T>>;

    fn find_child_clusters(
        root_node_idx: &usize,
        condensed_tree: &[CondensedNode<T>],
        size: usize,
    ) -> Vec<usize>;

    fn label_data(
        wining_clusters: &[usize],
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
    ) -> Vec<Classification>;

    fn get_cluster_size(
        cluster_idx: usize,
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
    ) -> usize;

    fn find_child_samples(
        root_node_idx: usize,
        node_size: usize,
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
    ) -> Vec<usize>;
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

impl<T: Float, const N: usize> Hdbscan<T, N> for Vec<Point<T, N>> {
    fn cluster(input: &Self, parameters: &HdbscanParameters) -> Vec<Classification>
    where
        Self: Sized,
    {
        let core_distances = Self::kd_core_distances(input, parameters);
        let ms_tree = MSTree::new(input, &core_distances);
        let sl_tree = SLTree::new(&ms_tree);
        let condensed_tree = Self::condense_tree(&sl_tree, input.len(), parameters);
        let winning_clusters = Self::winning_clusters(&condensed_tree, input.len(), parameters);

        Self::label_data(&winning_clusters, &condensed_tree, input.len(), parameters)
    }

    fn kd_core_distances(input: &Self, parameters: &HdbscanParameters) -> Vec<T> {
        let capacity = (input.len() as f32).sqrt() as usize;
        let mut kd_tree = KdTree::<T, N>::new(capacity);
        input
            .iter()
            .enumerate()
            .for_each(|(i, point)| kd_tree.add(point.clone(), i));
        input
            .iter()
            .map(|point| {
                kd_tree
                    .nearest(point, parameters.min_samples)
                    .iter()
                    .last()
                    .unwrap()
                    .0
            })
            .collect()
    }

    fn condense_tree(
        sl_tree: &SLTree<T>,
        size: usize,
        parameters: &HdbscanParameters,
    ) -> Vec<CondensedNode<T>> {
        let top_node = (size - 1) * 2;
        let node_indices = Self::search_sl_tree(sl_tree, top_node, size);

        let mut new_node_indices = vec![0; top_node + 1];
        new_node_indices[top_node] = size;
        let mut next_parent_id = size + 1;

        let mut visited = vec![false; node_indices.len()];
        let mut condensed_tree = Vec::new();
        node_indices.iter().for_each(|&node_idx| {
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

                let is_left_cluster = left_child_size > parameters.min_points;
                let is_right_cluster = right_child_size > parameters.min_points;

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

    fn winning_clusters(
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
    ) -> Vec<usize> {
        let n_clusters = condensed_tree.len() - size + 1;
        let stabilities = (0..size)
            .filter(|n| parameters.single_cluster || *n != 0)
            .map(|n| size + n)
            .map(|cluster_id| {
                (
                    cluster_id,
                    RefCell::new(Self::calc_stability(cluster_id, condensed_tree, n_clusters)),
                )
            })
            .collect::<BTreeMap<usize, RefCell<T>>>();
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
                            .unwrap_or(&RefCell::new(T::ZERO))
                            .borrow()
                    })
                    .fold(T::ZERO, |acc, n| acc + n);
            if *stability.borrow() > combined_child_stability
                && Self::get_cluster_size(*cluster_idx, condensed_tree, size, parameters)
                    < parameters.max_points
            {
                //wjhy set it to true then check...
                *selected_clusters.get_mut(cluster_idx).unwrap() = true;

                Self::find_child_clusters(cluster_idx, condensed_tree, size)
                    .iter()
                    .for_each(|node_idx| {
                        let is_selected = selected_clusters.get(node_idx);
                        if let Some(true) = is_selected {
                            *selected_clusters.get_mut(node_idx).unwrap() = false;
                        }
                    });
            } else {
                stabilities
                    .get(cluster_idx)
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

    fn calc_stability(cluster_idx: usize, condensed_tree: &[CondensedNode<T>], size: usize) -> T {
        let lambda = if cluster_idx == size {
            T::ZERO
        } else {
            condensed_tree
                .iter()
                .find(|node| node.node_idx == cluster_idx)
                .map(|node| node.lambda)
                .unwrap_or(T::ZERO)
        };

        condensed_tree
            .iter()
            .filter(|node| node.parent_node_idx == cluster_idx)
            .map(|node| (node.lambda - lambda) * T::usize(node.size))
            .fold(T::ZERO, |acc, n| acc + n)
    }

    fn intermediate_child_clusters(
        cluster_idx: usize,
        condensed_tree: &[CondensedNode<T>],
        size: usize,
    ) -> Vec<&CondensedNode<T>> {
        condensed_tree
            .iter()
            .filter(|node| node.parent_node_idx == cluster_idx)
            .filter(|node| node.node_idx >= size)
            .collect()
    }

    fn find_child_clusters(
        root_node_idx: &usize,
        condensed_tree: &[CondensedNode<T>],
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
        wining_clusters: &[usize],
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
    ) -> Vec<Classification> {
        let mut labels = vec![-1; size];
        for (current_cluster_idx, cluster_idx) in wining_clusters.iter().enumerate() {
            let node_size = Self::get_cluster_size(*cluster_idx, condensed_tree, size, parameters);
            Self::find_child_samples(*cluster_idx, node_size, condensed_tree, size, parameters)
                .into_iter()
                .for_each(|id| labels[id] = current_cluster_idx as isize);
        }
        labels
            .iter()
            .map(|l| if *l == -1 { Noise } else { Core(*l as usize) })
            .collect()
    }

    fn get_cluster_size(
        cluster_idx: usize,
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
    ) -> usize {
        if parameters.single_cluster && cluster_idx == size {
            condensed_tree
                .iter()
                .filter(|node| node.node_idx >= size)
                .filter(|node| node.parent_node_idx == cluster_idx)
                .map(|node| node.size)
                .sum()
        } else {
            condensed_tree
                .iter()
                .find(|node| node.node_idx == cluster_idx)
                .map(|node| node.size)
                .unwrap_or(1)
        }
    }

    fn find_child_samples(
        root_node_idx: usize,
        node_size: usize,
        condensed_tree: &[CondensedNode<T>],
        size: usize,
        parameters: &HdbscanParameters,
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
                        if parameters.single_cluster && current_node_idx == size {
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

#[cfg(test)]
mod hdbscan_tests {
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn hdbscan_simple() {
        let data = vec![
            Point::new([9.308548692822459, 2.1673586347139224]),
            Point::new([-5.6424039931897765, -1.9620561766472002]),
            Point::new([-9.821995596375428, -3.1921112766174997]),
            Point::new([-4.992109362834896, -2.0745015313494455]),
            Point::new([10.107315875917662, 2.4489015959094216]),
            Point::new([-7.962477597931141, -5.494741864480315]),
            Point::new([10.047917462523671, 5.1631966716389766]),
            Point::new([-5.243921934674187, -2.963359100733349]),
            Point::new([-9.940544426622527, -3.2655473073528816]),
            Point::new([8.30445373000034, 2.129694332932624]),
            Point::new([-9.196460281784482, -3.987773678358418]),
            Point::new([-10.513583123594056, -2.5364233580562887]),
            Point::new([9.072668506714033, 3.405664632524281]),
            Point::new([-7.031861004012987, -2.2616818331210844]),
            Point::new([9.627963795272553, 4.502533177849574]),
            Point::new([-10.442760023564471, -5.0830680881481065]),
            Point::new([8.292151321984209, 3.8776876670218834]),
            Point::new([-6.51560033683665, -3.8185628318207585]),
            Point::new([-10.887633624071544, -4.416570704487158]),
            Point::new([-9.465804800021168, -2.2222090878656884]),
        ];
        let parameters = HdbscanParameters {
            min_points: 3,
            max_points: 20,
            single_cluster: false,
            min_samples: 5,
        };
        let mask = <Vec<Point<f32, 2>> as Hdbscan<f32, 2>>::cluster(&data, &parameters);
        let known_mask = if mask[0] == Core(0) {
            vec![
                Core(0),
                Core(1),
                Core(1),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(1),
                Core(0),
                Core(1),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(1),
                Core(1),
            ]
        } else {
            vec![
                Core(1),
                Core(0),
                Core(0),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(0),
                Core(1),
                Core(0),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(1),
                Core(0),
                Core(0),
                Core(0),
            ]
        };
        mask.iter().zip(known_mask.iter()).for_each(|(m, k)| {
            assert!(*m == *k);
        });
    }

    #[test]
    fn hdbscan_time() {
        let mut data = Vec::new();
        (0..2000).for_each(|i| data.push(Point::new([f32::usize(i), f32::usize(i + 1)])));
        let now = SystemTime::now();
        let parameters = HdbscanParameters {
            min_points: 2,
            max_points: 1000,
            single_cluster: false,
            min_samples: 32,
        };
        let _ = <Vec<Point<f32, 2>> as Hdbscan<f32, 2>>::cluster(&data, &parameters);
        let _ = dbg!(now.elapsed()); //about twice as fast as python.. but could it be better?
    }
}

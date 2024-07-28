use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, VecDeque}, marker::PhantomData,
};

use crate::{
    cluster::Classification::{Core, Noise},
    graph::{kd_tree::KdTree, ms_tree::MSTree, sl_tree::SLTree},
    shared::{float::Float, real::Real, point::Point},
};

use super::Classification;
pub struct Hdbscan<P:Point> {
    min_points: usize,
    max_points: usize,
    single_cluster: bool,
    min_samples: usize,
    phantom_data: PhantomData<P>
}

impl<P:Point> Hdbscan<P> {
    pub fn new(min_points: usize, max_points: usize, single_cluster: bool, min_samples: usize) -> Self{
        let phantom_data = PhantomData;
        Self {
            min_points,
            max_points,
            single_cluster,
            min_samples,
            phantom_data,
        }
    }

    pub fn cluster(&self,data: &[P]) -> Vec<Classification>{
        let core_distances = self.kd_core_distances(data);
        let ms_tree = MSTree::new(data, &core_distances);
        let sl_tree = SLTree::new(&ms_tree);
        let condensed_tree =self.condense_tree(&sl_tree, data.len());
        let winning_clusters = self.winning_clusters(&condensed_tree, data.len());

        self.label_data(&winning_clusters, &condensed_tree, data.len())
    }

    fn kd_core_distances(&self,data: &[P]) -> Vec<P::Primitive> {
        let capacity = (data.len() as f32).sqrt() as usize;
        let mut kd_tree = KdTree::new(capacity);
        data
            .iter()
            .enumerate()
            .for_each(|(i, point)| kd_tree.add(point.clone(), i));
        data
            .iter()
            .map(|point| {
                kd_tree
                    .nearest(point, self.min_samples)
                    .iter()
                    .last()
                    .unwrap()
                    .0
            })
            .collect()
    }

    fn condense_tree(&self,
        sl_tree: &SLTree<P>,
        size: usize,
    ) -> Vec<CondensedNode<P>> {
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

                let is_left_cluster = left_child_size > self.min_points;
                let is_right_cluster = right_child_size > self.min_points;

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
            .iter()
            .for_each(|&child_id| {
                if child_id < size {
                    condensed_tree.push(CondensedNode::new(child_id, new_node_idx, lambda, 1));
                }
                visited[child_id] = true;
            });
    }

    fn winning_clusters(
        &self,
        condensed_tree: &[CondensedNode<P>],
        size: usize,
    ) -> Vec<usize> {
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
                && self.get_cluster_size(*cluster_idx, condensed_tree, size)
                    < self.max_points
            {
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

    fn calc_stability(cluster_idx: usize, condensed_tree: &[CondensedNode<P>], size: usize) -> P::Primitive {
        let lambda = if cluster_idx == size {
            P::Primitive::ZERO
        } else {
            condensed_tree
                .iter()
                .find(|node| node.node_idx == cluster_idx)
                .map(|node| node.lambda)
                .unwrap_or(P::Primitive::ZERO)
        };

        condensed_tree
            .iter()
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
            .iter()
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

        for (current_cluster_idx, cluster_idx) in wining_clusters.iter().enumerate() {
            let node_size = self.get_cluster_size(*cluster_idx, condensed_tree, size);
            self.find_child_samples(*cluster_idx, node_size, condensed_tree, size)
                .into_iter()
                .for_each(|id| labels[id] = current_cluster_idx as isize);
        }
        labels
            .iter()
            .map(|l| if *l == -1 { Noise } else { Core(*l as usize) })
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
#[cfg(test)]
mod hdbscan_tests {
    use crate::shared::point::StaticPoint;
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn hdbscan_simple() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([-5.6424039931897765, -1.9620561766472002]),
            StaticPoint::new([-9.821995596375428, -3.1921112766174997]),
            StaticPoint::new([-4.992109362834896, -2.0745015313494455]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
            StaticPoint::new([-7.962477597931141, -5.494741864480315]),
            StaticPoint::new([10.047917462523671, 5.1631966716389766]),
            StaticPoint::new([-5.243921934674187, -2.963359100733349]),
            StaticPoint::new([-9.940544426622527, -3.2655473073528816]),
            StaticPoint::new([8.30445373000034, 2.129694332932624]),
            StaticPoint::new([-9.196460281784482, -3.987773678358418]),
            StaticPoint::new([-10.513583123594056, -2.5364233580562887]),
            StaticPoint::new([9.072668506714033, 3.405664632524281]),
            StaticPoint::new([-7.031861004012987, -2.2616818331210844]),
            StaticPoint::new([9.627963795272553, 4.502533177849574]),
            StaticPoint::new([-10.442760023564471, -5.0830680881481065]),
            StaticPoint::new([8.292151321984209, 3.8776876670218834]),
            StaticPoint::new([-6.51560033683665, -3.8185628318207585]),
            StaticPoint::new([-10.887633624071544, -4.416570704487158]),
            StaticPoint::new([-9.465804800021168, -2.2222090878656884]),
        ];
        let hdbscan = Hdbscan::new(3,20,false,5);
        let mask = hdbscan.cluster(&data);
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
    fn hdbscan_single_on() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
            StaticPoint::new([10.047917462523671, 5.1631966716389766]),
            StaticPoint::new([8.30445373000034, 2.129694332932624]),
            StaticPoint::new([9.072668506714033, 3.405664632524281]),
            StaticPoint::new([9.627963795272553, 4.502533177849574]),
            StaticPoint::new([8.292151321984209, 3.8776876670218834]),
            StaticPoint::new([10.308548692822459, 2.1673586347139224]),
            StaticPoint::new([11.107315875917662, 2.4489015959094216]),
            StaticPoint::new([11.047917462523671, 5.1631966716389766]),
            StaticPoint::new([9.30445373000034, 2.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
            StaticPoint::new([9.308548692822459, 3.1673586347139224]),
            StaticPoint::new([10.107315875917662, 3.4489015959094216]),
            StaticPoint::new([10.047917462523671, 4.1631966716389766]),
            StaticPoint::new([8.30445373000034, 3.129694332932624]),
            StaticPoint::new([9.072668506714033, 4.405664632524281]),
            StaticPoint::new([9.627963795272553, 5.502533177849574]),
            StaticPoint::new([8.292151321984209, 4.8776876670218834]),
            StaticPoint::new([10.308548692822459, 3.1673586347139224]),
            StaticPoint::new([11.107315875917662, 3.4489015959094216]),
            StaticPoint::new([11.047917462523671, 4.1631966716389766]),
            StaticPoint::new([9.30445373000034, 3.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
        ];

        let hdbscan = Hdbscan::new(3,20,true,5);
        let mask = hdbscan.cluster(&data);
        mask.iter().for_each(|m| {
            assert!(*m == Core(0));
        });
    }
    #[test]
    fn hdbscan_single_off() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
            StaticPoint::new([10.047917462523671, 5.1631966716389766]),
            StaticPoint::new([8.30445373000034, 2.129694332932624]),
            StaticPoint::new([9.072668506714033, 3.405664632524281]),
            StaticPoint::new([9.627963795272553, 4.502533177849574]),
            StaticPoint::new([8.292151321984209, 3.8776876670218834]),
            StaticPoint::new([10.308548692822459, 2.1673586347139224]),
            StaticPoint::new([11.107315875917662, 2.4489015959094216]),
            StaticPoint::new([11.047917462523671, 5.1631966716389766]),
            StaticPoint::new([9.30445373000034, 2.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
            StaticPoint::new([9.308548692822459, 3.1673586347139224]),
            StaticPoint::new([10.107315875917662, 3.4489015959094216]),
            StaticPoint::new([10.047917462523671, 4.1631966716389766]),
            StaticPoint::new([8.30445373000034, 3.129694332932624]),
            StaticPoint::new([9.072668506714033, 4.405664632524281]),
            StaticPoint::new([9.627963795272553, 5.502533177849574]),
            StaticPoint::new([8.292151321984209, 4.8776876670218834]),
            StaticPoint::new([10.308548692822459, 3.1673586347139224]),
            StaticPoint::new([11.107315875917662, 3.4489015959094216]),
            StaticPoint::new([11.047917462523671, 4.1631966716389766]),
            StaticPoint::new([9.30445373000034, 3.129694332932624]),
            StaticPoint::new([10.072668506714033, 4.405664632524281]),
            StaticPoint::new([10.627963795272553, 5.502533177849574]),
            StaticPoint::new([9.292151321984209, 4.8776876670218834]),
        ];
        let hdbscan = Hdbscan::new(3,20,false,5);
        let mask = hdbscan.cluster(&data);
        dbg!(&mask);
        let known_mask = vec![
            Noise,
            Noise,
            Core(0),
            Noise,
            Noise,
            Core(0),
            Noise,
            Noise,
            Noise,
            Noise,
            Noise,
            Core(0),
            Noise,
            Core(0),
            Noise,
            Core(0),
            Core(0),
            Noise,
            Core(0),
            Noise,
            Noise,
            Noise,
            Noise,
            Noise,
            Noise,
            Core(0),
            Noise,
            Core(0),
        ];

        mask.iter().zip(known_mask.iter()).for_each(|(m, k)| {
            assert!(*m == *k);
        });
    }

    #[test]
    fn hdbscan_time() {
        let mut data = Vec::new();
        (0..20000).for_each(|i| data.push(StaticPoint::new([f32::usize(i), f32::usize(i + 1)])));
        let now = SystemTime::now();
        let hdbscan = Hdbscan::new(2,1000,false,32);
        let _ = hdbscan.cluster(&data);
        let _ = dbg!(now.elapsed()); //about twice as fast as python.. but could it be better?


        let mut data = Vec::new();
        (0..20000).for_each(|i| data.push(i as f32));
        let now = SystemTime::now();
        let hdbscan = Hdbscan::new(2,1000,false,32);
        let _ = hdbscan.cluster(&data);
        let _ = dbg!(now.elapsed()); //faster, but not by much
    }
}

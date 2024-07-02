//I don't like this version very much, it needs to go back in the oven for a bit
/*
    issues: 
        seems really complicated
        doesn't divide to core/edge/noise
        couple of repeated work locations
        couple of uncommon practices (refcell)
        variables didn't track through very well
        the kdtree vs balltree issue 

*/
use std::{cell::RefCell, collections::{BTreeMap, HashMap, VecDeque}};

use crate::{
    cluster::Classification::{Core, Edge, Noise},
    geometric::{kd_tree::KdTree, ms_tree::MSTree, sl_tree::SLTree},
    shared::float::Float,
};

use super::Classification;

trait HDBSCAN<T: Float> {
    fn cluster(input: &Self, epsilon: &T, min_points: usize) -> Vec<isize>
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

    fn calc_stability(cluster_idx:usize, condensed_tree: &Vec<CondensedNode<T>>, size: usize) -> T;

    fn winning_clusters(condensed_tree: &Vec<CondensedNode<T>>, size: usize)-> Vec<usize> ;

    fn intermediate_child_clusters(cluster_idx:usize, condensed_tree: &Vec<CondensedNode<T>>,size:usize) -> Vec<&CondensedNode<T>>;

    fn find_child_clusters(root_node_idx: &usize,condensed_tree: &Vec<CondensedNode<T>>,size:usize) -> Vec<usize>;

    fn label_data(wining_clusters: &Vec<usize>, condensed_tree: &Vec<CondensedNode<T>>,size:usize) -> Vec<isize>;

    fn get_cluster_size(cluster_idx:usize, condensed_tree: &Vec<CondensedNode<T>>) -> usize;

    fn find_child_samples(
        root_node_idx: usize,
        node_size: usize,
        condensed_tree: &Vec<CondensedNode<T>>,
        size:usize
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

impl<T: Float> HDBSCAN<T> for Vec<Vec<T>> {
    fn cluster(input: &Self, epsilon: &T, min_points: usize) -> Vec<isize>
    where
        Self: Sized,
    {
        let core_distances = Self::kd_core_distances(input, min_points);
        let ms_tree = MSTree::new(input, &core_distances);
        let sl_tree = SLTree::new(&ms_tree);
        let condensed_tree = Self::condense_tree(&sl_tree, input.len(), min_points);
        let winning_clusters = Self::winning_clusters(&condensed_tree,input.len());

        Self::label_data(&winning_clusters, &condensed_tree, input.len())
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

    fn winning_clusters(condensed_tree: &Vec<CondensedNode<T>>, size: usize)-> Vec<usize> {
        let n_clusters = condensed_tree.len() - size +1;
        let stabilities =         (0..size)
        .map(|n| size + n)
        .map(|cluster_id| (
            cluster_id, RefCell::new(Self::calc_stability(cluster_id, &condensed_tree,n_clusters ))
        ))
        .collect::<BTreeMap<usize, RefCell<T>>>();
    let mut selected_clusters = stabilities.keys().map(|id| (id.clone(),false)).collect::<HashMap<usize, bool>>();
    for(cluster_idx, stability) in stabilities.iter().rev(){
        let combined_child_stability = Self::intermediate_child_clusters(*cluster_idx,condensed_tree,size).iter()
        .map(|node| {
            stabilities.get(&node.node_idx)
                .unwrap_or(&RefCell::new(T::ZERO)).borrow().clone()
        })
        .fold(T::ZERO, |acc,n| acc+n);
        if *stability.borrow() > combined_child_stability{
            //wjhy set it to true then check...
            *selected_clusters.get_mut(&cluster_idx).unwrap() = true;

            Self::find_child_clusters(&cluster_idx, &condensed_tree,size).iter().for_each(|node_idx|{
                let is_selected = selected_clusters.get(node_idx);
                if let Some(true) = is_selected {
                    *selected_clusters.get_mut(node_idx).unwrap() = false;
                }
            });
        }else{
            stabilities.get(&cluster_idx).unwrap().replace(combined_child_stability);
        }
    }
    selected_clusters.into_iter().filter(|(_x,keep)|*keep).map(|(id,_)| id).collect()
    }

    fn calc_stability(cluster_idx:usize, condensed_tree: &Vec<CondensedNode<T>>,size: usize) -> T{
        let lambda = if cluster_idx==size{
            T::ZERO
        }else{
            condensed_tree.iter().find(|node| node.node_idx == cluster_idx).map(|node| node.lambda).unwrap_or(T::ZERO)
        };

        condensed_tree.iter().filter(|node| node.parent_node_idx == cluster_idx).map(|node| (node.lambda-lambda)*T::usize(node.size)).fold(T::ZERO, |acc, n| acc+n)
    }

    fn intermediate_child_clusters(cluster_idx:usize, condensed_tree: &Vec<CondensedNode<T>>,size:usize) -> Vec<&CondensedNode<T>>{
        condensed_tree.iter().filter(|node| node.parent_node_idx == cluster_idx).filter(|node| node.node_idx >= size).collect()
    }

    fn find_child_clusters(root_node_idx: &usize,condensed_tree: &Vec<CondensedNode<T>>,size:usize) -> Vec<usize>{
        let mut process_queue = VecDeque::from([root_node_idx]);
        let mut child_clusters= Vec::new();

        while !process_queue.is_empty() {
            let current_node_id = match process_queue.pop_front() {
                Some(node_id) => node_id,
                None => break,
            };

            for node in condensed_tree {
                if node.node_idx<size {
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

    fn label_data(wining_clusters: &Vec<usize>, condensed_tree: &Vec<CondensedNode<T>>,size:usize) -> Vec<isize>{
        let mut current_cluster_idx = 0;
        let mut labels = vec![-1; size];
        for cluster_idx in wining_clusters{
            let node_size = Self::get_cluster_size(*cluster_idx, condensed_tree);
            Self::find_child_samples(*cluster_idx, node_size,&condensed_tree,size).into_iter().for_each(|id| labels[id] = current_cluster_idx);
            current_cluster_idx+=1;
        }
        labels
    }

    fn get_cluster_size(cluster_idx:usize, condensed_tree: &Vec<CondensedNode<T>>) -> usize{
        condensed_tree.iter().find(|node| node.node_idx ==cluster_idx).map(|node| node.size).unwrap_or(1)
    }

    fn find_child_samples(
        root_node_idx: usize,
        node_size: usize,
        condensed_tree:  &Vec<CondensedNode<T>>,
        size:usize
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
                    if node.node_idx<size {
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

use crate::shared::float::Float;

use super::ms_tree::MSTree;

pub struct SLTreeNode<T: Float> {
    pub left_node_idx: usize,
    pub right_node_idx: usize,
    pub distance: T,
    pub size: usize,
}

pub struct SLTree<T: Float> {
    pub sl_tree_vec: Vec<SLTreeNode<T>>,
}

impl<T: Float> SLTreeNode<T> {
    fn new(left_node_idx: usize, right_node_idx: usize, distance: T, size: usize) -> Self {
        Self {
            left_node_idx,
            right_node_idx,
            distance,
            size,
        }
    }
}

impl<T: Float> SLTree<T> {
    pub fn new(input: &MSTree<T>) -> Self {
        let length = input.ms_tree_vec.len();
        let mut sl_tree_vec = Vec::with_capacity(length - 1);

        let double_length = 2 * length - 1;
        let mut parent = vec![double_length; double_length];
        let mut next_label = length;
        let mut size_vec = ((0..length).map(|_| 1))
            .chain((length..double_length).map(|_| 0))
            .collect::<Vec<_>>();
        input.ms_tree_vec.iter().for_each(|ms_tree_node| {
            let left_child = Self::find(ms_tree_node.left_node_idx, &mut parent);
            let right_child = Self::find(ms_tree_node.right_node_idx, &mut parent);
            let distance = ms_tree_node.distance;

            let size = size_vec[left_child] + size_vec[right_child];

            sl_tree_vec.push(SLTreeNode::new(left_child, right_child, distance, size));

            parent[left_child] = next_label;
            parent[right_child] = next_label;
            size_vec[next_label] = size;
            next_label += 1;
        });

        Self { sl_tree_vec }
    }

    fn find(node_idx: usize, parent: &mut Vec<usize>) -> usize {
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


#[cfg(test)]
mod sl_tree_tests {
    use super::*;

    #[test]
    fn create_tree_static() {



    }

    #[test]
    fn create_tree_dynamic() {todo!()}
}
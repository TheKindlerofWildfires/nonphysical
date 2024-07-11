use crate::shared::{float::Float, point::Point};

pub struct MSTreeNode<T: Float> {
    pub left_node_idx: usize,
    pub right_node_idx: usize,
    pub distance: T,
}

pub struct MSTree<T: Float> {
    pub ms_tree_vec: Vec<MSTreeNode<T>>,
}

impl<T: Float> MSTreeNode<T> {
    fn new(left_node_idx: usize, right_node_idx: usize, distance: T) -> Self {
        Self {
            left_node_idx,
            right_node_idx,
            distance,
        }
    }
}

//prim's algorithm
impl<T: Float> MSTree<T> {
    pub fn new<const N: usize>(input: &Vec<Point<T,N>>,distance_overrides: &Vec<T>) -> Self{
        let samples = input.len();
        let mut in_tree = vec![false; samples];
        let mut distances = vec![T::MAX; samples];

        distances[0] = T::ZERO;

        let mut ms_tree_vec = Vec::with_capacity(samples);
        let mut left_node_idx = 0;
        let mut right_node_idx = 0;

        (1..samples).for_each(|_| {
            in_tree[left_node_idx] = true;
            let mut current_min_dist = T::MAX;
            (0..samples).for_each(|i| {
                if !in_tree[i] {
                    let mutual_reach = Self::mutual_reach(left_node_idx, i, input,&distance_overrides);
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

    fn mutual_reach<const N: usize>(node_a_idx: usize, node_b_idx: usize, input: &Vec<Point<T,N>>,distances: &Vec<T>) -> T{
        let dist_a = distances[node_a_idx];
        let dist_b = distances[node_b_idx];

        let dist = input[node_a_idx].distance(&input[node_b_idx]);

        dist.greater(dist_a).greater(dist_b)
    }

    fn sort(&mut self){
        self.ms_tree_vec.sort_by(|a,b| a.distance.partial_cmp(&b.distance).unwrap())
    }
}
#[cfg(test)]
mod ms_tree_tests {
    use super::*;

    #[test]
    fn create_tree_static() {todo!()}

    #[test]
    fn create_tree_dynamic() {todo!()}
}
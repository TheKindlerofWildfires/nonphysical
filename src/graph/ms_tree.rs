use crate::shared::float::Float;

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
    pub fn new(input: &Vec<Vec<T>>,distance_overrides: &Vec<T>) -> Self{
        let mut in_tree = vec![false; input.len()];
        let mut distances = vec![T::MAX; input.len()];

        distances[0] = T::ZERO;

        let mut ms_tree_vec = Vec::with_capacity(input.len());
        let mut left_node_idx = 0;
        let mut right_node_idx = 0;

        (1..input.len()).for_each(|_| {
            in_tree[left_node_idx] = true;
            let mut current_min_dist = T::MAX;
            (0..input.len()).for_each(|i| {
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

        MSTree { ms_tree_vec }
    }

    fn mutual_reach(node_a_idx: usize, node_b_idx: usize, input: &Vec<Vec<T>>,distances: &Vec<T>) -> T{
        let dist_a = distances[node_a_idx];
        let dist_b = distances[node_b_idx];

        let dist = Self::distance(&input[node_a_idx], &input[node_b_idx]);

        dist.greater(dist_a).greater(dist_b)
    }
    fn distance(a: &Vec<T>, b: &Vec<T>) -> T {
        let out = a
            .iter()
            .zip(b.iter())
            .fold(T::ZERO, |dist, (ap, bp)| dist + (*ap - *bp).norm());
        out
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
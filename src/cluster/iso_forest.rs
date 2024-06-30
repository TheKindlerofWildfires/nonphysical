use crate::{
    cluster::Classification::{Core, Edge, Noise},
    geometric::iso_tree::IsoTree,
    random::pcg::PermutedCongruentialGenerator,
    shared::float::Float,
};

use super::Classification;

trait IsoForest<T: Float> {
    fn cluster(
        input: &Self,
        tree_count: usize,
        samples: usize,
        extension_level: usize,
        seed: usize,
    ) -> (Vec<IsoTree<T>>, T);

    fn score(
        input: &Self,
        trees: Vec<IsoTree<T>>,
        average_path: T,
        core_threshold: T,
        edge_threshold: T,
    ) -> Vec<Classification>;
}

impl<T: Float> IsoForest<T> for Vec<Vec<T>> {
    fn cluster(
        input: &Self,
        tree_count: usize,
        samples: usize,
        extension_level: usize,
        seed: usize,
    ) -> (Vec<IsoTree<T>>, T) {
        debug_assert!(input.len() >= samples);
        let dimension = input[0].len();
        debug_assert!(dimension > extension_level);
        let max_depth = T::usize(samples).ln().to_usize();
        let mut pcg = PermutedCongruentialGenerator::<T>::new(seed as u32, seed as u32);

        let sub_tree_count = input.len() / samples;
        //modification to algorithm: always checks every tree from permutation mod samples,makes it more fair
        let trees = (0..tree_count)
            .flat_map(|_| {
                let mut indices = (0..input.len()).collect();
                pcg.shuffle_usize(&mut indices);
                let sub_trees = indices
                    .chunks(samples)
                    .take(sub_tree_count)
                    .map(|index_chunk| {
                        let tree_data = index_chunk.iter().map(|i| input[*i].clone()).collect();
                        IsoTree::new(&tree_data, max_depth, extension_level, &mut pcg)
                    })
                    .collect::<Vec<_>>();
                sub_trees
            })
            .collect::<Vec<_>>();

        let average_path_length = IsoTree::c_factor(samples);

        (trees, average_path_length)
    }

    fn score(
        input: &Self,
        trees: Vec<IsoTree<T>>,
        average_path: T,
        core_threshold: T,
        edge_threshold: T,
    ) -> Vec<Classification> {
        debug_assert!(core_threshold > edge_threshold);
        input
            .iter()
            .map(|point| {
                let path_length = trees
                    .iter()
                    .map(|tree| IsoTree::<T>::path_length(&tree.root, point))
                    .fold(T::ZERO, |acc, length| acc + length)
                    / T::usize(trees.len());

                let anomaly_score = T::usize(2).powt(&(-path_length / average_path));

                if anomaly_score > core_threshold {
                    Core(0)
                } else if anomaly_score > edge_threshold {
                    Edge(0)
                } else {
                    Noise
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod iso_forest_tests {
    use super::*;

    #[test]
    fn create_forest() {todo!()}

    #[test]
    fn score_forest() {todo!()}

    #[test]
    fn score_novel() {todo!()}
}

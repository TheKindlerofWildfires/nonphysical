use crate::{
    cluster::Classification::{Core, Edge, Noise},
    graph::iso_tree::IsoTree,
    random::pcg::PermutedCongruentialGenerator,
    shared::{float::Float, point::Point,primitive::Primitive},
};

use alloc::vec::Vec;


use super::Classification;

pub struct IsoForest<P: Point> {
    pub trees: Vec<IsoTree<P>>,
    pub average_path: P::Primitive,
    core_threshold: P::Primitive,
    edge_threshold: P::Primitive,
}

impl<P: Point> IsoForest<P> {
    pub fn new(
        seed_data: &[P],
        tree_count: usize,
        samples: usize,
        extension_level: usize,
        seed: usize,
        core_threshold: P::Primitive,
        edge_threshold: P::Primitive,
    ) -> Self {
        debug_assert!(seed_data.len() >= samples);
        debug_assert!(core_threshold < edge_threshold);

        //debug_assert!(N > extension_level);
        let max_depth = P::Primitive::usize(samples).ln().as_usize();
        let mut rng = PermutedCongruentialGenerator::new(seed as u32, (seed + 1) as u32);

        let sub_tree_count =
            seed_data.len() / samples + if seed_data.len() % samples != 0 { 1 } else { 0 };
        let repeats = tree_count / sub_tree_count
            + if tree_count % sub_tree_count != 0 {
                1
            } else {
                0
            };
        //modification to algorithm: always checks every tree from permutation mod samples,makes it more fair
        let trees = (0..repeats)
            .flat_map(|_| {
                let mut indices = (0..seed_data.len()).collect::<Vec<_>>();
                rng.shuffle_usize(&mut indices);
                let sub_trees = indices
                    .chunks(samples)
                    .take(sub_tree_count)
                    .map(|index_chunk| {
                        let tree_data = index_chunk
                            .iter()
                            .map(|i| seed_data[*i])
                            .collect::<Vec<_>>();

                        IsoTree::new(&tree_data, max_depth, extension_level, &mut rng)
                    })
                    .collect::<Vec<_>>();
                sub_trees
            })
            .collect::<Vec<_>>();
        let average_path = IsoTree::<P>::c_factor(samples);

        Self {
            trees,
            average_path,
            core_threshold,
            edge_threshold,
        }
    }

    pub fn cluster(&self, data: &[P]) -> Vec<Classification> {
        self.score(data)
            .into_iter()
            .map(|score| {
                if score < self.core_threshold {
                    Core(0)
                } else if score < self.edge_threshold {
                    Edge(0)
                } else {
                    Noise
                }
            })
            .collect()
    }
    pub fn score(&self, data: &[P]) -> Vec<P::Primitive> {
        data.iter()
            .map(|point| {
                let path_length = self
                    .trees
                    .iter()
                    .map(|tree| IsoTree::path_length(&tree.root, point))
                    .fold(P::Primitive::ZERO, |acc, length| acc + length)
                    / P::Primitive::usize(self.trees.len());
                P::Primitive::usize(2).powf(-path_length / self.average_path)
            })
            .collect()
    }
}
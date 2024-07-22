use crate::{
    cluster::Classification::{Core, Edge, Noise},
    graph::iso_tree::IsoTree,
    random::pcg::PermutedCongruentialGenerator,
    shared::{float::Float, point::Point},
};

use super::Classification;

trait IsoForest<T: Float, const N: usize> {
    fn cluster(
        input: &Self,
        tree_count: usize,
        samples: usize,
        extension_level: usize,
        seed: usize,
    ) -> (Vec<IsoTree<T, N>>, T);
    fn classify(
        input: &Self,
        trees: &[IsoTree<T, N>],
        average_path: T,
        core_threshold: T,
        edge_threshold: T,
    ) -> Vec<Classification>;
    fn score(input: &Self, trees: &[IsoTree<T, N>], average_path: T) -> Vec<T>;
}

impl<T: Float, const N: usize> IsoForest<T, N> for Vec<Point<T, N>> {
    fn cluster(
        input: &Self,
        tree_count: usize,
        samples: usize,
        extension_level: usize,
        seed: usize,
    ) -> (Vec<IsoTree<T, N>>, T) {
        debug_assert!(input.len() >= samples);
        debug_assert!(N > extension_level);
        let max_depth = T::usize(samples).ln().to_usize();
        let mut pcg = PermutedCongruentialGenerator::<T>::new(seed as u32, seed as u32);

        let sub_tree_count = input.len() / samples + if input.len() % samples != 0 { 1 } else { 0 };
        //modification to algorithm: always checks every tree from permutation mod samples,makes it more fair
        let trees = (0..tree_count)
            .flat_map(|_| {
                let mut indices = (0..input.len()).collect::<Vec<_>>();
                pcg.shuffle_usize(&mut indices);
                let sub_trees = indices
                    .chunks(samples)
                    .take(sub_tree_count)
                    .map(|index_chunk| {
                        let tree_data = index_chunk
                            .iter()
                            .map(|i| input[*i].clone())
                            .collect::<Vec<_>>();

                        IsoTree::new(&tree_data, max_depth, extension_level, &mut pcg)
                    })
                    .collect::<Vec<_>>();
                sub_trees
            })
            .collect::<Vec<_>>();
        let average_path_length = IsoTree::<T, N>::c_factor(samples);

        (trees, average_path_length)
    }

    fn classify(
        input: &Self,
        trees: &[IsoTree<T, N>],
        average_path: T,
        core_threshold: T,
        edge_threshold: T,
    ) -> Vec<Classification> {
        debug_assert!(core_threshold < edge_threshold);
        Self::score(input, trees, average_path)
            .iter()
            .map(|score| {
                if *score < core_threshold {
                    Core(0)
                } else if *score < edge_threshold {
                    Edge(0)
                } else {
                    Noise
                }
            })
            .collect()
    }

    fn score(input: &Self, trees: &[IsoTree<T, N>], average_path: T) -> Vec<T> {
        input
            .iter()
            .map(|point| {
                let path_length = trees
                    .iter()
                    .map(|tree| IsoTree::<T, N>::path_length(&tree.root, point))
                    .fold(T::ZERO, |acc, length| acc + length)
                    / T::usize(trees.len());

                T::usize(2).powt(&(-path_length / average_path))
            })
            .collect()
    }
}

#[cfg(test)]
mod iso_forest_tests {

    use super::*;

    #[test]
    fn create_forest() {
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
            Point::new([100.0, 100.0]),
            Point::new([-100.0, 100.0]),
            Point::new([100.0, -100.0]),
            Point::new([-100.0, -100.0]),
        ];

        let (trees, average_path) =
            <Vec<Point<f32, 2>> as IsoForest<f32, 2>>::cluster(&data, 20, 10, 0, 1);
        assert!(trees.len() == 60);
        assert!((average_path - 3.7488806).square_norm() < f32::EPSILON);
    }

    #[test]
    fn score_forest() {
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
            Point::new([100.0, 100.0]),
            Point::new([-100.0, 100.0]),
            Point::new([100.0, -100.0]),
            Point::new([-100.0, -100.0]),
        ];

        let (trees, average_path) =
            <Vec<Point<f32, 2>> as IsoForest<f32, 2>>::cluster(&data, 100, 20, 1, 1);
        let scores = <Vec<Point<f32, 2>> as IsoForest<f32, 2>>::score(&data, &trees, average_path);
        let known_scores = vec![
            0.56223166, 0.5465164, 0.5488712, 0.5472224, 0.56223166, 0.55240774, 0.5671994,
            0.548802, 0.5488712, 0.5590197, 0.5484069, 0.5511991, 0.5628488, 0.5473299, 0.5628287,
            0.5521582, 0.5619537, 0.54854256, 0.55106795, 0.5512792, 0.70974106, 0.70208126,
            0.6949086, 0.68195057,
        ];
        scores
            .iter()
            .zip(known_scores.iter())
            .for_each(|(s, ks)| assert!((*s - *ks).square_norm() < f32::EPSILON));
    }

    #[test]
    fn score_novel() {
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

        let noise_data = vec![
            Point::new([100.0, 100.0]),
            Point::new([-100.0, 100.0]),
            Point::new([100.0, -100.0]),
            Point::new([-100.0, -100.0]),
        ];

        let (trees, average_path) =
            <Vec<Point<f32, 2>> as IsoForest<f32, 2>>::cluster(&data, 60, 20, 1, 1);
        let scores =
            <Vec<Point<f32, 2>> as IsoForest<f32, 2>>::score(&noise_data, &trees, average_path);
        let known_scores = vec![0.6438421, 0.58029604, 0.60804564, 0.5543376];

        scores.iter().zip(known_scores.iter()).for_each(|(s, ks)| {
            assert!((*s - *ks).square_norm() < f32::EPSILON);
        });

        let combined_data = data.iter().chain(noise_data.iter()).cloned().collect();

        let scores =
            <Vec<Point<f32, 2>> as IsoForest<f32, 2>>::score(&combined_data, &trees, average_path);
        let known_scores = vec![
            0.5693566,
            0.48800576,
            0.47635287,
            0.50720954,
            0.5770133,
            0.5118281,
            0.5752988,
            0.5085732,
            0.47635287,
            0.556946,
            0.48162508,
            0.49083218,
            0.5428835,
            0.47606847,
            0.5543319,
            0.50869894,
            0.55792373,
            0.49518526,
            0.5102761,
            0.4808876,
            0.6438421,
            0.58029604,
            0.60804564,
            0.5543376,
        ];
        scores.iter().zip(known_scores.iter()).for_each(|(s, ks)| {
            dbg!(s,ks);
            assert!((*s - *ks).square_norm() < f32::EPSILON);
        });
    }
}

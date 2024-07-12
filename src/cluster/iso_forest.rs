use crate::{
    cluster::Classification::{Core, Edge, Noise},
    graph::iso_tree::IsoTree,
    random::pcg::PermutedCongruentialGenerator,
    shared::{float::Float, point::Point},
};

use super::Classification;

trait IsoForest<T: Float,const N:usize> {
    fn cluster(
        input: &Self,
        tree_count: usize,
        samples: usize,
        extension_level: usize,
        seed: usize,
    ) -> (Vec<IsoTree<T,N>>, T);

    fn score(
        input: &Self,
        trees: &[IsoTree<T,N>],
        average_path: T,
        core_threshold: T,
        edge_threshold: T,
    ) -> Vec<Classification>;
}

impl<T: Float, const N: usize> IsoForest<T,N> for Vec<Point<T,N>> {
    fn cluster(
        input: &Self,
        tree_count: usize,
        samples: usize,
        extension_level: usize,
        seed: usize,
    ) -> (Vec<IsoTree<T,N>>, T) {
        debug_assert!(input.len() >= samples);
        debug_assert!(N > extension_level);
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
        let average_path_length = IsoTree::<T,N>::c_factor(samples);

        (trees, average_path_length)
    }

    fn score(
        input: &Self,
        trees: &[IsoTree<T,N>],
        average_path: T,
        core_threshold: T,
        edge_threshold: T,
    ) -> Vec<Classification> {
        debug_assert!(core_threshold < edge_threshold);
        input
            .iter()
            .map(|point| {
                let path_length = trees
                    .iter()
                    .map(|tree| IsoTree::<T,N>::path_length(&tree.root, point))
                    .fold(T::ZERO, |acc, length| acc + length)
                    / T::usize(trees.len());

                let anomaly_score = T::usize(2).powt(&(-path_length / average_path));
                if anomaly_score < core_threshold {
                    Core(0)
                } else if anomaly_score < edge_threshold {
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
    use crate::cluster;

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

        let (trees, average_path) = <Vec<Point<f32,2>> as IsoForest<f32,2>>::cluster(&data, 20, 10, 0, 1);
        assert!(trees.len() == 40);
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

        let (trees, average_path) = <Vec<Point<f32,2>> as IsoForest<f32,2>>::cluster(&data, 20, 10, 0, 1);
        let scores =
            <Vec<Point<f32,2>> as IsoForest<f32,2>>::score(&data, &trees, average_path, 0.45, 0.55);

        let known_scores = vec![
            Edge(0),
            Core(0),
            Core(0),
            Core(0),
            Edge(0),
            Core(0),
            Edge(0),
            Core(0),
            Core(0),
            Core(0),
            Core(0),
            Edge(0),
            Edge(0),
            Core(0),
            Edge(0),
            Edge(0),
            Core(0),
            Core(0),
            Edge(0),
            Core(0),
            Noise,
            Noise,
            Noise,
            Noise,
        ];
        scores
            .iter()
            .zip(known_scores.iter())
            .for_each(|(s, ks)|{
                dbg!(s,ks);
                assert!(*s == *ks)
            }
           );
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

        let noise_data =vec![
            Point::new([100.0, 100.0]),
            Point::new([-100.0, 100.0]),
            Point::new([100.0, -100.0]),
            Point::new([-100.0, -100.0]),
        ];

        let (trees, average_path) = <Vec<Point<f32,2>> as IsoForest<f32,2>>::cluster(&data, 20, 10, 0, 1);
        let scores =
            <Vec<Point<f32,2>> as IsoForest<f32,2>>::score(&noise_data, &trees, average_path, 0.45, 0.50);

        let known_scores = vec![Noise, Noise, Noise, Noise];

        scores
            .iter()
            .zip(known_scores.iter())
            .for_each(|(s, ks)| {
                assert!(*s == *ks);});

        let scores = <Vec<Point<f32,2>> as IsoForest<f32,2>>::score(&data, &trees, average_path, 0.45, 0.50);

        dbg!(scores);
        let known_scores = vec![
            Edge(0),
            Core(0),
            Core(0),
            Core(0),
            Edge(0),
            Core(0),
            Edge(0),
            Core(0),
            Core(0),
            Edge(0),
            Core(0),
            Core(0),
            Edge(0),
            Core(0),
            Edge(0),
            Core(0),
            Edge(0),
            Core(0),
            Core(0),
            Core(0),];
    }
}
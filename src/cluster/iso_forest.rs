use crate::{
    cluster::Classification::{Core, Edge, Noise},
    graph::iso_tree::IsoTree,
    random::pcg::PermutedCongruentialGenerator,
    shared::{float::Float, point::Point, real::Real},
};

use super::Classification;

pub struct IsoForest<P: Point> {
    trees: Vec<IsoTree<P>>,
    average_path: P::Primitive,
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
                            .map(|i| seed_data[*i].clone())
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
            .iter()
            .map(|score| {
                if *score < self.core_threshold {
                    Core(0)
                } else if *score < self.edge_threshold {
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

#[cfg(test)]
mod iso_forest_tests {

    use std::time::SystemTime;

    use super::*;

    use crate::shared::point::StaticPoint;

    #[test]
    fn create_forest() {
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
            StaticPoint::new([100.0, 100.0]),
            StaticPoint::new([-100.0, 100.0]),
            StaticPoint::new([100.0, -100.0]),
            StaticPoint::new([-100.0, -100.0]),
        ];

        let iso_forest: IsoForest<StaticPoint<f32, 2>> =
            IsoForest::new(&data, 20, 10, 0, 1, 0.0, 0.1);

        assert!(iso_forest.trees.len() == 21);
        assert!((iso_forest.average_path - 3.7488806).l2_norm() < f32::EPSILON);
    }

    #[test]
    fn score_forest() {
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
            StaticPoint::new([100.0, 100.0]),
            StaticPoint::new([-100.0, 100.0]),
            StaticPoint::new([100.0, -100.0]),
            StaticPoint::new([-100.0, -100.0]),
        ];
        let iso_forest: IsoForest<StaticPoint<f32, 2>> =
            IsoForest::new(&data, 100, 10, 1, 1, 0.0, 0.1);
        let scores = iso_forest.score(&data);
        let known_scores = vec![
            0.5267391, 0.48126653, 0.48582554, 0.48349816, 0.53105694, 0.48471618, 0.53105694,
            0.48126653, 0.48702633, 0.52544886, 0.4834478, 0.49014044, 0.5267391, 0.48193628,
            0.5267391, 0.49014044, 0.52544886, 0.48021436, 0.49307784, 0.48450702, 0.6727004,
            0.6713952, 0.6727004, 0.6713952,
        ];
        scores
            .iter()
            .zip(known_scores.iter())
            .for_each(|(s, ks)| assert!((*s - *ks).l2_norm() < f32::EPSILON));
    }

    #[test]
    fn score_novel() {
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

        let noise_data = vec![
            StaticPoint::new([100.0, 100.0]),
            StaticPoint::new([-100.0, 100.0]),
            StaticPoint::new([100.0, -100.0]),
            StaticPoint::new([-100.0, -100.0]),
        ];
        let iso_forest: IsoForest<StaticPoint<f32, 2>> =
            IsoForest::new(&data, 60, 20, 1, 1, 0.0, 0.1);
        let scores = iso_forest.score(&noise_data);
        let known_scores = vec![0.6145915, 0.56156874, 0.6145915, 0.56156874];
        scores.iter().zip(known_scores.iter()).for_each(|(s, ks)| {
            assert!((*s - *ks).l2_norm() < f32::EPSILON);
        });

        let combined_data = data
            .iter()
            .chain(noise_data.iter())
            .cloned()
            .collect::<Vec<_>>();

        let scores = iso_forest.score(&combined_data);
        let known_scores = vec![
            0.5114004, 0.47956806, 0.45588937, 0.50679964, 0.51736003, 0.45492068, 0.51736003,
            0.49616563, 0.45588937, 0.5175009, 0.454131, 0.470897, 0.512141, 0.4601812, 0.5136253,
            0.470897, 0.5175009, 0.4714252, 0.47527155, 0.45588937, 0.6145915, 0.56156874,
            0.6145915, 0.56156874,
        ];
        scores.iter().zip(known_scores.iter()).for_each(|(s, ks)| {
            assert!((*s - *ks).l2_norm() < f32::EPSILON);
        });
    }

    #[test]
    fn iso_forest_time_static1() {
        let mut rng = PermutedCongruentialGenerator::new(0, 1);
        let x: Vec<f32> = rng.interval(2048);
        let data = x.iter().map(|x| StaticPoint::new([*x])).collect::<Vec<_>>();
        let now = SystemTime::now();
        let iso_forest: IsoForest<StaticPoint<f32, 1>> =
            IsoForest::new(&data, 128, 128, 0, 1, 0.0, 0.1);
        let _ = iso_forest.score(&data);
        let _ = dbg!(now.elapsed());
    }

    #[test]
    fn iso_forest_time_static2() {
        let mut rng = PermutedCongruentialGenerator::new(0, 1);
        let x: Vec<f32> = rng.interval(2048);
        let y = rng.interval(2048);
        let data = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| StaticPoint::new([*x, *y]))
            .collect::<Vec<_>>();
        let now = SystemTime::now();
        let iso_forest: IsoForest<StaticPoint<f32, 2>> =
            IsoForest::new(&data, 128, 128, 1, 1, 0.0, 0.1);
        let _ = iso_forest.score(&data);
        let _ = dbg!(now.elapsed());
    }

    #[test]
    fn iso_forest_time_single() {
        let mut rng = PermutedCongruentialGenerator::new(0, 1);
        let x: Vec<f32> = rng.interval(2048);
        let data = x;
        let now = SystemTime::now();
        let iso_forest: IsoForest<f32> = IsoForest::new(&data, 128, 128, 0, 1, 0.0, 0.1);
        let _ = iso_forest.score(&data);
        let _ = dbg!(now.elapsed());
    }
}

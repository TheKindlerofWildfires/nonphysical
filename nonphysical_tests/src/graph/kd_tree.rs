

#[cfg(test)]
mod kd_tree_tests {
    use std::time::SystemTime;

    use nonphysical_core::{graph::kd_tree::{KdNode, KdTree}, shared::{primitive::Primitive,point::{Point, StaticPoint}}};
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn kd_tree_first() {
        let mut kd_tree = KdTree::<StaticPoint<F32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([F32(1.0),F32(2.0)]), 0);

        let known_buckets = vec![0];
        let known_max_bounds = StaticPoint::new([F32(1.0),F32( 2.0)]);
        let known_min_bounds = StaticPoint::new([F32(1.0),F32( 2.0)]);

        let known_points = vec![StaticPoint::new([F32(1.0),F32( 2.0)])];

        match kd_tree.node {
            KdNode::Leaf(leaf) => {
                leaf.bucket
                    .into_iter()
                    .zip(known_buckets.into_iter())
                    .for_each(|(bp, kbp)| {
                        assert!(bp == kbp);
                    });
                leaf.points
                    .into_iter()
                    .zip(known_points.into_iter())
                    .for_each(|(p, kp)| {
                        p.data.into_iter().zip(kp.data.into_iter()).for_each(|(pp, kpp)| {
                            assert!(pp == kpp);
                        });
                    });
            }
            KdNode::Branch(_) => assert!(false),
        }

        kd_tree
            .shared
            .max_bounds
            .data
            .into_iter()
            .zip(known_max_bounds.data.into_iter())
            .for_each(|(mb, kmb)| {
                assert!(mb == kmb);
            });

        kd_tree
            .shared
            .min_bounds
            .data
            .into_iter()
            .zip(known_min_bounds.data.into_iter())
            .for_each(|(mb, kmb)| {
                assert!(mb == kmb);
            });
    }

    #[test]
    fn kd_tree_second() {
        let mut kd_tree = KdTree::<StaticPoint<F32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([F32(1.0),F32( 2.0)]), 0);
        kd_tree.add(StaticPoint::new([F32(-1.0),F32( 3.0)]), 1);

        let known_buckets = vec![0, 1];
        let known_max_bounds = StaticPoint::new([F32(1.0),F32( 3.0)]);
        let known_min_bounds = StaticPoint::new([F32(-1.0),F32( 2.0)]);

        let known_points = vec![StaticPoint::new([F32(1.0),F32( 2.0)]), StaticPoint::new([F32(-1.0),F32( 3.0)])];
        match kd_tree.node {
            KdNode::Leaf(leaf) => {
                leaf.bucket
                    .into_iter()
                    .zip(known_buckets.into_iter())
                    .for_each(|(bp, kbp)| {
                        assert!(bp == kbp);
                    });
                leaf.points
                    .into_iter()
                    .zip(known_points.into_iter())
                    .for_each(|(p, kp)| {
                        p.data.into_iter().zip(kp.data.into_iter()).for_each(|(pp, kpp)| {
                            assert!(pp == kpp);
                        });
                    });
            }
            KdNode::Branch(_) => assert!(false),
        }

        kd_tree
            .shared
            .max_bounds
            .data
            .into_iter()
            .zip(known_max_bounds.data.into_iter())
            .for_each(|(mb, kmb)| {
                assert!(mb == kmb);
            });

        kd_tree
            .shared
            .min_bounds
            .data
            .into_iter()
            .zip(known_min_bounds.data.into_iter())
            .for_each(|(mb, kmb)| {
                assert!(mb == kmb);
            });
    }

    #[test]
    fn kd_tree_split() {
        let mut kd_tree = KdTree::<StaticPoint<F32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([F32(1.0),F32( 2.0)]), 0);
        kd_tree.add(StaticPoint::new([F32(-1.0),F32( 3.0)]), 1);
        kd_tree.add(StaticPoint::new([F32(-2.0),F32( 3.0)]), 2);

        match kd_tree.node {
            KdNode::Leaf(_) => assert!(false),
            KdNode::Branch(branch) => {
                let known_max_bounds = StaticPoint::new([F32(1.0),F32( 3.0)]);
                let known_min_bounds = StaticPoint::new([F32(-2.0),F32( 2.0)]);

                kd_tree
                    .shared
                    .max_bounds
                    .data
                    .into_iter()
                    .zip(known_max_bounds.data.into_iter())
                    .for_each(|(mb, kmb)| {
                        assert!(mb == kmb);
                    });

                kd_tree
                    .shared
                    .min_bounds
                    .data
                    .into_iter()
                    .zip(known_min_bounds.data.into_iter())
                    .for_each(|(mb, kmb)| {
                        assert!(mb == kmb);
                    });
                assert!(branch.split_dimension == 0);
                assert!(branch.split_value == F32(-0.5));

                match branch.left.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = StaticPoint::new([F32(-1.0),F32( 3.0)]);
                        let known_min_bounds = StaticPoint::new([F32(-2.0),F32( 3.0)]);

                        let known_points =
                            vec![StaticPoint::new([F32(-2.0),F32( 3.0)]), StaticPoint::new([F32(-1.0),F32( 3.0)])];

                        leaf.bucket
                            .into_iter()
                            .zip(known_buckets.into_iter())
                            .for_each(|(bp, kbp)| {
                                assert!(bp == kbp);
                            });

                        branch
                            .left
                            .shared
                            .max_bounds
                            .data
                            .into_iter()
                            .zip(known_max_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        branch
                            .left
                            .shared
                            .min_bounds
                            .data
                            .into_iter()
                            .zip(known_min_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        leaf.points
                            .into_iter()
                            .zip(known_points.into_iter())
                            .for_each(|(p, kp)| {
                                p.data.into_iter().zip(kp.data.into_iter()).for_each(|(pp, kpp)| {
                                    assert!(pp == kpp);
                                });
                            });
                    }
                    KdNode::Branch(_) => assert!(false),
                }
                match branch.right.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![0];
                        let known_max_bounds = StaticPoint::new([F32(1.0),F32( 2.0)]);
                        let known_min_bounds = StaticPoint::new([F32(1.0),F32( 2.0)]);

                        let known_points = vec![StaticPoint::new([F32(1.0),F32( 2.0)])];

                        leaf.bucket
                            .into_iter()
                            .zip(known_buckets.into_iter())
                            .for_each(|(bp, kbp)| {
                                assert!(bp == kbp);
                            });

                        branch
                            .right
                            .shared
                            .max_bounds
                            .data
                            .into_iter()
                            .zip(known_max_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        branch
                            .right
                            .shared
                            .min_bounds
                            .data
                            .into_iter()
                            .zip(known_min_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        leaf.points
                            .into_iter()
                            .zip(known_points.into_iter())
                            .for_each(|(p, kp)| {
                                p.data.into_iter().zip(kp.data.into_iter()).for_each(|(pp, kpp)| {
                                    assert!(pp == kpp);
                                });
                            });
                    }
                    KdNode::Branch(_) => assert!(false),
                }
            }
        }
    }

    #[test]
    fn kd_tree_split_add() {
        let mut kd_tree = KdTree::<StaticPoint<F32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([F32(1.0),F32( 2.0)]), 0);
        kd_tree.add(StaticPoint::new([F32(-1.0),F32( 3.0)]), 1);
        kd_tree.add(StaticPoint::new([F32(-2.0),F32( 3.0)]), 2);
        kd_tree.add(StaticPoint::new([F32(2.0),F32( 4.0)]), 4);

        match kd_tree.node {
            KdNode::Leaf(_) => assert!(false),
            KdNode::Branch(branch) => {
                let known_max_bounds = StaticPoint::new([F32(2.0),F32( 4.0)]);
                let known_min_bounds = StaticPoint::new([F32(-2.0),F32( 2.0)]);
                kd_tree
                    .shared
                    .max_bounds
                    .data
                    .into_iter()
                    .zip(known_max_bounds.data.into_iter())
                    .for_each(|(mb, kmb)| {
                        assert!(mb == kmb);
                    });

                kd_tree
                    .shared
                    .min_bounds
                    .data
                    .into_iter()
                    .zip(known_min_bounds.data.into_iter())
                    .for_each(|(mb, kmb)| {
                        assert!(mb == kmb);
                    });
                assert!(branch.split_dimension == 0);
                assert!(branch.split_value == F32(-0.5));

                match branch.left.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = StaticPoint::new([F32(-1.0),F32( 3.0)]);
                        let known_min_bounds = StaticPoint::new([F32(-2.0),F32( 3.0)]);

                        let known_points =
                            vec![StaticPoint::new([F32(-2.0),F32( 3.0)]), StaticPoint::new([F32(-1.0),F32( 3.0)])];

                        leaf.bucket
                            .into_iter()
                            .zip(known_buckets.into_iter())
                            .for_each(|(bp, kbp)| {
                                assert!(bp == kbp);
                            });

                        branch
                            .left
                            .shared
                            .max_bounds
                            .data
                            .into_iter()
                            .zip(known_max_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        branch
                            .left
                            .shared
                            .min_bounds
                            .data
                            .into_iter()
                            .zip(known_min_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        leaf.points
                            .into_iter()
                            .zip(known_points.into_iter())
                            .for_each(|(p, kp)| {
                                p.data.into_iter().zip(kp.data.into_iter()).for_each(|(pp, kpp)| {
                                    assert!(pp == kpp);
                                });
                            });
                    }
                    KdNode::Branch(_) => assert!(false),
                }
                match branch.right.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![0];
                        let known_max_bounds = StaticPoint::new([F32(2.0),F32( 4.0)]);
                        let known_min_bounds = StaticPoint::new([F32(1.0),F32( 2.0)]);

                        let known_points =
                            vec![StaticPoint::new([F32(1.0),F32( 2.0)]), StaticPoint::new([F32(2.0),F32( 4.0)])];

                        leaf.bucket
                            .into_iter()
                            .zip(known_buckets.into_iter())
                            .for_each(|(bp, kbp)| {
                                assert!(bp == kbp);
                            });

                        branch
                            .right
                            .shared
                            .max_bounds
                            .data
                            .into_iter()
                            .zip(known_max_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        branch
                            .right
                            .shared
                            .min_bounds
                            .data
                            .into_iter()
                            .zip(known_min_bounds.data.into_iter())
                            .for_each(|(mb, kmb)| {
                                assert!(mb == kmb);
                            });

                        leaf.points
                            .into_iter()
                            .zip(known_points.into_iter())
                            .for_each(|(p, kp)| {
                                p.data.into_iter().zip(kp.data.into_iter()).for_each(|(pp, kpp)| {
                                    assert!(pp == kpp);
                                });
                            });
                    }
                    KdNode::Branch(_) => assert!(false),
                }
            }
        }
    }

    #[test]
    fn kd_tree_nearest() {
        let mut kd_tree = KdTree::<StaticPoint<F32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([F32(1.0),F32( 2.0)]), 0);
        kd_tree.add(StaticPoint::new([F32(-1.0),F32( 3.0)]), 1);
        kd_tree.add(StaticPoint::new([F32(-2.0),F32( 3.0)]), 2);
        kd_tree.add(StaticPoint::new([F32(2.0),F32( 4.0)]), 4);

        let point = StaticPoint::new([F32(1.5),F32(2.0)]);
        let near = kd_tree.nearest(&point, 4);

        let known_dist = vec![0.5, 2.5, 3.5, 4.5];
        let known_index = vec![0, 4, 1, 2];
        near.into_iter()
            .zip(known_dist.into_iter())
            .zip(known_index.into_iter())
            .for_each(|((np, kd), ki)| {
                assert!(np.0 == F32(kd));
                assert!(np.1 == ki);
            });
    }

    #[test]
    fn kd_speed() {
        let now = SystemTime::now();
        let mut kd_tree = KdTree::<StaticPoint<F32, 2>>::new(2);
        (0..500).for_each(|i| kd_tree.add(StaticPoint::new([F32::usize(i), F32::usize(i + 1)]), i));

        let _ = println!("{:?}", now.elapsed());
    }
}

use crate::shared::{float::Float, point::Point, real::Real};
use core::cmp::{min, Ordering};
use std::time::SystemTime;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::BinaryHeap;

pub struct KdLeaf<P: Point> {
    capacity: usize,
    points: Vec<P>,
    bucket: Vec<usize>,
}
pub struct KdBranch<P: Point> {
    left: Box<KdTree<P>>,
    right: Box<KdTree<P>>,
    split_value: P::Primitive,
    split_dimension: usize,
}

pub struct KdShared<P: Point> {
    size: usize,
    min_bounds: P,
    max_bounds: P,
}

pub enum KdNode<P: Point> {
    Leaf(KdLeaf<P>),
    Branch(KdBranch<P>),
}

pub struct KdTree<P: Point> {
    shared: KdShared<P>,
    node: KdNode<P>,
}

impl<P: Point> KdLeaf<P> {
    fn new(capacity: usize) -> Self {
        let points = Vec::with_capacity(capacity);
        let bucket = Vec::with_capacity(capacity);
        Self {
            capacity,
            points,
            bucket,
        }
    }
    fn add_to_bucket(
        &mut self,
        shared: &KdShared<P>,
        point: P,
        data: usize,
    ) -> Option<KdBranch<P>> {
        self.points.push(point);
        self.bucket.push(data);

        if shared.size > self.capacity {
            Some(self.split(shared))
        } else {
            None
        }
    }

    fn split(&mut self, shared: &KdShared<P>) -> KdBranch<P> {
        let (_, split_dimension) = shared.max_bounds.ordered_farthest(&shared.min_bounds);

        let min = shared.min_bounds.data(split_dimension);
        let max = shared.max_bounds.data(split_dimension);
        let split_value = min + (max - min) / P::Primitive::usize(2);

        let left = Box::new(KdTree::<P>::new(self.capacity));
        let right = Box::new(KdTree::<P>::new(self.capacity));
        let mut branch = KdBranch::new(left, right, split_value, split_dimension);

        while !self.points.is_empty() {
            let point = self.points.swap_remove(0);
            let data = self.bucket.swap_remove(0);
            if branch.branch_left(shared, &point) {
                branch.left.add(point, data); //unnecessary match happens below this
            } else {
                branch.right.add(point, data); //unnecessary match happens below this
            }
        }

        branch
    }
}

impl<P: Point> KdBranch<P> {
    fn new(
        left: Box<KdTree<P>>,
        right: Box<KdTree<P>>,
        split_value: P::Primitive,
        split_dimension: usize,
    ) -> Self {
        Self {
            left,
            right,
            split_value,
            split_dimension,
        }
    }

    fn branch_left(&self, shared: &KdShared<P>, point: &P) -> bool {
        //reduce the
        if point.data(self.split_dimension) == self.split_value {
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                % 2
                == 0
        } else if shared.min_bounds.data(self.split_dimension) == self.split_value {
            point.data(self.split_dimension) <= self.split_value
        } else {
            point.data(self.split_dimension) < self.split_value
        }
    }
}

impl<P: Point> KdShared<P> {
    fn new() -> Self {
        let min_bounds = Point::MAX;
        let max_bounds = Point::MIN;
        let size = 0;
        Self {
            size,
            min_bounds,
            max_bounds,
        }
    }
}

impl<P: Point> KdTree<P> {
    pub fn new(capacity: usize) -> Self {
        let leaf = KdLeaf::new(capacity);
        let node = KdNode::Leaf(leaf);
        let shared = KdShared::new();
        Self { shared, node }
    }

    pub fn len(&self) -> usize {
        self.shared.size
    }
    pub fn is_empty(&self) -> bool {
        self.shared.size == 0
    }

    pub fn add(&mut self, point: P, data: usize) {
        self.extend(&point);
        self.shared.size += 1;
        match &mut self.node {
            KdNode::Leaf(leaf) => {
                if let Some(branch) = leaf.add_to_bucket(&self.shared, point, data) {
                    self.node = KdNode::Branch(branch);
                };
            }
            KdNode::Branch(branch) => {
                let next = match branch.branch_left(&self.shared, &point) {
                    true => branch.left.as_mut(),
                    false => branch.right.as_mut(),
                };
                next.add(point, data);
            }
        }
    }

    fn extend(&mut self, point: &P) {
        self.shared.min_bounds = point.lesser(&self.shared.min_bounds);
        self.shared.max_bounds = point.greater(&self.shared.max_bounds);
    }

    pub fn nearest(&self, point: &P, k: usize) -> Vec<(P::Primitive, usize)> {
        debug_assert!(k != 0);
        let k = min(k, self.len());

        let mut pending: BinaryHeap<HeapElement<P::Primitive, &KdTree<P>>> =
            BinaryHeap::<HeapElement<P::Primitive, &Self>>::new();
        let mut evaluated = BinaryHeap::<HeapElement<P::Primitive, &usize>>::new();

        pending.push(HeapElement {
            distance: P::Primitive::ZERO,
            element: self,
        });

        while !pending.is_empty()
            && (evaluated.len() < k
                || (-pending.peek().unwrap().distance <= evaluated.peek().unwrap().distance))
        {
            self.nearest_step(point, k, P::Primitive::MAX, &mut pending, &mut evaluated);
        }
        evaluated
            .into_sorted_vec()
            .into_iter()
            .take(k)
            .map(|h| (h.distance, *h.element))
            .collect()
    }

    fn nearest_step<'b>(
        &self,
        point: &P,
        k: usize,
        max_dist: P::Primitive,
        pending: &mut BinaryHeap<HeapElement<P::Primitive, &'b Self>>,
        evaluated: &mut BinaryHeap<HeapElement<P::Primitive, &'b usize>>,
    ) {
        debug_assert!(evaluated.len() <= k);
        let mut current = pending.pop().unwrap().element;
        let eval_dist = match evaluated.len() == k && max_dist < evaluated.peek().unwrap().distance
        {
            true => evaluated.peek().unwrap().distance,
            false => max_dist,
        };

        loop {
            match &current.node {
                KdNode::Branch(branch) => {
                    let candidate = match branch.branch_left(&self.shared, point) {
                        true => {
                            let temp = branch.right.as_ref();
                            current = branch.left.as_ref();
                            temp
                        }
                        false => {
                            let temp = branch.left.as_ref();
                            current = branch.right.as_ref();
                            temp
                        }
                    };

                    let candidate_to_space = point.distance_to_range(
                        &candidate.shared.min_bounds,
                        &candidate.shared.max_bounds,
                    );
                    if candidate_to_space <= eval_dist {
                        pending.push(HeapElement {
                            distance: -candidate_to_space,
                            element: candidate,
                        })
                    }
                }
                KdNode::Leaf(leaf) => {
                    let points = leaf.points.iter();
                    let bucket = leaf.bucket.iter();
                    points
                        .zip(bucket)
                        .map(|(p, d)| HeapElement {
                            distance: point.l1_distance(&p),
                            element: d,
                        })
                        .for_each(|element| {
                            if element.distance <= max_dist {
                                if evaluated.len() < k {
                                    evaluated.push(element);
                                } else if element.distance < evaluated.peek().unwrap().distance {
                                    evaluated.pop();
                                    evaluated.push(element);
                                }
                            }
                        });
                    break;
                }
            }
        }
    }
}

pub struct HeapElement<R: Real, O> {
    pub distance: R,
    pub element: O,
}

impl<R: Real, O> Ord for HeapElement<R, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl<R: Real, O> PartialOrd for HeapElement<R, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<R: Real, O> Eq for HeapElement<R, O> {}

impl<R: Real, O> PartialEq for HeapElement<R, O> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

#[cfg(test)]
mod kd_tree_tests {
    use alloc::vec;
    use std::time::SystemTime;

    use crate::shared::point::StaticPoint;

    use super::*;

    #[test]
    fn kd_tree_first() {
        let mut kd_tree = KdTree::<StaticPoint<f32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([1.0, 2.0]), 0);

        let known_buckets = vec![0];
        let known_max_bounds = StaticPoint::new([1.0, 2.0]);
        let known_min_bounds = StaticPoint::new([1.0, 2.0]);

        let known_points = vec![StaticPoint::new([1.0, 2.0])];

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
        let mut kd_tree = KdTree::<StaticPoint<f32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([1.0, 2.0]), 0);
        kd_tree.add(StaticPoint::new([-1.0, 3.0]), 1);

        let known_buckets = vec![0, 1];
        let known_max_bounds = StaticPoint::new([1.0, 3.0]);
        let known_min_bounds = StaticPoint::new([-1.0, 2.0]);

        let known_points = vec![StaticPoint::new([1.0, 2.0]), StaticPoint::new([-1.0, 3.0])];
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
        let mut kd_tree = KdTree::<StaticPoint<f32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([1.0, 2.0]), 0);
        kd_tree.add(StaticPoint::new([-1.0, 3.0]), 1);
        kd_tree.add(StaticPoint::new([-2.0, 3.0]), 2);

        match kd_tree.node {
            KdNode::Leaf(_) => assert!(false),
            KdNode::Branch(branch) => {
                let known_max_bounds = StaticPoint::new([1.0, 3.0]);
                let known_min_bounds = StaticPoint::new([-2.0, 2.0]);

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
                assert!(branch.split_value == -0.5);

                match branch.left.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = StaticPoint::new([-1.0, 3.0]);
                        let known_min_bounds = StaticPoint::new([-2.0, 3.0]);

                        let known_points =
                            vec![StaticPoint::new([-2.0, 3.0]), StaticPoint::new([-1.0, 3.0])];

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
                        let known_max_bounds = StaticPoint::new([1.0, 2.0]);
                        let known_min_bounds = StaticPoint::new([1.0, 2.0]);

                        let known_points = vec![StaticPoint::new([1.0, 2.0])];

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
        let mut kd_tree = KdTree::<StaticPoint<f32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([1.0, 2.0]), 0);
        kd_tree.add(StaticPoint::new([-1.0, 3.0]), 1);
        kd_tree.add(StaticPoint::new([-2.0, 3.0]), 2);
        kd_tree.add(StaticPoint::new([2.0, 4.0]), 4);

        match kd_tree.node {
            KdNode::Leaf(_) => assert!(false),
            KdNode::Branch(branch) => {
                let known_max_bounds = StaticPoint::new([2.0, 4.0]);
                let known_min_bounds = StaticPoint::new([-2.0, 2.0]);
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
                assert!(branch.split_value == -0.5);

                match branch.left.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = StaticPoint::new([-1.0, 3.0]);
                        let known_min_bounds = StaticPoint::new([-2.0, 3.0]);

                        let known_points =
                            vec![StaticPoint::new([-2.0, 3.0]), StaticPoint::new([-1.0, 3.0])];

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
                        let known_max_bounds = StaticPoint::new([2.0, 4.0]);
                        let known_min_bounds = StaticPoint::new([1.0, 2.0]);

                        let known_points =
                            vec![StaticPoint::new([1.0, 2.0]), StaticPoint::new([2.0, 4.0])];

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
        let mut kd_tree = KdTree::<StaticPoint<f32, 2>>::new(2);
        kd_tree.add(StaticPoint::new([1.0, 2.0]), 0);
        kd_tree.add(StaticPoint::new([-1.0, 3.0]), 1);
        kd_tree.add(StaticPoint::new([-2.0, 3.0]), 2);
        kd_tree.add(StaticPoint::new([2.0, 4.0]), 4);

        let point = StaticPoint::new([1.5, 2.0]);
        let near = kd_tree.nearest(&point, 4);

        let known_dist = vec![0.5, 2.5, 3.5, 4.5];
        let known_index = vec![0, 4, 1, 2];
        near.into_iter()
            .zip(known_dist.into_iter())
            .zip(known_index.into_iter())
            .for_each(|((np, kd), ki)| {
                assert!(np.0 == kd);
                assert!(np.1 == ki);
            });
    }

    #[test]
    fn kd_speed() {
        let now = SystemTime::now();
        let mut kd_tree = KdTree::<StaticPoint<f32, 2>>::new(2);
        (0..500).for_each(|i| kd_tree.add(StaticPoint::new([f32::usize(i), f32::usize(i + 1)]), i));

        let _ = println!("{:?}", now.elapsed());
    }
}

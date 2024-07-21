use core::cmp::{min, Ordering};
use std::{collections::BinaryHeap, time::SystemTime};

use crate::shared::{float::Float, point::Point};

pub struct KdLeaf<T: Float, const N: usize> {
    capacity: usize,
    points: Vec<Point<T, N>>,
    bucket: Vec<usize>,
}
pub struct KdBranch<T: Float, const N: usize> {
    left: Box<KdTree<T, N>>,
    right: Box<KdTree<T, N>>,
    split_value: T,
    split_dimension: usize,
}

pub struct KdShared<T: Float, const N: usize> {
    size: usize,
    min_bounds: Point<T, N>,
    max_bounds: Point<T, N>,
}

pub enum KdNode<T: Float, const N: usize> {
    Leaf(KdLeaf<T, N>),
    Branch(KdBranch<T, N>),
}

pub struct KdTree<T: Float, const N: usize> {
    shared: KdShared<T, N>,
    node: KdNode<T, N>,
}

impl<T: Float, const N: usize> KdLeaf<T, N> {
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
        shared: &KdShared<T, N>,
        point: Point<T, N>,
        data: usize,
    ) -> Option<KdBranch<T, N>> {
        self.points.push(point);
        self.bucket.push(data);

        if shared.size > self.capacity {
            Some(self.split(shared))
        } else {
            None
        }
    }

    fn split(&mut self, shared: &KdShared<T, N>) -> KdBranch<T, N> {
        let (_, split_dimension) = (0..N).fold((T::MIN, 0), |acc, dim| {
            let difference = shared.max_bounds.data[dim] - shared.min_bounds.data[dim];
            if difference > acc.0 {
                (difference, dim)
            } else {
                acc
            }
        });

        let min = shared.min_bounds.data[split_dimension];
        let max = shared.max_bounds.data[split_dimension];
        let split_value = min + (max - min) / T::usize(2);

        let left = Box::new(KdTree::<T, N>::new(self.capacity));
        let right = Box::new(KdTree::<T, N>::new(self.capacity));
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

impl<T: Float, const N: usize> KdBranch<T, N> {
    fn new(
        left: Box<KdTree<T, N>>,
        right: Box<KdTree<T, N>>,
        split_value: T,
        split_dimension: usize,
    ) -> Self {
        Self {
            left,
            right,
            split_value,
            split_dimension,
        }
    }

    fn branch_left(&self, shared: &KdShared<T, N>, point: &Point<T, N>) -> bool {
        //reduce the
        if point.data[self.split_dimension] == self.split_value{
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap()
            .as_nanos()%2==0
        }
        else if shared.min_bounds.data[self.split_dimension] == self.split_value {
            point.data[self.split_dimension] <= self.split_value
        } else {
            point.data[self.split_dimension] < self.split_value
        }
    }
}

impl<T: Float, const N: usize> KdShared<T, N> {
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

impl<T: Float, const N: usize> KdTree<T, N> {
    pub fn new(capacity: usize) -> Self {
        let leaf = KdLeaf::new(capacity);
        let node = KdNode::Leaf(leaf);
        let shared = KdShared::new();
        Self { shared, node }
    }

    pub fn len(&self) -> usize {
        self.shared.size
    }
    pub fn is_empty(&self) -> bool{
        self.shared.size==0
    }

    pub fn add(&mut self, point: Point<T, N>, data: usize) {
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

    fn extend(&mut self, point: &Point<T, N>) {
        self.shared
            .min_bounds
            .data
            .iter_mut()
            .zip(self.shared.max_bounds.data.iter_mut())
            .zip(point.data.iter())
            .for_each(|((min_b, max_b), pb)| {
                if pb < min_b {
                    *min_b = *pb;
                }
                if pb > max_b {
                    *max_b = *pb;
                }
            });
    }

    pub fn nearest(&self, point: &Point<T, N>, k: usize) -> Vec<(T, usize)> {
        debug_assert!(k != 0);
        let k = min(k, self.len());

        let mut pending: BinaryHeap<HeapElement<T, &KdTree<T, N>>> =
            BinaryHeap::<HeapElement<T, &Self>>::new();
        let mut evaluated = BinaryHeap::<HeapElement<T, &usize>>::new();

        pending.push(HeapElement {
            distance: T::ZERO,
            element: self,
        });

        while !pending.is_empty()
            && (evaluated.len() < k
                || (-pending.peek().unwrap().distance <= evaluated.peek().unwrap().distance))
        {
            self.nearest_step(point, k, T::MAX, &mut pending, &mut evaluated);
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
        point: &Point<T, N>,
        k: usize,
        max_dist: T,
        pending: &mut BinaryHeap<HeapElement<T, &'b Self>>,
        evaluated: &mut BinaryHeap<HeapElement<T, &'b usize>>,
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

                    let candidate_to_space = self.distance_to_space(
                        point,
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
                            distance: point.distance(p),
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

    fn distance_to_space(
        &self,
        p1: &Point<T, N>,
        min_bounds: &Point<T, N>,
        max_bounds: &Point<T, N>,
    ) -> T {
        let mut p2 = Point::ZERO;
        (0..N).for_each(|i| {
            if p1.data[i] > max_bounds.data[i] {
                p2.data[i] = max_bounds.data[i];
            } else if p1.data[i] < min_bounds.data[i] {
                p2.data[i] = min_bounds.data[i];
            } else {
                p2.data[i] = p1.data[i];
            }
        });
        p1.distance(&p2)
    }
}

pub struct HeapElement<T: Float, O> {
    pub distance: T,
    pub element: O,
}

impl<T: Float, O> Ord for HeapElement<T, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl<T: Float, O> PartialOrd for HeapElement<T, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float, O> Eq for HeapElement<T, O> {}

impl<T: Float, O> PartialEq for HeapElement<T, O> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

#[cfg(test)]
mod kd_tree_tests {
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn kd_tree_first() {
        let mut kd_tree = KdTree::<f32, 2>::new(2);
        kd_tree.add(Point::new([1.0, 2.0]), 0);

        let known_buckets = vec![0];
        let known_max_bounds = Point::new([1.0, 2.0]);
        let known_min_bounds = Point::new([1.0, 2.0]);

        let known_points = vec![Point::new([1.0, 2.0])];

        match kd_tree.node {
            KdNode::Leaf(leaf) => {
                leaf.bucket
                    .iter()
                    .zip(known_buckets.iter())
                    .for_each(|(bp, kbp)| {
                        assert!(*bp == *kbp);
                    });
                leaf.points
                    .iter()
                    .zip(known_points.iter())
                    .for_each(|(p, kp)| {
                        p.data.iter().zip(kp.data.iter()).for_each(|(pp, kpp)| {
                            assert!(*pp == *kpp);
                        });
                    });
            }
            KdNode::Branch(_) => assert!(false),
        }

        kd_tree
            .shared
            .max_bounds
            .data
            .iter()
            .zip(known_max_bounds.data.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });

        kd_tree
            .shared
            .min_bounds
            .data
            .iter()
            .zip(known_min_bounds.data.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });
    }

    #[test]
    fn kd_tree_second() {
        let mut kd_tree = KdTree::<f32, 2>::new(2);
        kd_tree.add(Point::new([1.0, 2.0]), 0);
        kd_tree.add(Point::new([-1.0, 3.0]), 1);

        let known_buckets = vec![0, 1];
        let known_max_bounds = Point::new([1.0, 3.0]);
        let known_min_bounds = Point::new([-1.0, 2.0]);

        let known_points = vec![Point::new([1.0, 2.0]), Point::new([-1.0, 3.0])];
        match kd_tree.node {
            KdNode::Leaf(leaf) => {
                leaf.bucket
                    .iter()
                    .zip(known_buckets.iter())
                    .for_each(|(bp, kbp)| {
                        assert!(*bp == *kbp);
                    });
                leaf.points
                    .iter()
                    .zip(known_points.iter())
                    .for_each(|(p, kp)| {
                        p.data.iter().zip(kp.data.iter()).for_each(|(pp, kpp)| {
                            assert!(*pp == *kpp);
                        });
                    });
            }
            KdNode::Branch(_) => assert!(false),
        }

        kd_tree
            .shared
            .max_bounds
            .data
            .iter()
            .zip(known_max_bounds.data.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });

        kd_tree
            .shared
            .min_bounds
            .data
            .iter()
            .zip(known_min_bounds.data.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });
    }

    #[test]
    fn kd_tree_split() {
        let mut kd_tree = KdTree::<f32, 2>::new(2);
        kd_tree.add(Point::new([1.0, 2.0]), 0);
        kd_tree.add(Point::new([-1.0, 3.0]), 1);
        kd_tree.add(Point::new([-2.0, 3.0]), 2);

        match kd_tree.node {
            KdNode::Leaf(_) => assert!(false),
            KdNode::Branch(branch) => {
                let known_max_bounds = Point::new([1.0, 3.0]);
                let known_min_bounds = Point::new([-2.0, 2.0]);

                kd_tree
                    .shared
                    .max_bounds
                    .data
                    .iter()
                    .zip(known_max_bounds.data.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });

                kd_tree
                    .shared
                    .min_bounds
                    .data
                    .iter()
                    .zip(known_min_bounds.data.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });
                assert!(branch.split_dimension == 0);
                assert!(branch.split_value == -0.5);

                match branch.left.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = Point::new([-1.0, 3.0]);
                        let known_min_bounds = Point::new([-2.0, 3.0]);

                        let known_points = vec![Point::new([-2.0, 3.0]), Point::new([-1.0, 3.0])];

                        leaf.bucket
                            .iter()
                            .zip(known_buckets.iter())
                            .for_each(|(bp, kbp)| {
                                assert!(*bp == *kbp);
                            });

                        branch
                            .left
                            .shared
                            .max_bounds
                            .data
                            .iter()
                            .zip(known_max_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .left
                            .shared
                            .min_bounds
                            .data
                            .iter()
                            .zip(known_min_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.data.iter().zip(kp.data.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
                                });
                            });
                    }
                    KdNode::Branch(_) => assert!(false),
                }
                match branch.right.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![0];
                        let known_max_bounds = Point::new([1.0, 2.0]);
                        let known_min_bounds = Point::new([1.0, 2.0]);

                        let known_points = vec![Point::new([1.0, 2.0])];

                        leaf.bucket
                            .iter()
                            .zip(known_buckets.iter())
                            .for_each(|(bp, kbp)| {
                                assert!(*bp == *kbp);
                            });

                        branch
                            .right
                            .shared
                            .max_bounds
                            .data
                            .iter()
                            .zip(known_max_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .right
                            .shared
                            .min_bounds
                            .data
                            .iter()
                            .zip(known_min_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.data.iter().zip(kp.data.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
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
        let mut kd_tree = KdTree::<f32, 2>::new(2);
        kd_tree.add(Point::new([1.0, 2.0]), 0);
        kd_tree.add(Point::new([-1.0, 3.0]), 1);
        kd_tree.add(Point::new([-2.0, 3.0]), 2);
        kd_tree.add(Point::new([2.0, 4.0]), 4);

        match kd_tree.node {
            KdNode::Leaf(_) => assert!(false),
            KdNode::Branch(branch) => {
                let known_max_bounds = Point::new([2.0, 4.0]);
                let known_min_bounds = Point::new([-2.0, 2.0]);
                kd_tree
                    .shared
                    .max_bounds
                    .data
                    .iter()
                    .zip(known_max_bounds.data.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });

                kd_tree
                    .shared
                    .min_bounds
                    .data
                    .iter()
                    .zip(known_min_bounds.data.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });
                assert!(branch.split_dimension == 0);
                assert!(branch.split_value == -0.5);

                match branch.left.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = Point::new([-1.0, 3.0]);
                        let known_min_bounds = Point::new([-2.0, 3.0]);

                        let known_points = vec![Point::new([-2.0, 3.0]), Point::new([-1.0, 3.0])];

                        leaf.bucket
                            .iter()
                            .zip(known_buckets.iter())
                            .for_each(|(bp, kbp)| {
                                assert!(*bp == *kbp);
                            });

                        branch
                            .left
                            .shared
                            .max_bounds
                            .data
                            .iter()
                            .zip(known_max_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .left
                            .shared
                            .min_bounds
                            .data
                            .iter()
                            .zip(known_min_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.data.iter().zip(kp.data.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
                                });
                            });
                    }
                    KdNode::Branch(_) => assert!(false),
                }
                match branch.right.node {
                    KdNode::Leaf(leaf) => {
                        let known_buckets = vec![0];
                        let known_max_bounds = Point::new([2.0, 4.0]);
                        let known_min_bounds = Point::new([1.0, 2.0]);

                        let known_points = vec![Point::new([1.0, 2.0]), Point::new([2.0, 4.0])];

                        leaf.bucket
                            .iter()
                            .zip(known_buckets.iter())
                            .for_each(|(bp, kbp)| {
                                assert!(*bp == *kbp);
                            });

                        branch
                            .right
                            .shared
                            .max_bounds
                            .data
                            .iter()
                            .zip(known_max_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .right
                            .shared
                            .min_bounds
                            .data
                            .iter()
                            .zip(known_min_bounds.data.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.data.iter().zip(kp.data.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
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
        let mut kd_tree = KdTree::<f32, 2>::new(2);
        kd_tree.add(Point::new([1.0, 2.0]), 0);
        kd_tree.add(Point::new([-1.0, 3.0]), 1);
        kd_tree.add(Point::new([-2.0, 3.0]), 2);
        kd_tree.add(Point::new([2.0, 4.0]), 4);

        let point = Point::new([1.5, 2.0]);
        let near = kd_tree.nearest(&point, 4);

        let known_dist = vec![0.5, 2.5, 3.5, 4.5];
        let known_index = vec![0, 4, 1, 2];
        near.iter()
            .zip(known_dist.iter())
            .zip(known_index.iter())
            .for_each(|((np, kd), ki)| {
                assert!(np.0 == *kd);
                assert!(np.1 == *ki);
            });
    }

    #[test]
    fn kd_speed() {
        let now = SystemTime::now();
        let mut kd_tree = KdTree::<f32, 2>::new(2);
        (0..500).for_each(|i| kd_tree.add(Point::new([f32::usize(i), f32::usize(i + 1)]), i));

        let _ = dbg!(now.elapsed());
    }
}

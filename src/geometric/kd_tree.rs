use core::cmp::{min, Ordering};
use std::collections::BinaryHeap;

use crate::shared::float::Float;

pub struct KdLeaf<T: Float> {
    capacity: usize,
    points: Vec<Vec<T>>,
    bucket: Vec<usize>,
}
pub struct KdBranch<T: Float> {
    left: Box<KdTree<T>>,
    right: Box<KdTree<T>>,
    split_value: T,
    split_dimension: usize,
}

pub struct KdShared<T: Float> {
    dimensions: usize,
    size: usize,
    min_bounds: Vec<T>,
    max_bounds: Vec<T>,
}

pub enum KdOffshoot<T: Float> {
    Leaf(KdLeaf<T>),
    Branch(KdBranch<T>),
}

pub struct KdTree<T: Float> {
    shared: KdShared<T>,
    offshoot: KdOffshoot<T>,
}

impl<T: Float> KdLeaf<T> {
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
        shared: &KdShared<T>,
        point: Vec<T>,
        data: usize,
    ) -> Option<KdBranch<T>> {
        self.points.push(point);
        self.bucket.push(data);

        if shared.size > self.capacity {
            Some(self.split(shared))
        } else {
            None
        }
    }

    fn split(&mut self, shared: &KdShared<T>) -> KdBranch<T> {
        let mut max = T::MIN;
        let split_dimension = (0..shared.dimensions).fold(0, |acc, dim| {
            let difference = shared.max_bounds[dim] - shared.min_bounds[dim];
            if difference > max {
                max = difference;
                dim
            } else {
                acc
            }
        });

        let min = shared.min_bounds[split_dimension];
        let max = shared.max_bounds[split_dimension];
        let split_value = min + (max - min) / T::usize(2);

        let left = Box::new(KdTree::<T>::new(shared.dimensions, self.capacity));
        let right = Box::new(KdTree::<T>::new(shared.dimensions, self.capacity));
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

impl<T: Float> KdBranch<T> {
    fn new(
        left: Box<KdTree<T>>,
        right: Box<KdTree<T>>,
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

    fn branch_left(&self, shared: &KdShared<T>, point: &Vec<T>) -> bool {
        if shared.min_bounds[self.split_dimension] == self.split_value {
            point[self.split_dimension] <= self.split_value
        } else {
            point[self.split_dimension] < self.split_value
        }
    }
}

impl<T: Float> KdShared<T> {
    fn new(dimensions: usize) -> Self {
        let min_bounds = vec![T::MAX; dimensions];
        let max_bounds = vec![T::MIN; dimensions];
        let size = 0;
        Self {
            dimensions,
            size,
            min_bounds,
            max_bounds,
        }
    }
}

impl<T: Float> KdTree<T> {
    pub fn new(dimensions: usize, capacity: usize) -> Self {
        let leaf = KdLeaf::new(capacity);
        let offshoot = KdOffshoot::Leaf(leaf);
        let shared = KdShared::new(dimensions);
        Self { shared, offshoot }
    }

    pub fn len(&self) -> usize {
        self.shared.size
    }

    pub fn add(&mut self, point: Vec<T>, data: usize) {
        self.extend(&point);
        self.shared.size += 1;
        match &mut self.offshoot {
            KdOffshoot::Leaf(leaf) => {
                let may_branch = leaf.add_to_bucket(&self.shared, point, data);
                match may_branch {
                    Some(branch) => self.offshoot = KdOffshoot::Branch(branch),
                    None => {}
                }
            }
            KdOffshoot::Branch(branch) => {
                let next = match branch.branch_left(&self.shared, &point) {
                    true => branch.left.as_mut(),
                    false => branch.right.as_mut(),
                };
                next.add(point, data);
            }
        }
    }

    fn extend(&mut self, point: &Vec<T>) {
        self.shared
            .min_bounds
            .iter_mut()
            .zip(self.shared.max_bounds.iter_mut())
            .zip(point.iter())
            .for_each(|((min_b, max_b), pb)| {
                if pb < min_b {
                    *min_b = *pb;
                }
                if pb > max_b {
                    *max_b = *pb;
                }
            });
    }

    pub fn nearest(&self, point: &Vec<T>, k: usize) -> Vec<(T, usize)> {
        debug_assert!(k != 0);
        let k = min(k, self.len());

        let mut pending: BinaryHeap<HeapElement<T, &KdTree<T>>> =
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
        point: &Vec<T>,
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
            match &current.offshoot {
                KdOffshoot::Branch(branch) => {
                    let candidate = match branch.branch_left(&self.shared, &point) {
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
                        &point,
                        &candidate.shared.min_bounds,
                        &candidate.shared.max_bounds,
                    );
                    if candidate_to_space <= eval_dist {
                        pending.push(HeapElement {
                            distance: -candidate_to_space,
                            element: &candidate,
                        })
                    }
                }
                KdOffshoot::Leaf(leaf) => {
                    let points = leaf.points.iter();
                    let bucket = leaf.bucket.iter();
                    points
                        .zip(bucket)
                        .map(|(p, d)| HeapElement {
                            distance: self.distance(&point, p),
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

    fn distance_to_space(&self, p1: &Vec<T>, min_bounds: &Vec<T>, max_bounds: &Vec<T>) -> T {
        let mut p2 = vec![T::ZERO; p1.len()];
        (0..p1.len()).for_each(|i| {
            if p1[i] > max_bounds[i] {
                p2[i] = max_bounds[i];
            } else if p1[i] < min_bounds[i] {
                p2[i] = min_bounds[i];
            } else {
                p2[i] = p1[i];
            }
        });
        self.distance(p1, &p2)
    }

    fn distance(&self, a: &Vec<T>, b: &Vec<T>) -> T {
        let out = a
            .iter()
            .zip(b.iter())
            .fold(T::ZERO, |dist, (ap, bp)| dist + (*ap - *bp).norm());
        out
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
        self.distance.partial_cmp(&other.distance)
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
    use super::*;

    #[test]
    fn kd_tree_first() {
        let mut kd_tree = KdTree::<f32>::new(2, 2);
        kd_tree.add(vec![1.0, 2.0], 0);

        let known_buckets = vec![0];
        let known_max_bounds = vec![1.0, 2.0];
        let known_min_bounds = vec![1.0, 2.0];

        let known_points = vec![vec![1.0, 2.0]];

        match kd_tree.offshoot {
            KdOffshoot::Leaf(leaf) => {
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
                        p.iter().zip(kp.iter()).for_each(|(pp, kpp)| {
                            assert!(*pp == *kpp);
                        });
                    });
            }
            KdOffshoot::Branch(_) => assert!(false),
        }

        kd_tree
            .shared
            .max_bounds
            .iter()
            .zip(known_max_bounds.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });

        kd_tree
            .shared
            .min_bounds
            .iter()
            .zip(known_min_bounds.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });
    }

    #[test]
    fn kd_tree_second() {
        let mut kd_tree = KdTree::<f32>::new(2, 2);
        kd_tree.add(vec![1.0, 2.0], 0);
        kd_tree.add(vec![-1.0, 3.0], 1);

        let known_buckets = vec![0, 1];
        let known_max_bounds = vec![1.0, 3.0];
        let known_min_bounds = vec![-1.0, 2.0];

        let known_points = vec![vec![1.0, 2.0], vec![-1.0, 3.0]];
        match kd_tree.offshoot {
            KdOffshoot::Leaf(leaf) => {
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
                        p.iter().zip(kp.iter()).for_each(|(pp, kpp)| {
                            assert!(*pp == *kpp);
                        });
                    });
            }
            KdOffshoot::Branch(_) => assert!(false),
        }

        kd_tree
            .shared
            .max_bounds
            .iter()
            .zip(known_max_bounds.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });

        kd_tree
            .shared
            .min_bounds
            .iter()
            .zip(known_min_bounds.iter())
            .for_each(|(mb, kmb)| {
                assert!(*mb == *kmb);
            });
    }

    #[test]
    fn kd_tree_split() {
        let mut kd_tree = KdTree::<f32>::new(2, 2);
        kd_tree.add(vec![1.0, 2.0], 0);
        kd_tree.add(vec![-1.0, 3.0], 1);
        kd_tree.add(vec![-2.0, 3.0], 2);

        match kd_tree.offshoot {
            KdOffshoot::Leaf(_) => assert!(false),
            KdOffshoot::Branch(branch) => {
                let known_max_bounds = vec![1.0, 3.0];
                let known_min_bounds = vec![-2.0, 2.0];

                kd_tree
                    .shared
                    .max_bounds
                    .iter()
                    .zip(known_max_bounds.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });

                kd_tree
                    .shared
                    .min_bounds
                    .iter()
                    .zip(known_min_bounds.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });
                assert!(branch.split_dimension == 0);
                assert!(branch.split_value == -0.5);

                match branch.left.offshoot {
                    KdOffshoot::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = vec![-1.0, 3.0];
                        let known_min_bounds = vec![-2.0, 3.0];

                        let known_points = vec![vec![-2.0, 3.0], vec![-1.0, 3.0]];

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
                            .iter()
                            .zip(known_max_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .left
                            .shared
                            .min_bounds
                            .iter()
                            .zip(known_min_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.iter().zip(kp.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
                                });
                            });
                    }
                    KdOffshoot::Branch(_) => assert!(false),
                }
                match branch.right.offshoot {
                    KdOffshoot::Leaf(leaf) => {
                        let known_buckets = vec![0];
                        let known_max_bounds = vec![1.0, 2.0];
                        let known_min_bounds = vec![1.0, 2.0];

                        let known_points = vec![vec![1.0, 2.0]];

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
                            .iter()
                            .zip(known_max_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .right
                            .shared
                            .min_bounds
                            .iter()
                            .zip(known_min_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.iter().zip(kp.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
                                });
                            });
                    }
                    KdOffshoot::Branch(_) => assert!(false),
                }
            }
        }
    }
    
    #[test]
    fn kd_tree_split_add(){
        let mut kd_tree = KdTree::<f32>::new(2, 2);
        kd_tree.add(vec![1.0,2.0], 0);
        kd_tree.add(vec![-1.0,3.0], 1);
        kd_tree.add(vec![-2.0,3.0], 2);
        kd_tree.add(vec![2.0,4.0], 4);

        match kd_tree.offshoot {
            KdOffshoot::Leaf(_) => assert!(false),
            KdOffshoot::Branch(branch) => {
                let known_max_bounds = vec![2.0, 4.0];
                let known_min_bounds = vec![-2.0, 2.0];
                kd_tree
                    .shared
                    .max_bounds
                    .iter()
                    .zip(known_max_bounds.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });

                kd_tree
                    .shared
                    .min_bounds
                    .iter()
                    .zip(known_min_bounds.iter())
                    .for_each(|(mb, kmb)| {
                        assert!(*mb == *kmb);
                    });
                assert!(branch.split_dimension == 0);
                assert!(branch.split_value == -0.5);

                match branch.left.offshoot {
                    KdOffshoot::Leaf(leaf) => {
                        let known_buckets = vec![2, 1];
                        let known_max_bounds = vec![-1.0, 3.0];
                        let known_min_bounds = vec![-2.0, 3.0];

                        let known_points = vec![vec![-2.0, 3.0], vec![-1.0, 3.0]];

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
                            .iter()
                            .zip(known_max_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .left
                            .shared
                            .min_bounds
                            .iter()
                            .zip(known_min_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.iter().zip(kp.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
                                });
                            });
                    }
                    KdOffshoot::Branch(_) => assert!(false),
                }
                match branch.right.offshoot {
                    KdOffshoot::Leaf(leaf) => {
                        let known_buckets = vec![0];
                        let known_max_bounds = vec![2.0, 4.0];
                        let known_min_bounds = vec![1.0, 2.0];

                        let known_points = vec![vec![1.0, 2.0],vec![2.0,4.0]];

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
                            .iter()
                            .zip(known_max_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        branch
                            .right
                            .shared
                            .min_bounds
                            .iter()
                            .zip(known_min_bounds.iter())
                            .for_each(|(mb, kmb)| {
                                assert!(*mb == *kmb);
                            });

                        leaf.points
                            .iter()
                            .zip(known_points.iter())
                            .for_each(|(p, kp)| {
                                p.iter().zip(kp.iter()).for_each(|(pp, kpp)| {
                                    assert!(*pp == *kpp);
                                });
                            });
                    }
                    KdOffshoot::Branch(_) => assert!(false),
                }
            }
        }
    }
    
    #[test]
    fn kd_tree_nearest(){
        let mut kd_tree = KdTree::<f32>::new(2, 2);
        kd_tree.add(vec![1.0,2.0], 0);
        kd_tree.add(vec![-1.0,3.0], 1);
        kd_tree.add(vec![-2.0,3.0], 2);
        kd_tree.add(vec![2.0,4.0], 4);

        let point = vec![1.5,2.0];
        let near = kd_tree.nearest(&point, 4);

        let known_dist = vec![0.5,2.5,3.5,4.5];
        let known_index = vec![0,4,1,2];
        near.iter().zip(known_dist.iter()).zip(known_index.iter()).for_each(|((np, kd),ki)|{
            assert!(np.0==*kd);
            assert!(np.1==*ki);
        });
    }
}
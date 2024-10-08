use crate::shared::{float::Float, point::Point, primitive::Primitive};
use core::cmp::{min, Ordering};
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::BinaryHeap;

pub struct KdLeaf<P: Point> {
    pub capacity: usize,
    pub points: Vec<P>,
    pub bucket: Vec<usize>,
}
pub struct KdBranch<P: Point> {
    pub left: Box<KdTree<P>>,
    pub right: Box<KdTree<P>>,
    pub split_value: P::Primitive,
    pub split_dimension: usize,
}

pub struct KdShared<P: Point> {
    pub size: usize,
    pub min_bounds: P,
    pub max_bounds: P,
}

pub enum KdNode<P: Point> {
    Leaf(KdLeaf<P>),
    Branch(KdBranch<P>),
}

pub struct KdTree<P: Point> {
    pub shared: KdShared<P>,
    pub node: KdNode<P>,
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

        let min = shared.min_bounds.coeff(split_dimension);
        let max = shared.max_bounds.coeff(split_dimension);
        let split_value = min + (max - min) / P::Primitive::usize(2);

        let left = Box::new(KdTree::<P>::new(self.capacity));
        let right = Box::new(KdTree::<P>::new(self.capacity));
        let mut branch = KdBranch::new(left, right, split_value, split_dimension);
        let mut count = 0;
        while !self.points.is_empty() {
            let point = self.points.swap_remove(0);
            let data = self.bucket.swap_remove(0);
            if branch.branch_left(shared, &point,count) {
                branch.left.add(point, data); //unnecessary match happens below this
            } else {
                branch.right.add(point, data); //unnecessary match happens below this
            }
            count+=1;
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

    fn branch_left(&self, shared: &KdShared<P>, point: &P, size: usize) -> bool {
        
        if point.coeff(self.split_dimension) == self.split_value {
            size%2 == 0 //Handle the repeated collision case
        } else if shared.min_bounds.coeff(self.split_dimension) == self.split_value {
            point.coeff(self.split_dimension) <= self.split_value
        } else {
            point.coeff(self.split_dimension) < self.split_value
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
                let next = match branch.branch_left(&self.shared, &point,self.shared.size) {
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
                    let candidate = match branch.branch_left(&self.shared, point,self.shared.size) {
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
                            distance: point.l1_distance(p),
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

pub struct HeapElement<P: Primitive, O> {
    pub distance: P,
    pub element: O,
}

impl<P: Primitive, O> Ord for HeapElement<P, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl<P: Primitive, O> PartialOrd for HeapElement<P, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Primitive, O> Eq for HeapElement<P, O> {}

impl<P: Primitive, O> PartialEq for HeapElement<P, O> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
use crate::{
    random::pcg::PermutedCongruentialGenerator,
    shared::{float::Float, point::Point,primitive::Primitive},
};
use alloc::vec::Vec;
use alloc::boxed::Box;

pub struct IsoLeaf {
    pub count: usize,
}
pub struct IsoBranch<P: Point> {
    pub left: Box<IsoNode<P>>,
    pub right: Box<IsoNode<P>>,
    pub split_vector: P,
    pub split_point: P,
}

pub enum IsoNode<P: Point> {
    Leaf(IsoLeaf),
    Branch(IsoBranch<P>),
}

pub struct IsoTree<P: Point> {
    pub root: IsoNode<P>,
}

impl IsoLeaf {
    fn new(count: usize) -> Self {
        Self { count }
    }
}

impl<P: Point> IsoBranch<P> {
    fn new(
        left: Box<IsoNode<P>>,
        right: Box<IsoNode<P>>,
        split_vector: P,
        split_point: P,
    ) -> Self {
        Self {
            left,
            right,
            split_vector,
            split_point,
        }
    }
}

impl<P: Point> IsoTree<P> {
    pub fn new(
        points: &[P],
        max_depth: usize,
        extension_level: usize,
        rng: &mut PermutedCongruentialGenerator,
    ) -> Self {
        let root = Self::make_node(points, 0, max_depth, extension_level, rng);
        Self { root }
    }

    fn make_node(
        points: &[P],
        current_depth: usize,
        max_depth: usize,
        extension_level: usize,
        rng: &mut PermutedCongruentialGenerator,
    ) -> IsoNode<P> {
        let point_count = points.len();

        if current_depth >= max_depth || point_count <= 1 {
            IsoNode::Leaf(IsoLeaf::new(point_count))
        } else {
            let split_point = {

                let point_max = points.iter().fold(P::MIN, |acc,p|p.greater(&acc));
                let point_min = points.iter().fold(P::MAX, |acc,p|p.lesser(&acc));
                P::uniform(&point_min, &point_max,rng)

            };
            //Creates a split vector
            let normals = rng.normal(P::Primitive::ZERO, P::Primitive::ONE, extension_level);
            let split_vector = P::partial_random(normals,rng);

            let mut points_left = Vec::with_capacity(points.len()/2);
            let mut points_right = Vec::with_capacity(points.len()/2);

            points.iter().for_each(|point| {
                match Self::branch_left(point, &split_point, &split_vector) {
                    true => points_left.push(point.clone()),
                    false => points_right.push(point.clone()),
                }
            });

            let left = Box::new(Self::make_node(
                &points_left,
                current_depth + 1,
                max_depth,
                extension_level,
                rng,
            ));
            let right = Box::new(Self::make_node(
                &points_right,
                current_depth + 1,
                max_depth,
                extension_level,
                rng,
            ));

            IsoNode::Branch(IsoBranch::new(left, right, split_vector, split_point))
        }
    }
    fn branch_left(
        point: &P,
        split_point: &P,
        split_vector: &P,
    ) -> bool {
        point.sub(split_point).dot(split_vector) <= P::Primitive::ZERO
        
    }

    pub fn path_length(node: &IsoNode<P>, point: &P) ->  P::Primitive {
        match node {
            IsoNode::Leaf(leaf) => {
                if leaf.count <= 1 {
                    P::Primitive::ZERO
                } else {
                    Self::c_factor(leaf.count)
                }
            }
            IsoNode::Branch(branch) => {
                let child =
                    match Self::branch_left(point, &branch.split_point, &branch.split_vector) {
                        true => &branch.left,
                        false => &branch.right,
                    };
                P::Primitive::ONE + Self::path_length(child, point)
            }
        }
    }

    pub fn c_factor(n: usize) -> P::Primitive {
        P::Primitive::usize(2) * ((P::Primitive::usize(n) - P::Primitive::ONE).ln() + P::Primitive::GAMMA)
            - (P::Primitive::usize(2) * (P::Primitive::usize(n) - P::Primitive::ONE) / P::Primitive::usize(n))
    }
}
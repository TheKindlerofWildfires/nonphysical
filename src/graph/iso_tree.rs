use crate::{
    random::pcg::PermutedCongruentialGenerator,
    shared::{float::Float, point::Point,real::Real},
};
use alloc::vec::Vec;
use alloc::boxed::Box;

pub struct IsoLeaf {
    count: usize,
}
pub struct IsoBranch<P: Point> {
    left: Box<IsoNode<P>>,
    right: Box<IsoNode<P>>,
    split_vector: P,
    split_point: P,
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

        let node = if current_depth >= max_depth || point_count <= 1 {
            IsoNode::Leaf(IsoLeaf::new(point_count))
        } else {
            let split_point = {

                let point_max = points.into_iter().fold(P::MIN, |acc,p|p.greater(&acc));
                let point_min = points.into_iter().fold(P::MAX, |acc,p|p.lesser(&acc));
                P::uniform(&point_min, &point_max,rng)

            };
            //Creates a split vector
            let normals = rng.normal(P::Primitive::ZERO, P::Primitive::ONE, extension_level);
            let split_vector = P::partial_random(normals,rng);

            let mut points_left = Vec::with_capacity(points.len()/2);
            let mut points_right = Vec::with_capacity(points.len()/2);

            points.into_iter().for_each(|point| {
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
        };
        node
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

#[cfg(test)]
mod iso_tree_tests {
    use core::borrow::Borrow;
    use alloc::vec;

    use crate::shared::point::StaticPoint;

    use super::*;
    #[test]
    fn print_iso_tree(){
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([-5.6424039931897765, -1.9620561766472002]),
            StaticPoint::new([-9.821995596375428, -3.1921112766174997]),
            StaticPoint::new([-4.992109362834896, -2.0745015313494455]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
        ];
        let mut rng = PermutedCongruentialGenerator::new(1, 1);
        let tree: IsoTree<StaticPoint<f32, 2>> = IsoTree::new(&data, 4, 1, &mut rng);
        let node = tree.root;
        print_node(&node, 0);


    }

    fn print_node(node: &IsoNode<StaticPoint<f32, 2>>,depth: usize){
        match node {
            IsoNode::Leaf(leaf) =>{
                println!("{}: Leaf: {}", depth, leaf.count);
            },
            IsoNode::Branch(branch) => {
                println!("{}: Branch: {:?} {:?}", depth,branch.split_point, branch.split_vector);
                print_node(&branch.left, depth+1);
                print_node(&branch.right, depth+1);
            },
        }

    }
    #[test]
    fn create_tree_static() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([-5.6424039931897765, -1.9620561766472002]),
            StaticPoint::new([-9.821995596375428, -3.1921112766174997]),
            StaticPoint::new([-4.992109362834896, -2.0745015313494455]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
        ];
        let mut rng = PermutedCongruentialGenerator::new(1, 1);
        let tree = IsoTree::new(&data, 4, 1, &mut rng);
        let node = tree.root;
        match node {
            IsoNode::Leaf(_) => {
                unreachable!()
            }
            IsoNode::Branch(branch) => {
                match branch.left.borrow() {
                    IsoNode::Leaf(leaf) => {
                        assert!(leaf.count == 0);
                    }
                    IsoNode::Branch(_) => {
                        unreachable!()
                    }
                }
                match branch.right.borrow() {
                    IsoNode::Leaf(_) => {
                        unreachable!()
                    }
                    IsoNode::Branch(branch) => {
                        match branch.left.borrow() {
                            IsoNode::Leaf(_) => {
                                unreachable!()
                            }
                            IsoNode::Branch(branch) => {
                                match branch.left.borrow() {
                                    IsoNode::Leaf(leaf) => {
                                        assert!(leaf.count == 0);
                                    }
                                    IsoNode::Branch(_) => {
                                        unreachable!()
                                    }
                                }
                                match branch.right.borrow() {
                                    IsoNode::Leaf(_) => {
                                        unreachable!()
                                    }
                                    IsoNode::Branch(branch) => {
                                        match branch.left.borrow() {
                                            IsoNode::Leaf(leaf) => {
                                                assert!(leaf.count == 0);
                                            }
                                            IsoNode::Branch(_) => {
                                                unreachable!()
                                            }
                                        }
                                        match branch.right.borrow() {
                                            IsoNode::Leaf(leaf) => {
                                                assert!(leaf.count == 2);
                                            }
                                            IsoNode::Branch(_) => {
                                                unreachable!()
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        match branch.right.borrow() {
                            IsoNode::Leaf(_) => {
                                unreachable!()
                            }
                            IsoNode::Branch(branch) => {
                                match branch.left.borrow() {
                                    IsoNode::Leaf(leaf) => {
                                        assert!(leaf.count == 1);
                                    }
                                    IsoNode::Branch(_) => {
                                        unreachable!()
                                    }
                                }
                                match branch.right.borrow() {
                                    IsoNode::Leaf(_) => {
                                        unreachable!();
                                    }
                                    IsoNode::Branch(branch) => {
                                        match branch.left.borrow() {
                                            IsoNode::Leaf(leaf) => {
                                                assert!(leaf.count == 0);
                                            }
                                            IsoNode::Branch(_) => {
                                                unreachable!()
                                            }
                                        }
                                        match branch.right.borrow() {
                                            IsoNode::Leaf(leaf) => {
                                                assert!(leaf.count == 2);
                                            }
                                            IsoNode::Branch(_) => {
                                                unreachable!()
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

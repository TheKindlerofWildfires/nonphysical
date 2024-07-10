use crate::{
    random::pcg::PermutedCongruentialGenerator,
    shared::{complex::Complex, float::Float, point::Point},
};

pub struct IsoLeaf {
    count: usize,
}
pub struct IsoBranch<T: Float, const N: usize> {
    left: Box<IsoNode<T, N>>,
    right: Box<IsoNode<T, N>>,
    split_vector: Point<T, N>,
    split_point: Point<T, N>,
}

pub enum IsoNode<T: Float, const N: usize> {
    Leaf(IsoLeaf),
    Branch(IsoBranch<T, N>),
}

pub struct IsoTree<T: Float, const N: usize> {
    pub root: IsoNode<T, N>,
}

impl IsoLeaf {
    fn new(count: usize) -> Self {
        Self { count }
    }
}

impl<T: Float, const N: usize> IsoBranch<T, N> {
    fn new(
        left: Box<IsoNode<T, N>>,
        right: Box<IsoNode<T, N>>,
        split_vector: Point<T, N>,
        split_point: Point<T, N>,
    ) -> Self {
        Self {
            left,
            right,
            split_vector,
            split_point,
        }
    }
}

impl<T: Float, const N: usize> IsoTree<T, N> {
    pub fn new(
        points: &Vec<Point<T, N>>,
        max_depth: usize,
        extension_level: usize,
        rng: &mut PermutedCongruentialGenerator<T>,
    ) -> Self {
        let root = Self::make_node(points, 0, max_depth, extension_level, rng);
        Self { root }
    }

    fn make_node(
        points: &Vec<Point<T, N>>,
        current_depth: usize,
        max_depth: usize,
        extension_level: usize,
        rng: &mut PermutedCongruentialGenerator<T>,
    ) -> IsoNode<T, N> {
        let point_count = points.len();

        let node = if current_depth >= max_depth || point_count <= 1 {
            IsoNode::Leaf(IsoLeaf::new(point_count))
        } else {
            let split_point = {
                let mut point_max = points[0].clone();
                let mut point_min = points[0].clone();
                points.iter().skip(1).for_each(|s| {
                    s.data
                        .iter()
                        .zip(point_max.data.iter_mut())
                        .zip(point_min.data.iter_mut())
                        .for_each(|((sp, mxp), mnp)| {
                            *mxp = sp.greater(*mxp);
                            *mnp = sp.lesser(*mnp);
                        })
                });
                let mut p = Point::ZERO;
                p.data
                    .iter_mut()
                    .zip(point_max.data.iter())
                    .zip(point_min.data.iter())
                    .for_each(|((sp, mxp), mnp)| {
                        *sp = T::usize(rng.next_u32() as usize) / T::usize(u32::MAX as usize)
                            * (*mxp - *mnp)
                            + *mnp;
                    });
                p
            };
            //zeros a split vector
            let mut split_vector = Point::ZERO;
            rng.normal(Complex::ZERO, T::ONE, 1)[0].real;

            //sets the vars to normal randoms
            split_vector
                .data
                .iter_mut()
                .for_each(|n| *n = rng.normal(Complex::ZERO, T::ONE, 1)[0].real);
            //rezeros unextended values (this feels backwards)
            let mut unextended = (0..N).collect();
            rng.shuffle_usize(&mut unextended);
            unextended[0..N - extension_level - 1]
                .iter_mut()
                .for_each(|i| split_vector.data[*i] = T::ZERO);

            let mut points_left = Vec::new();
            let mut points_right = Vec::new();

            points.iter().for_each(|point| {
                match Self::branch_left(&point, &split_point, &split_vector) {
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
        point: &Point<T, N>,
        split_point: &Point<T, N>,
        split_vector: &Point<T, N>,
    ) -> bool {
        point
            .data
            .iter()
            .zip(split_point.data.iter())
            .map(|(pp, sp)| *pp - *sp)
            .zip(split_vector.data.iter())
            .fold(T::ZERO, |acc, (pp, v)| acc + pp * (*v))
            < T::ZERO
    }

    pub fn path_length(node: &IsoNode<T, N>, point: &Point<T, N>) -> T {
        let length = match node {
            IsoNode::Leaf(leaf) => {
                if leaf.count <= 1 {
                    T::ZERO
                } else {
                    Self::c_factor(leaf.count)
                }
            }
            IsoNode::Branch(branch) => {
                let child =
                    match Self::branch_left(&point, &branch.split_point, &branch.split_vector) {
                        true => &branch.left,
                        false => &branch.right,
                    };
                T::ONE + Self::path_length(child, point)
            }
        };
        length
    }

    pub fn c_factor(n: usize) -> T {
        T::usize(2) * ((T::usize(n) - T::ONE).ln() + T::GAMMA)
            - (T::usize(2) * (T::usize(n) - T::ONE) / T::usize(n))
    }
}

#[cfg(test)]
mod iso_tree_tests {
    use std::borrow::Borrow;

    use super::*;

    #[test]
    fn create_tree_static() {
        let data = vec![
            Point::new([9.308548692822459, 2.1673586347139224]),
            Point::new([-5.6424039931897765, -1.9620561766472002]),
            Point::new([-9.821995596375428, -3.1921112766174997]),
            Point::new([-4.992109362834896, -2.0745015313494455]),
            Point::new([10.107315875917662, 2.4489015959094216]),
        ];
        let mut rng = PermutedCongruentialGenerator::<f32>::new(1, 1);
        let tree = IsoTree::new(&data, 4, 1, &mut rng);
        let node = tree.root;
        match node {
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
                            IsoNode::Leaf(_) => {
                                unreachable!();
                            }
                            IsoNode::Branch(branch) => {
                                match branch.left.borrow() {
                                    IsoNode::Leaf(leaf) => {
                                        assert!(leaf.count == 1)
                                    }
                                    IsoNode::Branch(_) => {
                                        unreachable!()
                                    }
                                }
                                match branch.right.borrow() {
                                    IsoNode::Leaf(leaf) => {
                                        assert!(leaf.count == 1)
                                    }
                                    IsoNode::Branch(_) => {
                                        unreachable!()
                                    }
                                }
                            }
                        }
                        match branch.right.borrow() {
                            IsoNode::Leaf(leaf) => {
                                dbg!(leaf.count);
                            }
                            IsoNode::Branch(branch) => {
                                match branch.left.borrow() {
                                    IsoNode::Leaf(leaf) => {
                                        assert!(leaf.count == 1)
                                    }
                                    IsoNode::Branch(_) => {
                                        unreachable!()
                                    }
                                }
                                match branch.right.borrow() {
                                    IsoNode::Leaf(leaf) => {
                                        assert!(leaf.count == 1)
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
                    IsoNode::Leaf(leaf) => {
                        assert!(leaf.count == 1)
                    }
                    IsoNode::Branch(_) => {
                        unreachable!()
                    }
                };
            }
        }
    }

    #[test]
    fn create_tree_dynamic() {
        todo!()
    }

    #[test]
    fn path_length() {
        todo!()
    }
}

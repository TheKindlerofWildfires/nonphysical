

#[cfg(test)]
mod iso_tree_tests {
    use std::borrow::Borrow;

    use nonphysical_core::{graph::iso_tree::{IsoNode, IsoTree}, random::pcg::PermutedCongruentialGenerator, shared::point::{Point, StaticPoint}};
    use nonphysical_std::shared::primitive::F32;

    #[test]
    fn print_iso_tree(){
        let data = vec![
            StaticPoint::new([F32(9.308548692822459), F32(2.1673586347139224)]),
            StaticPoint::new([F32(-5.6424039931897765), F32( -1.9620561766472002)]),
            StaticPoint::new([F32(-9.821995596375428), F32( -3.1921112766174997)]),
            StaticPoint::new([F32(-4.992109362834896), F32( -2.0745015313494455)]),
            StaticPoint::new([F32(10.107315875917662), F32( 2.4489015959094216)]),
        ];
        let mut rng = PermutedCongruentialGenerator::new(1, 1);
        let tree: IsoTree<StaticPoint<F32, 2>> = IsoTree::new(&data, 4, 1, &mut rng);
        let node = tree.root;
        print_node(&node, 0);


    }

    fn print_node(node: &IsoNode<StaticPoint<F32, 2>>,depth: usize){
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
            StaticPoint::new([F32(9.308548692822459), F32(2.1673586347139224)]),
            StaticPoint::new([F32(-5.6424039931897765), F32( -1.9620561766472002)]),
            StaticPoint::new([F32(-9.821995596375428), F32( -3.1921112766174997)]),
            StaticPoint::new([F32(-4.992109362834896), F32( -2.0745015313494455)]),
            StaticPoint::new([F32(10.107315875917662), F32(2.4489015959094216)]),
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

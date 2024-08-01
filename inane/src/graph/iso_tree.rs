

#[cfg(test)]
mod iso_tree_tests {
    use std::borrow::Borrow;

    use baseless::{graph::iso_tree::{IsoNode, IsoTree}, random::pcg::PermutedCongruentialGenerator, shared::point::{Point, StaticPoint}};

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

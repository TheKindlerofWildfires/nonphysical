use crate::{
    cluster::Classification::{Core, Edge, Noise},
    shared::point::Point,
};

use super::Classification;
use alloc::vec;
use alloc::vec::Vec;

pub struct Dbscan<P:Point>{
    epsilon: P::Primitive,
    min_points: usize
}
impl<P:Point> Dbscan<P>{
    pub fn new(epsilon: P::Primitive, min_points: usize) -> Self{
        Self { epsilon, min_points }

    }
    pub fn cluster(&self, data: &[P])-> Vec<Classification>{
        let mut classifications = vec![Noise; data.len()];
        let mut visited = vec![false; data.len()];
        let mut cluster = 0;
        let mut queue = Vec::new();

        (0..data.len()).for_each(|i| {
            if !visited[i] {
                visited[i] = true;
                queue.push(i);

                let mut new_cluster=0;
                //expand the cluster
                while let Some(idx) = queue.pop() {
                    let neighbors = data
                        .iter()
                        .enumerate()
                        .filter(|(_, pt)| data[idx].l1_distance(pt) < self.epsilon)
                        .map(|(idx, _)| idx)
                        .collect::<Vec<_>>();
                    if neighbors.len() > self.min_points {
                        new_cluster = 1;
                        classifications[idx] = Core(cluster);
                        neighbors.iter().for_each(|neighbor_idx| {
                            if classifications[*neighbor_idx] == Noise {
                                classifications[*neighbor_idx] = Edge(cluster)
                            }
                            if !visited[*neighbor_idx] {
                                visited[*neighbor_idx] = true;
                                queue.push(*neighbor_idx);
                            }
                        });
                    }
                }
                cluster+=new_cluster;
            }
        });
        classifications
    }
}



#[cfg(test)]
mod dbscan_tests{
    use alloc::vec;
    use crate::shared::point::StaticPoint;

    use super::*;

    #[test]
    fn dbscan_simple() {
        let data = vec![
            StaticPoint::new([9.308548692822459, 2.1673586347139224]),
            StaticPoint::new([-5.6424039931897765, -1.9620561766472002]),
            StaticPoint::new([-9.821995596375428, -3.1921112766174997]),
            StaticPoint::new([-4.992109362834896, -2.0745015313494455]),
            StaticPoint::new([10.107315875917662, 2.4489015959094216]),
            StaticPoint::new([-7.962477597931141, -5.494741864480315]),
            StaticPoint::new([10.047917462523671, 5.1631966716389766]),
            StaticPoint::new([-5.243921934674187, -2.963359100733349]),
            StaticPoint::new([-9.940544426622527, -3.2655473073528816]),
            StaticPoint::new([8.30445373000034, 2.129694332932624]),
            StaticPoint::new([-9.196460281784482, -3.987773678358418]),
            StaticPoint::new([-10.513583123594056, -2.5364233580562887]),
            StaticPoint::new([9.072668506714033, 3.405664632524281]),
            StaticPoint::new([-7.031861004012987, -2.2616818331210844]),
            StaticPoint::new([9.627963795272553, 4.502533177849574]),
            StaticPoint::new([-10.442760023564471, -5.0830680881481065]),
            StaticPoint::new([8.292151321984209, 3.8776876670218834]),
            StaticPoint::new([-6.51560033683665, -3.8185628318207585]),
            StaticPoint::new([-10.887633624071544, -4.416570704487158]),
            StaticPoint::new([-9.465804800021168, -2.2222090878656884]),
        ];
        let dbscan = Dbscan::new(9.0,5);
        let mask = dbscan.cluster(&data);
        let known_mask = vec![Core(0), Core(1), Core(1), Core(1), Core(0), Core(1), Core(0), Core(1), Core(1), Core(0), Core(1), Core(1), Core(0), Core(1), Core(0), Core(1), Core(0), Core(1), Core(1), Core(1)];
        mask.iter().zip(known_mask.iter()).for_each(|(m,k)|{
            assert!(*m==*k);
        });
        

    }
}
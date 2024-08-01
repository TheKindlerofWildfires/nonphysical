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
                        neighbors.into_iter().for_each(|neighbor_idx| {
                            if classifications[neighbor_idx] == Noise {
                                classifications[neighbor_idx] = Edge(cluster)
                            }
                            if !visited[neighbor_idx] {
                                visited[neighbor_idx] = true;
                                queue.push(neighbor_idx);
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

use crate::{
    cluster::Classification::Core, shared::{float::Float, point::Point, primitive::Primitive}
};
use super::Classification;
use alloc::vec::Vec;
use alloc::vec;

pub struct Kmeans<P:Point>{
    pub centroids: Vec<P>,
    iterations: usize,
}
impl<P:Point> Kmeans<P>{
    pub fn new(seed_data: &[P], clusters: usize,iterations: usize) -> Self{
        let mut taken: Vec<bool> = vec![false; seed_data.len()];
        let mut centroids = vec![P::ORIGIN;clusters];
        taken[0] = true; 
        centroids[0] = seed_data[0];

        (1..clusters).for_each(|i| {
            let mut max_index = 0;
            let mut max_distance = P::Primitive::MIN;

            seed_data.into_iter().enumerate().for_each(|(i, c)| {
                if !taken[i] {
                    let mut min_distance = P::Primitive::MAX;

                    centroids.iter().for_each(|centroid| {
                        let dx = c.l1_distance(&centroid);
                        if dx < min_distance {
                            min_distance = dx;
                        }
                    });

                    if min_distance > max_distance {
                        max_distance = min_distance;
                        max_index = i;
                    }
                }

            });
            taken[max_index] = true;
            centroids[i] = seed_data[max_index];
        });
        Self { centroids,iterations }
    }

    pub fn cluster(&mut self, data: &[P]) -> Vec<Classification>{
        let mut counts = vec![0; self.centroids.len()];

        let mut membership = vec![Core(0); data.len()];
        (0..self.iterations).for_each(|_| {
            data.into_iter().enumerate().for_each(|(i, c)| {
                let old = membership[i];
                match old {
                    Core(op) => {
                        let mut cluster = old;
                
                        let mut dist = c.l1_distance(&self.centroids[op]);
        
                        self.centroids.iter().enumerate().for_each(|(j, centroid)| {
                            let square_distance = c.l1_distance(centroid);
                            if square_distance < dist {
                                dist = square_distance;
                                cluster = Core(j);
                            }
                        });
                        membership[i] = cluster;
                    },
                    _ => {unreachable!()}
                }
                
            });
            counts.iter_mut().for_each(|x| *x = 0);
            self.centroids.iter_mut().for_each(|c| *c = P::ORIGIN);
    
            data.into_iter().zip(membership.iter()).for_each(|(c,m)|{
                match m {
                    Core(mp) => {
                        counts[*mp] +=1;
                        self.centroids[*mp] = self.centroids[*mp].add(*c);
                    },
                    _ => {unreachable!()}

                }

            });

            self.centroids.iter_mut().zip(counts.iter()).for_each(|(centroid, count)|{
                match count {
                    0 => {
                        *centroid = P::ORIGIN;
                    },
                    size => {
                        let scaler = P::Primitive::usize(*size).recip();
                        centroid.scale(scaler);
                    }
                }
            });
        });
        membership
    }   
}
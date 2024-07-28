use crate::{
    cluster::Classification::Core, shared::{float::Float, point::Point, real::Real}
};
use super::Classification;


pub struct Kmeans<P:Point>{
    centroids: Vec<P>,
    iterations: usize,
}
impl<P:Point> Kmeans<P>{
    fn new(seed_data: &[P], clusters: usize,iterations: usize) -> Self{
        let mut taken: Vec<bool> = vec![false; seed_data.len()];
        let mut centroids = vec![P::ZERO;clusters];
        taken[0] = true; 
        centroids[0] = seed_data[0].clone();

        (1..clusters).for_each(|i| {
            let mut max_index = 0;
            let mut max_distance = P::Primitive::MIN;

            seed_data.iter().enumerate().for_each(|(i, c)| {
                if !taken[i] {
                    let mut min_distance = P::Primitive::MAX;

                    centroids.iter().for_each(|centroid| {
                        let dx = c.l1_distance(centroid);
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
            centroids[i] = seed_data[max_index].clone();
        });
        Self { centroids,iterations }
    }

    fn cluster(&mut self, data: &[P]) -> Vec<Classification>{
        let mut counts = vec![0; self.centroids.len()];

        let mut membership = vec![Core(0); data.len()];
        (0..self.iterations).for_each(|_| {
            data.iter().enumerate().for_each(|(i, c)| {
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
            self.centroids.iter_mut().for_each(|c| *c = P::ZERO);
    
            data.iter().zip(membership.iter()).for_each(|(c,m)|{
                match m {
                    Core(mp) => {
                        counts[*mp] +=1;
                        self.centroids[*mp] = self.centroids[*mp].add(c);
                    },
                    _ => {unreachable!()}

                }

            });

            self.centroids.iter_mut().zip(counts.iter()).for_each(|(centroid, count)|{
                match count {
                    0 => {
                        *centroid = P::ZERO;
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

#[cfg(test)]
mod kmeans_tests{
    use crate::shared::{float::Float, point::{Point, StaticPoint}};

    use super::*;

    #[test]
    fn kmeans_simple() {
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

        let mut kmeans: Kmeans<StaticPoint<f32, 2>> = Kmeans::new(&data, 2,32);

        let mask = kmeans.cluster(&data);

        let binding = vec![0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1];
        let known_mask = binding.iter().map(|i| Core(*i));

        mask.iter().zip(known_mask).for_each(|(m,k)|{
            assert!(*m==k);
        });
        
        let known_centroids = vec![vec![9.2515742, 3.38500524],vec![-8.2813197, -3.3291236]];

        kmeans.centroids.iter().zip(known_centroids.iter()).for_each(|(c,k)|{
            c.data.iter().zip(k.iter()).for_each(|(cp, kp)|{
                assert!((*cp-*kp).l2_norm()<f32::EPSILON);
            })
        });

    }
}
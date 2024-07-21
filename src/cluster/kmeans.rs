use crate::{
    cluster::Classification::Core, random::pcg::PermutedCongruentialGenerator, shared::{float::Float, point::Point}
};
use super::Classification;
trait Kmeans<T: Float,const N: usize> {
    fn new(input: &Self, count: usize,seed:usize) -> Self;

    fn cluster(
        input: &Self,
        centroids: &mut Self,
        iterations: usize,
    ) -> Vec<Classification>
    where
        Self: Sized;
}

impl<T: Float,const N: usize> Kmeans<T,N> for Vec<Point<T,N>> {
    fn new(input: &Self, count: usize,seed:usize) ->Self {
        let mut taken = vec![false; input.len()];
        let mut centroids = Vec::with_capacity(count);
        let mut pcg = PermutedCongruentialGenerator::<T>::new(seed as u32, seed as u32+1);
        let first = pcg.next_u32() as usize % input.len();
        taken[first] = true; 
        centroids.push(input[first].clone());

        (1..count).for_each(|_| {
            let mut max_index = 0;
            let mut max_distance = T::MIN;

            input.iter().enumerate().for_each(|(i, c)| {
                if !taken[i] {
                    let mut min_distance = T::MIN;

                    centroids.iter().for_each(|centroid| {
                        let dx = c.distance(centroid);
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
            centroids.push(input[max_index].clone());
        });
        centroids
    }

    //doesn't have the early exit
    fn cluster(
        input: &Self,
        centroids: &mut Self,
        iterations: usize,
    ) -> Vec<Classification>
    where
        Self: Sized,
    {
        let mut counts = vec![0; centroids.len()];

        let mut membership = vec![Core(0); input.len()];
        (0..iterations).for_each(|_| {
            input.iter().enumerate().for_each(|(i, c)| {
                let old = membership[i];
                match old {
                    Core(op) => {
                        let mut cluster = old;
                
                        let mut dist = c.distance(&centroids[op]);
        
                        centroids.iter().enumerate().for_each(|(j, centroid)| {
                            let square_distance = c.distance(centroid);
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
            centroids.iter_mut().for_each(|c| c.data.iter_mut().for_each(|d|*d = T::ZERO));
    
            input.iter().zip(membership.iter()).for_each(|(c,m)|{
                match m {
                    Core(mp) => {
                        counts[*mp] +=1;
    
                        centroids[*mp].data.iter_mut().zip(c.data.iter()).for_each(|(centroid_p,cp)|{
                            *centroid_p += *cp 
                        });
                    },
                    _ => {unreachable!()}

                }

            });

            centroids.iter_mut().zip(counts.iter()).for_each(|(centroid, count)|{
                match count {
                    0 => {
                        centroid.data.iter_mut().for_each(|cp| *cp = T::ZERO);
                    },
                    size => {
                        centroid.data.iter_mut().for_each(|cp| *cp /=T::usize(*size))
                    }
                }
            });
        });
        membership
    }

}

#[cfg(test)]
mod kmeans_tests{
    use crate::shared::point::Point;

    use super::*;

    #[test]
    fn kmeans_simple() {
        let data = vec![
            Point::new([9.308548692822459, 2.1673586347139224]),
            Point::new([-5.6424039931897765, -1.9620561766472002]),
            Point::new([-9.821995596375428, -3.1921112766174997]),
            Point::new([-4.992109362834896, -2.0745015313494455]),
            Point::new([10.107315875917662, 2.4489015959094216]),
            Point::new([-7.962477597931141, -5.494741864480315]),
            Point::new([10.047917462523671, 5.1631966716389766]),
            Point::new([-5.243921934674187, -2.963359100733349]),
            Point::new([-9.940544426622527, -3.2655473073528816]),
            Point::new([8.30445373000034, 2.129694332932624]),
            Point::new([-9.196460281784482, -3.987773678358418]),
            Point::new([-10.513583123594056, -2.5364233580562887]),
            Point::new([9.072668506714033, 3.405664632524281]),
            Point::new([-7.031861004012987, -2.2616818331210844]),
            Point::new([9.627963795272553, 4.502533177849574]),
            Point::new([-10.442760023564471, -5.0830680881481065]),
            Point::new([8.292151321984209, 3.8776876670218834]),
            Point::new([-6.51560033683665, -3.8185628318207585]),
            Point::new([-10.887633624071544, -4.416570704487158]),
            Point::new([-9.465804800021168, -2.2222090878656884]),
        ];

        let mut centroids = <Vec<Point<f32,2>> as Kmeans<f32,2>>::new(&data,2,0);
        let mask = <Vec<Point<f32,2>> as Kmeans<f32,2>>::cluster(&data, &mut centroids, 32);

        let binding = vec![1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0];
        let known_mask = binding.iter().map(|i| Core(*i));

        mask.iter().zip(known_mask).for_each(|(m,k)|{
            assert!(*m==k);
        });
        
        let known_centroids = vec![vec![-8.2813197, -3.3291236],vec![9.2515742, 3.38500524],];

        centroids.iter().zip(known_centroids.iter()).for_each(|(c,k)|{
            c.data.iter().zip(k.iter()).for_each(|(cp, kp)|{
                assert!((*cp-*kp).square_norm()<f32::EPSILON);
            })
        });

    }
}
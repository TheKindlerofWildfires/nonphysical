use crate::{
    random::pcg::PermutedCongruentialGenerator,
    cluster::Classification::Core,
    shared::{complex::Complex, float::Float},
};
use super::Classification;
trait Kmeans<T: Float> {
    fn new(input: &Self, count: usize) -> Vec<Vec<Complex<T>>>;

    fn cluster(
        input: &Self,
        centroids: &mut Vec<Vec<Complex<T>>>,
        iterations: usize,
    ) -> Vec<Classification>
    where
        Self: Sized;

    fn square_distance(a: &Vec<Complex<T>>, b: &Vec<Complex<T>>) -> T;
}

impl<T: Float> Kmeans<T> for Vec<Vec<Complex<T>>> {
    fn new(input: &Self, count: usize) -> Vec<Vec<Complex<T>>> {
        let mut taken = vec![false; input.len()];
        let mut centroids = Vec::with_capacity(count);
        let mut pcg = PermutedCongruentialGenerator::<T>::new_timed();
        let first = pcg.next_u32() as usize % input.len();
        taken[first] = true; 
        centroids.push(input[first].iter().map(|in_val| in_val.clone()).collect());

        (1..count).for_each(|_| {
            let mut max_index = 0;
            let mut max_distance = T::MIN;

            input.iter().enumerate().for_each(|(i, c)| {
                if !taken[i] {
                    let mut min_distance = T::MIN;

                    centroids.iter().for_each(|centroid| {
                        let dx = Self::square_distance(c, centroid);
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
            centroids.push(input[max_index].iter().map(|in_val| in_val.clone()).collect());
        });
        centroids
    }

    //doesn't have the early exit
    fn cluster(
        input: &Self,
        centroids: &mut Vec<Vec<Complex<T>>>,
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
                
                        let mut dist = Self::square_distance(c, &centroids[op]);
        
                        centroids.iter().enumerate().for_each(|(j, centroid)| {
                            let square_distance = Self::square_distance(c, centroid);
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
            centroids.iter_mut().for_each(|c| c.iter_mut().for_each(|d|*d = Complex::ZERO));
    
            input.iter().zip(membership.iter()).for_each(|(c,m)|{
                match m {
                    Core(mp) => {
                        counts[*mp] +=1;
    
                        centroids[*mp].iter_mut().zip(c.iter()).for_each(|(centroid_p,cp)|{
                            *centroid_p += *cp 
                        });
                    },
                    _ => {unreachable!()}

                }

            });

            centroids.iter_mut().zip(counts.iter()).for_each(|(centroid, count)|{
                match count {
                    0 => {
                        centroid.iter_mut().for_each(|cp| *cp = Complex::ZERO);
                    },
                    size => {
                        centroid.iter_mut().for_each(|cp| *cp = *cp/T::usize(*size))
                    }
                }
            });
        });
        membership
    }

    fn square_distance(a: &Vec<Complex<T>>, b: &Vec<Complex<T>>) -> T {
        a.iter()
            .zip(b.iter())
            .fold(T::ZERO, |s, (ap, bp)| s + (*ap - *bp).square_norm())
    }
}

#[cfg(test)]
mod kmeans_tests{
    use super::*;

    #[test]
    fn test_kmeans() {
        let data = vec![
            vec![Complex::new(9.308548692822459,0.0),Complex::new(2.1673586347139224,0.0)],
            vec![Complex::new(-5.6424039931897765,0.0),Complex::new(-1.9620561766472002,0.0)],
            vec![Complex::new(-9.821995596375428,0.0),Complex::new(-3.1921112766174997,0.0)],
            vec![Complex::new(-4.992109362834896,0.0),Complex::new(-2.0745015313494455,0.0)],
            vec![Complex::new(10.107315875917662,0.0),Complex::new(2.4489015959094216,0.0)],
            vec![Complex::new(-7.962477597931141,0.0),Complex::new(-5.494741864480315,0.0)],
            vec![Complex::new(10.047917462523671,0.0),Complex::new(5.1631966716389766,0.0)],
            vec![Complex::new(-5.243921934674187,0.0),Complex::new(-2.963359100733349,0.0)],
            vec![Complex::new(-9.940544426622527,0.0),Complex::new(-3.2655473073528816,0.0)],
            vec![Complex::new(8.30445373000034,0.0),Complex::new(2.129694332932624,0.0)],
            vec![Complex::new(-9.196460281784482,0.0),Complex::new(-3.987773678358418,0.0)],
            vec![Complex::new(-10.513583123594056,0.0),Complex::new(-2.5364233580562887,0.0)],
            vec![Complex::new(9.072668506714033,0.0),Complex::new(3.405664632524281,0.0)],
            vec![Complex::new(-7.031861004012987,0.0),Complex::new(-2.2616818331210844,0.0)],
            vec![Complex::new(9.627963795272553,0.0),Complex::new(4.502533177849574,0.0)],
            vec![Complex::new(-10.442760023564471,0.0),Complex::new(-5.0830680881481065,0.0)],
            vec![Complex::new(8.292151321984209,0.0),Complex::new(3.8776876670218834,0.0)],
            vec![Complex::new(-6.51560033683665,0.0),Complex::new(-3.8185628318207585,0.0)],
            vec![Complex::new(-10.887633624071544,0.0),Complex::new(-4.416570704487158,0.0)],
            vec![Complex::new(-9.465804800021168,0.0),Complex::new(-2.2222090878656884,0.0)],
        ];

        let mut centroids = <Vec<Vec<Complex<f32>>> as Kmeans<f32>>::new(&data,2);
        let mask = <Vec<Vec<Complex<f32>>> as Kmeans<f32>>::cluster(&data, &mut centroids, 16);

        let binding = vec![1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0];
        let known_mask = binding.iter().map(|i| Core(*i));

        mask.iter().zip(known_mask).for_each(|(m,k)|{
            assert!(*m==k);
        });
        
        let known_centroids = vec![vec![Complex::new(-8.2813197,0.0), Complex::new(-3.3291236,0.0)],vec![Complex::new(9.2515742,0.0), Complex::new(3.38500524,0.0)],];

        centroids.iter().zip(known_centroids.iter()).for_each(|(c,k)|{
            c.iter().zip(k.iter()).for_each(|(cp, kp)|{
                assert!((*cp-*kp).square_norm()<f32::EPSILON);
            })
        });

    }
}
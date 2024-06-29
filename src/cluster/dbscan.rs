use crate::{
    cluster::Classification::{Core, Edge, Noise},
    shared::{complex::Complex, float::Float},
};

use super::Classification;


trait DBSCAN<T: Float> {
    fn new();

    fn cluster(input: &Self, epsilon: &T, min_points: usize) -> Vec<Classification>
    where
        Self: Sized;

    fn square_distance(a: &Vec<Complex<T>>, b: &Vec<Complex<T>>) -> T;
}

impl<T: Float> DBSCAN<T> for Vec<Vec<Complex<T>>> {
    fn new() {
        todo!()
    }

    fn cluster(input: &Self, epsilon: &T, min_points: usize) -> Vec<Classification>
    where
        Self: Sized,
    {
        let mut classifications = vec![Noise; input.len()];
        let mut visited = vec![false; input.len()];
        let mut cluster = 0;
        let mut queue = Vec::new();

        (0..input.len()).for_each(|i| {
            if !visited[i] {
                visited[i] = true;
                queue.push(i);

                let mut new_cluster=0;
                //expand the cluster
                while let Some(idx) = queue.pop() {
                    let neighbors = input
                        .iter()
                        .enumerate()
                        .filter(|(_, pt)| Self::square_distance(&input[idx], pt) < *epsilon)
                        .map(|(idx, _)| idx)
                        .collect::<Vec<_>>();
                    if neighbors.len() > min_points {
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

    fn square_distance(a: &Vec<Complex<T>>, b: &Vec<Complex<T>>) -> T {
        a.iter()
            .zip(b.iter())
            .fold(T::ZERO, |s, (ap, bp)| s + (*ap - *bp).square_norm())
    }
}


#[cfg(test)]
mod dbscan_tests{
    use super::*;

    #[test]
    fn test_dbscan() {
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

        let mask = <Vec<Vec<Complex<f32>>> as DBSCAN<f32>>::cluster(&data,&9.0, 5);
        let known_mask = vec![Core(0), Edge(1), Core(1), Edge(1), Core(0), Core(1), Edge(0), Edge(1), Core(1), Core(0), Core(1), Core(1), Core(0), Core(1), Core(0), Core(1), Core(0), Core(1), Core(1), Core(1)];

        mask.iter().zip(known_mask.iter()).for_each(|(m,k)|{
            assert!(*m==*k);
        });
        

    }
}
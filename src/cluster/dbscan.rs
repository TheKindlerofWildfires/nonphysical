use crate::{
    cluster::Classification::{Core, Edge, Noise},
    shared::{float::Float, point::Point},
};

use super::Classification;


trait DBSCAN<T: Float,const N: usize> {
    fn cluster(input: &Self, epsilon: &T, min_points: usize) -> Vec<Classification>
    where
        Self: Sized;
}

impl<T: Float,const N: usize> DBSCAN<T,N> for Vec<Point<T,N>> {
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
                        .filter(|(_, pt)| input[idx].distance(pt) < *epsilon)
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

}


#[cfg(test)]
mod dbscan_tests{
    use super::*;

    #[test]
    fn dbscan_simple() {
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

        let mask = <Vec<Point<f32,2>> as DBSCAN<f32,2>>::cluster(&data,&9.0, 5);
        let known_mask = vec![Core(0), Core(1), Core(1), Core(1), Core(0), Core(1), Core(0), Core(1), Core(1), Core(0), Core(1), Core(1), Core(0), Core(1), Core(0), Core(1), Core(0), Core(1), Core(1), Core(1)];
        mask.iter().zip(known_mask.iter()).for_each(|(m,k)|{
            assert!(*m==*k);
        });
        

    }
}
use std::marker::PhantomData;

use crate::{
    cluster::Classification::{Core, Edge, Noise},
    random::pcg::PermutedCongruentialGenerator,
    shared::{float::Float, point::Point, primitive::Primitive},
};

use super::Classification;
use alloc::vec;
use alloc::vec::Vec;

pub struct SelfSelectiveCompetitiveLearning<P: Point> {
    rng: PermutedCongruentialGenerator,
    iterations: usize,
    phantom_data: PhantomData<P>
}

#[derive(Clone,Debug)]
struct SSCluster<P: Point> {
    p: P,
    r: P,
    a: P,
    c: P,
    nr: P::Primitive,
    na: P::Primitive,
    nc: P::Primitive,
}
impl<Prim: Primitive, P: Point<Primitive = Prim>> SelfSelectiveCompetitiveLearning<P> {
    pub fn new(state: u32, iterations: usize) -> Self {
        let rng = PermutedCongruentialGenerator::new(state, state+1);

        Self { rng, iterations, phantom_data: PhantomData::default() }
    }

    pub fn cluster(&mut self, data: &[P]) -> Vec<Classification> {
        //Init the data
        let min_p = data.iter().fold(P::MAX, |acc, p| acc.lesser(p));
        let max_p = data.iter().fold(P::MIN, |acc, p| acc.greater(p));

        let p_init = P::random_uniform(&min_p, &max_p, &mut self.rng);
        /*let eps = data
            .iter()
            .fold(P::Primitive::ZERO, |acc, d| acc + d.l2_distance(&p_init))
            / P::Primitive::usize(data.len());*/
        //This parameter matters a lot, if it's too big theres only one cluster, if it's too small we get like 15
        let eps = (max_p-min_p).max_data()/P::Primitive::usize(100);
        let cluster_init = self.new_cluster(p_init, (min_p, max_p));
        let mut clusters = vec![cluster_init];
        loop {
            //Learning loop
            let mut iteration = 0;
            while iteration < self.iterations {
                let mut order = (0..data.len()).collect::<Vec<_>>();
                self.rng.shuffle_usize(&mut order);
                order.iter().for_each(|i| {
                    //Find winning P
                    let x = data[*i].clone();
                    let (_, win_cluster_idx) = clusters.iter().enumerate().fold(
                        (P::Primitive::MAX, 0),
                        |acc, (i, cluster)| {
                            let distance = x.l2_distance(&cluster.p);
                            if distance < acc.0 {
                                (distance, i)
                            } else {
                                acc
                            }
                        },
                    );
                    let win_cluster = &mut clusters[win_cluster_idx];
                    //Calculate asymptotic/center/distance properties
                    let pa = win_cluster.p.l2_distance(&win_cluster.a);
                    let px = win_cluster.p.l2_distance(&x);
                    let pr = win_cluster.p.l2_distance(&win_cluster.r);
                    let theta_a = P::Primitive::usize(if pa >= px { 1 } else { 0 });
                    let theta_r = P::Primitive::usize(if px >= pr { 1 } else { 0 }); //r starts right at p, so this is a no go

                    let delta = pa / (px + pa);
                    let rho = px / (px + pr);
                    let beta = pr / (px + pr);
                    win_cluster.na += delta * theta_a;
                    win_cluster.nc += P::Primitive::ONE;
                    win_cluster.nr += P::Primitive::ONE;
                    let mut a_adjust = x - win_cluster.a;
                    a_adjust.scale(delta / win_cluster.na * theta_a);
                    win_cluster.a += a_adjust;
                    let mut c_adjust = x - win_cluster.c;
                    c_adjust.scale(win_cluster.nc.recip());
                    win_cluster.c += c_adjust;
                    let mut r_adjust = x - win_cluster.r;
                    r_adjust.scale(rho / win_cluster.nr * theta_r);
                    win_cluster.r += r_adjust;
                    let mut p_adjust = x - win_cluster.p;
                    p_adjust.scale(delta * beta);
                    win_cluster.p += p_adjust;
                    
                });
                iteration += 1;
                //check if the PA is below epsilon
                let max_pa = clusters.iter().fold(P::Primitive::MIN, |acc, cluster| {
                    let pa = cluster.p.l1_distance(&cluster.a);
                    if pa > acc {
                        pa
                    } else {
                        acc
                    }
                });
                if max_pa < eps {
                    break;
                }
            }
            //splitting code
            let (max_pc, max_pc_idx) =
                clusters
                    .iter()
                    .enumerate()
                    .fold((P::Primitive::MIN, 0), |acc, (i, cluster)| {
                        let pc = cluster.p.l1_distance(&cluster.c);
                        if pc > acc.0 {
                            (pc, i)
                        } else {
                            acc
                        }
                    });
            
            if max_pc > eps {
                let split_cluster = &clusters[max_pc_idx];
                let new_p = split_cluster.r;
                let new_cluster = self.new_cluster(new_p, (min_p, max_p));
                clusters.push(new_cluster);
                //reset the clsuters
                let max_distance = min_p.l1_distance(&max_p);
                clusters.iter_mut().for_each(|cluster| {
                    cluster.r = cluster.p;
                    while cluster.p.l1_distance(&cluster.a) < max_distance / Primitive::usize(2) {
                        cluster.a = P::random_uniform(&min_p, &max_p, &mut self.rng);
                    }
                    cluster.c = cluster.a;
                    cluster.na = P::Primitive::ZERO;
                    cluster.nc = P::Primitive::ZERO;
                    cluster.nr = P::Primitive::ZERO;
                });
            } else {
                break;
            }

        }
        dbg!(clusters.len());
        Vec::new()
    }
    fn new_cluster(&mut self, p: P, (mn, mx): (P, P)) -> SSCluster<P> {
        let r = p;
        let mut a = P::random_uniform(&mn, &mx, &mut self.rng);
        //If the cluster is closer to p than half the max distance it's too close
        let max_distance = mn.l1_distance(&mx);
        while p.l1_distance(&a) < max_distance / Primitive::usize(2) {
            a = P::random_uniform(&mn, &mx, &mut self.rng);
        }
        let c = a;
        SSCluster {
            p,
            r,
            a,
            c,
            nr: P::Primitive::ZERO,
            na: P::Primitive::ZERO,
            nc: P::Primitive::ZERO,
        }
    }
}

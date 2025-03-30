use alloc::vec::Vec;

use crate::shared::float::Float;

pub mod adjacency_matrix;
pub mod adjacency_list;


pub trait Graph<F:Float>{
    fn new() -> Self;
    fn sized(nodes: usize) -> Self;
    fn add_edge(&mut self, from: usize, to: usize, weight: F);
    fn edge_weight(&self, from: usize, to: usize)->F;
    fn neighbors_source(&self, source: usize)->Vec<usize>;
    fn neighbors_dest(&self, dest: usize)->Vec<usize>;
}
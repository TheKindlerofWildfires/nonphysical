use alloc::vec::Vec;

use crate::shared::{
    float::Float,
    matrix::{matrix_heap::MatrixHeap, Matrix},
};

use super::Graph;

struct AdjacencyList<F: Float> {
    matrix: MatrixHeap<F>, //needs to be a hashmap
}

impl<F: Float> Graph<F> for AdjacencyList<F> {
    fn new() -> Self {
        let matrix = MatrixHeap::zero(0, 0);
        Self { matrix }
    }

    fn sized(nodes: usize) -> Self {
        let matrix = MatrixHeap::single(nodes, nodes, F::NAN);
        Self { matrix }
    }

    fn add_edge(&mut self, from: usize, to: usize, weight: F) {
        *self.matrix.coeff_ref(from, to) = weight;
    }

    fn edge_weight(&self, from: usize, to: usize) -> F {
        self.matrix.coeff(from, to)
    }

    fn neighbors_source(&self, source: usize) -> Vec<usize> {
        self.matrix
            .data_row(source)
            .enumerate()
            .filter(|(_, f)| f.finite())
            .map(|(i, _)| i)
            .collect()
    }

    fn neighbors_dest(&self, dest: usize) -> Vec<usize> {
        self.matrix
            .data_col(dest)
            .enumerate()
            .filter(|(_, f)| f.finite())
            .map(|(i, _)| i)
            .collect()
    }
}

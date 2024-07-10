#![forbid(unsafe_code)]
#![allow(dead_code)]


pub mod linear;
pub mod shared;
pub mod signal;
//pub mod neural;
pub mod random;
pub mod filter;
pub mod cluster;
pub mod graph;


pub mod playground;

fn main() {
    playground::play_svd();
    //playground::play_neural();
    //playground::play_matrix();
}

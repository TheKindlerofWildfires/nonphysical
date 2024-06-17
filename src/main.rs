#![forbid(unsafe_code)]
#![allow(dead_code)]
use std::process::exit;


pub mod linear;
pub mod shared;
pub mod signal;
pub mod neural;
pub mod random;


pub mod playground;

fn main() {
    playground::play_neural();
}

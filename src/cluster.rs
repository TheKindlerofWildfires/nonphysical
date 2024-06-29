pub mod kmeans;
pub mod dbscan;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Classification {
    Core(usize),
    Edge(usize),
    Noise,
}
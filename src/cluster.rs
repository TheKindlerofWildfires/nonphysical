pub mod kmeans;
pub mod dbscan;
pub mod hdbscan;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Classification {
    Core(usize),
    Edge(usize),
    Noise,
}
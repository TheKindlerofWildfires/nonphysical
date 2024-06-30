pub mod kmeans;
pub mod dbscan;
pub mod hdbscan;
pub mod iso_forest;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Classification {
    Core(usize),
    Edge(usize),
    Noise,
}
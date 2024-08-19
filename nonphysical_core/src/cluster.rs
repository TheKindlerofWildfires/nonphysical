pub mod kmeans;
pub mod dbscan;
#[cfg(feature = "std")]
pub mod hdbscan;
pub mod iso_forest;
pub mod sscl;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Classification {
    Core(usize),
    Edge(usize),
    Noise,
}
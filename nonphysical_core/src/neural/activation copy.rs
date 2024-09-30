use crate::shared::float::Float;

use super::layer::Layer;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod softplus;
pub mod softsign;
pub mod tanh;
pub mod selu;
pub mod elu;
pub mod exponential;
pub mod leaky_relu;
pub mod relu6;
pub mod silu;
pub mod hard_silu;
pub mod gelu;
pub mod hard_sigmoid;
pub mod linear;
pub mod mish;
pub mod log_softmax;
pub trait Activation<F:Float> : Layer<F>{

}
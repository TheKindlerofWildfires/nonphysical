use super::primitive::Primitive;

/*
    This Trait defines functions which are specific to real numbers
*/
pub trait Real: Primitive {}

impl Real for f32 {}
impl Real for f64 {}

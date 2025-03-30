use alloc::vec::Vec;

use crate::{
    neural::layer::Layer,
    shared::{float::Float, matrix::{matrix_heap::MatrixHeap, Matrix}},
};

use super::Activation;

pub struct ReLu<F: Float> {
    parameters: ReLuParameters<F>,
}

pub struct ReLuParameters<F: Float> {
    negative_slope: F,
    max_value: F,
    threshold: F,
}

impl<F: Float> Activation<F> for ReLu<F> {}

impl<F: Float> Layer<F> for ReLu<F> {
    type Matrix = MatrixHeap<F>;
    type Parameters = ReLuParameters<F>;
    fn new(parameters: Self::Parameters, _size: usize, _previous_size: usize) -> Self
    where
        Self: Sized,
    {
        Self { parameters }
    }
    /*
        f(x) =  x>threshold => min(x,max_value)
                x<threshold => x*negative_slope
     */ 
    fn forward(&self, x: &Self::Matrix) -> Self::Matrix {
        Matrix::new((x.rows, x.data().map(|&f| {
            match f>self.parameters.threshold{
                true => if f<self.parameters.max_value{
                    f
                }else{
                    self.parameters.max_value
                },
                false => f*self.parameters.negative_slope,
            }
        }).collect::<Vec<_>>()))
    }
    /*
        f'(x) = x>threshold => x>0 max_value? 0 : 1
                x<threshold => negative_slope
    
     */
    fn backward(
        &self,
        gradient: &Self::Matrix, //incoming grad
        memory: &Self::Matrix, //historical input
        _lambda: F,
        _epsilon: F,
    ) -> Self::Matrix {

        let mut output = gradient.clone();
        output.data_ref().zip(memory.data()).for_each(|(grad,&mem)|{
            match mem>self.parameters.threshold{
                true =>{
                    if mem>self.parameters.max_value{
                        *grad = F::ZERO
                    }
                    //otherwise 1
                },
                false =>{
                    *grad/=self.parameters.negative_slope
                }
            }
        });
        output
    }
    
    fn forward_ref(&self, x: &mut Self::Matrix) {
        x.data_ref().for_each(|f|{
            *f=match *f>self.parameters.threshold{
                true => if *f<self.parameters.max_value{
                    *f
                }else{
                    self.parameters.max_value
                },
                false => *f*self.parameters.negative_slope,
            }
        });
    }
    
    fn backward_ref(&self, gradient: &mut Self::Matrix, memory: &Self::Matrix, _lambda: F, _epsilon: F) {
        gradient.data_ref().zip(memory.data()).for_each(|(grad,&mem)|{
            match mem>self.parameters.threshold{
                true =>{
                    if mem>self.parameters.max_value{
                        *grad = F::ZERO
                    }
                    //otherwise 1
                },
                false =>{
                    *grad/=self.parameters.negative_slope
                }
            }
        });
    }
}

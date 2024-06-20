use std::process::exit;

use crate::{
    random::pcg::PermutedCongruentialGenerator,
    shared::{complex::Complex, float::Float, matrix::Matrix},
};

use super::{
    layer::{
        self, AddLayer, IdentityLayer, Layer, LayerType, MultiplyLayer, PerceptronLayer, TanhLayer
    },
    predictor::{Predictor, SoftPredictor},
};
pub struct Network<T: Float> {
    layers: Vec<Box<dyn Layer<T>>>, //I couldn't think of a better way to do this without dynamic dispatch
    lambda: T,
    epsilon: T,
    batch_size: usize,
    epochs: usize,
    memories: Vec<Matrix<T>>,
}

impl<T: Float + 'static> Network<T> {
    //this lifetime math is weird
    pub fn new(
        layer_shape: Vec<usize>,
        layer_map: Vec<LayerType>,
        lambda: T,
        epsilon: T,
        batch_size: usize,
        epochs: usize,
    ) -> Self {
        let layers = Self::compile_layers(&layer_shape, &layer_map);
        Self {
            layers,
            lambda,
            epsilon,
            batch_size,
            epochs,
            memories: Vec::new(),
        }
    }

    pub fn new_direct(
        layers: Vec<Box<dyn Layer<T>>>,
        lambda: T,
        epsilon: T,
        batch_size: usize,
        epochs: usize,
    ) -> Self {
        Self {
            layers,
            lambda,
            epsilon,
            batch_size,
            epochs,
            memories: Vec::new(),
        }
    }
    pub fn fit(&mut self, x: &Matrix<T>, y: &Matrix<T>) {
        let mut pcg = PermutedCongruentialGenerator::<T>::new_timed();
        (0..self.epochs).for_each(|epoch| {
            let mut sum_loss = Complex::<T>::zero();
            let mut index_table: Vec<_> = (0..x.rows).collect();
            //pcg.shuffle_usize(&mut index_table);

            let rows_x = x.data_rows().collect::<Vec<_>>();
            let rows_y = y.data_rows().collect::<Vec<_>>();
            index_table
                .chunks_exact(self.batch_size)
                .for_each(|index_row| {
                    //this clones the batch into new matrices
                    let batch_x_data = index_row
                        .into_iter()
                        .flat_map(|&i| rows_x[i].into_iter().map(|r| r.clone()))
                        .collect::<Vec<_>>();
                    let mut batch_x = Matrix::new(self.batch_size, batch_x_data);

                    let batch_y_data = index_row
                        .into_iter()
                        .flat_map(|&i| rows_y[i].into_iter().map(|r| r.clone()))
                        .collect::<Vec<_>>();
                    let mut batch_y = Matrix::new(self.batch_size, batch_y_data);

                    //transmute batch_x to pred_y, but batch_x isn't used again
                    self.forward(&mut batch_x);
                    let loss = self.loss(&batch_y);
                    if epoch % 100 == 0 {
                        println!("On Epoch {}", epoch);
                        dbg!(loss);
                    }

                    //push the details backwards up the matrix
                    self.backwards(&mut batch_y);
                })
        });
    }

    fn compile_layers(
        layer_shapes: &Vec<usize>,
        layer_map: &Vec<LayerType>,
    ) -> Vec<Box<dyn Layer<T>>> {
        let mut previous_size = layer_shapes.first().unwrap(); //assume it's identity like

        layer_shapes
            .iter()
            .zip(layer_map)
            .map(|(size, layer_type)| {
                let layer: Box<dyn Layer<T>> = match layer_type {
                    LayerType::Multiply => Box::new(MultiplyLayer::new(*size, *previous_size)),
                    LayerType::Add => Box::new(AddLayer::new(*size, *previous_size)),
                    LayerType::Tanh => Box::new(TanhLayer::new(*size, *previous_size)),
                    LayerType::PerceptronLayer => {
                        Box::new(PerceptronLayer::new(*size, *previous_size))
                    }
                    LayerType::Identity => Box::new(IdentityLayer::new(*size, *previous_size)),
                };
                previous_size = size;
                layer
            })
            .collect::<Vec<_>>()
    }

    fn forward(&mut self, input: &mut Matrix<T>) {
        //the copy operations here are not very well done
        let mut i = 0;
        self.layers.iter_mut().for_each(|layer| {
            *input = layer.forward(input);
            self.memories.push(input.explicit_copy());
            i+=1;
        });
    }

    fn backwards(&mut self, output: &mut Matrix<T>) {
        //hack while testing
        *output = SoftPredictor::diff(&self.memories.last().unwrap(), output);
        for i in (1..self.layers.len()).rev(){
            *output = self.layers[i].backward(output, &self.memories[i-1],self.lambda, self.epsilon);
        }
    }
    

    fn loss(&self, output: &Matrix<T>) -> T {
        SoftPredictor::<T>::loss(&self.memories.last().unwrap(), output)
    }
}

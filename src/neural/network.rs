use crate::{
    random::pcg::PermutedCongruentialGenerator,
    shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector},
};

use super::layer::Layer;
pub struct Network<T: Float> {
    layer_shape: Vec<usize>,
    layers: Vec<Layer<T>>,
    lambda: T,
    learning_step: T,
    batch_size: usize,
    epochs: usize,
}

impl<T: Float> Network<T> {
    pub fn new(
        layer_shape: Vec<usize>,
        lambda: T,
        learning_step: T,
        batch_size: usize,
        epochs: usize,
    ) -> Self {
        let layers = Self::create_layers(&layer_shape);

        Self {
            layer_shape,
            layers,
            lambda,
            learning_step,
            batch_size,
            epochs,
        }
    }
    pub fn fit(&mut self, data: &Matrix<T>, labels: &Vec<usize>) {
        let mut pcg = PermutedCongruentialGenerator::<T>::new_timed();
        (0..self.epochs).for_each(|epoch| {
            println!("On Epoch {}", epoch);
            let mut sum_loss = Complex::<T>::zero();
            let mut index_table: Vec<_> = (0..data.rows).collect();
            pcg.shuffle_usize(&mut index_table);

            let mut batch_number = 0;

            let rows = data.data_rows().collect::<Vec<_>>();
            index_table
                .chunks_exact(self.batch_size)
                .for_each(|index_row| {
                    //this clones the batch into a new matrix
                    let batch_data = index_row
                        .into_iter()
                        .flat_map(|&i| rows[i].into_iter().map(|r| r.clone()))
                        .collect::<Vec<_>>();

                    let batch_matrix = Matrix::<T>::new(self.batch_size, batch_data);
                    let batch_label = index_row
                        .into_iter()
                        .map(|&i| labels[i])
                        .collect::<Vec<_>>();

                    let predictions = self.evaluate(&batch_matrix);

                    let loss = self.loss(&predictions, &batch_label);

                    sum_loss += loss;
                    dbg!(sum_loss);

                    let score = self.score(&predictions, &batch_label);

                    self.update(score, batch_matrix);
                })
        });
    }

    fn create_layers(layer_shape: &Vec<usize>) -> Vec<Layer<T>> {
        let length = layer_shape.len() - 1;
        let mut previous_size = 0;
        layer_shape
            .iter()
            .enumerate()
            .map(|(i, layer_size)| {
                let j = match i {
                    0 => Layer::<T>::new_identity(*layer_size, Layer::<T>::linear),
                    n if n == length => {
                        Layer::<T>::new_random(*layer_size, previous_size, Layer::<T>::linear)
                    }
                    _ => Layer::<T>::new_random(*layer_size, previous_size, Layer::<T>::relu),
                };
                previous_size = *layer_size;
                j
            })
            .collect()
    }

    fn evaluate(&mut self, input: &Matrix<T>) -> Matrix<T> {
        let mut index = 0;

        //could do better with a fold operation here I think
        let mut tmp = Matrix::<T>::zero(0, 0);
        self.layers.iter_mut().for_each(|layer| {
            if index == 0 {
                tmp = layer.forward(input);
            } else {
                tmp = layer.forward(&tmp);
            }
            index += 1;
        });

        let output = Layer::<T>::soft_max(&tmp);
        output
    }

    fn loss(&mut self, output: &Matrix<T>, labels: &Vec<usize>) -> Complex<T> {
        self.cross_entropy(output, labels) + self.l2_reg(&self.layers)
    }

    fn score(&mut self, score: &Matrix<T>, labels: &Vec<usize>) -> Matrix<T> {
        let mut output = Matrix::<T>::zero(score.columns, score.rows);

        //this loop can probably be a single clone and  then subtract the identity matrix
        for r in 0..score.rows {
            for c in 0..score.columns {
                if labels[r] == c {
                    *output.coeff_ref(r, c) = score.coeff(r, c) - Complex::<T>::one()
                } else {
                    *output.coeff_ref(r, c) = score.coeff(r, c);
                }
            }
        }
        output
    }

    fn cross_entropy(&self, output: &Matrix<T>, labels: &Vec<usize>) -> Complex<T> {
        let mut o1 = Vec::with_capacity(output.rows);
        dbg!(output.rows, output.columns);
        //this seems to expect a square matrix, but output came in as a 1x10 matrix
        for c in 0..output.rows {
            o1.push(output.coeff(c, labels[c]));
        }

        let mut loss = Complex::<T>::zero();

        for c in 0..o1.len() {
            loss += -o1[c].ln(); //principle log
        }

        loss = loss / T::usize(o1.len());

        loss
    }

    fn l2_reg(&self, layers: &Vec<Layer<T>>) -> Complex<T> {
        let mut l2 = Complex::<T>::zero();

        for layer in layers {
            l2 += layer.weights.data().fold(Complex::<T>::zero(), |acc, c| {
                acc + Complex::<T>::new(c.square_norm(), T::zero())
            }) * T::float(0.5)
                * self.lambda;
        }

        l2
    }

    //doing a bunch of copies for poc
    fn update(&mut self, score: Matrix<T>, batch: Matrix<T>) {
        let mut index = self.layers.len() - 1;
        let mut dz = Matrix::<T>::new(score.rows, score.data().map(|c| c.clone()).collect());

        loop {
            let zm1 = match index > 0 {
                true => Matrix::<T>::new(
                    self.layers[index - 1].output.rows,
                    self.layers[index - 1]
                        .output
                        .data()
                        .map(|c| c.clone())
                        .collect(),
                ),
                false => Matrix::<T>::new(batch.rows, batch.data().map(|c| c.clone()).collect()),
            };

            let dz_copy = Matrix::<T>::new(dz.rows, dz.data().map(|c| c.clone()).collect());
            let zm1_copy = Matrix::<T>::new(zm1.rows, zm1.data().map(|c| c.clone()).collect());
            let tmp = zm1_copy.transposed()*dz_copy;

            let weight_copy = Matrix::<T>::new(
                self.layers[index].weights.rows,
                self.layers[index]
                    .weights
                    .data()
                    .map(|c| c.clone())
                    .collect(),
            );
            let dw = tmp + (weight_copy * self.lambda);
            let db = score
                .data_rows()
                .map(|row| {
                    Complex::<T>::new(
                        <Vec<&'_ Complex<T>> as Vector<T>>::norm_sum(row.iter()),
                        T::zero(),
                    )
                })
                .collect::<Vec<_>>();

            dz = dz * self.layers[index].weights.transposed();

            self.layers[index].update_weights(&dw, self.learning_step);
            self.layers[index].update_biases(&db, self.learning_step);

            if index == 0 {
                break;
            }
            self.drelu(&dz, &zm1);

            index -= 1;
        }
    }

    fn drelu(&self, input: &Matrix<T>, zm1: &Matrix<T>) -> Matrix<T> {
        let mut output = Matrix::zero(input.rows, input.columns);

        for r in 0..input.rows {
            for c in 0..input.columns {
                if zm1.coeff(r, c).norm() <= T::zero() {
                    *output.coeff_ref(r, c) = Complex::<T>::zero();
                } else {
                    *output.coeff_ref(r, c) = input.coeff(r, c);
                }
            }
        }

        output
    }
}

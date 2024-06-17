use crate::{
    random::pcg::PermutedCongruentialGenerator,
    shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector},
};

pub struct Layer<T: Float> {
    pub weights: Matrix<T>,
    pub biases: Matrix<T>,
    pub output: Matrix<T>,
    delta_weights: Matrix<T>,
    delta_biases: Matrix<T>,
    function: fn(&mut Complex<T>),
}

impl<T: Float> Layer<T> {
    pub fn new_identity(size: usize, function: fn(&mut Complex<T>)) -> Self {
        Self {
            weights: Matrix::<T>::identity(size, size),
            biases: Matrix::<T>::zero(1, size),
            output: Matrix::<T>::zero(size, size),
            delta_weights: Matrix::<T>::zero(size, size),
            delta_biases: Matrix::<T>::zero(size, 1),
            function: function,
        }
    }

    pub fn new_random(size: usize, previous_size: usize, function: fn(&mut Complex<T>)) -> Self {
        Self {
            weights: Self::kaiming_he(previous_size, size),
            biases: Matrix::<T>::zero(1, size),
            output: Matrix::<T>::zero(previous_size, size),
            delta_weights: Matrix::<T>::zero(previous_size, size),
            delta_biases: Matrix::<T>::zero(size, 1),
            function: function,
        }
    }

    fn kaiming_he(rows: usize, columns: usize) -> Matrix<T> {
        let mut pcg = PermutedCongruentialGenerator::<T>::new_timed();
        let mut data = pcg.normal(Complex::<T>::zero(), T::one(), rows * columns);
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(
            data.iter_mut(),
            T::usize(columns).sqrt().recip(),
        );
        Matrix::new(rows, data)
    }

    pub fn update_weights(&mut self, input: &Matrix<T>, step: T) {
        self.weights
            .data_ref()
            .zip(input.data())
            .for_each(|(w, i)| *w = *w - *i * step);
    }

    pub fn update_biases(&mut self, input: &Vec<Complex<T>>, step: T) {
        self.biases
            .data_ref()
            .zip(input.iter())
            .for_each(|(w, i)| *w = *w - *i * step);
    }

    pub fn forward(&mut self, input: &Matrix<T>) -> Matrix<T> {
        let wc = Matrix::<T>::new(
            self.weights.rows,
            self.weights.data().map(|c| c.clone()).collect(),
        );
        let input_copy = Matrix::<T>::new(input.rows, input.data().map(|c| c.clone()).collect());
        //matrix multiplication here got the dimensionality wrong
        let mut output = input_copy * wc;
        let bc = Matrix::<T>::new(
            self.biases.rows,
            self.biases.data().map(|c| c.clone()).collect(),
        );
        output = output + bc; //what are the odds this is reliably right?
                              //apply activation function
        output.data_ref().for_each(|c| (self.function)(c));

        let oc = Matrix::<T>::new(output.rows, output.data().map(|c| c.clone()).collect());

        self.output = oc;

        output
    }

    pub fn relu(x: &mut Complex<T>) {
        if x.norm() < T::zero() {
            *x = Complex::<T>::zero();
        }
    }

    pub fn linear(_: &mut Complex<T>) {}

    pub fn soft_max(x: &Matrix<T>) -> Matrix<T> {
        let mut input_sub_max = Matrix::zero(x.rows, x.columns);

        //the abuse of norm max here over real max may be ugly when using actual complex numbers
        //this is possible to combine into single loop
        input_sub_max.data_rows_ref().for_each(|row| {
            let max = -Complex::<T>::new(
                <Vec<&'_ Complex<T>> as Vector<T>>::norm_max(row.iter()),
                T::zero(),
            );
            <Vec<&'_ Complex<T>> as Vector<T>>::add(row.iter_mut(), max);
        });

        input_sub_max.data_ref().for_each(|c| *c = c.exp());

        input_sub_max.data_rows_ref().for_each(|row| {
            let inv_sum = <Vec<&'_ Complex<T>> as Vector<T>>::norm_sum(row.iter()).recip();
            <Vec<&'_ Complex<T>> as Vector<T>>::scale(row.iter_mut(), inv_sum);
        });
        input_sub_max
    }
}

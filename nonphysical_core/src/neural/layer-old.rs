use core::{marker::PhantomData, mem};

use crate::{
    random::pcg::PermutedCongruentialGenerator,
    shared::{complex::Complex, float::Float, matrix::Matrix, vector::Vector},
};

pub enum LayerType {
    Multiply,
    Add,
    Tanh,
    Identity,
    PerceptronLayer,
}
/*
    Activations
        relu
        sigmoid
        softmax
        softplus
        softsign
        tanh
        selu
        elu
        exponential
        leaky_relu
        relu6
        silu
        hard_silu
        gelu
        hard_sigmoid
        linear
        mish
        log_softmax

    Layers
        Input
        InputSpec
        Dense
        EinsumDense
        Activation
        Embedding
        Masking
        Lambda
        Identity

    Convolutions
        Conv1-3
        Sepconv 1-2
        Depconv1-2
        Convtran 1-3

    Pooling
        mapool 1-3
        avgpool 1-3
        globalmp 1-3
        globalavg 1-3
    Normalization
        ??? (batch/layer etc)
    Regularization
        ??? (dropout and stuff)
    Attention
        ???
    Reshaping
        ???
    Merging
        ???
    
    Plan: Activation class-> Impls for all activations
    read up on all the core layers
    read up on the contraints
*/
pub trait Layer<T: Float> {
    fn new(size: usize, previous_size: usize) -> Self
    where
        Self: Sized;
    fn forward(&mut self, x: &Matrix<T>) -> Matrix<T>;
    fn backward(&mut self, x: &Matrix<T>, memory: &Matrix<T>, lambda: T, epsilon: T) -> Matrix<T>;

    fn kaiming_he(rows: usize, columns: usize) -> Matrix<T>
    where
        Self: Sized,
    {
        let mut pcg = PermutedCongruentialGenerator::new_timed();
        let mut data = pcg.normal(Complex::zero(), T::one(), rows * columns);
        <Vec<&'_ Complex<T>> as Vector<T>>::scale(
            data.iter_mut(),
            T::usize(columns).sqrt().recip(),
        );
        Matrix::new(rows, data)
    }

    fn normal(rows: usize, columns: usize) -> Matrix<T>
    where
        Self: Sized,
    {
        let mut pcg = PermutedCongruentialGenerator::new_timed();
        let data = pcg.normal(Complex::zero(), T::one(), rows * columns);
        Matrix::new(rows, data)
    }
}

pub struct MultiplyLayer<T: Float> {
    weights: Matrix<T>,
}

impl<T: Float> Layer<T> for MultiplyLayer<T> {
    fn new(size: usize, previous_size: usize) -> Self {
        Self {
            weights: Self::kaiming_he(previous_size, size),
        }
    }

    fn forward(&mut self, x: &Matrix<T>) -> Matrix<T> {
        //opportunity to stream directly into preallocated memory
        x.dot(&self.weights)
    }

    fn backward(&mut self, x: &Matrix<T>, memory: &Matrix<T>, lambda: T, epsilon: T) -> Matrix<T> {
        let mut dw = memory.transposed().dot(x);
        let dx = x.dot(&self.weights.transposed());
        let ph = self.weights.clone() * lambda;
        dw = dw.acc(&ph);
        self.weights = self.weights.acc(&(dw * -epsilon));
        dx
    }
}

pub struct AddLayer<T: Float> {
    biases: Matrix<T>,
}

impl<T: Float> Layer<T> for AddLayer<T> {
    fn new(size: usize, _previous_size: usize) -> Self {
        Self {
            biases: Self::normal(1, size),
        }
    }

    fn forward(&mut self, x: &Matrix<T>) -> Matrix<T> {
        let mut output = x.clone();
        output.data_rows_ref().for_each(|row| {
            row.into_iter()
                .zip(self.biases.data())
                .for_each(|(c_row, c_bias)| *c_row = *c_row + *c_bias)
        });
        output
    }

    fn backward(
        &mut self,
        x: &Matrix<T>,
        _memory: &Matrix<T>,
        _lambda: T,
        epsilon: T,
    ) -> Matrix<T> {
        let dx = x.clone();
        //let db = Matrix::single(1, x.rows, Complex::<T>::one()).dot(x); //could pre-alloc
        self.biases = self.biases.acc(&(db * -epsilon));

        dx
    }
}

pub struct TanhLayer<T: Float> {
    phantom_data: PhantomData<T>,
}

impl<T: Float> Layer<T> for TanhLayer<T> {
    fn new(_size: usize, _previous_size: usize) -> Self {
        Self {
            phantom_data: PhantomData::default(),
        }
    }

    fn forward(&mut self, x: &Matrix<T>) -> Matrix<T> {
        Matrix::new(x.rows, x.data().map(|c| c.tanh()).collect::<Vec<_>>())
    }

    fn backward(
        &mut self,
        x: &Matrix<T>,
        memory: &Matrix<T>,
        _lambda: T,
        _epsilon: T,
    ) -> Matrix<T> {
        let mut output = self.forward(&memory);
        output
            .data_ref()
            .for_each(|c| *c = Complex::one() - *c * *c);
        output
            .data_ref()
            .zip(x.data())
            .for_each(|(oc, xc)| *oc = *oc * *xc);
        output
    }
}

pub struct IdentityLayer<T: Float> {
    phantom_data: PhantomData<T>,
}

impl<T: Float> Layer<T> for IdentityLayer<T> {
    fn new(_size: usize, _previous_size: usize) -> Self {
        Self {
            phantom_data: PhantomData::default(),
        }
    }

    fn forward(&mut self, x: &Matrix<T>) -> Matrix<T> {
        x.clone()
    }

    fn backward(
        &mut self,
        x: &Matrix<T>,
        _memory: &Matrix<T>,
        _lambda: T,
        _epsilon: T,
    ) -> Matrix<T> {
        x.clone()
    }
}

pub struct PerceptronLayer<T: Float> {
    mul_layer: MultiplyLayer<T>,
    add_layer: AddLayer<T>,
    activation_layer: TanhLayer<T>,
    memories: Vec<Matrix<T>>,
}

impl<T: Float> PerceptronLayer<T> {
    pub fn layer_one() -> Self {
        let kw1 = vec![
            Complex::new(T::float(0.03235616), T::zero()),
            Complex::new(T::float(-0.13235897), T::zero()),
            Complex::new(T::float(1.08383858), T::zero()),
            Complex::new(T::float(1.03899355), T::zero()),
            Complex::new(T::float(0.10956438), T::zero()),
            Complex::new(T::float(0.26740128), T::zero()),
        ];
        let kb1 = vec![
            Complex::new(T::float(-0.88778575), T::zero()),
            Complex::new(T::float(-1.98079647), T::zero()),
            Complex::new(T::float(-0.34791215), T::zero()),
        ];
        let mut mul_layer = MultiplyLayer::new(3, 2);
        let mut add_layer = AddLayer::new(3, 2);
        let activation_layer = TanhLayer::new(3, 2);
        mul_layer.weights = Matrix::new(2, kw1);
        add_layer.biases = Matrix::new(1, kb1);

        Self {
            mul_layer,
            add_layer,
            activation_layer,
            memories: Vec::new(),
        }
    }

    pub fn layer_two() -> Self {
        let kw1 = vec![
            Complex::new(T::float(0.09026812), T::zero()),
            Complex::new(T::float(0.71030866), T::zero()),
            Complex::new(T::float(0.69419433), T::zero()),
            Complex::new(T::float(-0.22362324), T::zero()),
            Complex::new(T::float(-0.17453457), T::zero()),
            Complex::new(T::float(-0.60538234), T::zero()),
        ];
        let kb1 = vec![
            Complex::new(T::float(-1.42001794), T::zero()),
            Complex::new(T::float(-1.70627019), T::zero()),
        ];
        let mut mul_layer = MultiplyLayer::new(2, 3);
        let mut add_layer = AddLayer::new(2, 3);
        let activation_layer = TanhLayer::new(2, 3);
        mul_layer.weights = Matrix::new(3, kw1);
        add_layer.biases = Matrix::new(1, kb1);
        Self {
            mul_layer,
            add_layer,
            activation_layer,
            memories: Vec::new(),
        }
    }
}

impl<T: Float> Layer<T> for PerceptronLayer<T> {
    fn new(size: usize, previous_size: usize) -> Self
    where
        Self: Sized,
    {
        let s = Self {
            mul_layer: MultiplyLayer::new(size, previous_size),
            add_layer: AddLayer::new(size, previous_size),
            activation_layer: TanhLayer::new(size, previous_size),
            memories: Vec::new(),
        };
        s
    }

    fn forward(&mut self, x: &Matrix<T>) -> Matrix<T> {
        let mut output = self.mul_layer.forward(x); //broken
        self.memories.push(output.clone());
        output = self.add_layer.forward(&output);
        self.memories.push(output.clone());
        output = self.activation_layer.forward(&output);
        output
    }

    fn backward(&mut self, x: &Matrix<T>, memory: &Matrix<T>, lambda: T, epsilon: T) -> Matrix<T> {
        let mut output = self
            .activation_layer
            .backward(x, &self.memories[1], lambda, epsilon);
        output = self
            .add_layer
            .backward(&output, &self.memories[0], lambda, epsilon);
        output = self.mul_layer.backward(&output, memory, lambda, epsilon);
        output
    }
}

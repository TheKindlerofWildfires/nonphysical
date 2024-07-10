use crate::filter::gaussian::GaussianFilter;
use crate::linear::gemm::Gemm;
use crate::linear::svd::SingularValueDecomposition;
/* 
use crate::neural::layer::{IdentityLayer, Layer, LayerType, PerceptronLayer};
use crate::neural::network::{self, Network};*/
use crate::random::pcg::PermutedCongruentialGenerator;
use crate::shared::complex::Complex;
use crate::shared::float::Float;
use crate::shared::matrix::{self, Matrix};
use crate::signal::fourier::FastFourierTransform;
use crate::signal::gabor::GaborTransform;

use std::io::Write;
use std::time::SystemTime;

pub fn play_fourier() {
    let mut signal: Vec<Complex<f32>> = (0..8)
        .map(|i| Complex::<f32>::new(i as f32, (i + 1) as f32))
        .collect();

    let fft_l = FastFourierTransform::new(signal.len());
    fft_l.fft(&mut signal);

    let now: SystemTime = SystemTime::now();

    let fft_r = FastFourierTransform::new(256);
    for j in 0..256 {
        let signal: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::<f32>::new((i + j) as f32, (i + 1) as f32))
            .collect();
        fft_r.fft(&mut signal.into_boxed_slice());
    }
    println!("Fourier time {:?} ", now.elapsed().unwrap());
}

pub fn play_gabor() {
    let mut file = std::fs::File::open("spectest.np").unwrap();
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
    let mut signal = Vec::new();
    buffer.chunks_exact(8).for_each(|chunk| {
        let vb: [u8; std::mem::size_of::<f64>()] = chunk.try_into().unwrap();
        let value = f64::from_le_bytes(vb);
        signal.push(Complex::<f32>::new(value as f32, 0.0));
    });
    let now: SystemTime = SystemTime::now();

    let sigma: Vec<f32> = vec![f32::float(12.0)];
    let window: Vec<Complex<f32>> =  <Vec<Complex<f32>> as GaussianFilter<f32>>::window(vec![256], sigma);
    println!("Window time {:?} ", now.elapsed().unwrap());

    let now: SystemTime = SystemTime::now();
    let gabor = GaborTransform::new(4, window);

    let ideal_len = gabor.square_len();
    dbg!(ideal_len);

    let spec = gabor.gabor(&mut signal[0..ideal_len]);
    //dbg!(spec);

    println!("Gabor time {:?} ", now.elapsed().unwrap());

    let mut file = std::fs::File::create("spec.np").unwrap();
    for value in spec.data {
        file.write_all(&(value.real * value.real + value.imag * value.imag).to_le_bytes())
            .unwrap();
    }
}

pub fn play_svd() {
    let ci = 10;
    let svd_data = (0..ci * ci)
        .map(|i| Complex::<f32>::new(i as f32, 0.0))
        .collect::<Vec<_>>();
    //dbg!(&svd_data);
    let mut svd_mat = Matrix::<f32>::new(ci, svd_data);
    //dbg!(&svd_mat.data);
    let now: SystemTime = SystemTime::now();
    //let (_u, s, _v) = <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd_full(&mut svd_mat);
    let s = <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd(&mut svd_mat);

    println!("SVD time {:?} ", now.elapsed().unwrap());
    dbg!(&s[0..10]);
}
/* 
pub fn play_neural() {
    println!("Started neural network");
    let layer_shape = vec![2, 3, 2];
    let layer_map = vec![
        LayerType::Identity,
        LayerType::PerceptronLayer,
        LayerType::PerceptronLayer,
    ];

    let direct_layers: Vec<Box<dyn Layer<f32>>> = vec![
        Box::new(IdentityLayer::new(10, 10)),
        Box::new(PerceptronLayer::layer_one()),
        Box::new(PerceptronLayer::layer_two()),
    ];
    let mut network = Network::<f32>::new_direct(direct_layers, 0.01, 0.01, 200, 10000);
    //let mut network = Network::<f32>::new(layer_shape, layer_map, 0.01, 0.01, 10, 10000);
    println!("Network created");
    println!("Data created");

    let known_x = vec![
        Complex::new(0.19001768, 0.0),
        Complex::new(0.96972856, 0.0),
        Complex::new(1.68646301, 0.0),
        Complex::new(-0.12498708, 0.0),
        Complex::new(-0.97119129, 0.0),
        Complex::new(0.2908547, 0.0),
        Complex::new(2.15220755, 0.0),
        Complex::new(0.524335, 0.0),
        Complex::new(0.79587943, 0.0),
        Complex::new(0.77384165, 0.0),
        Complex::new(0.59170903, 0.0),
        Complex::new(-0.24813843, 0.0),
        Complex::new(1.06261354, 0.0),
        Complex::new(-0.67081915, 0.0),
        Complex::new(-1.21770474, 0.0),
        Complex::new(0.8378305, 0.0),
        Complex::new(1.17288724, 0.0),
        Complex::new(-0.148433, 0.0),
        Complex::new(0.45395092, 0.0),
        Complex::new(0.20912687, 0.0),
    ];

    let known_y = vec![
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ];

    let (x_mat, y_mat) = simulate_moons();
    //let mut x_mat = Matrix::new(10,known_x);
    //let y_mat = Matrix::new(10,known_y);
    network.fit(&x_mat, &y_mat);

    //let mut x_mat = Matrix::new(10,known_x);
    //let y_mat = Matrix::new(10,known_y);

    //network.predict(&mut x_mat);
    dbg!(y_mat);
    println!("Network fitted");
}

pub fn simulate_moons() -> (Matrix<f32>, Matrix<f32>) {
    let mut file = std::fs::File::open("x.np").unwrap();
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
    let mut x = Vec::new();
    buffer.chunks_exact(4).for_each(|chunk| {
        let vb: [u8; std::mem::size_of::<f32>()] = chunk.try_into().unwrap();
        let value = f32::from_le_bytes(vb);
        x.push(Complex::<f32>::new(value as f32, 0.0));
    });
    let x_matrix = Matrix::<f32>::new(200, x);

    let mut file = std::fs::File::open("y.np").unwrap();
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
    let mut y = Vec::new();
    buffer.chunks_exact(4).for_each(|chunk| {
        let vb: [u8; std::mem::size_of::<f32>()] = chunk.try_into().unwrap();
        let value = f32::from_le_bytes(vb);
        y.push(Complex::<f32>::new(value as f32, 0.0));
    });

    let y_matrix = Matrix::<f32>::new(200, y);

    (x_matrix, y_matrix)
}
*/
pub fn play_matrix() {
    let mut pcg = PermutedCongruentialGenerator::<f32>::new_timed();
    for n in [2, 10, 20, 100, 200, 1000,2000] {
        let data = (0..n * n).map(|_| Complex::<f32>::new(pcg.next_u32() as f32, 0.0));
        let m1 = Matrix::new(n, data.collect());
        let now = SystemTime::now();
        <matrix::Matrix<f32> as Gemm<f32>>::gemm(&m1,&m1);
        //m1.gemm(&m1);

        println!("Time {:?} {}", now.elapsed().unwrap(), n);
    }
}

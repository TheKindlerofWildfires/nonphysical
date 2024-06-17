use crate::linear::svd::SingularValueDecomposition;
use crate::neural::network::Network;
use crate::shared::complex::Complex;
use crate::shared::matrix::Matrix;
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

    let window: Vec<Complex<f32>> = GaborTransform::gaussian(256, 12.0);
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
    let ci = 1000;
    let svd_data = (0..ci * ci)
        .map(|i| Complex::<f32>::new(i as f32, 0.0))
        .collect::<Vec<_>>();
    //dbg!(&svd_data);
    let mut svd_mat = Matrix::<f32>::new(ci, svd_data);
    //dbg!(&svd_mat.data);
    let now: SystemTime = SystemTime::now();
    let (u, s, v) = <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd_full(&mut svd_mat);
    //let s = <Matrix<f32> as SingularValueDecomposition<f32>>::jacobi_svd(&mut svd_mat);

    println!("SVD time {:?} ", now.elapsed().unwrap());
    dbg!(&s[0..10]);
}

pub fn play_neural(){
    println!("Started neural network");
    let layer_shape = vec![1,3,3,10];
    let mut network = Network::<f32>::new(layer_shape,0.1,0.1,10,10);
    println!("Network created");

    let labels:Vec<_> = (0..10).map(|i| i).collect();
    let data_vec:Vec<_> = (0..10).map(|i| Complex::<f32>::new((i as f32)*2.0+1.0,0.0)).collect();
    let data_mat = Matrix::<f32>::new(10,data_vec);
    println!("Data created");

    network.fit(&data_mat, &labels);
    println!("Network fitted");
}

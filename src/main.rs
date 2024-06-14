#![forbid(unsafe_code)]
#![allow(dead_code)]
use std::io::Write;
use std::process::exit;
use std::time::SystemTime;

use linear::svd::SingularValueDecomposition;
use shared::complex::Complex;
use shared::matrix::Matrix;
use signal::fourier::FastFourierTransform;
use signal::gabor::GaborTransform;

pub mod linear;
pub mod shared;
pub mod signal;

fn main() {
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

    //write it back to the file
    /*
    let mut file = std::fs::File::create("spec.np").unwrap();
    for value in spec.data {
        file.write_all(&(value.real * value.real + value.imag * value.imag).to_le_bytes())
            .unwrap();
    }*/

    let ci = 10000;
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

    //dbg!(&u.data);
    //dbg!(&v.data);

    //perf notes -> it's about 10x faster
    /*
    size    |    lin alg    |   lins    |   rust
    10      |    0.0        |   0.0     |   21us
    100     |    0.0063     |   0.003   |   509us 
    1000    |    0.7        |   0.6     |   27 ms
    2000    |    2.4        |           | 141 ms
    3000    |    5.7        |           | 366 ms
    5000    |    19s        |  16s      | 1.28s (12/10)
    10000   |    92hr?      |           | 5.8 s
    */
}

use std::io::Write;
use std::time::SystemTime;

use shared::complex::Complex64;
use signal::fourier::FastFourierTransform;
use signal::gabor::GaborTransform;

pub mod shared;
pub mod signal;

fn main() {
    let mut signal: Vec<Complex64> = (0..16)
        .map(|i| Complex64::new(i as f32, (i + 1) as f32))
        .collect();
    let fft_l = FastFourierTransform::new(signal.len());
    fft_l.fft(&mut signal);

    let now: SystemTime = SystemTime::now();
    let fft_r = FastFourierTransform::new(256);
    for j in 0..256 {
        let mut signal: Vec<Complex64> = (0..256)
            .map(|i| Complex64::new((i + j) as f32, (i + 1) as f32))
            .collect();
        fft_r.fft(&mut signal);
    }
    println!("Fourier time {:?} ", now.elapsed().unwrap());

    let mut file = std::fs::File::open("spectest.np").unwrap();
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).unwrap();
    let mut signal = Vec::new();
    buffer.chunks_exact(8).for_each(|chunk| {
        let vb: [u8; std::mem::size_of::<f64>()] = chunk.try_into().unwrap();
        let value = f64::from_le_bytes(vb);
        signal.push(Complex64::new(value as f32, 0.0));
    });
    let now: SystemTime = SystemTime::now();

    let window: Vec<Complex64> = GaborTransform::gaussian(256, 12.0);

    let gabor = GaborTransform::new(1, window);

    let spec = gabor.gabor(&mut signal);
    //dbg!(spec);

    println!("Gabor time {:?} ", now.elapsed().unwrap());

    //write it back to the file

    let mut file = std::fs::File::create("spec.np").unwrap();
    for value in spec {
        file.write_all(&(value.real * value.real + value.imag * value.imag).to_le_bytes())
            .unwrap();
    }
}

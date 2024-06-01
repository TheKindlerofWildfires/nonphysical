use std::time::SystemTime;

use shared::complex::Complex64;
use signal::fft;

pub mod shared;
pub mod signal;

fn main() {
    let mut signal : Vec<Complex64> = (0..16).map(|i| Complex64::new(i as f32,(i+1) as f32)).collect();
    let fft_l = fft::FastFourierTransform::new(signal.len());
    fft_l.fft(&mut signal);

    let now: SystemTime = SystemTime::now();
    let fft_r =fft::FastFourierTransform::new(256);
    for j in 0..256{
        let mut signal : Vec<Complex64> = (0..256).map(|i| Complex64::new((i+j) as f32,(i+1) as f32)).collect();
        fft_r.fft(&mut signal);
    }
    dbg!(now.elapsed().unwrap());
}
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
//starting speed: 14.5-15.8ms
//faster twiddles: 14.8-15.2ms
//assert removal: 14.7-14.9 ms
//if removal: 14.7-15.4
//with dbg: 15.7-17.4
//complex1: 8.57 ms
//complex2: 8.4-9.5 ms (degreded in fft_n)
//cache twiddle: 7.6-8.2
//optimizations: 0.5
//twiddle loop move: 390 us
//more opts: 331 us
